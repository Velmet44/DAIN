"""dain_sim.cluster — one-command local cluster (S3).

Spawns a real coordinator (in-process uvicorn) plus N real node-agent
subprocesses, watches the registry for the requested duration, shuts everything
down gracefully (SIGTERM/CTRL_BREAK → agents deregister), prints a health
summary, and exits 0 only if the cluster was healthy throughout.

With `--chat` it instead runs an interactive REPL: prompts are streamed through
the same distributed pipeline (`/v1/completions`, SSE) the coordinator serves.

Usage: uv run python -m dain_sim.cluster --nodes 4 --duration 60
       uv run python -m dain_sim.cluster --nodes 6 --chat
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import httpx
from dain_common.logging_setup import configure_logging
from dain_common.schemas import NodeState
from dain_coordinator.settings import CoordinatorSettings
from dain_node.shard_export import DEV_MODEL_ID, export_tiny_llama

from dain_sim.dev import ADMIN_HEADERS, ADMIN_KEY, API_KEY, JOIN_TOKEN
from dain_sim.server import start_server, stop_server


@dataclass
class NodeProc:
    node_id: str
    workdir: str
    process: subprocess.Popen
    saw_offline: bool = False
    final_state: str = "?"
    transitions: int = 0
    heartbeats: int = 0
    score: float | None = None
    shutdown_reason: str | None = None


@dataclass
class ClusterReport:
    healthy: bool
    reason: str
    nodes: list[NodeProc] = field(default_factory=list)


def _spawn_flags() -> int:
    return subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0


def _graceful_stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        with contextlib.suppress(OSError, ValueError):
            proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.terminate()


def _node_env(
    port: int, node_id: str, workdir: str, heartbeat_s: float, idx: int
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "DAIN_COORD_URL": f"ws://127.0.0.1:{port}",
            "DAIN_JOIN_TOKEN": JOIN_TOKEN,
            "DAIN_NODE_ID": node_id,
            "DAIN_HEARTBEAT_S": str(heartbeat_s),
            "DAIN_NODE_STATE_PATH": os.path.join(workdir, "node_state.json"),
            "DAIN_MODEL": DEV_MODEL_ID,
            "DAIN_MODEL_CACHE": os.path.join(workdir, "shard_cache"),
            # Varied reported bandwidth exercises score differentiation.
            "DAIN_NET_BW_MBPS": str(150 + idx * 25),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return env


def _print_result(
    code: int, reason: str, procs: list[NodeProc], keep_dir: bool, workdir_root: str
) -> None:
    print("\n== health summary ==")
    print(f"{'node':10s} {'final':9s} {'transitions':11s} {'heartbeats':10s} {'score':6s} shutdown")
    for proc in procs:
        score = f"{proc.score:.3f}" if proc.score is not None else "-"
        print(
            f"{proc.node_id:10s} {proc.final_state:9s} {proc.transitions:<11d} "
            f"{proc.heartbeats:<10d} {score:6s} {proc.shutdown_reason}"
        )
    print(f"\n== result: {'HEALTHY' if code == 0 else 'UNHEALTHY'} ({reason}) ==")
    if keep_dir:
        print(f"[cluster] artifacts kept in {workdir_root}")
    else:
        with contextlib.suppress(OSError):
            for proc in procs:
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(proc.workdir, "node_state.json"))
                with contextlib.suppress(OSError):
                    os.rmdir(proc.workdir)
            os.rmdir(workdir_root)


async def _summarize(server, procs: list[NodeProc]) -> ClusterReport:
    async with httpx.AsyncClient(timeout=5.0) as client:
        for proc in procs:
            detail = (
                await client.get(
                    f"{server.base_url}/admin/nodes/{proc.node_id}",
                    headers=ADMIN_HEADERS,
                )
            ).json()
            proc.final_state = detail["state"]
            proc.transitions = len(detail["history"])
            proc.heartbeats = (detail["last_seq"] or -1) + 1
            proc.score = detail["score"]
            reasons = [h["reason"] for h in detail["history"] if h["to_state"] == "offline"]
            proc.shutdown_reason = reasons[-1] if reasons else None

    all_heartbeating = all(p.heartbeats > 0 for p in procs)
    clean_shutdown = all(p.shutdown_reason == "deregistered" for p in procs)
    no_spurious = all(not p.saw_offline for p in procs)
    if not all_heartbeating:
        return ClusterReport(False, "some nodes never heartbeated", procs)
    if not no_spurious:
        return ClusterReport(False, "spurious OFFLINE flips during the run", procs)
    if not clean_shutdown:
        return ClusterReport(False, "some nodes failed to deregister cleanly", procs)
    return ClusterReport(True, f"{len(procs)}/{len(procs)} nodes stable, clean shutdown", procs)


async def _wait_online(client, base_url: str, timeout_s: float = 40.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        listing = (await client.get(f"{base_url}/admin/nodes", headers=ADMIN_HEADERS)).json()
        if sum(1 for n in listing if n["state"] == NodeState.ONLINE.value) >= 1:
            return True
        await asyncio.sleep(0.5)
    return False


async def _chat(client, base_url: str, max_tokens: int) -> None:
    """Interactive REPL: stream each prompt through the real cluster pipeline."""
    print("\n== chat ==")
    print(
        f"model={DEV_MODEL_ID} (hermetic test weights — output is proto-text), "
        f"max_tokens={max_tokens}"
    )
    print("live across the spawned nodes; empty prompt quits.\n")
    headers = {"X-API-Key": API_KEY}
    while True:
        try:
            prompt = await asyncio.to_thread(input, "> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            break
        frames: list[dict] = []
        async with client.stream(
            "POST",
            f"{base_url}/v1/completions",
            headers=headers,
            json={
                "model_id": DEV_MODEL_ID,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace").strip()
                print(f"[{response.status_code}] {body}")
                continue
            async for raw in response.aiter_lines():
                if not raw.startswith("data: "):
                    continue
                payload = raw[len("data: ") :]
                if payload == "[DONE]":
                    break
                frame = json.loads(payload)
                frames.append(frame)
                if frame.get("type") == "token" and frame.get("token"):
                    print(frame["token"], end="", flush=True)
        final = [f for f in frames if f.get("type") == "final"]
        usage = final[0].get("usage") or {} if final else {}
        print(f"\n[tokens={usage.get('tokens', len(frames))}]\n")


async def run_cluster(
    nodes_n: int,
    duration_s: float,
    heartbeat_s: float,
    keep_dir: bool,
    *,
    chat: bool = False,
    max_tokens: int = 40,
) -> int:
    workdir_root = tempfile.mkdtemp(prefix="dain-cluster-")
    configure_logging()

    store_dir = os.path.join(workdir_root, "model_store")
    export_tiny_llama(store_dir)

    settings = CoordinatorSettings(
        db_path=os.path.join(workdir_root, "coordinator.sqlite3"),
        model_store_dir=store_dir,
        heartbeat_interval_s=heartbeat_s,
        join_token=JOIN_TOKEN,
        api_key=API_KEY,
        admin_api_key=ADMIN_KEY,
    )
    server = await start_server(settings)
    print(f"[cluster] coordinator on :{server.port} (db={settings.db_path})")

    procs: list[NodeProc] = []
    python = sys.executable
    for idx in range(nodes_n):
        node_id = f"node-{idx:02d}"
        workdir = os.path.join(workdir_root, node_id)
        os.makedirs(workdir, exist_ok=True)
        proc = subprocess.Popen(
            [python, "-m", "dain_node"],
            cwd=workdir,
            env=_node_env(server.port, node_id, workdir, heartbeat_s, idx),
            creationflags=_spawn_flags(),
        )
        procs.append(NodeProc(node_id=node_id, workdir=workdir, process=proc))
        print(f"[cluster] spawned {node_id} pid={proc.pid}")

    stop_at = time.monotonic() + duration_s
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if chat:
                if not await _wait_online(client, server.base_url):
                    raise AssertionError("no node came ONLINE before the chat timeout")
                print(f"[cluster] {nodes_n} nodes ready — piping /v1/completions")
                await _chat(client, server.base_url, max_tokens)
            else:
                while time.monotonic() < stop_at:
                    await asyncio.sleep(min(2.0, max(0.5, duration_s / 20)))
                    response = await client.get(
                        f"{server.base_url}/admin/nodes", headers=ADMIN_HEADERS
                    )
                    listing = response.json()
                    states = {n["node_id"]: n["state"] for n in listing}
                    online = sum(1 for s in states.values() if s == NodeState.ONLINE.value)
                    time_left = stop_at - time.monotonic()
                    print(f"[cluster] t-{time_left:5.0f}s online={online}/{nodes_n}")
                    for proc in procs:
                        if states.get(proc.node_id) == NodeState.OFFLINE.value:
                            proc.saw_offline = True
    except (httpx.HTTPError, OSError, AssertionError) as exc:
        print(f"[cluster] ERROR: {exc}")
        await stop_server(server)
        _print_result(2, "chat/monitor failure", procs, keep_dir, workdir_root)
        return 2

    # Graceful shutdown: agents deregister on SIGTERM/CTRL_BREAK.
    for proc in procs:
        _graceful_stop(proc.process)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(p.process.poll() is None for p in procs):
        await asyncio.sleep(0.2)
    for proc in procs:
        if proc.process.poll() is None:
            proc.process.kill()
            print(f"[cluster] WARNING: {proc.node_id} ignored shutdown signal — killed")

    report = await _summarize(server, procs)
    await stop_server(server)
    code = 0 if report.healthy else 1
    _print_result(code, report.reason, report.nodes, keep_dir, workdir_root)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="DAIN local cluster (coordinator + N node agents)")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--duration", type=float, default=30.0, help="seconds to run")
    parser.add_argument("--heartbeat-s", type=float, default=5.0)
    parser.add_argument("--keep", action="store_true", help="keep temp artifacts (db, state files)")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="interactive REPL: stream prompts through the cluster pipeline",
    )
    parser.add_argument("--max-tokens", type=int, default=40, help="max tokens per chat reply")
    args = parser.parse_args()
    try:
        return asyncio.run(
            run_cluster(
                args.nodes,
                args.duration,
                args.heartbeat_s,
                keep_dir=args.keep,
                chat=args.chat,
                max_tokens=args.max_tokens,
            )
        )
    except KeyboardInterrupt:
        print("\n[cluster] interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
