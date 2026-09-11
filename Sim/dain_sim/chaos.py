"""dain_sim.chaos — deterministic fault-injection harness (S7).

Drives a real coordinator + N real node agents through a kill schedule and
reports whether the cluster met a fault-tolerance expectation (spec §13):

- `expect="complete"` — an in-flight job survives the loss of a stage node and
  completes via stage retry / backup reassignment.
- `expect="degraded"` — after nodes are killed the pool (still above `min_nodes`)
  keeps serving new jobs in degraded mode (fewer, larger stages).
- `expect="reject"`   — after the pool drops below `min_nodes`, new jobs get a
  clean 429/503 instead of hanging.

Every scenario also checks retry idempotency: the `(stage_idx, attempt)` keys
recorded across the run are unique, so no duplicate ledger keys would be emitted
when the accounting ledger lands (S8, §13/§15).

Usage: uv run python -m dain_sim.chaos --nodes 6 --kill-at 5s:node3 --expect-complete
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time

import httpx
from dain_common.schemas import NodeState
from dain_coordinator.settings import CoordinatorSettings
from dain_node.shard_export import DEV_MODEL_ID, export_tiny_llama

from dain_sim.dev import ADMIN_HEADERS, API_KEY, JOIN_TOKEN, ADMIN_KEY
from dain_sim.server import start_server, stop_server


def _spawn_env(port: int, workdir: str, node_id: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.update(
        {
            "DAIN_COORD_URL": f"ws://127.0.0.1:{port}",
            "DAIN_JOIN_TOKEN": JOIN_TOKEN,
            "DAIN_NODE_ID": node_id,
            "DAIN_HEARTBEAT_S": "0.5",
            "DAIN_NODE_STATE_PATH": os.path.join(workdir, f"{node_id}_state.json"),
            "DAIN_MODEL": DEV_MODEL_ID,
            "DAIN_MODEL_CACHE": os.path.join(workdir, f"{node_id}_cache"),
        }
    )
    return env


async def _wait_connected(client, base_url: str, count: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        listing = (await client.get(f"{base_url}/admin/nodes", headers=ADMIN_HEADERS)).json()
        if sum(1 for n in listing if n.get("connected")) >= count:
            return
        await asyncio.sleep(0.2)
    raise AssertionError(f"only {count} nodes not all connected in time")


async def _job_view(client, base_url: str, job_id: str) -> dict:
    resp = await client.get(f"{base_url}/v1/jobs/{job_id}", headers={"X-API-Key": API_KEY})
    return resp.json()


def _attempt_keys(job: dict) -> list[tuple[int, int]]:
    """Explicit (stage_idx, attempt) keys recorded for the job's stages.

    Each retry of a stage contributes one key. The ledger (S8) dedupes on
    `(job_id, stage_idx, attempt)`, so uniqueness here guarantees no duplicate
    ledger entries from a retry storm.
    """
    keys: list[tuple[int, int]] = []
    for stage in job.get("stages", []):
        keys.append((stage["stage_idx"], 0))
    for stage_idx_str, count in (job.get("retries") or {}).items():
        for attempt in range(1, int(count) + 1):
            keys.append((int(stage_idx_str), attempt))
    return keys


async def _reconcile_ledger(
    client, base_url: str, job_id: str, attempt_keys: list, stages: list, retries: dict
) -> dict:
    """Cross-check the coordinator's accounting ledger (S8, §15) against the job
    view: exactly one SUCCESS per (job, stage), earlier attempts RETRIED_AWAY,
    rows unique + verified, and the summary/export APIs consistent."""
    headers = {"X-API-Key": API_KEY}
    summary = (await client.get(f"{base_url}/ledger/summary", headers=headers)).json()
    rows = (
        await client.post(f"{base_url}/ledger/export", headers=headers, json={"format": "json"})
    ).json()
    csv = await client.post(f"{base_url}/ledger/export", headers=headers, json={"format": "csv"})

    job_rows = [r for r in rows if r["job_id"] == job_id]
    success = [r for r in job_rows if r["outcome"] == "success"]
    retried_away = [r for r in job_rows if r["outcome"] == "retried_away"]
    failed = [r for r in job_rows if r["outcome"] == "failed"]
    keys_rows = [(r["stage_idx"], r["attempt"]) for r in job_rows]
    per_stage_success: dict[int, int] = {}
    for r in success:
        per_stage_success[r["stage_idx"]] = per_stage_success.get(r["stage_idx"], 0) + 1
    node_credits = {n["node_id"]: n["credits"] for n in summary["nodes"]}

    ok = (
        summary["totals"]["events"] == len(rows)
        and abs(sum(node_credits.values()) - sum(r["credit"] for r in rows)) < 1e-6
        and len(job_rows) == len(attempt_keys)
        and len(keys_rows) == len(set(keys_rows))
        and len(per_stage_success) == len(stages)
        and all(c == 1 for c in per_stage_success.values())
        and all(r["credit"] > 0 for r in success)
        and all(r["credit"] == 0.0 for r in failed)
        and all(r["verified"] is True for r in job_rows)
        and csv.status_code == 200
        and csv.text.startswith("node_id,job_id")
    )
    return {
        "ok": ok,
        "attempt_keys": attempt_keys,
        "events": len(rows),
        "job_events": len(job_rows),
        "success": len(success),
        "retried_away": len(retried_away),
        "failed": len(failed),
        "per_stage_success": per_stage_success,
        "unique_keys": len(keys_rows) == len(set(keys_rows)),
        "node_credits": node_credits,
        "csv_status": csv.status_code,
    }


async def run_chaos(
    *,
    node_count: int,
    expect: str,
    min_nodes: int = 1,
    job_tokens: int = 24,
    layers_per_node_target: int = 4,
    prompt: str = "Once upon a time",
    workdir_root: str | None = None,
    kill_at_s: float = 0.0,
    kill_target: str = "sampling",
) -> dict:
    root = workdir_root or tempfile.mkdtemp(prefix="dain-chaos-")
    store_dir = os.path.join(root, "model_store")
    export_tiny_llama(store_dir)

    settings = CoordinatorSettings(
        db_path=os.path.join(root, "coordinator.sqlite3"),
        model_store_dir=store_dir,
        heartbeat_interval_s=0.5,
        offline_after_missed=3,
        monitor_tick_s=0.2,
        watchdog_tick_s=0.2,
        api_key=API_KEY,
        admin_api_key=ADMIN_KEY,
        job_timeout_s=90.0,
        min_nodes=min_nodes,
        layers_per_node_target=layers_per_node_target,
        max_stage_attempts=3,
        max_job_restarts=2,
    )
    server = await start_server(settings)
    procs: dict[str, subprocess.Popen] = {}
    report: dict = {"root": root}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for i in range(node_count):
                node_id = f"node-{i:02d}"
                node_log = open(os.path.join(root, f"{node_id}.log"), "w", encoding="utf-8")
                procs[node_id] = subprocess.Popen(
                    [sys.executable, "-m", "dain_node"],
                    env=_spawn_env(server.port, root, node_id),
                    cwd=root,
                    stdout=node_log,
                    stderr=node_log,
                )
                procs[node_id]._node_log = node_log  # type: ignore[attr-defined]
            await _wait_connected(client, server.base_url, node_count, 40.0)

            if expect == "complete":
                report.update(
                    await _scenario_complete(
                        client,
                        server.base_url,
                        procs,
                        job_tokens,
                        prompt,
                        kill_at_s=kill_at_s,
                        kill_target=kill_target,
                    )
                )
            elif expect == "degraded":
                report.update(
                    await _scenario_degraded(client, server.base_url, procs, job_tokens, prompt)
                )
            else:  # reject / baseline
                report.update(
                    await _scenario_reject(client, server.base_url, procs, job_tokens, prompt)
                )
        report["ok"] = True
    finally:
        for proc in procs.values():
            with contextlib.suppress(Exception):
                proc.kill()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(lambda: [p.wait() for p in procs.values()])
        for proc in procs.values():
            with contextlib.suppress(Exception):
                getattr(proc, "_node_log", None).close()  # type: ignore[attr-defined]
        await stop_server(server)
    return report


async def _scenario_complete(
    client, base_url, procs, job_tokens, prompt, kill_at_s: float = 0.0, kill_target: str = "sampling"
) -> dict:
    """Kill a stage node mid-job; the job must survive (stage retry onto a
    backup) and reach COMPLETED.

    With `kill_target == "sampling"` (the historical behavior) the victim is the
    node currently running the final sampling stage, chosen as soon as the first
    token streams. `--kill-at TIME:NODE` instead waits TIME into the stream and
    kills the named node, exercising the same retry path from a different point.
    """
    job_id: str | None = None
    text: list[str] = []
    final: dict = {}
    killed = False
    result: dict = {}
    stream_start = time.monotonic()

    async with client.stream(
        "POST",
        f"{base_url}/v1/completions",
        headers={"X-API-Key": API_KEY},
        json={"model_id": DEV_MODEL_ID, "prompt": prompt, "max_tokens": job_tokens, "stream": True},
    ) as response:
        status = response.status_code
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            frame = json.loads(data)
            job_id = job_id or frame.get("job_id")
            if frame.get("type") == "token":
                text.append(frame["token"])
                if job_id and not killed:
                    if kill_target == "sampling":
                        victim = await _sampling_node(client, base_url, job_id)
                    else:
                        victim = kill_target
                    if kill_at_s > 0:
                        wait = kill_at_s - (time.monotonic() - stream_start)
                        if wait > 0:
                            await asyncio.sleep(wait)
                    killed = True
                    if victim in procs:
                        procs[victim].kill()
                        await asyncio.to_thread(procs[victim].wait)
            elif frame.get("type") in ("final", "error"):
                final = frame
    result["status"] = status
    result["job_id"] = job_id or ""
    result["text"] = "".join(text)
    result["final"] = final

    job = await _job_view(client, base_url, job_id or "")
    keys = _attempt_keys(job)
    result["job_final_state"] = job.get("state")
    result["stages"] = job.get("stages")
    result["retries"] = job.get("retries")
    result["attempt_keys_unique"] = len(keys) == len(set(keys))
    result["attempt_keys"] = keys
    result["completed"] = job.get("state") == "completed"
    result["ledger"] = await _reconcile_ledger(
        client, base_url, job_id or "", keys, job.get("stages") or [], job.get("retries") or {}
    )
    return {"scenario": "complete", **result}


async def _sampling_node(client, base_url: str, job_id: str) -> str:
    job = (await client.get(f"{base_url}/v1/jobs/{job_id}", headers={"X-API-Key": API_KEY})).json()
    if job.get("stages"):
        return job["stages"][-1]["node_id"]
    return ""


async def _stream_and_view(client, base_url, tokens, prompt) -> tuple[int, dict]:
    """POST a streaming completion to completion; returns (status, job_view)."""
    job_id: str | None = None
    final_text: list[str] = []
    async with client.stream(
        "POST",
        f"{base_url}/v1/completions",
        headers={"X-API-Key": API_KEY},
        json={"model_id": DEV_MODEL_ID, "prompt": prompt, "max_tokens": tokens, "stream": True},
    ) as response:
        status = response.status_code
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            frame = json.loads(data)
            job_id = job_id or frame.get("job_id")
            if frame.get("type") == "token":
                final_text.append(frame["token"])
    job_id_str = job_id or ""
    job = await _job_view(client, base_url, job_id_str)
    return status, job


async def _scenario_degraded(client, base_url, procs, job_tokens, prompt) -> dict:
    """Kill 3 of 8 nodes, then serve a new job in degraded mode (fewer, larger
    stages) — still ON, above min_nodes."""
    victims = ["node-" + f"{i:02d}" for i in range(3)]
    for node_id in victims:
        if node_id in procs:
            procs[node_id].kill()
            await asyncio.to_thread(procs[node_id].wait)
    await asyncio.sleep(2.0)  # let recompute + OFFLINE settle
    status, job = await _stream_and_view(client, base_url, job_tokens, prompt)
    return {
        "scenario": "degraded",
        "status": status,
        "served": status == 200 and job.get("state") == "completed",
        "job_final_state": job.get("state"),
        "stages": job.get("stages"),
    }


async def _scenario_reject(client, base_url, procs, job_tokens, prompt) -> dict:
    """Kill all nodes; below min_nodes a new job must get a clean 429/503."""
    for node_id in list(procs.keys()):
        if node_id in procs:
            procs[node_id].kill()
            await asyncio.to_thread(procs[node_id].wait)
    await asyncio.sleep(2.0)
    listing = (await client.get(f"{base_url}/admin/nodes", headers=ADMIN_HEADERS)).json()
    online = sum(1 for n in listing if n["state"] == NodeState.ONLINE.value)
    status, _job = await _stream_and_view(client, base_url, 4, prompt)
    return {
        "scenario": "reject",
        "online_after": online,
        "status": status,
        "rejected_cleanly": status in (429, 503),
    }


def _parse_kill_at(spec: str) -> tuple[float, str]:
    """Parse --kill-at TIME:NODE where NODE is a node id or the special marker
    "sampling" (the node running the final sampling stage). Returns (seconds, target)."""
    if ":" in spec:
        raw_time, target = spec.split(":", 1)
    else:
        raw_time, target = spec, "sampling"
    low = raw_time.strip().lower()
    try:
        if low.endswith("ms"):
            kill_at_s = float(low[:-2]) / 1000.0
        elif low.endswith("m"):
            kill_at_s = float(low[:-1]) * 60.0
        elif low.endswith("s") or low.endswith("sec"):
            kill_at_s = float(low.removesuffix("sec").removesuffix("s"))
        else:
            kill_at_s = float(low)
    except ValueError:
        raise ValueError(f"--kill-at has an invalid TIME: {raw_time!r} (use e.g. 1s:node-03)") from None
    return kill_at_s, target.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="DAIN chaos harness (deterministic kills)")
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument(
        "--kill-at",
        type=str,
        default="1s:sampling",
        help="TIME:NODE — kill the named node (or `sampling`) that far into the "
             "job stream (e.g. 2s:node-03); `sampling` picks the final-stage node "
             "on the first token",
    )
    parser.add_argument("--expect", choices=["complete", "degraded", "reject"], default="complete")
    parser.add_argument("--tokens", type=int, default=40)
    args = parser.parse_args()
    try:
        kill_at_s, kill_target = _parse_kill_at(args.kill_at)
    except ValueError as exc:
        print(f"[chaos] {exc}", file=sys.stderr)
        return 2

    result = asyncio.run(
        run_chaos(
            node_count=args.nodes,
            expect=args.expect,
            job_tokens=args.tokens,
            kill_at_s=kill_at_s,
            kill_target=kill_target,
        )
    )
    expected_key = {"complete": "completed", "degraded": "served", "reject": "rejected_cleanly"}[
        args.expect
    ]
    print(json.dumps(result, indent=2))
    return 0 if result.get(expected_key) else 1


if __name__ == "__main__":
    raise SystemExit(main())
