"""S5 checkpoint: distributed pipeline parity — 4 node agents ≡ single node ≡ HF.

Runs one coordinator twice within the same store: first with one agent (whole
model), then with four agents (16 layers → 4 stages × 4 layers, chosen by
throughput-proportional sizing). Greedy generation must produce identical
token streams, and the job view must record per-stage latencies.
"""

import asyncio
import os
import subprocess
import sys
import time

import httpx
import torch
from dain_common.schemas import NodeState
from dain_coordinator.settings import CoordinatorSettings
from dain_node.shard_export import DEV_MODEL_ID, export_tiny_llama, tiny_config

from dain_sim.server import start_server, stop_server

JOIN_TOKEN = "dain-dev-join-token"
API_KEY = "dain-dev-key"
ADMIN_KEY = "dain-dev-admin-key"
ADMIN_HEADERS = {"X-Admin-Key": ADMIN_KEY}
PROMPT = "Once upon a time"
MAX_TOKENS = 24


def hf_reference_ids(prompt: str, steps: int) -> list[int]:
    """Cache-free greedy reference straight from the HF model (seed 1 weights)."""
    torch.manual_seed(1)
    model = __import__("transformers.models.llama.modeling_llama", fromlist=["x"]).LlamaForCausalLM(
        tiny_config()
    )
    model.eval()
    current = list(prompt.encode("utf-8"))
    out = []
    with torch.no_grad():
        for _ in range(steps):
            token = int(torch.argmax(model(torch.tensor([current])).logits[0, -1]).item())
            out.append(token)
            current.append(token)
    return out


def spawn_agent(server_port: int, workdir, node_id: str) -> subprocess.Popen:
    env = dict(
        os.environ,
        DAIN_COORD_URL=f"ws://127.0.0.1:{server_port}",
        DAIN_JOIN_TOKEN=JOIN_TOKEN,
        DAIN_NODE_ID=node_id,
        DAIN_HEARTBEAT_S="0.5",
        DAIN_NODE_STATE_PATH=str(workdir / f"{node_id}_state.json"),
        DAIN_MODEL=DEV_MODEL_ID,
        DAIN_MODEL_CACHE=str(workdir / f"{node_id}_cache"),
    )
    return subprocess.Popen([sys.executable, "-m", "dain_node"], cwd=workdir, env=env)


async def wait_connected(
    client: httpx.AsyncClient, server, count: int, timeout_s: float = 30.0
) -> None:
    """Nodes must not only be registered but have their WS attached (S5 dispatch)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        listing = (await client.get(f"{server.base_url}/admin/nodes", headers=ADMIN_HEADERS)).json()
        connected = sum(1 for n in listing if n.get("connected"))
        if connected >= count:
            return
        await asyncio.sleep(0.25)
    raise AssertionError(f"only {connected}/{count} nodes with live WS in time")


async def wait_online(
    client: httpx.AsyncClient, server, count: int, timeout_s: float = 25.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        listing = (await client.get(f"{server.base_url}/admin/nodes", headers=ADMIN_HEADERS)).json()
        online = sum(1 for n in listing if n["state"] == NodeState.ONLINE.value)
        if online >= count:
            return
        await asyncio.sleep(0.25)
    raise AssertionError(f"only {online}/{count} nodes ONLINE in time")


async def complete(client: httpx.AsyncClient, server, *, stream: bool) -> tuple[str, str]:
    if stream:
        frames = []
        async with client.stream(
            "POST",
            f"{server.base_url}/v1/completions",
            headers={"X-API-Key": API_KEY},
            json={"model_id": DEV_MODEL_ID, "prompt": PROMPT, "max_tokens": MAX_TOKENS},
        ) as response:
            assert response.status_code == 200
            job_id = None
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                frame = __import__("json").loads(line[len("data: ") :])
                job_id = job_id or frame.get("job_id")
                frames.append(frame)
        text = "".join(f["token"] for f in frames if f.get("type") == "token")
        return text, job_id
    response = await client.post(
        f"{server.base_url}/v1/completions",
        headers={"X-API-Key": API_KEY},
        json={
            "model_id": DEV_MODEL_ID,
            "prompt": PROMPT,
            "max_tokens": MAX_TOKENS,
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    return body["text"], body["job_id"]


def test_pipeline_parity_four_agents(tmp_path) -> None:
    async def main() -> None:
        store_dir = tmp_path / "model_store"
        export_tiny_llama(str(store_dir))
        reference_ids = hf_reference_ids(PROMPT, MAX_TOKENS)
        expected_text = bytes(reference_ids).decode("utf-8", errors="replace")

        settings = CoordinatorSettings(
            db_path=str(tmp_path / "coordinator.sqlite3"),
            model_store_dir=str(store_dir),
            heartbeat_interval_s=0.5,
            offline_after_missed=3,
            monitor_tick_s=0.25,
            api_key=API_KEY,
            admin_api_key=ADMIN_KEY,
            job_timeout_s=90.0,
            layers_per_node_target=4,
        )
        server = await start_server(settings)
        procs: list[subprocess.Popen] = []
        workdir = tmp_path / "agents"
        workdir.mkdir()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Single-node reference.
                procs.append(spawn_agent(server.port, workdir, "node-solo"))
                await wait_online(client, server, 1)
                text_ref, _ = await complete(client, server, stream=False)
                assert len(text_ref) >= 20

                # Replace the solo node with a 4-agent pool.
                procs[0].terminate()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    listing = (
                        await client.get(
                            f"{server.base_url}/admin/nodes",
                            headers=ADMIN_HEADERS,
                        )
                    ).json()
                    if all(n["state"] != "online" for n in listing):
                        break
                    await asyncio.sleep(0.2)
                for i in range(4):
                    procs.append(spawn_agent(server.port, workdir, f"node-dist-{i}"))
                await wait_connected(client, server, 4)

                # Distributed completion: 4 stages × 4 layers through the relay.
                text_dist, job_id = await complete(client, server, stream=True)
                assert text_dist == text_ref, (
                    f"distributed vs single-node mismatch:\n{text_dist!r}\n{text_ref!r}"
                )
                assert text_dist == expected_text, "distributed vs HF reference mismatch"

                view = (
                    await client.get(
                        f"{server.base_url}/v1/jobs/{job_id}", headers={"X-API-Key": API_KEY}
                    )
                ).json()
                assert len(view["stages"]) == 4, view["stages"]
                assert [s["layer_start"] for s in view["stages"]] == [0, 4, 8, 12]
                assert all(s["latency_ms"] is not None for s in view["stages"])
        finally:
            for proc in procs:
                proc.terminate()
            await asyncio.to_thread(lambda: [p.wait() for p in procs])
            await stop_server(server)

    asyncio.run(main())
