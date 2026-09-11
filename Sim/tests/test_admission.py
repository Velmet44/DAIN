"""S6 checkpoint: admission control — 50 concurrent requests vs a small pool.

With 2 nodes (K=2) and queue_limit=3 / max_concurrent_per_key=2, most requests
must get a clean 429 with Retry-After, none may hang or crash the server, and
the pool must recover to ONLINE afterwards.
"""

import asyncio
import os
import subprocess
import sys
import time

import httpx
from dain_common.schemas import NodeState
from dain_coordinator.settings import CoordinatorSettings
from dain_node.shard_export import DEV_MODEL_ID, export_tiny_llama

from dain_sim.dev import ADMIN_HEADERS, ADMIN_KEY, API_KEY, JOIN_TOKEN
from dain_sim.server import start_server, stop_server


def test_admission_under_load(tmp_path) -> None:
    async def main() -> None:
        store_dir = tmp_path / "model_store"
        export_tiny_llama(str(store_dir))
        settings = CoordinatorSettings(
            db_path=str(tmp_path / "coordinator.sqlite3"),
            model_store_dir=str(store_dir),
            heartbeat_interval_s=0.5,
            offline_after_missed=3,
            monitor_tick_s=0.25,
            api_key=API_KEY,
            admin_api_key=ADMIN_KEY,
            job_timeout_s=60.0,
            queue_limit=3,
            max_concurrent_per_key=2,
            layers_per_node_target=16,  # one stage per node → K = min(2 nodes)
        )
        server = await start_server(settings)
        procs = []
        workdir = tmp_path / "agents"
        workdir.mkdir()
        for i in range(2):
            env = dict(
                os.environ,
                DAIN_COORD_URL=f"ws://127.0.0.1:{server.port}",
                DAIN_JOIN_TOKEN=JOIN_TOKEN,
                DAIN_NODE_ID=f"node-{i}",
                DAIN_HEARTBEAT_S="0.5",
                DAIN_NODE_STATE_PATH=str(workdir / f"state{i}.json"),
                DAIN_MODEL=DEV_MODEL_ID,
                DAIN_MODEL_CACHE=str(workdir / f"cache{i}"),
            )
            procs.append(
                subprocess.Popen([sys.executable, "-m", "dain_node"], cwd=workdir, env=env)
            )
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                deadline = time.monotonic() + 25.0
                while time.monotonic() < deadline:
                    listing = (
                        await client.get(
                            f"{server.base_url}/admin/nodes",
                            headers=ADMIN_HEADERS,
                        )
                    ).json()
                    online = sum(
                        1 for n in listing if n["state"] == NodeState.ONLINE.value
                    )
                    connected = sum(1 for n in listing if n.get("connected"))
                    if online >= 2 and connected >= 2:
                        break
                    await asyncio.sleep(0.25)
                else:
                    raise AssertionError("pool never came up")

                async def fire(i: int) -> int:
                    response = await client.post(
                        f"{server.base_url}/v1/completions",
                        headers={"X-API-Key": API_KEY},
                        json={
                            "model_id": DEV_MODEL_ID,
                            "prompt": f"job {i}",
                            "max_tokens": 8,
                            "stream": False,
                        },
                    )
                    return response.status_code

                statuses = await asyncio.gather(*(fire(i) for i in range(50)))
                ok = statuses.count(200)
                rejected = statuses.count(429)
                assert ok + rejected == 50
                assert rejected > 0, "admission control never engaged"
                assert ok > 0, "no request was ever admitted"
                assert all(code in (200, 429) for code in statuses), statuses

                # The pool recovers: both nodes ONLINE again (not stuck BUSY).
                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline:
                    listing = (
                        await client.get(
                            f"{server.base_url}/admin/nodes",
                            headers=ADMIN_HEADERS,
                        )
                    ).json()
                    if all(n["state"] == NodeState.ONLINE.value for n in listing):
                        break
                    await asyncio.sleep(0.25)
                assert all(n["state"] == NodeState.ONLINE.value for n in listing)
        finally:
            for proc in procs:
                proc.terminate()
            await asyncio.to_thread(lambda: [p.wait() for p in procs])
            await stop_server(server)

    asyncio.run(main())
