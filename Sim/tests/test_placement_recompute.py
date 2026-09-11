"""S6 checkpoint: placement recompute — a dead node is excluded from new
placements while previously placed jobs are unaffected."""

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


def test_placement_recompute_after_node_loss(tmp_path) -> None:
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
            # 4 stages do real CPU inference; on a 4-core host full-suite
            # contention can stretch a non-streaming run well past the default
            # 60 s — match the S5 parity test's generous bound (S5 lesson: tests
            # wait on state with generous bounds, never on sleep()).
            job_timeout_s=90.0,
            layers_per_node_target=4,
        )
        server = await start_server(settings)
        procs: dict[str, subprocess.Popen] = {}
        workdir = tmp_path / "agents"
        workdir.mkdir()

        def spawn(node_id: str) -> subprocess.Popen:
            env = dict(
                os.environ,
                DAIN_COORD_URL=f"ws://127.0.0.1:{server.port}",
                DAIN_JOIN_TOKEN=JOIN_TOKEN,
                DAIN_NODE_ID=node_id,
                DAIN_HEARTBEAT_S="0.5",
                DAIN_NODE_STATE_PATH=str(workdir / f"{node_id}_state.json"),
                DAIN_MODEL=DEV_MODEL_ID,
                DAIN_MODEL_CACHE=str(workdir / f"{node_id}_cache"),
            )
            return subprocess.Popen([sys.executable, "-m", "dain_node"], cwd=workdir, env=env)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                for i in range(5):
                    procs[f"node-{i}"] = spawn(f"node-{i}")
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    listing = (
                        await client.get(
                            f"{server.base_url}/admin/nodes",
                            headers=ADMIN_HEADERS,
                        )
                    ).json()
                    connected = sum(1 for n in listing if n.get("connected"))
                    if connected >= 5:
                        break
                    await asyncio.sleep(0.25)

                # Job 1: 4 stages on 4 of the 5 nodes; the 5th is a warm backup.
                body = (
                    await client.post(
                        f"{server.base_url}/v1/completions",
                        headers={"X-API-Key": API_KEY},
                        json={
                            "model_id": DEV_MODEL_ID,
                            "prompt": "Once upon a time",
                            "max_tokens": 12,
                            "stream": False,
                        },
                    )
                ).json()
                assert body["finish_reason"] == "length"
                job1 = (
                    await client.get(
                        f"{server.base_url}/v1/jobs/{body['job_id']}",
                        headers={"X-API-Key": API_KEY},
                    )
                ).json()
                assert len(job1["stages"]) == 4
                assert len(job1["backups"]) == 1

                # Kill the node holding the last (sampling) stage.
                victim = job1["stages"][-1]["node_id"]
                procs[victim].kill()
                await asyncio.to_thread(procs[victim].wait)
                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline:
                    listing = (
                        await client.get(
                            f"{server.base_url}/admin/nodes",
                            headers=ADMIN_HEADERS,
                        )
                    ).json()
                    states = {n["node_id"]: n["state"] for n in listing}
                    if states.get(victim) in (None, NodeState.OFFLINE.value):
                        break
                    await asyncio.sleep(0.2)

                # Job 2: recomputed placement must avoid the dead node and still
                # cover every layer.
                body2 = (
                    await client.post(
                        f"{server.base_url}/v1/completions",
                        headers={"X-API-Key": API_KEY},
                        json={
                            "model_id": DEV_MODEL_ID,
                            "prompt": "Another prompt entirely",
                            "max_tokens": 12,
                            "stream": False,
                        },
                    )
                ).json()
                assert body2["finish_reason"] in ("length", "eos")
                job2 = (
                    await client.get(
                        f"{server.base_url}/v1/jobs/{body2['job_id']}",
                        headers={"X-API-Key": API_KEY},
                    )
                ).json()
                stage_nodes = {s["node_id"] for s in job2["stages"]}
                assert victim not in stage_nodes, stage_nodes
                assert [s["layer_start"] for s in job2["stages"]] == sorted(
                    s["layer_start"] for s in job2["stages"]
                )
                assert job2["stages"][0]["layer_start"] == 0
                assert job2["stages"][-1]["layer_end"] == 15
        finally:
            for proc in procs.values():
                proc.terminate()
            await asyncio.to_thread(lambda: [p.wait() for p in procs.values()])
            await stop_server(server)

    asyncio.run(main())
