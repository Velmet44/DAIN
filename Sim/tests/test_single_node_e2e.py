"""S4 checkpoint: coordinator + 1 node agent → prompt → ≥20 streamed SSE tokens.

Exercises the full request path: auth → /v1/completions → JOB_ASSIGN over the
node WS → shard download/verify → CPU generation → TOKEN_BATCH relay → SSE.
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


def test_single_node_streaming_e2e(tmp_path) -> None:
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
            join_token=JOIN_TOKEN,
            job_timeout_s=60.0,
        )
        server = await start_server(settings)
        workdir = tmp_path / "node"
        workdir.mkdir()
        env = dict(
            os.environ,
            DAIN_COORD_URL=f"ws://127.0.0.1:{server.port}",
            DAIN_JOIN_TOKEN=JOIN_TOKEN,
            DAIN_NODE_ID="node-infer",
            DAIN_HEARTBEAT_S="0.5",
            DAIN_NODE_STATE_PATH=str(workdir / "node_state.json"),
            DAIN_MODEL=DEV_MODEL_ID,
            DAIN_MODEL_CACHE=str(workdir / "shard_cache"),
        )
        proc = subprocess.Popen([sys.executable, "-m", "dain_node"], cwd=workdir, env=env)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Wait for the node to register AND attach its live WS. Registration
                # (state=ONLINE) alone is not enough: /v1/completions only dispatches
                # to a *connected* node, so firing before the WS handshake finishes
                # races into "no connected node pool" (429).
                deadline = time.monotonic() + 20.0
                while time.monotonic() < deadline:
                    listing = (
                        await client.get(
                            f"{server.base_url}/admin/nodes",
                            headers=ADMIN_HEADERS,
                        )
                    ).json()
                    if any(
                        n["state"] == NodeState.ONLINE.value and n.get("connected")
                        for n in listing
                    ):
                        break
                    await asyncio.sleep(0.25)
                else:
                    raise AssertionError("node never came ONLINE with a live WS")

                # Auth is enforced on the client API: keyless localhost is
                # trusted (session 15 console posture), but an explicit wrong
                # key is rejected even from loopback.
                unauth = await client.post(
                    f"{server.base_url}/v1/completions",
                    headers={"X-API-Key": "wrong-key-1"},
                    json={"model_id": DEV_MODEL_ID, "prompt": "hi", "max_tokens": 4},
                )
                assert unauth.status_code == 401

                # Models are listed from the store.
                models = (
                    await client.get(f"{server.base_url}/v1/models", headers={"X-API-Key": API_KEY})
                ).json()
                assert any(m["model_id"] == DEV_MODEL_ID for m in models["models"])

                # The S4 gate: streamed completion with valid SSE framing.
                frames: list[dict] = []
                done_sentinel = False
                async with client.stream(
                    "POST",
                    f"{server.base_url}/v1/completions",
                    headers={"X-API-Key": API_KEY},
                    json={
                        "model_id": DEV_MODEL_ID,
                        "prompt": "Once upon a time",
                        "max_tokens": 24,
                        "stream": True,
                    },
                ) as response:
                    assert response.status_code == 200
                    assert "text/event-stream" in response.headers["content-type"]
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[len("data: ") :]
                        if payload == "[DONE]":
                            done_sentinel = True
                            break
                        frames.append(__import__("json").loads(payload))

                token_chars = [f["token"] for f in frames if f.get("type") == "token"]
                assert len(token_chars) >= 20, f"only {len(token_chars)} tokens streamed"
                final = [f for f in frames if f.get("type") == "final"]
                assert len(final) == 1 and final[0]["finish_reason"] == "length"
                assert done_sentinel

                job_id = frames[0]["job_id"]
                view = (
                    await client.get(
                        f"{server.base_url}/v1/jobs/{job_id}", headers={"X-API-Key": API_KEY}
                    )
                ).json()
                assert view["state"] == "completed"
                assert view["tokens_generated"] == len(token_chars)

                # Non-streaming JSON path returns the same text.
                non_stream = (
                    await client.post(
                        f"{server.base_url}/v1/completions",
                        headers={"X-API-Key": API_KEY},
                        json={
                            "model_id": DEV_MODEL_ID,
                            "prompt": "Once upon a time",
                            "max_tokens": 10,
                            "stream": False,
                        },
                    )
                ).json()
                assert non_stream["finish_reason"] == "length"
                assert len(non_stream["text"]) == 10
        finally:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 10)
            except subprocess.TimeoutExpired:
                proc.kill()
            await stop_server(server)

    asyncio.run(main())
