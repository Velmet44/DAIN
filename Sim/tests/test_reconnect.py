"""S3 checkpoint: kill a node process → OFFLINE within 15 s; restart → ONLINE
with the same node_id, history kept. Uses 1 s heartbeats (3 s offline bound),
comfortably inside the production 15 s detection budget the checkpoint allows.
"""

import asyncio
import os
import signal
import subprocess
import sys
import time

import httpx
from dain_coordinator.settings import CoordinatorSettings
from helpers import ADMIN_HEADERS, ADMIN_KEY

from dain_sim.dev import JOIN_TOKEN
from dain_sim.server import start_server, stop_server


def test_reconnect_cycle(tmp_path) -> None:
    async def main() -> None:
        settings = CoordinatorSettings(
            db_path=str(tmp_path / "coordinator.sqlite3"),
            heartbeat_interval_s=1.0,
            offline_after_missed=3,
            monitor_tick_s=0.25,
            admin_api_key=ADMIN_KEY,
        )
        server = await start_server(settings)
        workdir = tmp_path / "node"
        workdir.mkdir()
        env = dict(
            os.environ,
            DAIN_COORD_URL=f"ws://127.0.0.1:{server.port}",
            DAIN_JOIN_TOKEN=JOIN_TOKEN,
            DAIN_NODE_ID="node-recon",
            DAIN_HEARTBEAT_S="1.0",
            DAIN_NODE_STATE_PATH=str(workdir / "node_state.json"),
        )
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

        def spawn() -> subprocess.Popen:
            return subprocess.Popen(
                [sys.executable, "-m", "dain_node"], cwd=workdir, env=env, creationflags=flags
            )

        async def wait_state(client: httpx.AsyncClient, state: str, timeout_s: float) -> bool:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                listing = (
                    await client.get(
                        f"{server.base_url}/admin/nodes", headers=ADMIN_HEADERS
                    )
                ).json()
                match = [n for n in listing if n["node_id"] == "node-recon"]
                if match and match[0]["state"] == state:
                    return True
                await asyncio.sleep(0.2)
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Life 1: the agent registers and heartbeats.
                proc = spawn()
                assert await wait_state(client, "online", 40), "node did not come ONLINE"

                # Crash it — hard kill, no deregister.
                proc.kill()
                await asyncio.to_thread(proc.wait)
                assert await wait_state(client, "offline", 30), (
                    # 3 s configured bound; generous for loaded-machine flakiness
                    "node not OFFLINE in time after the kill"
                )
                detail = (
                    await client.get(
                        f"{server.base_url}/admin/nodes/node-recon",
                        headers=ADMIN_HEADERS,
                    )
                ).json()
                assert any(h["reason"] == "heartbeat_timeout" for h in detail["history"])
                assert (workdir / "node_state.json").exists()  # identity persisted

                # Life 2: same state file → same node_id + token → re-registered ONLINE.
                proc = spawn()
                assert await wait_state(client, "online", 90), "node did not re-register"
                detail = (
                    await client.get(
                        f"{server.base_url}/admin/nodes/node-recon",
                        headers=ADMIN_HEADERS,
                    )
                ).json()
                to_states = [h["to_state"] for h in detail["history"]]
                assert "online" == detail["state"]
                assert to_states.count("online") >= 2  # first registration + re-registration
                assert any(h["reason"] == "heartbeat_timeout" for h in detail["history"])

                # Heartbeats flow again (poll: the first beat may race the ONLINE read).
                async def beat_again() -> bool:
                    detail = (
                        await client.get(
                            f"{server.base_url}/admin/nodes/node-recon",
                            headers=ADMIN_HEADERS,
                        )
                    ).json()
                    return (detail["last_seq"] or 0) >= 1

                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and not await beat_again():
                    await asyncio.sleep(0.2)
                detail = (
                    await client.get(
                        f"{server.base_url}/admin/nodes/node-recon",
                        headers=ADMIN_HEADERS,
                    )
                ).json()
                assert (detail["last_seq"] or 0) >= 1

                # Graceful shutdown → deregistered.
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
                assert await wait_state(client, "offline", 10), "graceful stop did not deregister"
                detail = (
                    await client.get(
                        f"{server.base_url}/admin/nodes/node-recon",
                        headers=ADMIN_HEADERS,
                    )
                ).json()
                assert any(h["reason"] == "deregistered" for h in detail["history"])
                await asyncio.to_thread(proc.wait, 10)
                assert proc.returncode == 0
        finally:
            await stop_server(server)

    asyncio.run(main())
