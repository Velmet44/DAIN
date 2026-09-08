"""In-process coordinator server for tests and the cluster orchestrator.

Runs a real uvicorn server on 127.0.0.1 inside the caller's event loop — the
same app a deployed coordinator runs, so sim results transfer.
"""

from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass

import uvicorn
from dain_coordinator.app import create_app
from dain_coordinator.settings import CoordinatorSettings


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class ClusterServer:
    settings: CoordinatorSettings
    port: int
    _server: uvicorn.Server
    _task: asyncio.Task

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    @property
    def db_path(self) -> str:
        return self.settings.db_path


async def start_server(settings: CoordinatorSettings) -> ClusterServer:
    port = free_port()
    app = create_app(settings)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(), name="coordinator-server")
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    if not server.started:
        raise RuntimeError("coordinator did not start in time")
    return ClusterServer(settings=settings, port=port, _server=server, _task=task)


async def stop_server(server: ClusterServer) -> None:
    server._server.should_exit = True  # noqa: SLF001 (handle owns the server)
    await server._task
