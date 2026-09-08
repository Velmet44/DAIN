"""Shared helpers for cross-component tests: in-process cluster + fake nodes.

`start_cluster()` runs a real coordinator (uvicorn) inside the test's event loop
on 127.0.0.1; `fake_node()` plays one node: REST registration + a heartbeat WS
loop with the wire Envelope protocol — the same code path the real agent uses
from S3 on.
"""

from __future__ import annotations

import asyncio
import socket
import time
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
import websockets
from dain_common.schemas import (
    CapabilityManifest,
    CPUInfo,
    Envelope,
    GPUInfo,
    Heartbeat,
    MessageType,
    MetricsReport,
    NetInfo,
    PowerInfo,
)
from dain_coordinator.app import create_app
from dain_coordinator.settings import CoordinatorSettings

JOIN_TOKEN = "dain-dev-join-token"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class Cluster:
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


async def start_cluster(settings: CoordinatorSettings) -> Cluster:
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
    return Cluster(settings=settings, port=port, _server=server, _task=task)


async def stop_cluster(cluster: Cluster) -> None:
    cluster._server.should_exit = True  # noqa: SLF001 (test handle)
    await cluster._task


async def register_node(
    client: httpx.AsyncClient, base_url: str, node_id: str, manifest: CapabilityManifest
) -> dict[str, Any]:
    response = await client.post(
        f"{base_url}/node/register",
        json={
            "node_id": node_id,
            "auth_token": JOIN_TOKEN,
            "manifest": manifest.model_dump(mode="json"),
            "agent_version": "0.1.0-test",
        },
    )
    response.raise_for_status()
    return response.json()


def gpu_manifest(name: str, tflops: float, vram: float, tdp: float) -> CapabilityManifest:
    return CapabilityManifest(
        gpu=GPUInfo(
            name=name,
            vram_total_gb=vram,
            vram_free_gb=vram * 0.9,
            tflops_claimed=tflops,
            mem_bw_gbs=tflops * 7.0,
        ),
        cpu=CPUInfo(cores=8, ram_total_gb=32.0, ram_free_gb=16.0),
        net=NetInfo(bw_mbps=300.0, lat_ms_p95=25.0),
        power=PowerInfo(idle_watts=12.0, tdp_watts=tdp),
    )


def cpu_only_manifest() -> CapabilityManifest:
    return CapabilityManifest(
        gpu=None,
        cpu=CPUInfo(cores=8, ram_total_gb=32.0, ram_free_gb=20.0),
        net=NetInfo(bw_mbps=200.0, lat_ms_p95=40.0),
    )


async def fake_node(
    ws_url: str,
    node_id: str,
    token: str,
    *,
    heartbeat_interval_s: float,
    duration_s: float,
    stagger_s: float = 0.0,
) -> int:
    """One node's heartbeat loop; returns the number of heartbeats sent."""
    await asyncio.sleep(stagger_s)
    sent = 0
    seq = 0
    deadline = time.monotonic() + duration_s
    async with websockets.connect(f"{ws_url}/node/ws?node_id={node_id}&token={token}") as ws:
        while time.monotonic() < deadline:
            metrics = MetricsReport(
                gpu_util_pct=30.0 + (seq % 10),
                vram_free_gb=10.0,
                cpu_util_pct=15.0,
                net_bw_mbps=300.0,
                temp_c=60.0,
            )
            envelope = Envelope.wrap(
                MessageType.HEARTBEAT,
                Heartbeat(node_id=node_id, seq=seq, metrics=metrics),
                ts=time.time(),
            )
            await ws.send(envelope.model_dump_json())
            sent += 1
            seq += 1
            await asyncio.sleep(heartbeat_interval_s)
    return sent
