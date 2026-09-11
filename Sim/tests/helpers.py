"""Shared test fakes: REST registration + heartbeat-loop node clients.

Cluster server plumbing lives in `dain_sim.server`; this module keeps only the
test-side fakes (direct-WS fake nodes used by protocol/stability tests).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
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

from dain_sim.dev import ADMIN_HEADERS, ADMIN_KEY, JOIN_TOKEN

__all__ = ["ADMIN_HEADERS", "ADMIN_KEY", "JOIN_TOKEN"]


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
