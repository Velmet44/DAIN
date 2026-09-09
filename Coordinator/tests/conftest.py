"""Shared fixtures: compressed-timing settings + registration helpers.

`make_settings()` compresses heartbeat timing ~50× so timeout/degraded paths run
in fractions of a second instead of the production 15 s.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from dain_common.schemas import CapabilityManifest, CPUInfo, GPUInfo, NetInfo, PowerInfo
from fastapi.testclient import TestClient

from dain_coordinator.app import create_app
from dain_coordinator.settings import CoordinatorSettings

# Known test-only admin key used by all Coordinator unit tests.
ADMIN_KEY = "dain-dev-admin-key"
ADMIN_HEADERS = {"X-Admin-Key": ADMIN_KEY}


def gpu(**overrides: Any) -> GPUInfo:
    values = {
        "name": "RTX 3060",
        "vram_total_gb": 12.0,
        "vram_free_gb": 11.5,
        "tflops_claimed": 51.0,
        "mem_bw_gbs": 360.0,
    }
    values.update(overrides)
    return GPUInfo(**values)


def manifest(gpu_info: GPUInfo | None = None, **overrides: Any) -> CapabilityManifest:
    values: dict[str, Any] = {
        "gpu": gpu_info if gpu_info is not None else gpu(),
        "cpu": CPUInfo(cores=8, ram_total_gb=32.0, ram_free_gb=16.0),
        "net": NetInfo(bw_mbps=300.0, lat_ms_p95=25.0),
        "power": PowerInfo(idle_watts=12.0, tdp_watts=170.0),
    }
    values.update(overrides)
    return CapabilityManifest(**values)


def cpu_only_manifest() -> CapabilityManifest:
    return CapabilityManifest(
        gpu=None,
        cpu=CPUInfo(cores=8, ram_total_gb=32.0, ram_free_gb=20.0),
        net=NetInfo(bw_mbps=200.0, lat_ms_p95=40.0),
    )


def make_settings(tmp_path, **overrides: Any) -> CoordinatorSettings:
    """Production settings with ~50× compressed heartbeat timing for tests."""
    return CoordinatorSettings(
        db_path=str(tmp_path / "coordinator.sqlite3"),
        heartbeat_interval_s=0.1,
        offline_after_missed=3,
        monitor_tick_s=0.05,
        admin_api_key=ADMIN_KEY,
        **overrides,
    )


def make_client(settings: CoordinatorSettings) -> TestClient:
    return TestClient(create_app(settings))


def register_payload(
    node_id: str, *, auth_token: str, manifest_model: CapabilityManifest | None = None
) -> dict:
    model = manifest_model if manifest_model is not None else manifest()
    return {
        "node_id": node_id,
        "auth_token": auth_token,
        "manifest": model.model_dump(mode="json"),
        "agent_version": "0.1.0",
    }


def wait_for(predicate, timeout_s: float = 3.0, poll_s: float = 0.02) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


@pytest.fixture()
def client(tmp_path):
    with make_client(make_settings(tmp_path)) as test_client:
        yield test_client
