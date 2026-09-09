"""Node agent runtime settings (S3)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_JOIN_TOKEN = "dain-dev-join-token"


@dataclass(frozen=True)
class NodeSettings:
    coord_url: str = "ws://localhost:8000"
    join_token: str = DEFAULT_JOIN_TOKEN
    node_id: str | None = None
    heartbeat_interval_s: float = 5.0
    state_path: str = "node_state.json"
    model_id: str | None = None
    model_cache_dir: str = "shard_cache"
    job_timeout_s: float = 120.0
    # Stub network figures until the S10 measurement work (spec §6 notes they are
    # reported values; the coordinator's scoring treats all claims as unverified).
    net_bw_mbps: float = 100.0
    net_lat_ms_p95: float = 50.0
    reconnect_min_s: float = 0.5
    reconnect_max_s: float = 8.0
    log_json: bool = False

    @property
    def http_base_url(self) -> str:
        return self.coord_url.replace("wss://", "https://").replace("ws://", "http://").rstrip("/")

    @property
    def ws_base_url(self) -> str:
        return self.coord_url.strip().rstrip("/")

    @classmethod
    def from_env(cls) -> NodeSettings:
        env = os.environ
        return cls(
            coord_url=env.get("DAIN_COORD_URL", "ws://localhost:8000"),
            join_token=env.get("DAIN_JOIN_TOKEN", DEFAULT_JOIN_TOKEN),
            node_id=env.get("DAIN_NODE_ID"),
            heartbeat_interval_s=float(env.get("DAIN_HEARTBEAT_S", "5.0")),
            state_path=env.get("DAIN_NODE_STATE_PATH", "node_state.json"),
            model_id=env.get("DAIN_MODEL"),
            model_cache_dir=env.get("DAIN_MODEL_CACHE", "shard_cache"),
            job_timeout_s=float(env.get("DAIN_JOB_TIMEOUT_S", "120")),
            net_bw_mbps=float(env.get("DAIN_NET_BW_MBPS", "100")),
            net_lat_ms_p95=float(env.get("DAIN_NET_LAT_MS", "50")),
            log_json=env.get("DAIN_LOG_JSON", "").lower() in ("1", "true", "yes"),
        )
