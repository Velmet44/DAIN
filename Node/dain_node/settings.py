"""Node agent runtime settings (S3).

Supports two construction paths:

* ``from_env()`` — reads ``DAIN_*`` environment variables (dev mode).
* ``from_config(data, base_dir)`` — reads a ``config.json`` dict with relative
  paths resolved against *base_dir* (standalone distribution mode).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_JOIN_TOKEN = "Jj3L7ewD"


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
    # Peer-to-peer shard sharing (S11 extension): this node serves its cached
    # shards to siblings over the LAN (port 0 = OS-assigned) and downloads from
    # peers instead of always pulling from the coordinator's model store.
    peer_enabled: bool = True
    peer_bind_host: str = "0.0.0.0"
    peer_port: int = 0
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
            peer_enabled=env.get("DAIN_PEER_ENABLED", "").lower() not in ("0", "false", "no"),
            peer_bind_host=env.get("DAIN_PEER_HOST", "0.0.0.0"),
            peer_port=int(env.get("DAIN_PEER_PORT", "0")),
            net_bw_mbps=float(env.get("DAIN_NET_BW_MBPS", "100")),
            net_lat_ms_p95=float(env.get("DAIN_NET_LAT_MS", "50")),
            log_json=env.get("DAIN_LOG_JSON", "").lower() in ("1", "true", "yes"),
        )

    @classmethod
    def from_config(cls, config: dict, base_dir: Path) -> NodeSettings:
        """Build settings from a ``config.json`` dict.

        Relative paths in the config are resolved against *base_dir* (the
        directory that contains the exe or the project root).
        """

        def _resolve(val: str | None, default: str) -> str:
            if not val:
                return default
            p = Path(val)
            return str(p if p.is_absolute() else base_dir / p)

        return cls(
            coord_url=config.get("coord_url") or "ws://localhost:8000",
            join_token=config.get("join_token", DEFAULT_JOIN_TOKEN),
            node_id=config.get("node_id") or None,
            heartbeat_interval_s=float(config.get("heartbeat_s", 5.0)),
            state_path=_resolve(config.get("state_path"), str(base_dir / "node_state.json")),
            model_id=config.get("model") or None,
            model_cache_dir=_resolve(config.get("cache_dir"), str(base_dir / "shard_cache")),
            job_timeout_s=float(config.get("job_timeout_s", 120.0)),
            peer_enabled=bool(config.get("peer_enabled", True)),
            peer_bind_host=str(config.get("peer_host", "0.0.0.0")),
            peer_port=int(config.get("peer_port", 0)),
            net_bw_mbps=float(config.get("net_bw_mbps", 100.0)),
            net_lat_ms_p95=float(config.get("net_lat_ms", 50.0)),
            reconnect_min_s=float(config.get("reconnect_min_s", 0.5)),
            reconnect_max_s=float(config.get("reconnect_max_s", 8.0)),
            log_json=bool(config.get("log_json", False)),
        )
