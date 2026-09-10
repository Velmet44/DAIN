"""Coordinator runtime settings (S2).

Constructed from a ``config.json`` in ``Coordinator/`` (``from_config``), from
environment variables (``from_env``), or directly in tests with compressed
timings.  All timing knobs exist so the heartbeat/state machine can be
exercised in seconds instead of minutes.  When both a config file and
environment variables exist, environment variables win (per-field), so
launchers like ``Scripts/start-samepc.ps1`` keep taking precedence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dain_common.accounting import CreditWeights
from dain_common.config import ScoringConfig

DEFAULT_JOIN_TOKEN = "dain-dev-join-token"

# Every settings field → the env var that can override it.  Values follow the
# dataclass field names so config.json keys stay obvious ("heartbeat_interval_s").
COORDINATOR_ENV: dict[str, str] = {
    "host": "DAIN_HOST",
    "port": "DAIN_PORT",
    "db_path": "DAIN_DB_PATH",
    "heartbeat_interval_s": "DAIN_HEARTBEAT_S",
    "offline_after_missed": "DAIN_OFFLINE_AFTER_MISSED",
    "monitor_tick_s": "DAIN_MONITOR_TICK_S",
    "join_token": "DAIN_JOIN_TOKEN",
    "min_score": "DAIN_MIN_SCORE",
    "uptime_alpha": "DAIN_UPTIME_ALPHA",
    "overload_util_pct": "DAIN_OVERLOAD_UTIL_PCT",
    "overload_strikes_to_degrade": "DAIN_OVERLOAD_STRIKES",
    "temp_degrade_c": "DAIN_TEMP_DEGRADE_C",
    "log_json": "DAIN_LOG_JSON",
    "api_key": "DAIN_API_KEY",
    "model_store_dir": "DAIN_MODEL_STORE_DIR",
    "job_timeout_s": "DAIN_JOB_TIMEOUT_S",
    "queue_limit": "DAIN_QUEUE_LIMIT",
    "max_completion_tokens": "DAIN_MAX_COMPLETION_TOKENS",
    "layers_per_node_target": "DAIN_LAYERS_PER_NODE",
    "max_concurrent_per_key": "DAIN_MAX_CONCURRENT_PER_KEY",
    "backup_count": "DAIN_BACKUP_COUNT",
    "min_stages": "DAIN_MIN_STAGES",
    "max_stages": "DAIN_MAX_STAGES",
    "watchdog_tick_s": "DAIN_WATCHDOG_TICK_S",
    "stage_deadline_min_s": "DAIN_STAGE_DEADLINE_MIN_S",
    "stage_deadline_max_s": "DAIN_STAGE_DEADLINE_MAX_S",
    "max_stage_attempts": "DAIN_MAX_STAGE_ATTEMPTS",
    "max_job_restarts": "DAIN_JOB_RESTARTS",
    "min_nodes": "DAIN_MIN_NODES",
    "cors_origins": "DAIN_CORS_ORIGINS",
    "rate_limit_per_min": "DAIN_RATE_LIMIT_PER_MIN",
    "admin_api_key": "DAIN_ADMIN_API_KEY",
    "discovery_enabled": "DAIN_DISCOVERY_ENABLED",
    "discovery_port": "DAIN_DISCOVERY_PORT",
}

# Default config.json written next to the coordinator on first run.  Key names
# mirror the CoordinatorSettings fields so editing the file is self-documenting.
DEFAULT_CONFIG: dict[str, object] = {
    "host": "0.0.0.0",
    "port": 8000,
    "db_path": "coordinator.sqlite3",
    "heartbeat_interval_s": 5.0,
    "offline_after_missed": 3,
    "monitor_tick_s": 0.5,
    "join_token": "dain-dev-join-token",
    "min_score": 0.05,
    "uptime_alpha": 0.1,
    "overload_util_pct": 97.0,
    "overload_strikes_to_degrade": 3,
    "temp_degrade_c": 90.0,
    "log_json": False,
    "api_key": "dain-dev-key",
    "model_store_dir": "model_store",
    "job_timeout_s": 120.0,
    "queue_limit": 16,
    "max_completion_tokens": 512,
    "layers_per_node_target": 4,
    "max_concurrent_per_key": 4,
    "backup_count": 2,
    "min_stages": 1,
    "max_stages": 16,
    "watchdog_tick_s": 0.5,
    "stage_deadline_min_s": 5.0,
    "stage_deadline_max_s": 60.0,
    "max_stage_attempts": 3,
    "max_job_restarts": 1,
    "min_nodes": 1,
    "cors_origins": "*",
    "rate_limit_per_min": 60,
    "admin_api_key": "dain-dev-admin-key",
    "discovery_enabled": True,
    "discovery_port": 8456,
}


def _truthy(value: object) -> bool:
    """Accept JSON booleans and the usual string forms ("1", "true", "yes", ...)."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class CoordinatorSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    db_path: str = "coordinator.sqlite3"
    heartbeat_interval_s: float = 5.0
    offline_after_missed: int = 3
    monitor_tick_s: float = 0.5
    join_token: str = DEFAULT_JOIN_TOKEN
    min_score: float = 0.05
    uptime_alpha: float = 0.1
    overload_util_pct: float = 97.0
    overload_strikes_to_degrade: int = 3
    temp_degrade_c: float = 90.0
    log_json: bool = False
    # S4: client API + model store
    api_key: str = "dain-dev-key"
    model_store_dir: str = "model_store"
    job_timeout_s: float = 120.0
    queue_limit: int = 16
    max_completion_tokens: int = 512
    layers_per_node_target: int = 4
    max_concurrent_per_key: int = 4
    backup_count: int = 2
    # Spec §12 clamps K = ceil(L / layers_per_node_target) to [8, 16] for the
    # target 8–16 node production pool. The bounds are deployment configuration:
    # production reference values live in Deploy/.env.example (DAIN_MIN_STAGES=8),
    # while dev/sim pools are smaller and use a lower floor.
    min_stages: int = 1
    max_stages: int = 16
    # S7 (spec §13): fault tolerance knobs.
    watchdog_tick_s: float = 0.5
    stage_deadline_min_s: float = 5.0
    stage_deadline_max_s: float = 60.0
    max_stage_attempts: int = 3
    max_job_restarts: int = 1
    min_nodes: int = 1
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    # S8 (spec §15): credit weights applied by the ledger (illustrative defaults).
    accounting_weights: CreditWeights = field(default_factory=CreditWeights)
    # S9: browser client — CORS origins (comma-separated; "*" opens dev) and an
    # optional per-IP/per-key rate limit for the public completion API (0 off).
    cors_origins: tuple[str, ...] = ("*",)
    rate_limit_per_min: int = 60
    admin_api_key: str = "dain-dev-admin-key"
    # S10: LAN coordinator discovery — UDP responder nodes probe at startup so
    # they never need a hand-typed coord_url. Disable for public/WAN deploys.
    discovery_enabled: bool = True
    discovery_port: int = 8456

    @property
    def offline_timeout_s(self) -> float:
        """No heartbeat for this long → OFFLINE (3 missed intervals, spec §6/§13)."""
        return self.heartbeat_interval_s * self.offline_after_missed

    @classmethod
    def from_env(cls) -> CoordinatorSettings:
        env = os.environ
        return cls(
            host=env.get("DAIN_HOST", "0.0.0.0"),
            port=int(env.get("DAIN_PORT", "8000")),
            db_path=env.get("DAIN_DB_PATH", "coordinator.sqlite3"),
            heartbeat_interval_s=float(env.get("DAIN_HEARTBEAT_S", "5.0")),
            offline_after_missed=int(env.get("DAIN_OFFLINE_AFTER_MISSED", "3")),
            monitor_tick_s=float(env.get("DAIN_MONITOR_TICK_S", "0.5")),
            join_token=env.get("DAIN_JOIN_TOKEN", DEFAULT_JOIN_TOKEN),
            min_score=float(env.get("DAIN_MIN_SCORE", "0.05")),
            uptime_alpha=float(env.get("DAIN_UPTIME_ALPHA", "0.1")),
            overload_util_pct=float(env.get("DAIN_OVERLOAD_UTIL_PCT", "97")),
            overload_strikes_to_degrade=int(env.get("DAIN_OVERLOAD_STRIKES", "3")),
            temp_degrade_c=float(env.get("DAIN_TEMP_DEGRADE_C", "90")),
            api_key=env.get("DAIN_API_KEY", "dain-dev-key"),
            model_store_dir=env.get("DAIN_MODEL_STORE_DIR", "model_store"),
            job_timeout_s=float(env.get("DAIN_JOB_TIMEOUT_S", "120")),
            queue_limit=int(env.get("DAIN_QUEUE_LIMIT", "16")),
            max_completion_tokens=int(env.get("DAIN_MAX_COMPLETION_TOKENS", "512")),
            layers_per_node_target=int(env.get("DAIN_LAYERS_PER_NODE", "4")),
            max_concurrent_per_key=int(env.get("DAIN_MAX_CONCURRENT_PER_KEY", "4")),
            backup_count=int(env.get("DAIN_BACKUP_COUNT", "2")),
            min_stages=int(env.get("DAIN_MIN_STAGES", "1")),
            max_stages=int(env.get("DAIN_MAX_STAGES", "16")),
            watchdog_tick_s=float(env.get("DAIN_WATCHDOG_TICK_S", "0.5")),
            stage_deadline_min_s=float(env.get("DAIN_STAGE_DEADLINE_MIN_S", "5")),
            stage_deadline_max_s=float(env.get("DAIN_STAGE_DEADLINE_MAX_S", "60")),
            max_stage_attempts=int(env.get("DAIN_MAX_STAGE_ATTEMPTS", "3")),
            max_job_restarts=int(env.get("DAIN_JOB_RESTARTS", "1")),
            min_nodes=int(env.get("DAIN_MIN_NODES", "1")),
            cors_origins=tuple(
                o.strip() for o in env.get("DAIN_CORS_ORIGINS", "*").split(",") if o.strip()
            ),
            rate_limit_per_min=int(env.get("DAIN_RATE_LIMIT_PER_MIN", "60")),
            admin_api_key=env.get("DAIN_ADMIN_API_KEY", "dain-dev-admin-key"),
            discovery_enabled=env.get("DAIN_DISCOVERY_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            discovery_port=int(env.get("DAIN_DISCOVERY_PORT", "8456")),
            log_json=env.get("DAIN_LOG_JSON", "").lower() in ("1", "true", "yes"),
        )

    @classmethod
    def from_config(cls, data: dict, base_dir: Path) -> CoordinatorSettings:
        """Build settings from a ``config.json`` dict.

        Relative ``db_path`` / ``model_store_dir`` values are resolved against
        *base_dir* (the folder that owns the config).  Unknown keys are ignored
        so extra documentation fields never break loading.
        """

        def _path(val: object, default: str) -> str:
            if not val:
                return default
            p = Path(str(val))
            return str(p if p.is_absolute() else base_dir / p)

        def _cors(val: object) -> tuple[str, ...]:
            if isinstance(val, (list, tuple)):
                return tuple(str(o).strip() for o in val if str(o).strip())
            return tuple(
                o.strip() for o in str(val).split(",") if o.strip()
            )

        return cls(
            host=str(data.get("host", "0.0.0.0")),
            port=int(data.get("port", 8000)),
            db_path=_path(data.get("db_path"), str(base_dir / "coordinator.sqlite3")),
            heartbeat_interval_s=float(data.get("heartbeat_interval_s", 5.0)),
            offline_after_missed=int(data.get("offline_after_missed", 3)),
            monitor_tick_s=float(data.get("monitor_tick_s", 0.5)),
            join_token=str(data.get("join_token", DEFAULT_JOIN_TOKEN)),
            min_score=float(data.get("min_score", 0.05)),
            uptime_alpha=float(data.get("uptime_alpha", 0.1)),
            overload_util_pct=float(data.get("overload_util_pct", 97.0)),
            overload_strikes_to_degrade=int(data.get("overload_strikes_to_degrade", 3)),
            temp_degrade_c=float(data.get("temp_degrade_c", 90.0)),
            log_json=_truthy(data.get("log_json", False)),
            api_key=str(data.get("api_key", "dain-dev-key")),
            model_store_dir=_path(data.get("model_store_dir"), str(base_dir / "model_store")),
            job_timeout_s=float(data.get("job_timeout_s", 120.0)),
            queue_limit=int(data.get("queue_limit", 16)),
            max_completion_tokens=int(data.get("max_completion_tokens", 512)),
            layers_per_node_target=int(data.get("layers_per_node_target", 4)),
            max_concurrent_per_key=int(data.get("max_concurrent_per_key", 4)),
            backup_count=int(data.get("backup_count", 2)),
            min_stages=int(data.get("min_stages", 1)),
            max_stages=int(data.get("max_stages", 16)),
            watchdog_tick_s=float(data.get("watchdog_tick_s", 0.5)),
            stage_deadline_min_s=float(data.get("stage_deadline_min_s", 5.0)),
            stage_deadline_max_s=float(data.get("stage_deadline_max_s", 60.0)),
            max_stage_attempts=int(data.get("max_stage_attempts", 3)),
            max_job_restarts=int(data.get("max_job_restarts", 1)),
            min_nodes=int(data.get("min_nodes", 1)),
            cors_origins=_cors(data.get("cors_origins", "*")),
            rate_limit_per_min=int(data.get("rate_limit_per_min", 60)),
            admin_api_key=str(data.get("admin_api_key", "dain-dev-admin-key")),
            discovery_enabled=_truthy(data.get("discovery_enabled", True)),
            discovery_port=int(data.get("discovery_port", 8456)),
        )
