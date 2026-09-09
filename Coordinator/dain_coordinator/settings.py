"""Coordinator runtime settings (S2).

Constructed from environment variables in production (`from_env`) or directly in
tests with compressed timings. All timing knobs exist so the heartbeat/state
machine can be exercised in seconds instead of minutes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dain_common.config import ScoringConfig

DEFAULT_JOIN_TOKEN = "dain-dev-join-token"


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
            join_token=env.get("DAIN_JOIN_TOKEN", DEFAULT_JOIN_TOKEN),
            min_score=float(env.get("DAIN_MIN_SCORE", "0.05")),
            api_key=env.get("DAIN_API_KEY", "dain-dev-key"),
            model_store_dir=env.get("DAIN_MODEL_STORE_DIR", "model_store"),
            job_timeout_s=float(env.get("DAIN_JOB_TIMEOUT_S", "120")),
            queue_limit=int(env.get("DAIN_QUEUE_LIMIT", "16")),
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
            log_json=env.get("DAIN_LOG_JSON", "").lower() in ("1", "true", "yes"),
        )
