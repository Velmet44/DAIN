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
            log_json=env.get("DAIN_LOG_JSON", "").lower() in ("1", "true", "yes"),
        )
