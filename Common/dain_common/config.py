"""Configuration models shared across components (spec §7, §10).

Pure, immutable, validated. The coordinator loads these from its config file /
environment; the scoring math in `scoring.py` consumes `ScoringConfig`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_HEARTBEAT_INTERVAL_S = 5.0
DEFAULT_OFFLINE_AFTER_MISSED = 3


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProtocolConfig(_Model):
    """Timing constants for the control plane (spec §6.7, §13)."""

    heartbeat_interval_s: float = Field(default=DEFAULT_HEARTBEAT_INTERVAL_S, gt=0)
    offline_after_missed: int = Field(default=DEFAULT_OFFLINE_AFTER_MISSED, ge=1)

    @property
    def offline_timeout_s(self) -> float:
        return self.heartbeat_interval_s * self.offline_after_missed


class ScoringWeights(_Model):
    """Component weights for `S(node) = Π penalties · Σ w_i · n_i` (spec §7).

    Defaults split the spec's aggregate groups: network 0.15 → net_bw 0.10 +
    net_lat 0.05; reliability 0.15 → uptime 0.10 + failure_rate 0.05. The sum
    must equal 1.0.
    """

    throughput: float = Field(default=0.30, ge=0)
    vram: float = Field(default=0.20, ge=0)
    net_bw: float = Field(default=0.10, ge=0)
    net_lat: float = Field(default=0.05, ge=0)
    uptime: float = Field(default=0.10, ge=0)
    failure_rate: float = Field(default=0.05, ge=0)
    mem_bw: float = Field(default=0.10, ge=0)
    load: float = Field(default=0.05, ge=0)
    energy: float = Field(default=0.05, ge=0)

    @model_validator(mode="after")
    def _sums_to_one(self) -> ScoringWeights:
        total = (
            self.throughput
            + self.vram
            + self.net_bw
            + self.net_lat
            + self.uptime
            + self.failure_rate
            + self.mem_bw
            + self.load
            + self.energy
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"scoring weights must sum to 1.0 (got {total:.6f})")
        return self


class ScoringRefs(_Model):
    """Baseline values `m_ref` for the saturating normalizations (spec §7).

    These are the *configured* baselines; the coordinator may refresh them from
    pool statistics with hysteresis (EWMA, spec §7) — that refresh logic is a
    later-stage concern, the math here is pure.
    """

    tflops_ref: float = Field(default=100.0, gt=0)
    vram_ref: float = Field(default=24.0, gt=0)
    net_bw_ref: float = Field(default=100.0, gt=0)
    lat_ref: float = Field(default=50.0, gt=0)
    fail_rate_ref: float = Field(default=0.02, gt=0)
    mem_bw_ref: float = Field(default=500.0, gt=0)
    util_ref: float = Field(default=50.0, gt=0, le=100)
    energy_ref: float = Field(default=0.3, gt=0, description="TFLOPS per watt baseline")


class ScoringConfig(_Model):
    weights: ScoringWeights = ScoringWeights()
    refs: ScoringRefs = ScoringRefs()
    min_uptime_soft: float = Field(default=0.8, ge=0, le=1)
    soft_penalty: float = Field(default=0.5, gt=0, le=1)
