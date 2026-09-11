"""Node scoring math (spec §7) — pure functions, no I/O.

`S(node) = Π penalties · Σ w_i · n_i`, each `n_i` a saturating normalization
`m/(m+m_ref)` (or `1 − m/(m+m_ref)` where lower is better).

Structurally missing components (e.g. energy when a node reports no power data,
mem_bw on a CPU-only node) are excluded and their weight is redistributed over
the remaining components — a measured-but-low value is never conflated with an
absent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dain_common.config import ScoringConfig
from dain_common.schemas import CapabilityManifest, MetricsReport

# Slugs of ScoringWeights fields, in stable order for tests/observability.
WEIGHT_KEYS = (
    "throughput",
    "vram",
    "net_bw",
    "net_lat",
    "uptime",
    "failure_rate",
    "mem_bw",
    "load",
    "energy",
)


def norm(m: float, ref: float) -> float:
    """Saturating normalization for higher-is-better metrics, ∈ (0, 1)."""
    if ref <= 0:
        raise ValueError("ref must be > 0")
    if m < 0:
        raise ValueError("metric must be >= 0")
    return m / (m + ref)


def norm_inv(m: float, ref: float) -> float:
    """Saturating normalization for lower-is-better metrics, ∈ (0, 1)."""
    if ref <= 0:
        raise ValueError("ref must be > 0")
    if m < 0:
        raise ValueError("metric must be >= 0")
    return 1.0 - m / (m + ref)


def ewma(prev: float | None, sample: float, alpha: float) -> float:
    """Exponential moving average; `alpha` is the weight of the new sample."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    return sample if prev is None else alpha * sample + (1.0 - alpha) * prev


@dataclass(frozen=True)
class NodeReputation:
    """Coordinator-tracked reputation inputs (EWMA-updated over time, §7)."""

    uptime_ratio: float = 1.0
    failure_rate: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.uptime_ratio <= 1.0:
            raise ValueError("uptime_ratio must be in [0, 1]")
        if self.failure_rate < 0:
            raise ValueError("failure_rate must be >= 0")


@dataclass(frozen=True)
class ScoreResult:
    score: float
    feasible: bool
    components: dict[str, float] = field(default_factory=dict)


def is_feasible(manifest: CapabilityManifest, required_vram_gb: float) -> bool:
    """Hard feasibility: the node's free VRAM must cover the required shard.

    `required_vram_gb == 0` (CPU-servable partitions, embeddings/LM-head) is
    feasible for any node, including CPU-only ones.
    """
    if required_vram_gb <= 0:
        return True
    if manifest.gpu is None:
        return False
    return manifest.gpu.vram_free_gb >= required_vram_gb


def score_node(
    manifest: CapabilityManifest,
    metrics: MetricsReport,
    reputation: NodeReputation,
    *,
    required_vram_gb: float = 0.0,
    config: ScoringConfig | None = None,
) -> ScoreResult:
    """Compute the normalized node score in [0, 1] (spec §7).

    `metrics` should be the latest report (heartbeats carry one); missing
    optional metric fields fall back to manifest values where sensible.
    """
    config = config or ScoringConfig()
    if not is_feasible(manifest, required_vram_gb):
        return ScoreResult(score=0.0, feasible=False, components={})

    w, refs = config.weights, config.refs
    gpu = manifest.gpu
    tflops = gpu.tflops_claimed if gpu is not None else 0.0
    vram_free = (
        metrics.vram_free_gb
        if metrics.vram_free_gb is not None
        else (gpu.vram_free_gb if gpu is not None else 0.0)
    )
    mem_bw = gpu.mem_bw_gbs if gpu is not None else None
    net_bw = metrics.net_bw_mbps if metrics.net_bw_mbps is not None else manifest.net.bw_mbps
    gpu_util = metrics.gpu_util_pct if gpu is not None else None
    util = gpu_util if gpu_util is not None else metrics.cpu_util_pct
    energy_efficiency: float | None = None
    if gpu is not None and manifest.power is not None and manifest.power.tdp_watts:
        energy_efficiency = gpu.tflops_claimed / manifest.power.tdp_watts

    components: dict[str, float | None] = {
        "throughput": norm(tflops, refs.tflops_ref),
        "vram": norm(vram_free, refs.vram_ref),
        "net_bw": norm(net_bw, refs.net_bw_ref),
        "net_lat": norm_inv(manifest.net.lat_ms_p95, refs.lat_ref),
        "uptime": reputation.uptime_ratio,
        "failure_rate": norm_inv(reputation.failure_rate, refs.fail_rate_ref),
        "mem_bw": norm(mem_bw, refs.mem_bw_ref) if mem_bw is not None else None,
        "load": norm_inv(util, refs.util_ref) if util is not None else None,
        "energy": (
            norm(energy_efficiency, refs.energy_ref) if energy_efficiency is not None else None
        ),
    }

    weighted = 0.0
    total_weight = 0.0
    for key in WEIGHT_KEYS:
        value = components[key]
        if value is None:
            continue
        weight = getattr(w, key)
        weighted += weight * value
        total_weight += weight
    base = weighted / total_weight if total_weight > 0 else 0.0

    penalty = config.soft_penalty if reputation.uptime_ratio < config.min_uptime_soft else 1.0
    return ScoreResult(
        score=penalty * base,
        feasible=True,
        components={k: v for k, v in components.items() if v is not None},
    )
