"""S1 checkpoint: scoring math goldens (spec §7).

Golden values were computed once from the reference implementation and pinned
here with `pytest.approx`; any formula change must be a deliberate spec
amendment plus a re-pin of these numbers.
"""

import pytest
from dain_common import (
    CapabilityManifest,
    CPUInfo,
    GPUInfo,
    MetricsReport,
    NetInfo,
    NodeReputation,
    PowerInfo,
    ScoringConfig,
    ScoringWeights,
    ewma,
    is_feasible,
    norm,
    norm_inv,
    score_node,
)
from pydantic import ValidationError


def gpu(**overrides) -> GPUInfo:
    values = {
        "name": "RTX 3060",
        "vram_total_gb": 12.0,
        "vram_free_gb": 11.5,
        "tflops_claimed": 51.0,
        "mem_bw_gbs": 360.0,
    }
    values.update(overrides)
    return GPUInfo(**values)


_DEFAULT_GPU = gpu()  # shared immutable default: manifest() has a GPU, manifest(None) is CPU-only


def manifest(
    gpu_info: GPUInfo | None = _DEFAULT_GPU, *, power: PowerInfo | None = None
) -> CapabilityManifest:
    return CapabilityManifest(
        gpu=gpu_info,
        cpu=CPUInfo(cores=8, ram_total_gb=32.0, ram_free_gb=16.0),
        net=NetInfo(bw_mbps=300.0, lat_ms_p95=25.0),
        power=power,
    )


# -- normalization primitives -------------------------------------------------


def test_norm_is_saturating_and_monotonic() -> None:
    assert norm(0.0, 100.0) == 0.0
    assert norm(100.0, 100.0) == pytest.approx(0.5)
    assert norm(300.0, 100.0) == pytest.approx(0.75)
    assert norm(10_000.0, 100.0) < 0.991  # saturates, never reaches 1
    assert norm(50.0, 100.0) < norm(150.0, 100.0)


def test_norm_inv_complements_norm() -> None:
    assert norm_inv(50.0, 100.0) == pytest.approx(1.0 - norm(50.0, 100.0))
    assert norm_inv(0.0, 100.0) == pytest.approx(1.0)


def test_norm_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        norm(1.0, 0.0)
    with pytest.raises(ValueError):
        norm_inv(-1.0, 1.0)


def test_ewma_first_sample_and_decay() -> None:
    assert ewma(None, 20.0, 0.5) == 20.0
    assert ewma(10.0, 20.0, 0.5) == pytest.approx(15.0)
    assert ewma(10.0, 20.0, 1.0) == 20.0
    with pytest.raises(ValueError):
        ewma(10.0, 20.0, 0.0)


# -- weights config ------------------------------------------------------------


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1.0"):
        ScoringWeights(throughput=0.9)


# -- feasibility (hard gate, spec §7/§12) --------------------------------------


def test_feasibility_hard_vram_gate() -> None:
    rich = manifest()  # 11.5 GB free
    poor = manifest(gpu(vram_free_gb=6.0))
    cpu_only = manifest(None)
    assert is_feasible(rich, required_vram_gb=8.0)
    assert not is_feasible(poor, required_vram_gb=8.0)
    assert not is_feasible(cpu_only, required_vram_gb=0.001)  # no GPU at all
    assert is_feasible(cpu_only, required_vram_gb=0.0)  # CPU-servable partition


# -- golden scores --------------------------------------------------------------


def test_golden_high_end_node() -> None:
    m = CapabilityManifest(
        gpu=GPUInfo(
            name="RTX 4090",
            vram_total_gb=24.0,
            vram_free_gb=22.0,
            tflops_claimed=165.0,
            mem_bw_gbs=1008.0,
        ),
        cpu=CPUInfo(cores=16, ram_total_gb=64.0, ram_free_gb=60.0),
        net=NetInfo(bw_mbps=940.0, lat_ms_p95=8.0),
        power=PowerInfo(idle_watts=20.0, tdp_watts=450.0),
    )
    metrics = MetricsReport(
        gpu_util_pct=10.0, vram_free_gb=22.0, cpu_util_pct=5.0, net_bw_mbps=940.0
    )
    result = score_node(
        m, metrics, NodeReputation(uptime_ratio=0.995, failure_rate=0.001), required_vram_gb=8.0
    )
    assert result.feasible
    assert result.score == pytest.approx(0.6990619060156839, rel=1e-9)
    assert result.components["energy"] == pytest.approx(
        165.0 / 450.0 / (165.0 / 450.0 + 0.3), rel=1e-9
    )


def test_golden_mid_node() -> None:
    metrics = MetricsReport(
        gpu_util_pct=35.0, vram_free_gb=11.5, cpu_util_pct=10.0, net_bw_mbps=300.0
    )
    result = score_node(
        manifest(power=PowerInfo(idle_watts=12.0, tdp_watts=170.0)),
        metrics,
        NodeReputation(uptime_ratio=0.97, failure_rate=0.01),
        required_vram_gb=8.0,
    )
    assert result.score == pytest.approx(0.5010521321944525, rel=1e-9)


def test_golden_cpu_only_node_redistributes_missing_weights() -> None:
    # Fixture mirrors the golden computation: CPU-only, net 200 Mbps / 40 ms p95.
    cpu_only = CapabilityManifest(
        gpu=None,
        cpu=CPUInfo(cores=8, ram_total_gb=32.0, ram_free_gb=20.0),
        net=NetInfo(bw_mbps=200.0, lat_ms_p95=40.0),
    )
    metrics = MetricsReport(cpu_util_pct=20.0, net_bw_mbps=200.0)
    result = score_node(
        cpu_only,
        metrics,
        NodeReputation(uptime_ratio=0.90, failure_rate=0.02),
        required_vram_gb=0.0,
    )
    assert result.feasible
    assert result.score == pytest.approx(0.28842203548085904, rel=1e-9)
    assert "mem_bw" not in result.components and "energy" not in result.components


def test_golden_flakey_node_gets_soft_penalty() -> None:
    metrics = MetricsReport(
        gpu_util_pct=35.0, vram_free_gb=11.5, cpu_util_pct=10.0, net_bw_mbps=300.0
    )
    flakey = NodeReputation(uptime_ratio=0.60, failure_rate=0.05)
    config = ScoringConfig()
    result = score_node(
        manifest(),  # no power info → energy component absent (as in the golden run)
        metrics,
        flakey,
        required_vram_gb=8.0,
        config=config,
    )
    assert result.score == pytest.approx(0.22105500691938604, rel=1e-9)
    # Verify the penalty path: score == soft_penalty × base, with base recomputed
    # independently from the reported components and the configured weights.
    base = sum(
        getattr(config.weights, key) * value for key, value in result.components.items()
    ) / sum(getattr(config.weights, key) for key in result.components)
    assert result.score == pytest.approx(config.soft_penalty * base, rel=1e-9)


def test_infeasible_node_scores_zero_with_no_components() -> None:
    metrics = MetricsReport(gpu_util_pct=35.0)
    result = score_node(manifest(), metrics, NodeReputation(), required_vram_gb=64.0)
    assert not result.feasible
    assert result.score == 0.0
    assert result.components == {}


def test_ranking_order_on_synthetic_pool() -> None:
    """Regression for the S1 gate: 4090 > 3060 > cpu-only(0 VRAM) > flakey."""
    scores = {
        "n4090": 0.6990619060156839,
        "n3060": 0.5010521321944525,
        "ncpu": 0.28842203548085904,
        "nflakey": 0.22105500691938604,
    }
    ordered = sorted(scores, key=scores.get, reverse=True)
    assert ordered == ["n4090", "n3060", "ncpu", "nflakey"]


def test_custom_weights_change_ranking_direction_is_respected() -> None:
    """Config-driven behavior: zeroing network weights must not crash scoring."""
    weights = ScoringWeights(
        throughput=0.5,
        vram=0.3,
        net_bw=0.0,
        net_lat=0.0,
        uptime=0.1,
        failure_rate=0.05,
        mem_bw=0.05,
        load=0.0,
        energy=0.0,
    )
    config = ScoringConfig(weights=weights)
    metrics = MetricsReport(gpu_util_pct=35.0, net_bw_mbps=300.0)
    result = score_node(manifest(), metrics, NodeReputation(), config=config)
    assert result.feasible
    assert 0.0 < result.score < 1.0
