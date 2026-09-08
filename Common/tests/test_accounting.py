"""S1 checkpoint: accounting math goldens (spec §15)."""

import pytest

from dain_common import (
    MODELS,
    CreditWeights,
    LedgerEvent,
    TaskOutcome,
    credit,
    flops_for_stages,
    flops_per_token,
    get_model,
)


def event(**overrides) -> LedgerEvent:
    values = {
        "node_id": "node-alpha",
        "job_id": "job-0001",
        "model_id": "qwen2.5-7b",
        "stage_idx": 3,
        "attempt": 0,
        "partition_id": "layers_12_18",
        "tokens_in": 100,
        "tokens_out": 50,
        "flops_est": 3.8e11,
        "compute_seconds": 2.0,
        "gpu_util_avg": 50.0,
        "outcome": TaskOutcome.SUCCESS,
        "ts": 1.75e9,
    }
    values.update(overrides)
    return LedgerEvent(**values)


# -- model registry goldens (values match Docs/spec.md §9/§18) -----------------


def test_model_registry_reference_models() -> None:
    assert set(MODELS) == {"tinyllama-1.1b", "qwen2.5-7b", "qwen3-32b", "olmoe-1b-7b"}
    olmoe = MODELS["olmoe-1b-7b"]
    assert olmoe.experts == 64
    assert olmoe.params_active < olmoe.params_total  # MoE: sparse activation


def test_flops_per_token_golden() -> None:
    assert flops_per_token("tinyllama-1.1b") == pytest.approx(2.2e9)
    assert flops_per_token("qwen2.5-7b") == pytest.approx(1.52e10)
    assert flops_per_token("qwen3-32b") == pytest.approx(6.56e10)
    assert flops_per_token("olmoe-1b-7b") == pytest.approx(2.0e9)  # active params only


def test_flops_for_stages_golden() -> None:
    # 2 × (7.6e9 / 28) × 7 layers × 100 tokens = 3.8e11
    assert flops_for_stages("qwen2.5-7b", n_layers=7, tokens=100) == pytest.approx(3.8e11)
    # Whole model must equal the per-token estimate × tokens
    whole = flops_for_stages("qwen2.5-7b", n_layers=28, tokens=1000)
    assert whole == pytest.approx(flops_per_token("qwen2.5-7b") * 1000)


def test_flops_for_stages_rejects_bad_layer_count() -> None:
    with pytest.raises(ValueError):
        flops_for_stages("qwen2.5-7b", n_layers=0, tokens=10)
    with pytest.raises(ValueError):
        flops_for_stages("qwen2.5-7b", n_layers=29, tokens=10)


def test_unknown_model_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_model("gpt-99")


# -- credit function (spec §15) -------------------------------------------------


def test_credit_golden_success() -> None:
    # 1.0 × 150 + 1e-10 × 3.8e11 + 0.05 × 2.0 × 0.5 = 150 + 38 + 0.05 = 188.05
    assert credit(event()) == pytest.approx(188.05, rel=1e-9)


def test_credit_outcome_factors() -> None:
    retried = event(outcome=TaskOutcome.RETRIED_AWAY)
    failed = event(outcome=TaskOutcome.FAILED)
    assert credit(retried) == pytest.approx(188.05 * 0.2, rel=1e-9)
    assert credit(failed) == 0.0


def test_credit_falls_back_to_cpu_util_then_full() -> None:
    cpu_only = event(gpu_util_avg=None, cpu_util_avg=25.0)
    no_telemetry = event(gpu_util_avg=None, cpu_util_avg=None)
    weights = CreditWeights(w_tok=0.0, w_flop=0.0, w_time=1.0)  # isolate the time term
    assert credit(cpu_only, weights) == pytest.approx(2.0 * 0.25, rel=1e-9)
    assert credit(no_telemetry, weights) == pytest.approx(2.0 * 1.0, rel=1e-9)


def test_credit_weights_are_configurable() -> None:
    heavy_tokens = credit(event(), CreditWeights(w_tok=10.0, w_flop=0.0, w_time=0.0))
    assert heavy_tokens == pytest.approx(150.0 * 10.0, rel=1e-9)
