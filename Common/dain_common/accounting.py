"""Compute accounting math (spec §15) — model registry, FLOP estimates, credits.

The coordinator derives FLOPs from these static tables; nodes never self-report
FLOPs (untrusted input, §15/§16). Parameter counts are first-order constants
for the reference models (spec §18) used by scoring, scheduling and the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

from dain_common.schemas import LedgerEvent, TaskOutcome


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    name: str
    layers: int
    hidden: int
    params_total: float
    params_active: float
    kv_dim: int
    experts: int | None = None


# Reference models (spec §18). Values match Docs/spec.md §9 feasibility notes.
MODELS: dict[str, ModelSpec] = {
    "tinyllama-1.1b": ModelSpec(
        model_id="tinyllama-1.1b",
        name="TinyLlama-1.1B",
        layers=22,
        hidden=2048,
        params_total=1.1e9,
        params_active=1.1e9,
        kv_dim=256,
    ),
    "qwen2.5-7b": ModelSpec(
        model_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        layers=28,
        hidden=3584,
        params_total=7.6e9,
        params_active=7.6e9,
        kv_dim=512,
    ),
    "qwen3-32b": ModelSpec(
        model_id="qwen3-32b",
        name="Qwen3-32B",
        layers=64,
        hidden=4096,
        params_total=32.8e9,
        params_active=32.8e9,
        kv_dim=1024,
    ),
    "olmoe-1b-7b": ModelSpec(
        model_id="olmoe-1b-7b",
        name="OLMoE-1B-7B",
        layers=16,
        hidden=2048,
        params_total=6.9e9,
        params_active=1.0e9,
        kv_dim=2048,
        experts=64,
    ),
}


def get_model(model_id: str) -> ModelSpec:
    spec = MODELS.get(model_id)
    if spec is None:
        raise KeyError(f"unknown model_id {model_id!r}; known: {sorted(MODELS)}")
    return spec


def flops_per_token(model_id: str) -> float:
    """First-order dense/MoE decode FLOPs: `2 × active parameters × 1 token` (§15)."""
    return 2.0 * get_model(model_id).params_active


def flops_for_stages(model_id: str, n_layers: int, tokens: int) -> float:
    """FLOPs for a contiguous stage of `n_layers` layers over `tokens` tokens.

    Uniform-per-layer approximation: `2 × (active params / L) × n_layers × tokens`.
    MoE routing variance is averaged into `params_active` (per-token expectation).
    """
    spec = get_model(model_id)
    if not 0 < n_layers <= spec.layers:
        raise ValueError(f"n_layers must be in (0, {spec.layers}] for {model_id}")
    if tokens < 0:
        raise ValueError("tokens must be >= 0")
    return 2.0 * (spec.params_active / spec.layers) * n_layers * tokens


@dataclass(frozen=True)
class CreditWeights:
    """Illustrative defaults (spec §15); the coordinator makes them configurable.

    Calibration note: on qwen2.5-7b the FLOP term contributes ~1.52 credits per
    token (2·7.6e9·1e-10) and the token term 1.0 per token, so the two agree in
    magnitude; the time term adds utilization-weighted wall-clock credit.
    """

    w_tok: float = 1.0
    w_flop: float = 1e-10
    w_time: float = 0.05


OUTCOME_FACTOR: dict[TaskOutcome, float] = {
    TaskOutcome.SUCCESS: 1.0,
    TaskOutcome.RETRIED_AWAY: 0.2,
    TaskOutcome.FAILED: 0.0,
}


def credit(event: LedgerEvent, weights: CreditWeights | None = None) -> float:
    """Credit for one ledger event (spec §15); 0 for FAILED by construction."""
    weights = weights or CreditWeights()
    if event.gpu_util_avg is not None:
        util_factor = event.gpu_util_avg / 100.0
    elif event.cpu_util_avg is not None:
        util_factor = event.cpu_util_avg / 100.0
    else:
        util_factor = 1.0
    base = (
        weights.w_tok * (event.tokens_in + event.tokens_out)
        + weights.w_flop * event.flops_est
        + weights.w_time * event.compute_seconds * util_factor
    )
    return base * OUTCOME_FACTOR[event.outcome]
