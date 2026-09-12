"""S6 checkpoint: scheduler unit tests (spec §12).

Synthetic pools with pinned scores/capacities; asserts K clamping, membership
by score, min-1-layer sizing, capacity exclusion and warm backups.
"""

from dain_common.schemas import (
    CapabilityManifest,
    CPUInfo,
    GPUInfo,
    ModelManifest,
    NetInfo,
    NodeState,
    QuantizationSpec,
    ShardRef,
    SoftwareInfo,
)

from dain_coordinator.partition import plan_placement
from dain_coordinator.store import NodeRow

MANIFEST = ModelManifest(
    model_id="dain-tiny-16L",
    name="t",
    layers=16,
    hidden=64,
    heads=4,
    kv_heads=2,
    intermediate=172,
    vocab_size=256,
    eos_token_id=0,
    shards=tuple(
        ShardRef(
            model_id="dain-tiny-16L",
            shard_id=f"layers_{a:02d}_{a + 3:02d}",
            content_hash="a" * 64,
            layer_start=a,
            layer_end=a + 3,
            size_bytes=4_000_000,
        )
        for a in range(0, 16, 4)
    ),
)


def node(
    node_id: str,
    *,
    score: float = 0.3,
    state: NodeState = NodeState.ONLINE,
    tflops: float | None = 50.0,
    vram_free: float | None = 11.0,
    ram_free: float = 16.0,
    cores: int = 8,
) -> NodeRow:
    gpu = (
        GPUInfo(
            name="RTX-test",
            vram_total_gb=vram_free + 1,
            vram_free_gb=vram_free,
            tflops_claimed=tflops,
        )
        if tflops is not None
        else None
    )
    return NodeRow(
        node_id=node_id,
        node_token="tok",
        agent_version="0.1.0",
        state=state,
        manifest=CapabilityManifest(
            gpu=gpu,
            cpu=CPUInfo(cores=cores, ram_total_gb=32.0, ram_free_gb=ram_free),
            net=NetInfo(bw_mbps=300.0, lat_ms_p95=25.0),
        ),
        metrics=None,
        score=score,
        score_components={},
        overload_strikes=0,
        uptime_ratio=1.0,
        failure_rate=0.0,
        last_seq=None,
        last_heartbeat=None,
        registered_at=0.0,
    )


def test_k_clamped_by_target_and_pool() -> None:
    pool = [node(f"n{i:02d}", score=0.5 - i * 0.01) for i in range(12)]
    plan = plan_placement(MANIFEST, pool, layers_per_node_target=4)
    assert plan is not None and len(plan.stages) == 4
    # covers all layers contiguously
    assert [s.layer_start for s in plan.stages] == [0, 4, 8, 12]
    assert plan.stages[-1].layer_end == 15


def test_selection_ranks_by_score() -> None:
    pool = [node(f"n{i:02d}", score=0.2 + i * 0.01) for i in range(8)]
    plan = plan_placement(MANIFEST, pool, layers_per_node_target=4, max_k=4)
    assert plan is not None
    selected = {s.node_id for s in plan.stages}
    assert selected == {"n07", "n06", "n05", "n04"}  # top-4 by score
    assert set(plan.backups) == {"n03", "n02"}  # warm backups: next ranked


def test_sizing_proportional_to_throughput() -> None:
    pool = [
        node("fast", score=0.9, tflops=300.0),
        node("slow", score=0.3, tflops=50.0),
    ]
    plan = plan_placement(MANIFEST, pool, layers_per_node_target=8, max_k=2)
    assert plan is not None and len(plan.stages) == 2
    by_node = {s.node_id: s for s in plan.stages}
    fast_layers = by_node["fast"].layer_end - by_node["fast"].layer_start + 1
    slow_layers = by_node["slow"].layer_end - by_node["slow"].layer_start + 1
    assert fast_layers + slow_layers == 16
    assert fast_layers > slow_layers  # throughput-proportional, min 1 layer each


def test_low_score_node_excluded_when_pool_has_better() -> None:
    pool = [node("weak", score=0.05), node("strong", score=0.9)] + [
        node(f"n{i:02d}", score=0.5) for i in range(6)
    ]
    plan = plan_placement(MANIFEST, pool, layers_per_node_target=4, max_k=4)
    assert plan is not None
    assert "weak" not in {s.node_id for s in plan.stages}


def test_capacity_exclusion() -> None:
    tiny_vram = node("tiny-ram", score=0.95, vram_free=0.001)
    pool = [tiny_vram] + [node(f"n{i:02d}", score=0.5) for i in range(6)]
    plan = plan_placement(MANIFEST, pool, layers_per_node_target=4, max_k=4)
    assert plan is not None
    assert "tiny-ram" not in {s.node_id for s in plan.stages}


def test_none_when_no_online_nodes() -> None:
    pool = [node(f"n{i:02d}", state=NodeState.OFFLINE) for i in range(5)]
    assert plan_placement(MANIFEST, pool) is None


def test_degraded_nodes_not_selected() -> None:
    pool = [node("deg", score=0.99, state=NodeState.DEGRADED)] + [
        node(f"n{i:02d}", score=0.3) for i in range(6)
    ]
    plan = plan_placement(MANIFEST, pool, layers_per_node_target=4)
    assert plan is not None
    assert "deg" not in {s.node_id for s in plan.stages}


def test_backups_present_when_pool_allows() -> None:
    pool = [node(f"n{i:02d}", score=0.9 - i * 0.05) for i in range(8)]
    plan = plan_placement(MANIFEST, pool, layers_per_node_target=4, backup_count=2)
    assert plan is not None
    assert len(plan.backups) == 2
    assert not (set(plan.backups) & {s.node_id for s in plan.stages})


def test_small_model_uses_single_stage_when_it_fits() -> None:
    small = MANIFEST.model_copy(update={"layers": 3})
    pool = [node(f"n{i:02d}", score=0.5 - i * 0.01) for i in range(4)]
    # ceil(3 / 4) = 1: a 3-layer model fits one node (spec §12 formula).
    plan = plan_placement(small, pool, layers_per_node_target=4, max_k=16)
    assert plan is not None and len(plan.stages) == 1
    assert plan.stages[0].layer_start == 0 and plan.stages[0].layer_end == 2


def test_small_model_spreads_with_fine_target() -> None:
    small = MANIFEST.model_copy(update={"layers": 3})
    pool = [node(f"n{i:02d}", score=0.5 - i * 0.01) for i in range(4)]
    plan = plan_placement(small, pool, layers_per_node_target=1, max_k=16)
    assert plan is not None and len(plan.stages) == 3
    assert all(s.layer_end - s.layer_start + 1 == 1 for s in plan.stages)


# -- backend-aware placement (session N) --------------------------------------

QUANT_MANIFEST = MANIFEST.model_copy(update={
    "format": "torch_pt",
    "quantization": QuantizationSpec(
        backend="torchao",
        scheme="int4_weight_only",
        bits=4,
        group_size=128,
        packing_layout="int4_cpu",
    ),
})


def _quant_node(
    node_id: str,
    *,
    score: float = 0.5,
    backends: tuple[str, ...] = ("fp16", "fp32", "torchao"),
    quant: tuple[str, ...] = ("int4_weight_only",),
    layouts: tuple[str, ...] = ("int4_cpu",),
    activation_dtypes: tuple[str, ...] = ("fp16", "bf16"),
    adapters: tuple[str, ...] = ("llama",),
) -> NodeRow:
    sw = SoftwareInfo(
        supported_backends=backends,
        supported_quantization=quant,
        supported_packing_layouts=layouts,
        supported_activation_dtypes=activation_dtypes,
        supported_adapters=adapters,
    )
    return NodeRow(
        node_id=node_id,
        node_token="tok",
        agent_version="0.1.0",
        state=NodeState.ONLINE,
        manifest=CapabilityManifest(
            gpu=GPUInfo(
                name="RTX-test", vram_total_gb=12.0,
                vram_free_gb=11.0, tflops_claimed=50.0,
            ),
            cpu=CPUInfo(cores=8, ram_total_gb=32.0, ram_free_gb=16.0),
            net=NetInfo(bw_mbps=300.0, lat_ms_p95=25.0),
            software=sw,
        ),
        metrics=None,
        score=score,
        score_components={},
        overload_strikes=0,
        uptime_ratio=1.0,
        failure_rate=0.0,
        last_seq=None,
        last_heartbeat=None,
        registered_at=0.0,
    )


def test_quantized_model_rejects_no_torchao_nodes() -> None:
    pool = [_quant_node(f"n{i:02d}", backends=("fp16", "fp32")) for i in range(6)]
    plan = plan_placement(QUANT_MANIFEST, pool, layers_per_node_target=4, max_k=4)
    assert plan is None


def test_quantized_model_rejects_wrong_layout() -> None:
    pool = [_quant_node(f"n{i:02d}", layouts=("tensor_core_tiled",)) for i in range(6)]
    plan = plan_placement(QUANT_MANIFEST, pool, layers_per_node_target=4, max_k=4)
    assert plan is None


def test_quantized_model_selects_capable_nodes() -> None:
    capable = [_quant_node(f"cap{i}", score=0.8 - i * 0.01) for i in range(6)]
    plan = plan_placement(QUANT_MANIFEST, capable, layers_per_node_target=4, max_k=4)
    assert plan is not None and len(plan.stages) == 4


def test_quantized_model_rejects_missing_adapter_capability() -> None:
    """A manifest exported for an adapter the node doesn't advertise must not
    be placed on it (adapters are an execution capability, not metadata)."""
    adapter_manifest = QUANT_MANIFEST.model_copy(update={"adapter_id": "llama"})
    pool = [_quant_node(f"n{i:02d}", adapters=()) for i in range(6)]
    plan = plan_placement(adapter_manifest, pool, layers_per_node_target=4, max_k=4)
    assert plan is None


def test_quantized_model_requires_adapter_capability() -> None:
    adapter_manifest = QUANT_MANIFEST.model_copy(update={"adapter_id": "llama"})
    pool = [_quant_node(f"n{i:02d}", adapters=("llama",)) for i in range(6)]
    plan = plan_placement(adapter_manifest, pool, layers_per_node_target=4, max_k=4)
    assert plan is not None and len(plan.stages) == 4


def test_quantized_model_rejects_unsupported_activation_dtype() -> None:
    """A bf16-exported manifest must not run on a node advertising fp16 only."""
    assert QUANT_MANIFEST.quantization is not None
    bf16 = QUANT_MANIFEST.model_copy(
        update={
            "quantization": QUANT_MANIFEST.quantization.model_copy(
                update={"activation_dtype": "bf16"}
            )
        }
    )
    pool = [_quant_node(f"n{i:02d}", activation_dtypes=("fp16",)) for i in range(6)]
    plan = plan_placement(bf16, pool, layers_per_node_target=4, max_k=4)
    assert plan is None


def test_legacy_model_still_places_on_any_node() -> None:
    pool = [node(f"n{i:02d}", score=0.5 - i * 0.01) for i in range(6)]
    plan = plan_placement(MANIFEST, pool, layers_per_node_target=4, max_k=4)
    assert plan is not None and len(plan.stages) == 4
