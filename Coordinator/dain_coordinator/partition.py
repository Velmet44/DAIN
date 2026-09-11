"""Scheduler (S6, spec §12): feasibility filter → score-ranked top-K →
throughput-proportional sizing → warm backups.

Selection is gated by the node **score** (spec §7/§12) while stage *sizes* are
proportional to measured throughput (spec §8.1 — pipeline throughput is
`1 / max(stage_latency)`). Backups are the next-ranked nodes with capacity;
they are designated at placement time and used for reassignment in S7.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass

from dain_common.schemas import ModelManifest, NodeState, StageAssignment

from dain_coordinator.store import NodeRow

log = logging.getLogger("dain.coordinator.partition")


def _is_backend_feasible(manifest: ModelManifest, row: NodeRow) -> bool:
    """Check whether *row* can host a stage for *manifest*.

    Legacy fp16/fp32 models run anywhere.  INT4-quantized models require the
    node to support the torchao backend, the matching quantization scheme, and
    — critically — the packing layout the model was exported for.
    """
    quant = getattr(manifest, "quantization", None)
    if quant is None or getattr(quant, "backend", "none") == "none":
        return True  # unquantized model — any torch node works
    sw = row.manifest.software
    if "torchao" not in sw.supported_backends:
        return False
    if quant.scheme not in sw.supported_quantization:
        return False
    layout = getattr(quant, "packing_layout", None)
    if layout and layout not in sw.supported_packing_layouts:
        return False
    return True


def throughput_proxy(row: NodeRow) -> float:
    if row.manifest.gpu is not None:
        return max(row.manifest.gpu.tflops_claimed, 0.001)
    return max(row.manifest.cpu.cores * 0.1, 0.001)  # relative CPU proxy


def capacity_gb(row: NodeRow) -> float:
    """Free memory a shard may occupy: VRAM for GPU nodes, RAM for CPU-only."""
    if row.manifest.gpu is not None:
        return row.manifest.gpu.vram_free_gb
    return row.manifest.cpu.ram_free_gb


@dataclass(frozen=True)
class PlacementPlan:
    stages: tuple[StageAssignment, ...]
    backups: tuple[str, ...] = ()
    degraded: bool = False


@dataclass(frozen=True)
class PlacementEvent:
    """One placement recompute, surfaced via `GET /v1/nodes` (spec §12).

    Triggers: `request` (per admission), `join`, `leave`, `degraded`,
    `recovered` — spec §12 requires recompute on join/leave/DEGRADED.
    """

    ts: float
    trigger: str
    model_id: str
    k: int | None
    degraded: bool
    node_ids: tuple[str, ...]
    backups: tuple[str, ...]


class PlacementRecorder:
    """Ring buffer of placement recompute events (observability, spec §12)."""

    def __init__(self, capacity: int = 64) -> None:
        self._events: list[PlacementEvent] = []
        self._capacity = capacity

    def record(self, event: PlacementEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._capacity:
            del self._events[: len(self._events) - self._capacity]
        log.info(
            "placement_event trigger=%s model=%s k=%s degraded=%s nodes=%s",
            event.trigger,
            event.model_id,
            event.k,
            event.degraded,
            ",".join(event.node_ids),
        )

    def events(self) -> list[PlacementEvent]:
        return list(self._events)

    def last(self) -> PlacementEvent | None:
        return self._events[-1] if self._events else None


def recompute_pool(
    manifest: ModelManifest,
    rows: list[NodeRow],
    *,
    layers_per_node_target: int,
    min_k: int,
    max_k: int,
    backup_count: int,
    min_nodes: int,
    trigger: str,
    recorder: PlacementRecorder,
) -> PlacementPlan | None:
    """Recompute the placement for `manifest` and record the event (spec §12)."""
    plan = plan_placement(
        manifest,
        rows,
        layers_per_node_target=layers_per_node_target,
        min_k=min_k,
        max_k=max_k,
        backup_count=backup_count,
        min_nodes=min_nodes,
    )
    recorder.record(
        PlacementEvent(
            ts=time.time(),
            trigger=trigger,
            model_id=manifest.model_id,
            k=len(plan.stages) if plan else None,
            degraded=plan.degraded if plan else False,
            node_ids=tuple(s.node_id for s in plan.stages) if plan else (),
            backups=plan.backups if plan else (),
        )
    )
    return plan


def event_view(event: PlacementEvent) -> dict:
    return asdict(event)


def _integer_split(total: int, weights: list[float]) -> list[int]:
    """Largest-remainder split of `total` into positive ints ∝ weights."""
    k = len(weights)
    if k > total:
        raise ValueError("more stages than layers")
    raw = [total * w / sum(weights) for w in weights]
    counts = [max(1, math.floor(x)) for x in raw]
    remainder = total - sum(counts)
    order = sorted(range(k), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    for i in range(remainder):
        counts[order[i % k]] += 1
    # Trim any overshoot from the min-1 clamping.
    while sum(counts) > total:
        heaviest = max(range(k), key=lambda i: counts[i])
        if counts[heaviest] <= 1:
            raise ValueError("cannot trim split below one layer per stage")
        counts[heaviest] -= 1
    return counts


def plan_placement(
    manifest: ModelManifest,
    nodes: list[NodeRow],
    *,
    layers_per_node_target: int = 4,
    min_k: int = 1,
    max_k: int = 16,
    backup_count: int = 2,
    min_nodes: int = 1,
) -> PlacementPlan | None:
    """Spec §12: feasible → score-ranked top-K → sized placement → warm backups.

    Returns None when the pool cannot host the model (caller → HTTP 429 /
    degraded mode). Below `min_nodes` the pool is considered unable to serve at
    all (spec §13 degraded-mode floor). When fewer than the desired K stages fit,
    the plan is served with fewer, larger stages and flagged `degraded`.
    """
    if not nodes:
        return None
    model_gb = sum(s.size_bytes for s in manifest.shards) / 1e9

    feasible = [
        n for n in nodes
        if n.state == NodeState.ONLINE and _is_backend_feasible(manifest, n)
    ]
    # Spec §12: ranked = sort_by_score_desc(feasible); deterministic tie-breaks.
    ranked = sorted(
        feasible,
        key=lambda n: (
            -(n.score if n.score is not None else -1.0),
            -throughput_proxy(n),
            n.node_id,
        ),
    )
    if len(ranked) < min_nodes:
        return None
    if not ranked:
        return None

    desired_k = max(math.ceil(manifest.layers / layers_per_node_target), min_k, 1)
    k = min(desired_k, max_k, len(ranked))
    if k < 1:
        return None
    degraded = k < desired_k
    share_gb = model_gb / k

    # Capacity filter applied to the ranked order: take the first k nodes whose
    # free memory covers the per-stage share.
    selected: list[NodeRow] = []
    rest: list[NodeRow] = []
    for row in ranked:
        if len(selected) < k and capacity_gb(row) >= share_gb:
            selected.append(row)
        else:
            rest.append(row)
    if len(selected) < k:
        return None

    counts = _integer_split(manifest.layers, [throughput_proxy(n) for n in selected])
    stages: list[StageAssignment] = []
    layer = 0
    for idx, (node, count) in enumerate(zip(selected, counts, strict=True)):
        stage_start, stage_end = layer, layer + count - 1
        stages.append(
            StageAssignment(
                stage_idx=idx,
                node_id=node.node_id,
                shard_id=f"layers_{stage_start:02d}_{stage_end:02d}",
                layer_start=stage_start,
                layer_end=stage_end,
            )
        )
        layer = stage_end + 1

    backups = tuple(row.node_id for row in rest[:backup_count])
    return PlacementPlan(stages=tuple(stages), backups=backups, degraded=degraded)
