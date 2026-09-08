"""Placement planning (S5/S6, spec §8/§12): partition table construction.

Sizing rule (spec §8.1): the score gates admission (feasibility, top-K), while
stage sizes are proportional to each node's measured throughput so per-stage
latencies equalize — pipeline throughput is `1 / max(stage_latency)`. Throughput
is a proxy until S10: claimed GPU TFLOPS, or CPU cores as a relative measure
for CPU-only nodes. Integer split uses largest-remainder with a minimum of one
layer per stage.
"""

from __future__ import annotations

import math

from dain_common.schemas import ModelManifest, NodeState, StageAssignment

from dain_coordinator.store import NodeRow


def throughput_proxy(row: NodeRow) -> float:
    if row.manifest.gpu is not None:
        return max(row.manifest.gpu.tflops_claimed, 0.001)
    return max(row.manifest.cpu.cores * 0.1, 0.001)  # relative CPU proxy


def capacity_gb(row: NodeRow) -> float:
    """Free memory a shard may occupy: VRAM for GPU nodes, RAM for CPU-only."""
    if row.manifest.gpu is not None:
        return row.manifest.gpu.vram_free_gb
    return row.manifest.cpu.ram_free_gb


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


def build_placement(
    manifest: ModelManifest,
    nodes: list[NodeRow],
    *,
    layers_per_node_target: int = 4,
    min_k: int = 1,
    max_k: int = 16,
) -> tuple[StageAssignment, ...] | None:
    """Choose ≤ max_k ONLINE nodes and split layers ∝ throughput.

    Returns the stage graph (contiguous ascending ranges, shard ids matching the
    model store) or None when the pool cannot host the model.
    """
    if not nodes:
        return None
    model_bytes = sum(s.size_bytes for s in manifest.shards)
    model_gb = model_bytes / 1e9
    feasible = [n for n in nodes if n.state == NodeState.ONLINE]
    # Per-stage capacity check: each stage holds ~model_gb / k, so a node fits
    # if its capacity covers its share — evaluated after K is chosen (below).
    ranked = sorted(feasible, key=throughput_proxy, reverse=True)

    k = min(
        max(math.ceil(manifest.layers / layers_per_node_target), min_k, 1),
        max_k,
        len(ranked),
    )
    if k < 1:
        return None
    selected = ranked[:k]
    share_gb = model_gb / k
    if any(capacity_gb(n) < share_gb for n in selected):
        return None  # a selected node cannot hold its stage share

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
    return tuple(stages)


def stage_shard_ids(manifest: ModelManifest, stages: tuple[StageAssignment, ...]) -> list[str]:
    """Shard files each stage must download (all shards overlapping its range)."""
    ids: list[str] = []
    for stage in stages:
        for shard in manifest.shards:
            if shard.layer_start is None or shard.layer_end is None:
                continue
            if shard.layer_start <= stage.layer_end and shard.layer_end >= stage.layer_start:
                if shard.shard_id not in ids:
                    ids.append(shard.shard_id)
    return ids
