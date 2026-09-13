"""Model portfolio assignment (S22a): the coordinator's desired state.

The controller loop recomputes, for every model in the store, which connected
nodes should hold it, and reconciles each node's portfolio through
`ASSIGNMENT` envelopes. Nodes provision in the background and report
`MODEL_STATUS`; only `files_ready`/`warm` nodes are schedulable for a model.

Policy (plan §3.1/§3.5):
- replicas first: a model is assigned whole (mode="replica") to the best
  `replicas_per_model` nodes whose free memory covers its runtime footprint;
  demand bumps the count up to `max_replicas_per_model`.
- models too big for any single node (even with overcommit) are flagged
  pipeline-only: every stage node of the current `plan_placement` gets a
  `mode="pipeline"` assignment for its own layer range.
- everything a node holds that the desired set no longer wants is revoked.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

from dain_common.schemas import (
    Envelope,
    MessageType,
    ModelAssignment,
    ModelManifest,
    ModelStatus,
    NodeState,
)

from dain_coordinator.partition import (
    _is_backend_feasible,
    _rank_replicas,
    capacity_gb,
    plan_placement,
    runtime_size_gb,
)
from dain_coordinator.settings import CoordinatorSettings
from dain_coordinator.store import NodeRow

log = logging.getLogger("dain.coordinator.assignments")


class ReadinessMap:
    """Latest MODEL_STATUS per (node, model). Ephemeral by design: a node
    re-reports after reconnect, so restarts self-heal."""

    def __init__(self) -> None:
        self._by_node: dict[str, dict[str, ModelStatus]] = {}

    def update(self, status: ModelStatus) -> None:
        self._by_node.setdefault(status.node_id, {})[status.model_id] = status

    def get(self, node_id: str, model_id: str) -> ModelStatus | None:
        return self._by_node.get(node_id, {}).get(model_id)

    def drop_node(self, node_id: str) -> None:
        self._by_node.pop(node_id, None)

    def schedulable(self, node_id: str, model_id: str) -> bool:
        """True when the node reported files_ready/warm (not downloading/error)."""
        status = self.get(node_id, model_id)
        return status is not None and status.state in ("files_ready", "warm")

    def warm(self, node_id: str, model_id: str) -> bool:
        status = self.get(node_id, model_id)
        return status is not None and status.state in ("warm", "serving")

    def any_downloading(self, model_id: str) -> ModelStatus | None:
        for statuses in self._by_node.values():
            status = statuses.get(model_id)
            if status is not None and status.state == "downloading":
                return status
        return None


class DemandTracker:
    """Recent request arrivals per model (in-memory; only feeds scaling)."""

    def __init__(self, window_s: float = 600.0) -> None:
        self._window_s = window_s
        self._events: deque[tuple[float, str]] = deque()

    def record(self, model_id: str) -> None:
        now = time.time()
        self._events.append((now, model_id))
        while self._events and now - self._events[0][0] > self._window_s:
            self._events.popleft()

    def recent(self, model_id: str) -> int:
        cutoff = time.time() - self._window_s
        return sum(1 for ts, mid in self._events if mid == model_id and ts >= cutoff)


@dataclass(frozen=True)
class AssignmentPlan:
    """The full desired portfolio: node_id -> assignments for that node."""

    by_node: dict[str, list[ModelAssignment]]
    pipeline_only: frozenset[str]

    def diff(self, current: dict[str, list[ModelAssignment]]) -> list[tuple[str, ModelAssignment]]:
        """(node_id, assignment) entries that changed vs the last applied plan:
        new/changed `ensure`s plus `revoke`s for models the node should drop."""
        new_by_key: dict[tuple[str, str], ModelAssignment] = {}
        for node_id, desired in self.by_node.items():
            for a in desired:
                new_by_key[(node_id, a.model_id)] = a

        changes: list[tuple[str, ModelAssignment]] = []
        for node_id, entries in current.items():
            for a in entries:
                new = new_by_key.get((node_id, a.model_id))
                if new is None:
                    changes.append(
                        (
                            node_id,
                            ModelAssignment(
                                model_id=a.model_id,
                                action="revoke",
                                mode=a.mode,
                                layer_start=a.layer_start,
                                layer_end=a.layer_end,
                            ),
                        )
                    )
        for (node_id, _model_id), a in new_by_key.items():
            cur = next((x for x in current.get(node_id, []) if x.model_id == a.model_id), None)
            if cur is None or (
                cur.action,
                cur.mode,
                cur.layer_start,
                cur.layer_end,
            ) != (a.action, a.mode, a.layer_start, a.layer_end):
                changes.append((node_id, a))
        return changes


def compute_assignments(
    manifests: list[ModelManifest],
    rows: list[NodeRow],
    *,
    connected: set[str],
    readiness: ReadinessMap,
    demand: DemandTracker,
    settings: CoordinatorSettings,
) -> AssignmentPlan:
    """Desired portfolio for every connected node (pure; no side effects)."""
    live = [r for r in rows if r.state == NodeState.ONLINE and r.node_id in connected]
    by_node: dict[str, list[ModelAssignment]] = {}
    pipeline_only: set[str] = set()

    for manifest in manifests:
        model_id = manifest.model_id
        candidates = [
            r
            for r in live
            if _is_backend_feasible(manifest, r)
            and capacity_gb(r) >= runtime_size_gb(manifest)
        ]
        if candidates:
            desired_n = min(
                settings.max_replicas_per_model,
                max(
                    settings.replicas_per_model,
                    2 if demand.recent(model_id) >= settings.demand_scale_threshold else 1,
                ),
                len(candidates),
            )
            chosen = _rank_replicas(candidates, readiness, model_id)[:desired_n]
            for row in chosen:
                by_node.setdefault(row.node_id, []).append(
                    ModelAssignment(model_id=model_id, action="ensure", mode="replica")
                )
            continue

        # No single node fits it: pipeline-only. Every stage of the current
        # placement provisions its own layer range (files tier only).
        pipeline_only.add(model_id)
        plan = plan_placement(
            manifest,
            [r for r in live if _is_backend_feasible(manifest, r)],
            layers_per_node_target=settings.layers_per_node_target,
            min_k=settings.min_stages,
            max_k=settings.max_stages,
            backup_count=settings.backup_count,
            min_nodes=settings.min_nodes,
            allow_overcommit=settings.allow_memory_overcommit,
        )
        if plan is None:
            log.warning("assignment_unhostable model=%s (no feasible placement)", model_id)
            continue
        for stage in plan.stages:
            by_node.setdefault(stage.node_id, []).append(
                ModelAssignment(
                    model_id=model_id,
                    action="ensure",
                    mode="pipeline",
                    layer_start=stage.layer_start,
                    layer_end=stage.layer_end,
                )
            )

    return AssignmentPlan(by_node=by_node, pipeline_only=frozenset(pipeline_only))


class AssignmentService:
    """Stateful assignment reconciler (controller-side).

    `recompute()` computes the desired portfolio, diffs it against the last
    applied plan, pushes `ASSIGNMENT` envelopes to connected nodes, and
    persists the plan so a coordinator restart does not re-provision or drop
    node portfolios. `send_portfolio()` replays a node's full desired set on
    WS connect (fresh assignments without waiting for the next tick).
    """

    def __init__(
        self,
        *,
        registry,
        connections,
        readiness: ReadinessMap,
        demand: DemandTracker,
        manifests_fn,
        settings_getter=None,
    ) -> None:
        self._registry = registry
        self._connections = connections
        self._readiness = readiness
        self._demand = demand
        self._manifests_fn = manifests_fn
        self._settings_getter = settings_getter
        self._current: dict[str, list[ModelAssignment]] = {}
        self._loaded = False

    # -- state -------------------------------------------------------------------

    def _load_once(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            persisted = self._registry.load_assignments()
        except Exception:  # noqa: BLE001 — a broken table must not kill the pool
            log.exception("assignments_load_failed")
            return
        self._current = {
            node_id: [
                ModelAssignment(
                    model_id=m, action="ensure", mode=mode, layer_start=ls, layer_end=le
                )
                for (m, mode, ls, le) in entries
            ]
            for node_id, entries in persisted.items()
        }

    def current_portfolio(self, node_id: str) -> list[ModelAssignment]:
        self._load_once()
        return list(self._current.get(node_id, []))

    def _settings(self):
        if self._settings_getter is not None:
            return self._settings_getter()
        from dain_coordinator.settings import CoordinatorSettings

        return CoordinatorSettings()

    # -- reconciliation ------------------------------------------------------------

    async def send_portfolio(self, node_id: str, websocket) -> int:
        """Push the node's full desired portfolio down its fresh WS. Returns count."""
        self._load_once()
        sent = 0
        for assignment in self.current_portfolio(node_id):
            envelope = Envelope.wrap(MessageType.ASSIGNMENT, assignment, ts=time.time())
            try:
                await websocket.send_text(envelope.model_dump_json())
                sent += 1
            except Exception:  # noqa: BLE001 — transport races on connect
                break
        if sent:
            log.info("assignments_sent node=%s count=%d", node_id, sent)
        return sent

    async def recompute(self, trigger: str) -> int:
        self._load_once()
        manifests = self._manifests_fn()
        if not manifests:
            return 0
        rows = self._registry.list_nodes()
        plan = compute_assignments(
            manifests,
            rows,
            connected=set(self._connections.connected_ids()),
            readiness=self._readiness,
            demand=self._demand,
            settings=self._settings(),
        )
        changes = plan.diff(self._current)
        if not changes:
            return 0
        delivered = 0
        for node_id, assignment in changes:
            envelope = Envelope.wrap(MessageType.ASSIGNMENT, assignment, ts=time.time())
            if await self._connections.send_envelope(node_id, envelope):
                delivered += 1
            else:
                log.debug("assignment_deferred node=%s (not connected)", node_id)
        self._current = plan.by_node
        self._registry.replace_assignments(
            [
                (node_id, a.model_id, a.mode, a.layer_start, a.layer_end)
                for node_id, entries in plan.by_node.items()
                for a in entries
            ]
        )
        log.info(
            "assignments_recomputed trigger=%s changes=%d delivered=%d pipeline_only=%s",
            trigger,
            len(changes),
            delivered,
            ",".join(sorted(plan.pipeline_only)) or "-",
        )
        return len(changes)
