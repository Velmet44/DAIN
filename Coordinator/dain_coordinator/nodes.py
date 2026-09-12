"""Node lifecycle domain logic (S2): registration, heartbeats, state machine.

Implements the transition table of spec §6 exactly, recomputes the node score on
every metrics report (spec §7), and persists every accepted transition to
state_history. WS message parsing/dispatch also lives here so the transport
layer (api.py) stays thin.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from dain_common.schemas import (
    Envelope,
    Heartbeat,
    MessageType,
    MetricsReport,
    NodeState,
    Register,
    RegisterAck,
    ShardRef,
    parse_payload,
)
from dain_common.scoring import NodeReputation, ScoreResult, ewma, score_node

from dain_coordinator.settings import CoordinatorSettings
from dain_coordinator.store import NodeRow, SQLiteRegistry, StateChange

log = logging.getLogger("dain.coordinator.nodes")

# Allowed edges, straight from the spec §6 table + ASCII diagram. From-state
# None is the initial registration.
ALLOWED_TRANSITIONS: set[tuple[NodeState | None, NodeState]] = {
    (None, NodeState.ONLINE),
    (NodeState.ONLINE, NodeState.BUSY),
    (NodeState.ONLINE, NodeState.DEGRADED),
    (NodeState.ONLINE, NodeState.OFFLINE),
    (NodeState.BUSY, NodeState.ONLINE),
    (NodeState.BUSY, NodeState.DEGRADED),
    (NodeState.BUSY, NodeState.OFFLINE),
    (NodeState.DEGRADED, NodeState.ONLINE),
    (NodeState.DEGRADED, NodeState.OFFLINE),
    (NodeState.OFFLINE, NodeState.ONLINE),
}


class MessageOutcome:
    """Result of handling one WS message: OK keeps the connection open."""

    OK = "ok"
    CLOSE_MALFORMED = "close_malformed"
    CLOSE_IDENTITY = "close_identity"


@dataclass(frozen=True)
class DeregisterResult:
    ok: bool
    status_code: int
    detail: str


class NodeService:
    def __init__(
        self,
        registry: SQLiteRegistry,
        settings: CoordinatorSettings,
        is_connected: Callable[[str], bool] | None = None,
    ) -> None:
        self.registry = registry
        self.settings = settings
        # Set by the app: liveness of the node's live WS connection. Used to tell
        # a restarted node (row ONLINE but no live socket) from a live one, so
        # join-token re-admission can be instant instead of waiting out the
        # heartbeat timeout without relaxing the anti-impersonation guard.
        self.is_connected = is_connected
        # Set by the app: callable(node_id, to_state) fired when the *pool
        # composition* changes for scheduling purposes (DEGRADED entry/recovery,
        # spec §12 placement recompute on DEGRADED transitions).
        self.on_pool_change: Callable[[str, NodeState], None] | None = None
        # Set by the app: callable(node_id) fired when a node is evicted to
        # OFFLINE by the heartbeat monitor (S7 fault reassignment on node loss).
        self.on_node_lost: Callable[[str], None] | None = None

    # -- scoring ---------------------------------------------------------------

    def _score(
        self, manifest, metrics: MetricsReport | None, reputation: NodeReputation
    ) -> ScoreResult:
        return score_node(
            manifest,
            metrics if metrics is not None else MetricsReport(),
            reputation,
            config=self.settings.scoring,
        )

    # -- registration ------------------------------------------------------------

    def register(self, payload: Register, *, model_store_url: str | None = None) -> RegisterAck:
        now = time.time()
        existing = self.registry.get_node(payload.node_id)
        # A node identifies with EITHER its stored per-node token (normal) or the
        # cluster join token (self-heal: the node dropped its node_token and the
        # agent retries with the join token — admit it and rotate a fresh one so
        # it can authenticate heartbeats again).
        node_token_ok = existing is not None and secrets.compare_digest(
            payload.auth_token, existing.node_token
        )
        join_token_ok = payload.auth_token != "" and secrets.compare_digest(
            payload.auth_token, self.settings.join_token
        )
        if not (node_token_ok or join_token_ok):
            reason = "invalid node token" if existing is not None else "invalid join token"
            log.warning("register_rejected node=%s reason=%s", payload.node_id, reason)
            return self._ack(False, reason=reason)

        # Join-token re-admission is for SELF-HEAL only: a node that dropped its
        # per-node token proves itself with the cluster secret — but only while
        # it is NOT actively serving. An ONLINE node must keep authenticating
        # with its own token; otherwise anyone holding the (widely-broadcast)
        # join token could impersonate and silently evict a working node.
        # When the app wires a liveness callback and the ONLINE row has NO live
        # connection, the row is a restarted node: re-admit immediately via the
        # join token instead of making it wait out the heartbeat timeout.
        live = self.is_connected is None or self.is_connected(payload.node_id)
        if (
            existing is not None
            and existing.state == NodeState.ONLINE
            and join_token_ok
            and live
        ):
            log.warning(
                "register_rejected node=%s reason=join_token_cannot_displace_online",
                payload.node_id,
            )
            return self._ack(False, reason="node is ONLINE; use its node token")

        reputation = (
            NodeReputation(uptime_ratio=existing.uptime_ratio, failure_rate=existing.failure_rate)
            if existing is not None
            else NodeReputation()
        )
        score = self._score(payload.manifest, existing.metrics if existing else None, reputation)
        if score.score < self.settings.min_score:
            row = self._new_row(
                payload,
                node_token=existing.node_token if existing else "",
                state=NodeState.OFFLINE,
                score=score,
                now=now,
            )
            if existing is not None:
                row.uptime_ratio, row.failure_rate = existing.uptime_ratio, existing.failure_rate
            self.registry.save_node(row)
            self.registry.append_history(
                StateChange(
                    payload.node_id,
                    existing.state if existing else None,
                    NodeState.OFFLINE,
                    "score_below_min",
                    now,
                )
            )
            log.info(
                "register_rejected node=%s reason=score_below_min score=%.3f",
                payload.node_id,
                score.score,
            )
            return self._ack(False, reason="score below minimum")

        rotated = join_token_ok and existing is not None
        node_token = existing.node_token if node_token_ok else secrets.token_hex(16)
        row = self._new_row(
            payload, node_token=node_token, state=NodeState.ONLINE, score=score, now=now
        )
        if existing is not None:
            row.uptime_ratio, row.failure_rate = existing.uptime_ratio, existing.failure_rate
        self.registry.save_node(row)
        prev = existing.state if existing is not None else None
        if prev != NodeState.ONLINE:
            self.registry.append_history(
                StateChange(payload.node_id, prev, NodeState.ONLINE, "registered", now)
            )
        if rotated:
            log.info(
                "node_token_rotated node=%s agent=%s (re-admitted via join token)",
                payload.node_id,
                payload.agent_version,
            )
        log.info(
            "registered node=%s state=%s score=%.3f agent=%s",
            payload.node_id,
            NodeState.ONLINE.value,
            score.score,
            payload.agent_version,
        )
        return self._ack(True, node_token=node_token, model_store_url=model_store_url)

    def authenticate(self, node_id: str, token: str) -> bool:
        row = self.registry.get_node(node_id)
        return row is not None and secrets.compare_digest(row.node_token, token)

    def deregister(self, node_id: str, token: str) -> DeregisterResult:
        row = self.registry.get_node(node_id)
        if row is None:
            return DeregisterResult(False, 404, "unknown node")
        if not secrets.compare_digest(row.node_token, token):
            return DeregisterResult(False, 403, "invalid node token")
        moved = self.transition(node_id, NodeState.OFFLINE, "deregistered")
        detail = "deregistered" if moved else f"no transition from {row.state.value}"
        return DeregisterResult(True, 200, detail)

    # -- heartbeats & metrics ------------------------------------------------------

    def heartbeat(
        self,
        node_id: str,
        seq: int,
        metrics: MetricsReport | None,
        cached_shards: tuple[ShardRef, ...] = (),
    ) -> None:
        row = self.registry.get_node(node_id)
        if row is None:
            log.warning("heartbeat_from_unknown node=%s", node_id)
            return
        if row.last_seq is not None and seq != row.last_seq + 1:
            log.warning(
                "heartbeat_seq_gap node=%s expected=%s got=%s", node_id, row.last_seq + 1, seq
            )
        row.last_seq = seq
        row.last_heartbeat = time.time()
        row.uptime_ratio = ewma(row.uptime_ratio, 1.0, self.settings.uptime_alpha)
        # Full replacement when the node reports inventory (None = "no change"
        # on a regular beat, so silent heartbeats never wipe the list; an
        # explicit empty tuple revokes shards the node no longer holds).
        if cached_shards is not None:
            row.cached_shards = cached_shards
        self._apply_metrics(row, metrics)

    def metrics_update(self, node_id: str, metrics: MetricsReport) -> None:
        row = self.registry.get_node(node_id)
        if row is None:
            log.warning("metrics_from_unknown node=%s", node_id)
            return
        row.last_heartbeat = time.time()
        self._apply_metrics(row, metrics)

    def _apply_metrics(self, row: NodeRow, metrics: MetricsReport | None) -> None:
        # Capacity numbers drift (apps open/close) while the manifest snapshot
        # is from registration day — refresh free RAM/VRAM from live metrics so
        # placement feasibility uses current memory, not stale memory (S17).
        if metrics is not None:
            manifest = row.manifest
            if metrics.ram_free_gb is not None and manifest.cpu.ram_free_gb != metrics.ram_free_gb:
                manifest = manifest.model_copy(
                    update={
                        "cpu": manifest.cpu.model_copy(
                            update={"ram_free_gb": metrics.ram_free_gb}
                        )
                    }
                )
            if (
                metrics.vram_free_gb is not None
                and manifest.gpu is not None
                and manifest.gpu.vram_free_gb != metrics.vram_free_gb
            ):
                manifest = manifest.model_copy(
                    update={
                        "gpu": manifest.gpu.model_copy(
                            update={"vram_free_gb": metrics.vram_free_gb}
                        )
                    }
                )
            row.manifest = manifest
        reputation = NodeReputation(uptime_ratio=row.uptime_ratio, failure_rate=row.failure_rate)
        score = self._score(row.manifest, metrics, reputation)
        row.metrics = metrics if metrics is not None else row.metrics
        row.score = score.score
        row.score_components = score.components
        degraded, reason = self._evaluate_degraded(row, score, metrics)

        if row.state == NodeState.OFFLINE:
            if not degraded:
                row.state = NodeState.ONLINE
                self.registry.save_node(row)
                self.registry.append_history(
                    StateChange(
                        row.node_id, NodeState.OFFLINE, NodeState.ONLINE, "reconnected", time.time()
                    )
                )
                log.info("reconnected node=%s", row.node_id)
            else:
                self.registry.save_node(row)
            return

        if degraded and row.state in (NodeState.ONLINE, NodeState.BUSY):
            previous_state = row.state
            row.state = NodeState.DEGRADED
            self.registry.save_node(row)
            self.registry.append_history(
                StateChange(row.node_id, previous_state, NodeState.DEGRADED, reason, time.time())
            )
            log.warning("degraded node=%s reason=%s", row.node_id, reason)
            self._notify_pool_change(row.node_id, NodeState.DEGRADED)
            return

        if not degraded and row.state == NodeState.DEGRADED:
            row.state = NodeState.ONLINE
            self.registry.save_node(row)
            self.registry.append_history(
                StateChange(
                    row.node_id,
                    NodeState.DEGRADED,
                    NodeState.ONLINE,
                    "metrics_recovered",
                    time.time(),
                )
            )
            log.info("recovered node=%s", row.node_id)
            self._notify_pool_change(row.node_id, NodeState.ONLINE)
            return

        self.registry.save_node(row)

    def _notify_pool_change(self, node_id: str, to_state: NodeState) -> None:
        if self.on_pool_change is not None:
            try:
                self.on_pool_change(node_id, to_state)
            except Exception:  # noqa: BLE001 — observability hooks must not break the node loop
                log.exception("pool_change_hook_failed node=%s", node_id)

    def _evaluate_degraded(
        self, row: NodeRow, score: ScoreResult, metrics: MetricsReport | None
    ) -> tuple[bool, str]:
        """Spec §6: DEGRADED on score < min_score, thermal, or sustained overload."""
        s = self.settings
        if score.score < s.min_score:
            return True, "score_below_min"
        if metrics is not None and metrics.temp_c is not None and metrics.temp_c > s.temp_degrade_c:
            return True, "thermal"
        if (
            metrics is not None
            and metrics.gpu_util_pct is not None
            and metrics.gpu_util_pct > s.overload_util_pct
        ):
            row.overload_strikes += 1
            if row.overload_strikes >= s.overload_strikes_to_degrade:
                return True, "overload"
        else:
            row.overload_strikes = 0
        return False, ""

    # -- state machine -------------------------------------------------------------

    def transition(self, node_id: str, to_state: NodeState, reason: str) -> bool:
        row = self.registry.get_node(node_id)
        if row is None:
            return False
        if (row.state, to_state) not in ALLOWED_TRANSITIONS:
            log.warning(
                "transition_rejected node=%s %s→%s", node_id, row.state.value, to_state.value
            )
            return False
        prev = row.state
        row.state = to_state
        self.registry.save_node(row)
        self.registry.append_history(StateChange(node_id, prev, to_state, reason, time.time()))
        log.info(
            "state_transition node=%s %s→%s reason=%s", node_id, prev.value, to_state.value, reason
        )
        return True

    def mark_busy(self, node_id: str) -> bool:
        """Scheduler-facing (S6): ONLINE → BUSY on workload assignment."""
        return self.transition(node_id, NodeState.BUSY, "workload_assigned")

    def apply_verification_penalty(self, node_id: str, note: str) -> None:
        """S8 (spec §15): a ledger verification flag drags the node's score down
        by cutting its persisted uptime_ratio below the soft-penalty threshold
        (scoring applies ×0.5 for uptime_ratio < min_uptime_soft on the next
        metrics report). The raw value is pinned so repeated flags cap the
        penalty at the configured soft_penalty."""
        row = self.registry.get_node(node_id)
        if row is None:
            log.warning("verification_penalty_unknown node=%s", node_id)
            return
        row.uptime_ratio = min(row.uptime_ratio, self.settings.scoring.min_uptime_soft * 0.75)
        self.registry.save_node(row)
        log.warning("verification_penalty node=%s note=%r", node_id, note)

    def mark_online(self, node_id: str) -> bool:
        """Scheduler-facing (S6): BUSY → ONLINE when a job finishes."""
        return self.transition(node_id, NodeState.ONLINE, "job_finished")

    def release_node(self, node_id: str) -> None:
        """Scheduler-facing (S5): best-effort release after a terminal job.

        No-op when the node is not BUSY (it may have degraded or gone offline
        during the job — those states must not be papered over here).
        """
        row = self.registry.get_node(node_id)
        if row is not None and row.state == NodeState.BUSY:
            self.mark_online(node_id)

    def enforce_timeouts(self, now: float) -> list[str]:
        """Monitor hook: any non-OFFLINE node silent for offline_timeout_s → OFFLINE."""
        evicted: list[str] = []
        for row in self.registry.list_nodes():
            if row.state == NodeState.OFFLINE:
                continue
            if now - row.heartbeat_ref() > self.settings.offline_timeout_s:
                if self.transition(row.node_id, NodeState.OFFLINE, "heartbeat_timeout"):
                    evicted.append(row.node_id)
        for node_id in evicted:
            if self.on_node_lost is not None:
                try:
                    self.on_node_lost(node_id)
                except Exception:  # noqa: BLE001 — observability hooks must not break the loop
                    log.exception("on_node_lost_hook_failed node=%s", node_id)
        return evicted

    # -- WS message dispatch ----------------------------------------------------------

    def handle_message(self, node_id: str, raw: str) -> str:
        """Parse + apply one WS message; returns a MessageOutcome constant."""
        try:
            envelope = Envelope.model_validate_json(raw)
            payload = parse_payload(envelope)
        except Exception:
            log.warning("malformed_message node=%s", node_id)
            return MessageOutcome.CLOSE_MALFORMED
        payload_node_id = getattr(payload, "node_id", None)
        if payload_node_id is not None and payload_node_id != node_id:
            log.warning("identity_mismatch node=%s claimed=%s", node_id, payload_node_id)
            return MessageOutcome.CLOSE_IDENTITY

        if envelope.type == MessageType.HEARTBEAT:
            assert isinstance(payload, Heartbeat)
            self.heartbeat(node_id, payload.seq, payload.metrics, payload.cached_shards)
        elif envelope.type == MessageType.METRICS_REPORT:
            assert isinstance(payload, MetricsReport)
            self.metrics_update(node_id, payload)
        elif envelope.type == MessageType.REGISTER:
            assert isinstance(payload, Register)
            self.register(payload)
        elif envelope.type == MessageType.JOB_STATUS:
            log.debug("job_status_deferred node=%s (processed by api.jobs)", node_id)
        elif envelope.type == MessageType.SHARD_MANIFEST:
            log.debug("shard_manifest_deferred node=%s", node_id)
        elif envelope.type == MessageType.LEDGER_EVENT:
            log.debug("ledger_event_deferred node=%s", node_id)
        else:
            log.debug("message_ignored node=%s type=%s", node_id, envelope.type.value)
        return MessageOutcome.OK

    # -- helpers ---------------------------------------------------------------------

    def _new_row(
        self,
        payload: Register,
        *,
        node_token: str,
        state: NodeState,
        score: ScoreResult,
        now: float,
    ) -> NodeRow:
        return NodeRow(
            node_id=payload.node_id,
            node_token=node_token,
            agent_version=payload.agent_version,
            state=state,
            manifest=payload.manifest,
            metrics=None,
            score=score.score,
            score_components=dict(score.components),
            overload_strikes=0,
            uptime_ratio=1.0,
            failure_rate=0.0,
            last_seq=None,
            last_heartbeat=None,
            registered_at=now,
            cached_shards=payload.cached_shards,
            peer_url=payload.peer_url,
        )

    def _ack(
        self,
        accepted: bool,
        node_token: str | None = None,
        reason: str | None = None,
        model_store_url: str | None = None,
    ) -> RegisterAck:
        return RegisterAck(
            accepted=accepted,
            heartbeat_interval_s=self.settings.heartbeat_interval_s,
            node_token=node_token,
            model_store_url=model_store_url,
            reason=reason,
        )
