"""Fault tolerance orchestration (S7, spec §13).

Responsible for detection (stage watchdog), reassignment (impression of an
OFFLINE / stalled stage onto a warm backup), and the retry control messages
that resume a job from its buffered prefix instead of the prompt.

Responsibilities live here rather than in jobs.py so the job tracker stays a
pure bookkeeper:
- `FaultManager.handle_node_lost(node_id)` — reassign every in-flight stage
  hosted on `node_id` (triggered by WS disconnect / heartbeat OFFLINE).
- `FaultManager.tick()` — the stage watchdog: a stage that makes no progress
  within `4 × p99(stage latency)` (clamped) is treated as failed and retried,
  same recovery path as node loss.
- `_reassign(...)` picks a warm backup, re-sends `JOB_ASSIGN` for the
  reassigned stage, updates the job's stage graph, and asks the stage's
  *upstream* to replay the activation it is still buffering (`STAGE_RETRY`).
  Only the failed stage onward is recomputed; the completed prefix survives.

Idempotency: every retry increments `job.retries[stage_idx]`, which becomes the
`attempt` carried on re-dispatched `JOB_ASSIGN`/`STAGE_RETRY` and on the eventual
`LEDGER_EVENT` (S8), so `(job_id, stage_idx, attempt)` stays unique — retries
never produce duplicate ledger keys (§13/§15).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from dain_common.schemas import (
    Envelope,
    GenerationParams,
    JobAssign,
    JobState,
    MessageType,
    NodeState,
    StageRetry,
)

from dain_coordinator.jobs import JobRecord, JobTracker
from dain_coordinator.settings import CoordinatorSettings

log = logging.getLogger("dain.coordinator.faults")


class FaultManager:
    def __init__(
        self,
        settings_getter: CoordinatorSettings | Callable[[], CoordinatorSettings],
        jobs: JobTracker,
        connections,
        registry,
        service,
        readiness=None,
    ) -> None:
        # Accept either a snapshot or a getter so admin edits to runtime settings
        # (layers_per_node_target, max_job_restarts, …) apply live instead of
        # being frozen at startup.
        self._settings: Callable[[], CoordinatorSettings] = (
            settings_getter if callable(settings_getter) else (lambda: settings_getter)
        )
        self.jobs = jobs
        self.connections = connections
        self.registry = registry
        self.service = service
        # Optional ReadinessMap: replacements prefer nodes that already hold
        # the model (files_ready/warm) over unprovisioned ones.
        self.readiness = readiness

    @property
    def settings(self) -> CoordinatorSettings:
        return self._settings()

    # -- triggers ---------------------------------------------------------------

    def handle_node_lost(self, node_id: str) -> None:
        """Reassign every in-flight stage hosted on `node_id` (spec §13).

        Called on WS disconnect / heartbeat-timeout OFFLINE. Runs synchronously:
        it mutates the job graph in place, so any later relay attempt uses the
        new mapping.
        """
        for job in self.jobs.active_jobs():
            for stage_idx, stage in enumerate(job.stages):
                if stage.node_id == node_id:
                    log.warning(
                        "node_lost_reassign job=%s stage=%d node=%s",
                        job.job_id,
                        stage_idx,
                        node_id,
                    )
                    self._reassign(job, stage_idx, reason="node_lost")

    async def tick(self) -> None:
        """Stage watchdog: fail/retry stages that stop making progress.

        `deadline = 4 × p99(stage latency)`, clamped to
        [stage_deadline_min_s, stage_deadline_max_s] (spec §13). Only the stage
        that stalled is reassigned; a whole-job stall (no progress at all since
        dispatch) is covered by the request timeout; only post-token stage
        inactivity is handled by the watchdog.
        """
        now = time.time()
        deadline = self.jobs.stage_stats.deadline(
            self.settings.stage_deadline_min_s, self.settings.stage_deadline_max_s
        )
        for job in self.jobs.active_jobs():
            if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.RETRYING):
                continue
            if job.last_token_at is None:
                continue
                # `restarts` is incremented inside _restart_from_entry — the only
                # place — so `>=` here means the job gets exactly
                # `max_job_restarts` useful restarts before being failed.
            for stage_idx in range(len(job.stages)):
                last = job.stage_last_activity.get(stage_idx)
                if last is None:
                    continue
                if now - last > deadline:
                    self._reassign(job, stage_idx, reason="watchdog")

    # -- reassignment -----------------------------------------------------------

    def _restart_from_entry(self, job: JobRecord) -> None:
        """Whole-job stall (or entry-stage loss): restart from the prompt.

        Re-dispatches the full stage graph on the current placement and resets
        per-stage progress. The KV cache that produced already-streamed context
        is gone, so the job starts over — spec §13 permits restart from prompt
        only when the KV-cache-holding node is lost. Stages whose node died in
        the meantime keep their assignment: the re-sent `JOB_ASSIGN` fails to
        deliver, and the watchdog reassigns those stages onto warm backups via
        the normal `_reassign` path.
        """
        if job.state in (JobState.COMPLETED, JobState.FAILED):
            return
        log.info("job_restart job=%s", job.job_id)
        now = time.time()
        for stage_idx in range(len(job.stages)):
            job.stage_started_at.pop(stage_idx, None)
            job.stage_finished_at.pop(stage_idx, None)
            job.stage_last_activity[stage_idx] = now
        job.tokens.clear()
        job.last_token_at = None
        job.first_token_at = None
        job.restarts += 1
        job.dispatched_at = now
        job.state = JobState.DISPATCHED
        self.jobs._push(job, {"type": "reset", "job_id": job.job_id})
        # Re-fire JOB_ASSIGN for every stage of the current placement; stage 0
        # carries the prompt so the graph starts over (nodes replace their
        # runtime for the same job_id, so repeat assigns are safe).
        for stage in job.stages:
            assign = JobAssign(
                job_id=job.job_id,
                model_id=job.model_id,
                my_stage_idx=stage.stage_idx,
                stages=job.stages,
                prompt=job.prompt if stage.stage_idx == 0 else None,
                params=GenerationParams(
                    max_tokens=int(job.params.get("max_tokens", 64)),
                    temperature=float(job.params.get("temperature", 0.0)),
                    seed=job.params.get("seed"),
                ),
            )
            self._fire(MessageType.JOB_ASSIGN, assign, stage.node_id)

    def _pick_replacement(self, job: JobRecord, failed_node: str) -> str | None:
        """A backup that is ONLINE, connected, and free for this job.

        Nodes that already hold the model (readiness files_ready/warm) rank
        ahead of unprovisioned ones — a replacement without the shards turns
        'instant failover' into a multi-GB fetch mid-request.
        """
        current = {s.node_id for s in job.stages}
        dark = set(current) | {failed_node}
        candidates = [n for n in job.backups if n not in dark]
        seen = set(job.backups) | dark
        for row in self.registry.list_nodes(NodeState.ONLINE):
            if row.node_id not in seen:
                candidates.append(row.node_id)
                seen.add(row.node_id)

        def ready(node_id: str) -> int:
            if self.readiness is None:
                return 0
            return 0 if self.readiness.schedulable(node_id, job.model_id) else 1

        ordered = sorted(set(candidates), key=lambda n: (ready(n), n))
        for cand in ordered:
            row = self.registry.get_node(cand)
            if (
                row is not None
                and row.state == NodeState.ONLINE
                and self.connections.is_connected(cand)
            ):
                return cand
        return None

    def _reassign(self, job: JobRecord, stage_idx: int, *, reason: str) -> None:
        if job.state in (JobState.COMPLETED, JobState.FAILED):
            return
        # Replica jobs are single-stage: a lost node is healed by moving the
        # whole stage to a backup (prompt rides the re-fired JOB_ASSIGN), not
        # by restarting on the dead node. The regeneration restarts the
        # transcript, so the client is told to reset its accumulated text.
        if stage_idx == 0 and job.serving_mode == "replica":
            replacement = self._pick_replacement(job, job.stages[0].node_id)
            if replacement is None:
                log.warning(
                    "no_replacement job=%s replica node=%s", job.job_id, job.stages[0].node_id
                )
                self.jobs.fail_job(job.job_id, f"no backup node for replica ({reason})")
                return
            attempts = job.retries.get(0, 0)
            if attempts >= self.settings.max_stage_attempts:
                self.jobs.fail_job(job.job_id, f"replica exhausted retries ({reason})")
                return
            failed_node = job.stages[0].node_id
            job.retries[0] = attempts + 1
            job.attempt_nodes[(0, attempts)] = failed_node
            job.stages = tuple(
                s.model_copy(update={"node_id": replacement}) if s.stage_idx == 0 else s
                for s in job.stages
            )
            job.state = JobState.RETRYING
            # The whole (single) stage now runs on the replacement; move the
            # anchor so session accounting (_active_sessions), fail_jobs_of_node
            # and the admin job view attribute the work to the live node.
            job.node_id = replacement
            log.warning(
                "replica_reassign job=%s node=%s->%s attempt=%d reason=%s",
                job.job_id,
                failed_node,
                replacement,
                attempts + 1,
                reason,
            )
            job.tokens.clear()
            job.last_token_at = None
            job.first_token_at = None
            self.jobs._push(
                job, {"type": "reset", "job_id": job.job_id}
            )
            self._fire(
                MessageType.JOB_ASSIGN,
                JobAssign(
                    job_id=job.job_id,
                    model_id=job.model_id,
                    my_stage_idx=0,
                    stages=job.stages,
                    prompt=job.prompt,
                    params=GenerationParams(
                        max_tokens=int(job.params.get("max_tokens", 64)),
                        temperature=float(job.params.get("temperature", 0.0)),
                        seed=job.params.get("seed"),
                    ),
                ),
                replacement,
            )
            job.stage_last_activity[0] = time.time()
            return
        # Entry stage (0): the KV prefix chain starts at the prompt, so a lone
        # entry reassign cannot resume — live downstream stages still hold caches
        # built on the old activations and would stream garbage (§13). Restart
        # the whole job from the prompt instead; the re-fired JOB_ASSIGN is what
        # revokes the old (possibly still-connected) entry task.
        if stage_idx == 0:
            if job.restarts >= self.settings.max_job_restarts:
                log.error(
                    "entry_restarts_exhausted job=%s restarts=%d reason=%s",
                    job.job_id,
                    job.restarts,
                    reason,
                )
                self.jobs.fail_job(job.job_id, f"entry stage lost beyond max restarts ({reason})")
                return
            self._restart_from_entry(job)
            return
        attempts = job.retries.get(stage_idx, 0)
        if attempts >= self.settings.max_stage_attempts:
            log.error(
                "stage_retries_exhausted job=%s stage=%d attempts=%d",
                job.job_id,
                stage_idx,
                attempts,
            )
            self.jobs.fail_job(job.job_id, f"stage {stage_idx} exhausted retries")
            return
        failed_node = job.stages[stage_idx].node_id
        replacement = self._pick_replacement(job, failed_node)
        if replacement is None:
            log.warning(
                "no_replacement job=%s stage=%d node=%s", job.job_id, stage_idx, failed_node
            )
            self.jobs.fail_job(job.job_id, f"no backup node for stage {stage_idx}")
            return

        attempts += 1
        job.retries[stage_idx] = attempts
        # S8 (spec §15): the ledger attributes the abandoned attempt to the node
        # that actually served it; the replacement runs the next attempt.
        job.attempt_nodes[(stage_idx, attempts - 1)] = failed_node
        new_stages = tuple(
            (
                job.stages[i].model_copy(update={"node_id": replacement})
                if i == stage_idx
                else job.stages[i]
            )
            for i in range(len(job.stages))
        )
        job.stages = new_stages
        job.state = JobState.RETRYING
        log.warning(
            "stage_reassign job=%s stage=%d node=%s->%s attempt=%d reason=%s",
            job.job_id,
            stage_idx,
            failed_node,
            replacement,
            attempts,
            reason,
        )
        if job.serving_mode != "replica":
            # Pipeline jobs keep exclusive BUSY semantics; replica jobs share
            # the node with other sessions (load is a metric, S22d).
            if not self.service.mark_busy(replacement):
                # The replacement raced to BUSY/offline between _pick_replacement
                # and now. Assigning anyway would stack two pipeline jobs on one
                # node, and tracking it as ours would make the terminal edge
                # release a hold that belongs to another job. Fail cleanly.
                log.warning(
                    "replacement_claim_failed job=%s replacement=%s", job.job_id, replacement
                )
                self.jobs.fail_job(
                    job.job_id, f"replacement {replacement} no longer free for stage {stage_idx}"
                )
                return
            job.busy_nodes.append(replacement)

        assign = JobAssign(
            job_id=job.job_id,
            model_id=job.model_id,
            my_stage_idx=stage_idx,
            stages=new_stages,
            prompt=job.prompt if stage_idx == 0 else None,
            params=GenerationParams(
                max_tokens=int(job.params.get("max_tokens", 64)),
                temperature=float(job.params.get("temperature", 0.0)),
                seed=job.params.get("seed"),
            ),
        )
        self._fire(MessageType.JOB_ASSIGN, assign, replacement)
        job.stage_last_activity[stage_idx] = time.time()

        # Ask the upstream stage to replay the buffered activation so the
        # replacement resumes from the completed prefix (§13).
        if stage_idx > 0:
            upstream = job.stages[stage_idx - 1]
            self._fire(
                MessageType.STAGE_RETRY,
                StageRetry(
                    job_id=job.job_id,
                    stage_idx=stage_idx,
                    attempt=attempts,
                    reason="reassign",
                ),
                upstream.node_id,
            )

    def _fire(self, msg_type: MessageType, payload, node_id: str) -> None:
        """Fire-and-forget an envelope to `node_id` non-blockingly."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._send_task(msg_type, payload, node_id),
            name=f"fault-{getattr(payload, 'job_id', '')}",
        )

    async def _send_task(self, msg_type: MessageType, payload, node_id: str) -> None:
        ok = await self.connections.send_envelope(
            node_id, Envelope.wrap(msg_type, payload, ts=time.time())
        )
        if not ok:
            log.warning("fault_msg_send_failed type=%s node=%s", msg_type.value, node_id)
