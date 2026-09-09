"""Job tracking + relay (S4): request lifecycle, token streaming, activation relay.

Job state machine (spec §5): QUEUED → DISPATCHED → RUNNING → STREAMING →
COMPLETED | FAILED. SSE frames are pushed onto a per-job queue that the
/v1/completions generator drains; terminal frames are idempotent (first writer
wins) so node JOB_STATUS and TOKEN_BATCH finals can race safely.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from dain_common.schemas import (
    ActivationRelayHeader,
    Envelope,
    JobState,
    MessageType,
    ModelManifest,
    StageAssignment,
    TokenBatch,
)

log = logging.getLogger("dain.coordinator.jobs")


@dataclass
class JobRecord:
    job_id: str
    model_id: str
    prompt: str
    params: dict
    manifest: ModelManifest
    node_id: str | None = None
    api_key: str | None = None
    backups: tuple[str, ...] = ()
    stages: tuple[StageAssignment, ...] = ()
    state: JobState = JobState.QUEUED
    tokens: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    dispatched_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    queue: asyncio.Queue | None = None
    stage_started_at: dict[int, float] = field(default_factory=dict)
    stage_finished_at: dict[int, float] = field(default_factory=dict)
    # S7 (spec §13): per-stage retry attempt counters and liveness bookkeeping.
    retries: dict[int, int] = field(default_factory=dict)
    restarts: int = 0
    stage_last_activity: dict[int, float] = field(default_factory=dict)
    last_token_at: float | None = None
    # S8 (spec §15): the ledger is emitted exactly once per terminal job;
    # `attempt_nodes` records which node actually served each `(stage_idx,
    # attempt)` so a retried stage logs against the node that ran the work.
    ledger_emitted: bool = False
    attempt_nodes: dict[tuple[int, int], str] = field(default_factory=dict)


class StageLatencyStats:
    """Rolling stage-latency window → p99 for the watchdog deadline (§13).

    `deadline = 4 × p99(stage latency)` clamped to [min_s, max_s]; before any
    sample exists the max bound (a plain configured timeout) applies.
    """

    def __init__(self, window: int = 64) -> None:
        self._samples: list[float] = []
        self._window = window

    def observe(self, latency_s: float) -> None:
        self._samples.append(latency_s)
        if len(self._samples) > self._window:
            del self._samples[: len(self._samples) - self._window]

    def p99(self) -> float | None:
        if not self._samples:
            return None
        ordered = sorted(self._samples)
        return ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]

    def deadline(self, min_s: float, max_s: float) -> float:
        p99 = self.p99()
        if p99 is None:
            return max_s
        return min(max(4.0 * p99, min_s), max_s)


class JobTracker:
    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}
        self._order: list[str] = []
        self.stage_stats = StageLatencyStats()
        # Set by the app: called once when a job reaches a terminal state, so
        # the S8 ledger can be emitted exactly once (S7 established the job
        # tracker as a pure bookkeeper — side effects live elsewhere).
        self.on_terminal: Callable[[JobRecord], None] | None = None

    def create(
        self,
        model_id: str,
        prompt: str,
        params: dict,
        manifest: ModelManifest,
        *,
        api_key: str | None = None,
        backups: tuple[str, ...] = (),
    ) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            job_id=job_id,
            model_id=model_id,
            prompt=prompt,
            params=params,
            manifest=manifest,
            api_key=api_key,
            backups=backups,
        )
        self.jobs[job_id] = record
        self._order.append(job_id)
        return record

    def attach(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=4096)
        self.jobs[job_id].queue = queue
        return queue

    def get(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def mark_dispatched(
        self, job_id: str, node_id: str, stages: tuple[StageAssignment, ...]
    ) -> None:
        job = self.jobs[job_id]
        job.state = JobState.DISPATCHED
        job.node_id = node_id
        job.stages = stages
        job.dispatched_at = time.time()
        for stage in stages:
            job.stage_started_at.setdefault(stage.stage_idx, time.time())
            job.stage_last_activity[stage.stage_idx] = time.time()
            job.attempt_nodes.setdefault((stage.stage_idx, 0), stage.node_id)

    def _push(self, job: JobRecord, frame: dict) -> None:
        if job.queue is not None:
            try:
                job.queue.put_nowait(frame)
            except asyncio.QueueFull:
                log.warning("sse_queue_full job=%s (client too slow)", job.job_id)

    def _transition(self, job: JobRecord, state: JobState) -> None:
        if state in (JobState.RUNNING, JobState.STREAMING) and job.state in (
            JobState.RUNNING,
            JobState.STREAMING,
        ):
            return  # activity, not a real transition
        previous = job.state
        job.state = state
        log.info("job_state job=%s state=%s", job.job_id, state.value)
        if (
            state in (JobState.COMPLETED, JobState.FAILED)
            and previous not in (JobState.COMPLETED, JobState.FAILED)
            and self.on_terminal is not None
        ):
            self.on_terminal(job)

    def on_job_status(
        self, job_id: str, stage_idx: int, state: JobState, tokens_done: int, detail: str | None
    ) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        job.stage_last_activity[stage_idx] = time.time()
        if state == JobState.RUNNING and job.state in (
            JobState.DISPATCHED,
            JobState.QUEUED,
            JobState.RETRYING,
        ):
            self._transition(job, JobState.RUNNING)
        if state == JobState.RUNNING and stage_idx in job.stage_started_at:
            job.stage_finished_at[stage_idx] = time.time()  # last activity per stage
        if state == JobState.COMPLETED and job.state not in (
            JobState.COMPLETED,
            JobState.FAILED,
        ):
            job.stage_finished_at.setdefault(stage_idx, time.time())
            self._observe_stage_latencies(job)
            job.finished_at = time.time()
            # Token finals drive the SSE; JOB_STATUS COMPLETED only bookkeeps —
            # but if the client never got a final (e.g. empty output), emit one.
            if job.finish_reason is None:
                job.finish_reason = "length"
                self._push(
                    job,
                    {
                        "type": "final",
                        "job_id": job_id,
                        "finish_reason": "length",
                        "usage": {"tokens": len(job.tokens)},
                    },
                )
                self._transition(job, JobState.COMPLETED)

    def _observe_stage_latencies(self, job: JobRecord) -> None:
        for stage_idx, started in job.stage_started_at.items():
            finished = job.stage_finished_at.get(stage_idx)
            if finished is not None and finished > started:
                self.stage_stats.observe(finished - started)

    def on_token_batch(self, batch: TokenBatch) -> None:
        job = self.jobs.get(batch.job_id)
        if job is None or job.state in (JobState.COMPLETED, JobState.FAILED):
            return
        job.last_token_at = time.time()
        if job.state in (JobState.DISPATCHED, JobState.RUNNING, JobState.RETRYING):
            if job.state != JobState.STREAMING:
                self._transition(job, JobState.STREAMING)
            if job.first_token_at is None:
                job.first_token_at = time.time()
        for token in batch.tokens:
            job.tokens.append(token)
            self._push(job, {"type": "token", "job_id": batch.job_id, "token": token})
        if batch.is_final:
            log.info(
                "token_final job=%s reason=%s tokens=%d",
                batch.job_id,
                batch.finish_reason,
                len(job.tokens),
            )
            job.finish_reason = batch.finish_reason or "length"
            job.finished_at = time.time()
            self._observe_stage_latencies(job)
            self._push(
                job,
                {
                    "type": "final",
                    "job_id": batch.job_id,
                    "finish_reason": job.finish_reason,
                    "usage": {"tokens": len(job.tokens)},
                },
            )
            self._transition(job, JobState.COMPLETED)

    def fail_job(self, job_id: str, detail: str) -> None:
        job = self.jobs.get(job_id)
        if job is None or job.state in (JobState.COMPLETED, JobState.FAILED):
            return
        job.error = detail
        job.finished_at = time.time()
        self._observe_stage_latencies(job)
        self._push(job, {"type": "error", "job_id": job_id, "detail": detail})
        self._transition(job, JobState.FAILED)

    def active_jobs(self) -> list[JobRecord]:
        return [
            j for j in self.jobs.values() if j.state not in (JobState.COMPLETED, JobState.FAILED)
        ]

    def fail_jobs_of_node(self, node_id: str, detail: str) -> int:
        failed = 0
        for job in self.jobs.values():
            if job.node_id == node_id and job.state not in (
                JobState.COMPLETED,
                JobState.FAILED,
            ):
                self.fail_job(job.job_id, detail)
                failed += 1
        return failed

    def view(self, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        return {
            "job_id": job.job_id,
            "model_id": job.model_id,
            "state": job.state.value,
            "node_id": job.node_id,
            "tokens_generated": len(job.tokens),
            "finish_reason": job.finish_reason,
            "error": job.error,
            "backups": list(job.backups),
            "restarts": job.restarts,
            "retries": {str(k): v for k, v in sorted(job.retries.items())},
            "created_at": job.created_at,
            "first_token_at": job.first_token_at,
            "finished_at": job.finished_at,
            "stages": [
                {
                    "stage_idx": s.stage_idx,
                    "node_id": s.node_id,
                    "layer_start": s.layer_start,
                    "layer_end": s.layer_end,
                    "latency_ms": (
                        round(
                            (job.stage_finished_at[s.stage_idx] - job.stage_started_at[s.stage_idx])
                            * 1000,
                            2,
                        )
                        if s.stage_idx in job.stage_finished_at
                        and s.stage_idx in job.stage_started_at
                        else None
                    ),
                }
                for s in job.stages
            ],
        }


class ActivationRelay:
    """Routes activation chunks between stage nodes (spec §9.5, S4 single-hop)."""

    def __init__(self, connections) -> None:
        self.connections = connections
        self.relayed_bytes = 0

    async def route(self, header: ActivationRelayHeader, payload: bytes, jobs: JobTracker) -> None:
        job = jobs.get(header.job_id)
        if job is None or not job.stages:
            log.warning("relay_unknown_job job=%s", header.job_id)
            return
        if header.role == "sampled_token":
            target = job.stages[0].node_id
        else:
            next_idx = header.stage_idx + 1
            if next_idx >= len(job.stages):
                log.warning("relay_no_next_stage job=%s stage=%d", header.job_id, header.stage_idx)
                return
            target = job.stages[next_idx].node_id
        if target is None:
            return
        ok = await self.connections.send_envelope(
            target,
            Envelope.wrap(MessageType.ACTIVATION_RELAY, header, ts=time.time()),
        )
        if ok and payload:
            ok = await self.connections.send_bytes(target, payload)
        if ok:
            self.relayed_bytes += len(payload)
            return
        # Target unreachable. Stage/node-level failure is the FaultManager's
        # job (spec §13): it detects the loss (disconnect/OFFLINE) or stall
        # (watchdog) and reassigns the stage onto a backup. Failing the whole
        # job here would pre-empt a recoverable retry, so we just record it.
        log.warning(
            "relay_target_unreachable job=%s target=%s stage=%d role=%s",
            header.job_id,
            target,
            header.stage_idx,
            header.role,
        )
