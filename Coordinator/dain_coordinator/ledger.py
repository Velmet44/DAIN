"""Compute accounting ledger (S8, spec §15).

Every terminal job emits one append-only `LedgerEvent` per `(stage_idx, attempt)`
into the coordinator store. Everything is coordinator-derived and trusted:

- `flops_est` comes from the Common FLOP tables (`flops_for_stages`) — nodes
  never self-report FLOPs (untrusted input, §16).
- `tokens_out` is the client-visible count the coordinator already streams;
  `tokens_in` is a deterministic prompt-length estimate.
- `compute_seconds` is the observed per-stage wall time from the job tracker.
- `outcome` per §15: the final attempt of a finished stage is SUCCESS; attempts
  that were reassigned away are RETRIED_AWAY (×0.2); a stage left unfinished
  when the job fails is FAILED (×0.0).
- a verification pass cross-checks each stage's duration against the p99 latency
  window and flags statistical outliers to the caller (score penalties, §15).

Deduplication is the store's `UNIQUE(job_id, stage_idx, attempt)` + INSERT OR
IGNORE — re-delivery never double-counts (§13/§15).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from dain_common.accounting import (
    CreditWeights,
    estimate_prompt_tokens,
    flops_for_stages,
)
from dain_common.accounting import (
    credit as credit_for,
)
from dain_common.schemas import JobState, LedgerEvent, TaskOutcome

from dain_coordinator.jobs import JobRecord, JobTracker
from dain_coordinator.settings import CoordinatorSettings
from dain_coordinator.store import SQLiteRegistry

log = logging.getLogger("dain.coordinator.ledger")


def _default_credit_weights() -> CreditWeights:
    return CreditWeights()


class Ledger:
    def __init__(
        self,
        settings: CoordinatorSettings,
        registry: SQLiteRegistry,
        jobs: JobTracker,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.jobs = jobs
        # Set by the app: callable(node_id, note) fired when verification flags
        # a node for a score penalty (spec §15).
        self.on_verification_flag: Callable[[str, str], None] | None = None

    # -- emission ---------------------------------------------------------------

    def emit_job_terminal(self, job: JobRecord) -> None:
        """Persist one ledger event per (stage, attempt) for a terminal job.

        Idempotent per JobRecord (`ledger_emitted`) and per event (store key).
        """
        if job.ledger_emitted or not job.stages:
            return
        job.ledger_emitted = True
        now = job.finished_at or time.time()
        for stage in job.stages:
            stage_idx = stage.stage_idx
            final_attempt = job.retries.get(stage_idx, 0)
            for attempt in range(final_attempt + 1):
                outcome = self._outcome(job, stage_idx, attempt)
                node_id = job.attempt_nodes.get((stage_idx, attempt), stage.node_id)
                event = self._event_for(job, stage, attempt, node_id, outcome, now)
                if event is None:
                    continue
                verified, note = self._verify(event, stage_idx)
                inserted = self.registry.append_ledger(event, verified=verified, note=note)
                if inserted:
                    log.info(
                        "ledger_event job=%s stage=%d attempt=%d node=%s outcome=%s "
                        "flops=%.3e time=%.2fs credits=%.3f",
                        event.job_id,
                        event.stage_idx,
                        event.attempt,
                        event.node_id,
                        event.outcome.value,
                        event.flops_est,
                        event.compute_seconds,
                        credit_for(event),
                    )
                if not verified and self.on_verification_flag is not None:
                    self.on_verification_flag(node_id, note or "verification flag")

    def _outcome(self, job: JobRecord, stage_idx: int, attempt: int) -> TaskOutcome:
        """§15: exactly one SUCCESS per finished stage; earlier attempts and
        unfinished stages degrade to RETRIED_AWAY / FAILED."""
        final_attempt = job.retries.get(stage_idx, 0)
        if attempt < final_attempt:
            return TaskOutcome.RETRIED_AWAY
        if job.state == JobState.COMPLETED or stage_idx in job.stage_finished_at:
            return TaskOutcome.SUCCESS
        return TaskOutcome.FAILED

    def _event_for(
        self,
        job: JobRecord,
        stage,
        attempt: int,
        node_id: str,
        outcome: TaskOutcome,
        ts: float,
    ) -> LedgerEvent | None:
        tokens_out = len(job.tokens)
        tokens_in = estimate_prompt_tokens(job.prompt)
        n_layers = stage.layer_end - stage.layer_start + 1
        gpu_util, cpu_util = self._util(node_id)
        try:
            flops = flops_for_stages(job.model_id, n_layers, tokens_in + tokens_out)
        except (KeyError, ValueError):
            log.warning("ledger_flops_unknown model=%s job=%s", job.model_id, job.job_id)
            flops = 0.0
        return LedgerEvent(
            node_id=node_id,
            job_id=job.job_id,
            model_id=job.model_id,
            stage_idx=stage.stage_idx,
            attempt=attempt,
            partition_id=stage.shard_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            flops_est=flops,
            compute_seconds=self._compute_seconds(job, stage.stage_idx),
            gpu_util_avg=gpu_util,
            cpu_util_avg=cpu_util,
            energy_kwh_est=None,
            outcome=outcome,
            ts=ts,
        )

    # -- derivation helpers ------------------------------------------------------

    def _compute_seconds(self, job: JobRecord, stage_idx: int) -> float:
        started = job.stage_started_at.get(stage_idx)
        finished = job.stage_finished_at.get(stage_idx)
        if started is None:
            return 0.0
        if finished is not None and finished > started:
            return round(finished - started, 3)
        # Stage never reported a finish: bound the open interval by the job's
        # terminal edge so a failed job still records wall time.
        if job.finished_at is not None and job.finished_at > started:
            return round(job.finished_at - started, 3)
        return 0.0

    def _util(self, node_id: str) -> tuple[float | None, float | None]:
        row = self.registry.get_node(node_id)
        if row is None or row.metrics is None:
            return None, None
        return row.metrics.gpu_util_pct, row.metrics.cpu_util_pct

    # -- verification (spec §15) --------------------------------------------------

    def _verify(self, event: LedgerEvent, stage_idx: int) -> tuple[bool, str | None]:
        """Cross-check `compute_seconds` against the coordinator-observed p99
        stage-latency window. An anomaly isn't proof of fraud (4-core contention
        legitimately stretches stages), so the threshold is deliberately loose —
        >10× p99 and >15 s flags the node for a score penalty."""
        p99 = self.jobs.stage_stats.p99()
        if event.compute_seconds <= 0 or p99 is None or p99 <= 0:
            return True, None
        ratio = event.compute_seconds / p99
        if event.compute_seconds > 15.0 and ratio > 10.0:
            return (
                False,
                f"stage {stage_idx} duration {event.compute_seconds:.1f}s is "
                f"{ratio:.1f}x p99 ({p99:.1f}s)",
            )
        return True, None

    # -- read side (used by the API) ----------------------------------------------

    def credit(self, event: LedgerEvent) -> float:
        return credit_for(event, self.settings.accounting_weights)

    def events_for_node(self, node_id: str) -> list[dict]:
        """Enriched row dicts for one node: event fields + `verified` + `credit`."""
        return [self._row_view(r) for r in self.registry.ledger_rows(node_id=node_id)]

    def events_all(self, since: float | None = None) -> list[dict]:
        return [self._row_view(r) for r in self.registry.ledger_rows(since=since)]

    def _row_view(self, row: dict) -> dict:
        event = self._from_row(row)
        return {
            **event.model_dump(),
            "verified": bool(row["verified"]),
            "verification_note": row["verification_note"],
            "credit": round(credit_for(event, self.settings.accounting_weights), 6),
        }

    @staticmethod
    def _from_row(row: dict) -> LedgerEvent:
        return LedgerEvent(
            node_id=row["node_id"],
            job_id=row["job_id"],
            model_id=row["model_id"],
            stage_idx=row["stage_idx"],
            attempt=row["attempt"],
            partition_id=row["partition_id"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            flops_est=row["flops_est"],
            compute_seconds=row["compute_seconds"],
            gpu_util_avg=row["gpu_util_avg"],
            cpu_util_avg=row["cpu_util_avg"],
            energy_kwh_est=row["energy_kwh_est"],
            outcome=TaskOutcome(row["outcome"]),
            ts=row["ts"],
        )
