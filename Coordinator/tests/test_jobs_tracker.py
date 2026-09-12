"""JobTracker memory bounds: the `_order` history log used by the admin
/jobs endpoint must never grow unbounded on a high-churn run, even while many
jobs stay active (those are never evicted from `self.jobs`).
"""

from __future__ import annotations

import time

from dain_common.schemas import JobState, StageAssignment, TokenBatch
from test_scheduler import MANIFEST

from dain_coordinator.jobs import JobTracker

STAGES = (
    StageAssignment(
        stage_idx=0, node_id="node-0", shard_id="layers_00_03", layer_start=0, layer_end=3
    ),
    StageAssignment(
        stage_idx=1, node_id="node-1", shard_id="layers_04_07", layer_start=4, layer_end=7
    ),
    StageAssignment(
        stage_idx=2, node_id="node-2", shard_id="layers_08_11", layer_start=8, layer_end=11
    ),
)


def test_order_history_stays_bounded_with_mostly_active_jobs() -> None:
    jobs = JobTracker(max_jobs=8)
    for i in range(64):
        record = jobs.create(
            "dain-tiny-16L",
            f"prompt-{i}",
            {"max_tokens": 4},
            MANIFEST,
        )
        jobs.mark_dispatched(record.job_id, f"node-{i % 3}", STAGES)
        if i % 4 == 0:
            # a quarter of jobs terminate; the rest stay live
            jobs.on_job_status(record.job_id, 0, JobState.COMPLETED, 4, None)
    assert len(jobs._order) <= 8
    # terminal trimming still works even though `_order` is capped
    terminal = [j for j in jobs.jobs.values() if j.state.value == "completed"]
    assert len(terminal) <= 8


def test_order_history_drops_ttl_retired_ids() -> None:
    jobs = JobTracker(max_jobs=8, job_ttl_s=0.01)
    for i in range(16):
        record = jobs.create("dain-tiny-16L", f"p{i}", {"max_tokens": 1}, MANIFEST)
        jobs.mark_dispatched(record.job_id, f"node-{i % 2}", STAGES)
        jobs.on_job_status(record.job_id, 0, JobState.COMPLETED, 1, None)
    jobs._evict()
    assert len(jobs._order) == len(jobs.jobs)


def test_view_latency_falls_back_to_job_finish_for_intermediate_stages() -> None:
    """Intermediate stages never send COMPLETED acks (they relay activations),
    so their finish is clamped by the job's terminal edge (S8 keeps RUNNING
    acks out of stage_finished_at). The view must still report wall time."""
    jobs = JobTracker()
    record = jobs.create("dain-tiny-16L", "hi", {"max_tokens": 2}, MANIFEST)
    jobs.mark_dispatched(record.job_id, "node-0", STAGES)
    for stage in record.stages:
        record.stage_started_at[stage.stage_idx] = time.time() - 5.0
    jobs.on_token_batch(_final_batch(record.job_id))
    view = jobs.view(record.job_id)
    assert view is not None
    assert all(s["latency_ms"] is not None for s in view["stages"]), view["stages"]


def test_token_activity_refreshes_all_stage_watchdogs() -> None:
    jobs = JobTracker()
    record = jobs.create("dain-tiny-16L", "hi", {"max_tokens": 2}, MANIFEST)
    jobs.mark_dispatched(record.job_id, "node-0", STAGES)
    before = {stage_idx: 1.0 for stage_idx in record.stage_last_activity}
    record.stage_last_activity = before.copy()
    jobs.on_token_batch(TokenBatch(job_id=record.job_id, tokens=("a",)))
    assert all(record.stage_last_activity[idx] > before[idx] for idx in before)


def _final_batch(job_id: str) -> TokenBatch:
    return TokenBatch(job_id=job_id, tokens=("a", "b"), is_final=True, finish_reason="length")
