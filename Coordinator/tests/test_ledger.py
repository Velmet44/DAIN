"""S8 checkpoint: accounting ledger (spec §15) — emission, dedupe, verification,
and the export API. Jobs are driven through the JobTracker directly (no real
nodes); the Ledger derivation is coordinator-side and must match the Common
credit goldens.
"""

from __future__ import annotations

import time

import pytest
from dain_common.schemas import (
    ModelManifest,
    ShardRef,
    StageAssignment,
    TaskOutcome,
    TokenBatch,
)
from fastapi.testclient import TestClient

MODEL = ModelManifest(
    model_id="dain-tiny-16L",
    name="dev",
    layers=16,
    hidden=64,
    heads=4,
    kv_heads=2,
    intermediate=172,
    vocab_size=256,
    eos_token_id=0,
    shards=(
        ShardRef(
            model_id="dain-tiny-16L",
            shard_id="layers_00_05",
            content_hash="a" * 16,
            layer_start=0,
            layer_end=5,
            size_bytes=1,
        ),
        ShardRef(
            model_id="dain-tiny-16L",
            shard_id="layers_06_10",
            content_hash="b" * 16,
            layer_start=6,
            layer_end=10,
            size_bytes=1,
        ),
        ShardRef(
            model_id="dain-tiny-16L",
            shard_id="layers_11_15",
            content_hash="c" * 16,
            layer_start=11,
            layer_end=15,
            size_bytes=1,
        ),
    ),
)

STAGES = (
    StageAssignment(
        stage_idx=0, node_id="node-0", shard_id="layers_00_05", layer_start=0, layer_end=5
    ),
    StageAssignment(
        stage_idx=1, node_id="node-1", shard_id="layers_06_10", layer_start=6, layer_end=10
    ),
    StageAssignment(
        stage_idx=2, node_id="node-2", shard_id="layers_11_15", layer_start=11, layer_end=15
    ),
)

API_KEY = "dain-dev-key"


def complete_job_with_retry(jobs) -> str:
    """Dispatch a 3-stage job, retry stage 2 once onto node-9, then finish."""
    now = time.time()
    record = jobs.create(
        "dain-tiny-16L",
        "Once upon a time",
        {"max_tokens": 40, "temperature": 0.0},
        MODEL,
        api_key=API_KEY,
        backups=("node-9",),
    )
    stages = list(STAGES)
    stages[2] = STAGES[2].model_copy(update={"node_id": "node-9"})
    jobs.mark_dispatched(record.job_id, "node-0", tuple(stages))
    # Record the abandoned attempt of stage 2 on the node that served it (S8).
    job = jobs.get(record.job_id)
    assert job is not None
    job.retries[2] = 1
    job.attempt_nodes[(2, 0)] = "node-2"
    job.stage_started_at = {0: now - 3.0, 1: now - 2.5, 2: now - 2.0}
    job.stage_finished_at = {0: now - 2.5, 1: now - 2.0, 2: now}
    jobs.on_token_batch(
        TokenBatch(
            job_id=record.job_id,
            tokens=("h", "e", "l", "l", "o") * 8,  # 40 tokens
            is_final=True,
            finish_reason="length",
        )
    )
    return record.job_id


def failed_job(jobs) -> str:
    record = jobs.create(
        "dain-tiny-16L",
        "Once upon a time",
        {"max_tokens": 40, "temperature": 0.0},
        MODEL,
        api_key=API_KEY,
    )
    jobs.mark_dispatched(record.job_id, "node-0", STAGES)
    jobs.fail_job(record.job_id, "stage 1 exhausted retries")
    return record.job_id

# -- emission & outcome derivation ----------------------------------------------


def test_ledger_emits_one_success_per_stage_with_retried_away(client: TestClient) -> None:
    jobs = client.app.state.jobs
    ledger = client.app.state.ledger
    job_id = complete_job_with_retry(jobs)

    events = ledger.events_all()
    assert len(events) == 4  # 3 stages + 1 retry of stage 2

    success = [e for e in events if e["outcome"] == TaskOutcome.SUCCESS.value]
    assert len(success) == 3  # exactly one SUCCESS per (job, stage)
    assert {e["stage_idx"] for e in success} == {0, 1, 2}

    retried = [e for e in events if e["outcome"] == TaskOutcome.RETRIED_AWAY.value]
    assert len(retried) == 1
    assert retried[0]["stage_idx"] == 2 and retried[0]["attempt"] == 0
    assert retried[0]["node_id"] == "node-2"  # the node that actually ran it

    # Credits hit the §15 factors: SUCCESS ×1.0 beats the ×0.2 RETRIED_AWAY,
    # and the stored credit matches an independent recompute from the row.
    # (rel is loose enough to absorb the read path's round(..., 6).)
    assert all(
        e["credit"] == pytest.approx(ledger.credit(ledger._from_row(e)), rel=1e-5)
        for e in events
    )
    assert min(e["credit"] for e in success) > max(e["credit"] for e in retried)
    assert min(e["credit"] for e in success) > 0

    # All events belong to the job and carry coordinator-derived fields.
    assert all(e["job_id"] == job_id for e in events)
    assert all(e["flops_est"] > 0 for e in events)
    assert all(e["verified"] is True for e in events)
    # partition_id mirrors the stage layer range.
    assert {e["partition_id"] for e in events} == {"layers_00_05", "layers_06_10", "layers_11_15"}
    # tokens_in is the deterministic prompt estimate; tokens_out == emitted.
    assert all(e["tokens_in"] == 4 for e in events)
    assert all(e["tokens_out"] == 40 for e in events)


def test_ledger_failed_job_weights_zero(client: TestClient) -> None:
    jobs = client.app.state.jobs
    ledger = client.app.state.ledger
    job_id = failed_job(jobs)

    events = ledger.events_all()
    assert len(events) == 3
    assert all(e["job_id"] == job_id for e in events)
    # No stage finished: every final attempt is FAILED (§15 → ×0.0 credit).
    assert all(e["outcome"] == TaskOutcome.FAILED.value for e in events)
    assert all(e["credit"] == 0.0 for e in events)


def test_ledger_dedupe_on_replay(client: TestClient) -> None:
    jobs = client.app.state.jobs
    ledger = client.app.state.ledger
    registry = client.app.state.registry
    complete_job_with_retry(jobs)

    assert len(ledger.events_all()) == 4
    # The store key is the real guard: appending an identical row again must
    # no-op (INSERT OR IGNORE on UNIQUE(job_id, stage_idx, attempt)).
    row = registry.ledger_rows()[0]
    first = ledger._from_row(row)
    assert registry.append_ledger(first, verified=True) is False
    assert len(registry.ledger_rows()) == 4


def test_ledger_verification_flags_outliers(client: TestClient) -> None:
    jobs = client.app.state.jobs
    ledger = client.app.state.ledger
    flags: list[tuple[str, str]] = []
    ledger.on_verification_flag = lambda node_id, note: flags.append((node_id, note))
    # Populate a normal latency window so the outlier stands out against p99.
    for _ in range(40):
        jobs.stage_stats.observe(0.5)

    record = jobs.create(
        "dain-tiny-16L",
        "Once upon a time",
        {"max_tokens": 40, "temperature": 0.0},
        MODEL,
        api_key=API_KEY,
    )
    now = time.time()
    stages = list(STAGES)
    stages[2] = STAGES[2].model_copy(update={"node_id": "node-9"})
    jobs.mark_dispatched(record.job_id, "node-0", tuple(stages))
    job = jobs.get(record.job_id)
    assert job is not None
    job.retries[2] = 1
    job.attempt_nodes[(2, 0)] = "node-2"
    # Stage 0 never reported a finish: it is still bounded by the job's terminal
    # edge (~60 s observed) while the latency window sees only the finished
    # stages (0.5 s) → >10× p99 → verification flag fires.
    job.stage_started_at = {0: now - 60.0, 1: now - 0.5, 2: now - 0.5}
    job.stage_finished_at = {1: now, 2: now}
    jobs.on_token_batch(
        TokenBatch(job_id=record.job_id, tokens=("x",) * 40, is_final=True, finish_reason="length")
    )

    stage0 = [e for e in ledger.events_all() if e["stage_idx"] == 0]
    assert len(stage0) == 1
    assert stage0[0]["compute_seconds"] == pytest.approx(60.0, rel=1e-3)
    assert stage0[0]["verified"] is False
    assert stage0[0]["verification_note"]
    assert flags, "verification flag must reach the caller (score penalty hook)"


# -- export API -----------------------------------------------------------------


def test_ledger_api_node_summary_export(client: TestClient) -> None:
    jobs = client.app.state.jobs
    complete_job_with_retry(jobs)
    headers = {"X-API-Key": API_KEY}

    node_rows = client.get("/ledger/node/node-0", headers=headers).json()
    assert all(e["node_id"] == "node-0" for e in node_rows)

    summary = client.get("/ledger/summary", headers=headers).json()
    per_node_credits = {n["node_id"]: n["credits"] for n in summary["nodes"]}
    assert set(per_node_credits) == {"node-0", "node-1", "node-2", "node-9"}
    assert summary["totals"]["events"] == 4

    exported = client.post("/ledger/export", headers=headers, json={"format": "json"}).json()
    assert len(exported) == 4
    assert summary["totals"]["credits"] == pytest.approx(
        sum(e["credit"] for e in exported), rel=1e-6
    )

    csv_resp = client.post("/ledger/export", headers=headers, json={"format": "csv"})
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    lines = csv_resp.text.strip().splitlines()
    assert lines[0].startswith("node_id,job_id")


def test_ledger_api_requires_api_key(client: TestClient) -> None:
    assert client.get("/ledger/summary").status_code == 401
    assert client.get("/ledger/node/node-0").status_code == 401
