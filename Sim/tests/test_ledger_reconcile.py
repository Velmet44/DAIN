"""S8 checkpoint: accounting ledger reconciliation (spec §15).

Runs the fault-injection "complete" scenario (a stage node is killed mid-job, the
stage retries onto a backup) and then reconciles the coordinator's append-only
ledger against the job view over the live HTTP API:

- exactly one SUCCESS event per (job, stage); every earlier attempt RETRIED_AWAY
- no duplicate (job_id, stage_idx, attempt) keys after a retry storm
- every event verified, positive credit for SUCCESS, zero for FAILED
- /ledger/summary totals agree with /ledger/export rows and node credit sums
"""

import asyncio

from dain_sim.chaos import run_chaos


def test_ledger_reconciles_after_retry_storm() -> None:
    report = asyncio.run(
        run_chaos(node_count=6, expect="complete", job_tokens=40, min_nodes=1)
    )
    assert report["completed"] is True, report
    assert report["status"] == 200, report
    assert report["retries"], report  # the kill must have caused a real retry

    ledger = report["ledger"]
    assert ledger["ok"], ledger
    # Job produces one row per (stage, attempt): attempts + retries.
    assert ledger["job_events"] == ledger["success"] + ledger["retried_away"] + ledger["failed"]
    assert ledger["success"] == len(report["stages"])
    assert ledger["retried_away"] == sum((report["retries"] or {}).values())
    assert ledger["unique_keys"] is True
    assert set(ledger["node_credits"]) >= {s["node_id"] for s in report["stages"]}
