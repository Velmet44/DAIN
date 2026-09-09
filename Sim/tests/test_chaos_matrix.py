"""S7 checkpoint: chaos matrix (fault tolerance, spec §13).

Drives real node agents through deterministic kill schedules and asserts the
cluster degrades gracefully:

1. kill_one_mid_job_completes      — killing a stage node mid-job completes via
                                     stage retry onto a warm backup.
2. kill_three_of_eight_still_serves — pool above `min_nodes` keeps serving (fewer,
                                     larger stages), no cluster outage.
3. kill_below_min_nodes_rejects     — pool below `min_nodes` returns a clean
                                     429/503, never hangs.
4. no_duplicate_ledger_keys         — `(stage_idx, attempt)` keys are unique, so
                                     the S8 accounting ledger would not double
                                     count a retry storm.

These are integration tests that spawn real `dain_node` subprocesses; the
Settings compress timings so the whole thing runs in seconds. On a contended
4-core host the bounds are generous (S5/S6 lesson: wait on state, never sleep()).
"""

import asyncio

from dain_sim.chaos import run_chaos


def test_kill_one_mid_job_completes() -> None:
    report = asyncio.run(
        run_chaos(node_count=6, expect="complete", job_tokens=40, min_nodes=1)
    )
    assert report["completed"] is True, report
    assert report["status"] == 200, report


def test_kill_three_of_eight_still_serves() -> None:
    report = asyncio.run(
        run_chaos(node_count=8, expect="degraded", job_tokens=40, min_nodes=3)
    )
    # 8 - 3 victims = 5 online, above min_nodes=3: must still serve in degraded
    # mode (fewer, larger stages recomputed) and complete.
    assert report["served"] is True, report
    assert report["status"] == 200, report


def test_kill_below_min_nodes_rejects() -> None:
    report = asyncio.run(
        run_chaos(node_count=4, expect="reject", job_tokens=4, min_nodes=4)
    )
    # All 4 killed, pool below min_nodes: clean 429/503, no hang.
    assert report["rejected_cleanly"] is True, report


def test_no_duplicate_ledger_keys() -> None:
    report = asyncio.run(
        run_chaos(node_count=6, expect="complete", job_tokens=40, min_nodes=1)
    )
    assert report["attempt_keys_unique"] is True, report
    # The retry actually happened: the sampling stage was reassigned at least once.
    assert report["retries"], report
