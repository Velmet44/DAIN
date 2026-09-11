"""S7 (spec §13) fault manager: entry-stage loss must restart the whole job
from the prompt (downstream stages hold caches built on the old activations),
while non-entry reassigns keep the completed prefix and replay via STAGE_RETRY.
"""

from __future__ import annotations

import asyncio

from dain_common.schemas import (
    JobState,
    MessageType,
    StageAssignment,
)
from test_scheduler import MANIFEST, node

from dain_coordinator.faults import FaultManager
from dain_coordinator.jobs import JobRecord, JobTracker
from dain_coordinator.settings import CoordinatorSettings
from dain_coordinator.store import NodeRow

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

BACKUPS = ("node-8", "node-9")


class StubConnections:
    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []
        self.connected: set[str] = {n.node_id for n in (node("node-8"), node("node-9"))}

    def is_connected(self, node_id: str) -> bool:
        return node_id in self.connected

    async def send_envelope(self, node_id: str, envelope: object) -> bool:
        self.sent.append((node_id, envelope))
        return True


class StubRegistry:
    def __init__(self) -> None:
        self.rows: dict[str, NodeRow] = {
            "node-8": node("node-8"),
            "node-9": node("node-9"),
        }

    def list_nodes(self, state: JobState | None = None) -> list[NodeRow]:
        return list(self.rows.values())

    def get_node(self, node_id: str) -> NodeRow | None:
        return self.rows.get(node_id)


class StubService:
    def __init__(self) -> None:
        self.busy: list[str] = []

    def mark_busy(self, node_id: str) -> None:
        self.busy.append(node_id)


def make_fault_manager() -> tuple[FaultManager, StubConnections, JobRecord]:
    settings = CoordinatorSettings(max_stage_attempts=3, max_job_restarts=1)
    jobs = JobTracker()
    record = jobs.create(
        "dain-tiny-16L",
        "Once upon a time",
        {"max_tokens": 8, "temperature": 0.0},
        MANIFEST,
        api_key="dain-dev-key",
        backups=BACKUPS,
    )
    job_id = record.job_id
    jobs.mark_dispatched(job_id, "node-0", STAGES)
    job = jobs.get(job_id)
    assert job is not None
    conn = StubConnections()
    reg = StubRegistry()
    svc = StubService()
    faults = FaultManager(settings, jobs, conn, reg, svc)
    return faults, conn, job


def _run(coro) -> None:
    asyncio.run(coro)


def test_entry_stage_loss_restarts_from_prompt() -> None:
    faults, conn, job = make_fault_manager()
    job.state = JobState.RUNNING
    job.first_token_at = 1.0
    job.stage_finished_at = {2: 2.0}

    async def scenario() -> None:
        faults._reassign(job, 0, reason="node_lost")
        await asyncio.sleep(0.05)

    _run(scenario())

    assert job.restarts == 1
    assert job.state == JobState.DISPATCHED
    assert job.stage_finished_at == {}  # per-stage progress reset
    assert job.retries == {}  # entry loss never consumes per-stage retries
    assert (0, 1) not in job.attempt_nodes
    assigns = [e for _, e in conn.sent if e.type == MessageType.JOB_ASSIGN]
    retries = [e for _, e in conn.sent if e.type == MessageType.STAGE_RETRY]
    assert len(assigns) == 3
    assert {a.payload["my_stage_idx"] for a in assigns} == {0, 1, 2}
    assert assigns[0].payload["prompt"] == "Once upon a time"
    assert retries == []  # no prefix to replay — the whole graph restarts


def test_entry_stage_loss_exhausts_restarts_fails_job() -> None:
    faults, conn, job = make_fault_manager()
    job.state = JobState.RUNNING
    job.restarts = faults.settings.max_job_restarts

    async def scenario() -> None:
        faults._reassign(job, 0, reason="node_lost")
        await asyncio.sleep(0.05)

    _run(scenario())

    assert job.state == JobState.FAILED
    assert "entry" in (job.error or "")
    assert job.restarts == faults.settings.max_job_restarts
    assert conn.sent == []  # nothing re-dispatched


def test_non_entry_reassign_keeps_prefix_and_replays() -> None:
    faults, conn, job = make_fault_manager()
    job.state = JobState.RUNNING
    job.first_token_at = 1.0

    async def scenario() -> None:
        faults._reassign(job, 2, reason="watchdog")
        await asyncio.sleep(0.05)

    _run(scenario())

    assert job.retries == {2: 1}
    assert job.attempt_nodes[(2, 0)] == "node-2"
    assert job.stages[2].node_id == "node-8"  # first warm backup
    assigns = [e for _, e in conn.sent if e.type == MessageType.JOB_ASSIGN]
    retry = [e for _, e in conn.sent if e.type == MessageType.STAGE_RETRY]
    assert len(assigns) == 1 and assigns[0].payload["my_stage_idx"] == 2
    assert assigns[0].payload["prompt"] is None  # prefix survives; only stage 2 replaced
    assert len(retry) == 1 and retry[0].payload["stage_idx"] == 2
    assert conn.sent[0][0] == "node-8"  # assign to replacement
    assert any(n == "node-1" for n, _ in conn.sent)  # replay request to upstream (stage 1)
