"""S22 instant-serving: replica-first placement, readiness gating, assignments."""

from dain_common.schemas import (
    ModelAssignment,
    ModelManifest,
    ModelStatus,
    ShardRef,
)

from dain_coordinator.assignments import (
    AssignmentService,
    DemandTracker,
    ReadinessMap,
    compute_assignments,
)
from dain_coordinator.partition import plan_serving
from dain_coordinator.store import NodeRow
from tests.test_scheduler import MANIFEST, node


def ready_node(node_id: str, **kw) -> NodeRow:
    return node(node_id, **kw)


def replica_manifest(shard_mb: float = 4.0) -> ModelManifest:
    return MANIFEST.model_copy(
        update={
            "shards": tuple(
                ShardRef(
                    model_id=MANIFEST.model_id,
                    shard_id=f"layers_{a:02d}_{a + 3:02d}",
                    content_hash="a" * 64,
                    layer_start=a,
                    layer_end=a + 3,
                    size_bytes=int(shard_mb * 1_000_000),
                )
                for a in range(0, 16, 4)
            )
        }
    )


class FakeReadiness:
    """Duck-typed ReadinessMap with explicit warm/ready sets."""

    def __init__(self, ready: set[str], warm: set[str], toks: float = 10.0):
        self.ready = ready
        self.warm_set = warm
        self.toks = toks

    def schedulable(self, node_id: str, model_id: str) -> bool:
        return node_id in self.ready

    def warm(self, node_id: str, model_id: str) -> bool:
        return node_id in self.warm_set

    def get(self, node_id: str, model_id: str):
        if node_id in self.ready:
            return ModelStatus(
                node_id=node_id,
                model_id=model_id,
                state="warm" if node_id in self.warm_set else "files_ready",
                toks_s=self.toks,
            )
        return None


# -- plan_serving (replica-first) -------------------------------------------------


def test_replica_plan_selects_warm_ready_node() -> None:
    readiness = FakeReadiness(ready={"n01", "n02"}, warm={"n02"})
    pool = [ready_node("n01", score=0.9), ready_node("n02", score=0.5)]
    plan = plan_serving(
        replica_manifest(), pool, readiness=readiness, active_sessions={}
    )
    assert plan is not None and plan.serving_mode == "replica"
    assert len(plan.stages) == 1
    assert plan.stages[0].node_id == "n02"  # warm beats higher score
    assert plan.stages[0].layer_start == 0 and plan.stages[0].layer_end == 15


def test_replica_plan_falls_back_to_pipeline_when_no_node_ready() -> None:
    readiness = FakeReadiness(ready=set(), warm=set())
    pool = [ready_node(f"n0{i}", score=0.5) for i in range(4)]
    plan = plan_serving(
        replica_manifest(), pool, readiness=readiness, active_sessions={}
    )
    assert plan is not None and plan.serving_mode == "pipeline"
    assert len(plan.stages) == 4  # classic placement, layers 0..15 contiguous


def test_replica_plan_honors_session_cap() -> None:
    readiness = FakeReadiness(ready={"n01"}, warm={"n01"})
    pool = [ready_node("n01", score=0.9)]
    plan = plan_serving(
        replica_manifest(),
        pool,
        readiness=readiness,
        active_sessions={"n01": 4},
        max_sessions_per_node=4,
    )
    # n1 saturated -> pipeline fallback across the same pool
    assert plan is not None and plan.serving_mode == "pipeline"


def test_replica_plan_overcommit_flags() -> None:
    readiness = FakeReadiness(ready={"nsmall"}, warm={"nsmall"})
    big = replica_manifest(shard_mb=4000.0)  # 16 GB packed -> 64 GB runtime
    pool = [ready_node("nsmall", score=0.9, vram_free=1.0)]
    assert (
        plan_serving(big, pool, readiness=readiness, active_sessions={}) is None
    )
    plan = plan_serving(
        big, pool, readiness=readiness, active_sessions={}, allow_overcommit=True
    )
    assert plan is not None and plan.serving_mode == "replica"
    assert plan.overcommitted is True


# -- compute_assignments ----------------------------------------------------------


def _settings(**kw):
    from dain_coordinator.settings import CoordinatorSettings

    return CoordinatorSettings(**kw)


def test_assignments_replica_on_fitting_nodes() -> None:
    readiness = FakeReadiness(ready=set(), warm=set())
    plan = compute_assignments(
        [replica_manifest()],
        [ready_node("n01", score=0.9), ready_node("n02", score=0.5)],
        connected={"n01", "n02"},
        readiness=readiness,
        demand=DemandTracker(),
        settings=_settings(max_replicas_per_model=1),
    )
    assert plan.pipeline_only == frozenset()
    assert set(plan.by_node) == {"n01"}  # replicas_per_model=1, best node
    assignment = plan.by_node["n01"][0]
    assert assignment.action == "ensure" and assignment.mode == "replica"


def test_assignments_pipeline_only_when_no_node_fits() -> None:
    readiness = FakeReadiness(ready=set(), warm=set())
    big = replica_manifest(shard_mb=4000.0)
    plan = compute_assignments(
        [big],
        [ready_node(f"n0{i}", score=0.5) for i in range(4)],
        connected={f"n0{i}" for i in range(4)},
        readiness=readiness,
        demand=DemandTracker(),
        settings=_settings(layers_per_node_target=4),
    )
    assert plan.pipeline_only == frozenset({MANIFEST.model_id})
    assert len(plan.by_node) == 4  # every stage node provisions its range
    ranges = sorted(
        (a.layer_start, a.layer_end)
        for entries in plan.by_node.values()
        for a in entries
    )
    assert ranges == [(0, 3), (4, 7), (8, 11), (12, 15)]


def test_assignment_diff_emits_revoke() -> None:
    old = {
        "n01": [
            ModelAssignment(model_id="m-a", action="ensure", mode="replica"),
            ModelAssignment(model_id="m-b", action="ensure", mode="replica"),
        ]
    }
    # Build a plan containing only m-a on n1 via the service-level diff:
    from dain_coordinator.assignments import AssignmentPlan

    new_plan = AssignmentPlan(
        by_node={"n01": [ModelAssignment(model_id="m-a", action="ensure", mode="replica")]},
        pipeline_only=frozenset(),
    )
    changes = new_plan.diff(old)
    revoked = [a for _, a in changes if a.action == "revoke"]
    ensured = [a for _, a in changes if a.action == "ensure"]
    assert [a.model_id for a in revoked] == ["m-b"]
    assert ensured == []  # m-a unchanged -> no re-send


def test_demand_scaling_bumps_replicas() -> None:
    readiness = FakeReadiness(ready=set(), warm=set())
    demand = DemandTracker(window_s=600.0)
    for _ in range(10):
        demand.record(MANIFEST.model_id)
    plan = compute_assignments(
        [replica_manifest()],
        [ready_node("n01", score=0.9), ready_node("n02", score=0.8)],
        connected={"n01", "n02"},
        readiness=readiness,
        demand=demand,
        settings=_settings(demand_scale_threshold=8, max_replicas_per_model=2),
    )
    assert set(plan.by_node) == {"n01", "n02"}


# -- readiness map + service persistence --------------------------------------------


def test_readiness_map_schedulable_states() -> None:
    rm = ReadinessMap()
    rm.update(ModelStatus(node_id="n01", model_id="m", state="downloading", progress=0.2))
    assert not rm.schedulable("n01", "m")
    rm.update(ModelStatus(node_id="n01", model_id="m", state="files_ready"))
    assert rm.schedulable("n01", "m") and not rm.warm("n01", "m")
    rm.update(ModelStatus(node_id="n01", model_id="m", state="warm", toks_s=3.2))
    assert rm.warm("n01", "m")
    rm.drop_node("n01")
    assert not rm.schedulable("n01", "m")


def test_assignment_service_persists_and_replays(tmp_path) -> None:
    from dain_coordinator.store import SQLiteRegistry

    registry = SQLiteRegistry(str(tmp_path / "db.sqlite3"))
    registry.open()
    registry.save_node(ready_node("n01", score=0.9))
    sent: list[tuple[str, object]] = []

    class FakeConnections:
        def connected_ids(self):
            return ["n01"]

        async def send_envelope(self, node_id, envelope):
            sent.append((node_id, envelope))
            return True

    readiness = FakeReadiness(ready=set(), warm=set())
    service = AssignmentService(
        registry=registry,
        connections=FakeConnections(),
        readiness=readiness,
        demand=DemandTracker(),
        manifests_fn=lambda: [replica_manifest()],
        settings_getter=lambda: _settings(replicas_per_model=1),
    )
    import asyncio

    changes = asyncio.run(service.recompute("test"))
    assert changes == 1
    assert [n for n, _ in sent] == ["n01"]

    # Restart: the persisted plan is loaded once and replayed on connect.
    service2 = AssignmentService(
        registry=registry,
        connections=FakeConnections(),
        readiness=readiness,
        demand=DemandTracker(),
        manifests_fn=lambda: [replica_manifest()],
        settings_getter=lambda: _settings(replicas_per_model=1),
    )
    portfolio = service2.current_portfolio("n01")
    assert len(portfolio) == 1 and portfolio[0].mode == "replica"
    registry.close()
