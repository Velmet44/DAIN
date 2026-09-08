"""S2 unit tests: the spec §6 transition table, enforced by NodeService."""

import pytest
from conftest import make_settings
from dain_common.schemas import NodeState

from dain_coordinator.nodes import NodeService
from dain_coordinator.store import SQLiteRegistry


@pytest.fixture()
def service(tmp_path) -> NodeService:
    registry = SQLiteRegistry(str(tmp_path / "r.sqlite3"))
    registry.open()
    settings = make_settings(tmp_path)
    return NodeService(registry, settings)


def _registered(service: NodeService, node_id: str = "node-a") -> None:
    from dain_common.schemas import Register

    payload = Register.model_validate(
        {
            "node_id": node_id,
            "auth_token": service.settings.join_token,
            "manifest": {
                "cpu": {"cores": 8, "ram_total_gb": 32.0, "ram_free_gb": 16.0},
                "net": {"bw_mbps": 300.0, "lat_ms_p95": 25.0},
            },
        }
    )
    ack = service.register(payload)
    assert ack.accepted


def test_allowed_transitions_table(service: NodeService) -> None:
    _registered(service)
    cases = [
        ("online", "busy", True, "workload_assigned"),  # ONLINE → BUSY
        ("busy", "online", True, "job_finished"),  # BUSY → ONLINE
        ("online", "degraded", True, "overload"),  # ONLINE → DEGRADED
        ("degraded", "online", True, "metrics_recovered"),  # DEGRADED → ONLINE
        ("online", "offline", True, "deregistered"),  # ONLINE → OFFLINE
        ("offline", "online", True, "registered"),  # OFFLINE → ONLINE (re-register path)
    ]
    for _frm, to, expected, reason in cases:
        assert service.transition("node-a", NodeState(to), reason) is expected


def test_invalid_transitions_rejected(service: NodeService) -> None:
    _registered(service)
    # No self-transition while ONLINE.
    assert service.transition("node-a", NodeState.ONLINE, "self") is False
    assert service.transition("node-a", NodeState.DEGRADED, "overload")  # → DEGRADED
    # DEGRADED → BUSY is not in the spec table.
    assert service.transition("node-a", NodeState.BUSY, "workload_assigned") is False
    # OFFLINE → BUSY is not in the spec table either.
    service.transition("node-a", NodeState.OFFLINE, "deregistered")
    assert service.transition("node-a", NodeState.BUSY, "jump") is False


def test_unknown_node_transitions_fail(service: NodeService) -> None:
    assert service.transition("ghost", NodeState.OFFLINE, "x") is False


def test_every_transition_is_persisted(service: NodeService) -> None:
    _registered(service)
    service.transition("node-a", NodeState.OFFLINE, "deregistered")
    service.transition("node-a", NodeState.ONLINE, "registered")
    history = service.registry.history("node-a")
    triples = [
        (h.from_state.value if h.from_state else None, h.to_state.value, h.reason) for h in history
    ]
    assert triples == [
        (None, "online", "registered"),
        ("online", "offline", "deregistered"),
        ("offline", "online", "registered"),
    ]


def test_busy_node_still_heartbeats_and_stays_busy(tmp_path) -> None:
    registry = SQLiteRegistry(str(tmp_path / "r.sqlite3"))
    registry.open()
    settings = make_settings(tmp_path)
    service = NodeService(registry, settings)
    _registered(service)
    service.mark_busy("node-a")
    service.heartbeat("node-a", 0, None)
    row = registry.get_node("node-a")
    assert row.state == NodeState.BUSY
    assert row.last_seq == 0
    assert row.last_heartbeat is not None
    registry.close()
