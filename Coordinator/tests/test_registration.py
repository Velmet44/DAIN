"""S2 checkpoint: registration, auth, heartbeats, state machine lifecycle.

Uses compressed timings (heartbeat 0.1 s, offline after 3 missed = 0.3 s) so the
heartbeat-timeout path runs in fractions of a second.
"""

import json

import pytest
from conftest import (
    ADMIN_HEADERS,
    cpu_only_manifest,
    gpu,
    make_client,
    make_settings,
    manifest,
    register_payload,
    wait_for,
)
from dain_common.schemas import Envelope, Heartbeat, MessageType, MetricsReport
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

JOIN = "dain-dev-join-token"


def send_heartbeat(
    client: TestClient, node_id: str, token: str, seq: int, metrics: MetricsReport | None = None
):
    envelope = Envelope.wrap(
        MessageType.HEARTBEAT,
        Heartbeat(node_id=node_id, seq=seq, metrics=metrics),
        ts=1.75e9 + seq,
    )
    with client.websocket_connect(f"/node/ws?node_id={node_id}&token={token}") as ws:
        ws.send_text(envelope.model_dump_json())


# -- registration & auth -------------------------------------------------------


def test_register_rejects_bad_join_token(client: TestClient) -> None:
    response = client.post(
        "/node/register", json=register_payload("node-a", auth_token="wrong-token")
    )
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is False
    assert "join token" in body["reason"]


def test_register_issues_persistent_token(client: TestClient) -> None:
    ack = client.post("/node/register", json=register_payload("node-a", auth_token=JOIN)).json()
    assert ack["accepted"] is True
    assert ack["node_token"] and len(ack["node_token"]) >= 16
    assert ack["heartbeat_interval_s"] == pytest.approx(0.1)

    # Re-registration with the issued token is accepted (state unchanged).
    again = client.post(
        "/node/register", json=register_payload("node-a", auth_token=ack["node_token"])
    ).json()
    assert again["accepted"] is True

    # Re-registration with a wrong token is rejected.
    bad = client.post(
        "/node/register", json=register_payload("node-a", auth_token=JOIN + "x")
    ).json()
    assert bad["accepted"] is False


def test_register_low_score_rejected(client: TestClient, tmp_path) -> None:
    settings = make_settings(tmp_path, min_score=0.5)
    with make_client(settings) as tight:
        ack = tight.post(
            "/node/register",
            json=register_payload("node-weak", auth_token=JOIN, manifest_model=cpu_only_manifest()),
        ).json()
    assert ack["accepted"] is False
    assert "score below minimum" in ack["reason"]
    detail = client.get("/admin/nodes/node-weak", headers=ADMIN_HEADERS).json()
    assert detail["state"] == "offline"


def test_ws_rejects_bad_credentials(client: TestClient) -> None:
    client.post("/node/register", json=register_payload("node-a", auth_token=JOIN))
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/node/ws?node_id=node-a&token=wrong") as ws:
            ws.send_text("hello")
    assert excinfo.value.code == 4401


def test_ws_rejects_malformed_message(client: TestClient) -> None:
    ack = client.post("/node/register", json=register_payload("node-a", auth_token=JOIN)).json()
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(f"/node/ws?node_id=node-a&token={ack['node_token']}") as ws:
            ws.send_text("this is not json")
            ws.receive_text()  # observe the server-side 4400 close
    assert excinfo.value.code == 4400


# -- heartbeat lifecycle ---------------------------------------------------------


def test_heartbeat_updates_score_and_seq(client: TestClient) -> None:
    ack = client.post("/node/register", json=register_payload("node-a", auth_token=JOIN)).json()
    send_heartbeat(
        client,
        "node-a",
        ack["node_token"],
        0,
        MetricsReport(gpu_util_pct=30.0, vram_free_gb=11.0, net_bw_mbps=300.0),
    )
    send_heartbeat(
        client,
        "node-a",
        ack["node_token"],
        1,
        MetricsReport(gpu_util_pct=40.0, vram_free_gb=10.9, net_bw_mbps=290.0),
    )
    detail = client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()
    assert detail["state"] == "online"
    assert detail["last_seq"] == 1
    assert detail["score"] is not None and 0.0 < detail["score"] < 1.0
    assert detail["score_components"]["throughput"] > 0
    assert detail["last_heartbeat"] is not None


def test_heartbeat_timeout_then_reregister(client: TestClient) -> None:
    """The S2 gate: register → ONLINE → silence → OFFLINE → re-register → ONLINE."""
    ack = client.post("/node/register", json=register_payload("node-a", auth_token=JOIN)).json()
    send_heartbeat(client, "node-a", ack["node_token"], 0)
    assert client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()["state"] == "online"

    # Stop heartbeating: offline after 3 × 0.1 s (monitor ticks every 0.05 s).
    assert wait_for(
        lambda: (
            client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()["state"] == "offline"
        ),
        timeout_s=2.0,
    )
    history = client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()["history"]
    assert any(h["to_state"] == "offline" and h["reason"] == "heartbeat_timeout" for h in history)

    # Re-register with the issued token → ONLINE, history preserved.
    re_ack = client.post(
        "/node/register", json=register_payload("node-a", auth_token=ack["node_token"])
    ).json()
    assert re_ack["accepted"] is True
    assert re_ack["node_token"] == ack["node_token"]
    detail = client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()
    assert detail["state"] == "online"
    transitions = [(h["from_state"], h["to_state"], h["reason"]) for h in detail["history"]]
    assert ("online", "offline", "heartbeat_timeout") in transitions
    assert ("offline", "online", "registered") in transitions


def test_degraded_on_thermal_then_recovered(client: TestClient) -> None:
    ack = client.post("/node/register", json=register_payload("node-a", auth_token=JOIN)).json()
    send_heartbeat(
        client,
        "node-a",
        ack["node_token"],
        0,
        MetricsReport(gpu_util_pct=40.0, temp_c=95.0),
    )
    assert client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()["state"] == "degraded"
    history = client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()["history"]
    assert any(h["to_state"] == "degraded" and h["reason"] == "thermal" for h in history)

    send_heartbeat(
        client,
        "node-a",
        ack["node_token"],
        1,
        MetricsReport(gpu_util_pct=40.0, temp_c=55.0),
    )
    detail = client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()
    assert detail["state"] == "online"
    assert any(
        h["to_state"] == "online" and h["reason"] == "metrics_recovered" for h in detail["history"]
    )


def test_overload_requires_sustained_strikes(client: TestClient) -> None:
    ack = client.post("/node/register", json=register_payload("node-a", auth_token=JOIN)).json()
    for seq in range(2):  # 2 strikes: below the default threshold of 3
        send_heartbeat(client, "node-a", ack["node_token"], seq, MetricsReport(gpu_util_pct=99.0))
    assert client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()["state"] == "online"
    send_heartbeat(client, "node-a", ack["node_token"], 2, MetricsReport(gpu_util_pct=99.0))
    assert client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()["state"] == "degraded"
    # A clean report resets strikes and recovers.
    send_heartbeat(client, "node-a", ack["node_token"], 3, MetricsReport(gpu_util_pct=30.0))
    assert client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()["state"] == "online"


# -- deregistration ---------------------------------------------------------------


def test_deregister_auth_and_states(client: TestClient) -> None:
    ack = client.post("/node/register", json=register_payload("node-a", auth_token=JOIN)).json()

    r = client.post("/node/deregister", json={"node_id": "node-a", "auth_token": "wrong"})
    assert r.status_code == 403
    r = client.post("/node/deregister", json={"node_id": "ghost", "auth_token": "x" * 16})
    assert r.status_code == 404

    r = client.post("/node/deregister", json={"node_id": "node-a", "auth_token": ack["node_token"]})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()["state"] == "offline"


def test_admin_listing_and_filters(client: TestClient) -> None:
    ack_b = client.post("/node/register", json=register_payload("node-b", auth_token=JOIN)).json()
    client.post("/node/register", json=register_payload("node-a", auth_token=JOIN))
    client.post("/node/deregister", json={"node_id": "node-b", "auth_token": ack_b["node_token"]})
    all_nodes = client.get("/admin/nodes", headers=ADMIN_HEADERS).json()
    assert {n["node_id"] for n in all_nodes} == {"node-a", "node-b"}
    online = client.get("/admin/nodes", params={"state": "online"}, headers=ADMIN_HEADERS).json()
    assert [n["node_id"] for n in online] == ["node-a"]
    assert (
        client.get("/admin/nodes", params={"state": "bogus"}, headers=ADMIN_HEADERS).status_code
        == 400
    )


def test_register_payload_is_wire_valid(client: TestClient) -> None:
    """The REST body must round-trip through the wire Envelope schema (spec §11)."""
    from dain_common.schemas import Register, parse_payload

    raw = register_payload("node-wire", auth_token=JOIN)
    registration = Register.model_validate(raw)  # REST body == wire REGISTER payload
    envelope = Envelope.wrap(MessageType.REGISTER, registration, ts=1.0)
    parsed = json.loads(envelope.model_dump_json())
    assert parsed["type"] == "register"
    assert parse_payload(envelope).node_id == "node-wire"


def test_heartbeat_refreshes_capacity_snapshot(client: TestClient) -> None:
    """Free RAM/VRAM in the manifest track live metrics (S17) — a node that
    registered with little free memory becomes placeable once memory frees up."""
    from dain_common.schemas import CPUInfo

    small_cpu = cpu_only_manifest().model_copy(
        update={"cpu": CPUInfo(cores=8, ram_total_gb=32.0, ram_free_gb=1.0)}
    )
    ack = client.post(
        "/node/register", json=register_payload("node-a", manifest_model=small_cpu, auth_token=JOIN)
    ).json()
    send_heartbeat(client, "node-a", ack["node_token"], 0, MetricsReport(cpu_util_pct=40.0))
    service = client.app.state.service
    assert service.registry.get_node("node-a").manifest.cpu.ram_free_gb == 1.0

    send_heartbeat(
        client, "node-a", ack["node_token"], 1, MetricsReport(cpu_util_pct=40.0, ram_free_gb=8.0)
    )
    assert service.registry.get_node("node-a").manifest.cpu.ram_free_gb == 8.0

    # GPU nodes refresh vram_free_gb the same way.
    gpu_manifest = manifest(gpu(vram_free_gb=2.0))
    ack2 = client.post(
        "/node/register",
        json=register_payload("node-b", manifest_model=gpu_manifest, auth_token=JOIN),
    ).json()
    send_heartbeat(
        client, "node-b", ack2["node_token"], 0, MetricsReport(gpu_util_pct=10.0, vram_free_gb=11.5)
    )
    assert service.registry.get_node("node-b").manifest.gpu.vram_free_gb == 11.5
