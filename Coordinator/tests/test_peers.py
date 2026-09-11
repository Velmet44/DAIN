"""P2P shard routing: cached-inventory persistence, peer lookup, size probe.

Covers the coordinator half of the LAN-fast shard transfer feature: register /
heartbeat now carry the node's cached-shard inventory + peer URL, the store
indexes them, and `/model/peers/...` hands a downloading node the live peers
that already hold the bytes (so downloads happen peer-to-peer, not one stream
through the hotspot).
"""

from __future__ import annotations

from conftest import make_client, make_settings, register_payload
from dain_common.schemas import (
    Envelope,
    Heartbeat,
    MessageType,
    MetricsReport,
    ShardRef,
)
from fastapi.testclient import TestClient

JOIN = "dain-dev-join-token"

REF_0 = ShardRef(model_id="model-g", shard_id="shard-0", content_hash="a" * 64, size_bytes=1024)
REF_1 = ShardRef(model_id="model-g", shard_id="shard-1", content_hash="b" * 64, size_bytes=2048)


def register(client: TestClient, node_id: str, **extra: object) -> TestClient:
    payload = register_payload(node_id, auth_token=JOIN)
    payload.update(extra)
    return client.post("/node/register", json=payload)


def send_heartbeat(
    client: TestClient, node_id: str, token: str, seq: int, cached_shards: tuple = ()
) -> None:
    envelope = Envelope.wrap(
        MessageType.HEARTBEAT,
        Heartbeat(
            node_id=node_id,
            seq=seq,
            metrics=MetricsReport(),
            cached_shards=cached_shards,
        ),
        ts=1.75e9 + seq,
    )
    with client.websocket_connect(f"/node/ws?node_id={node_id}&token={token}") as ws:
        ws.send_text(envelope.model_dump_json())


def auth(node_id: str, token: str) -> dict[str, str]:
    return {"X-Node-Id": node_id, "X-Node-Token": token}


# -- register / heartbeat persist the peer advertisement -------------------------


def test_register_persists_peer_url_and_cached_shards(client: TestClient) -> None:
    ack = register(
        client,
        "node-a",
        cached_shards=[r.model_dump(mode="json") for r in (REF_0, REF_1)],
        peer_url="http://10.0.0.5:54321",
    ).json()
    assert ack["accepted"] is True

    row = client.app.state.service.registry.get_node("node-a")
    assert row.peer_url == "http://10.0.0.5:54321"
    assert {r.shard_id for r in row.cached_shards} == {"shard-0", "shard-1"}
    assert row.cached_shards[0].content_hash == REF_0.content_hash


def test_register_without_inventory_defaults_empty(client: TestClient) -> None:
    ack = register(client, "node-a").json()
    assert ack["accepted"] is True
    row = client.app.state.service.registry.get_node("node-a")
    assert row.peer_url is None
    assert row.cached_shards == ()


def test_heartbeat_updates_cached_shards(client: TestClient) -> None:
    ack = register(client, "node-a").json()
    send_heartbeat(
        client, "node-a", ack["node_token"], 0, cached_shards=(REF_0.model_dump(mode="json"),)
    )
    send_heartbeat(
        client, "node-a", ack["node_token"], 1, cached_shards=tuple()
    )
    row = client.app.state.service.registry.get_node("node-a")
    assert row.cached_shards == ()


# -- peers lookup -----------------------------------------------------------------


def test_peers_endpoint_lists_connected_holders(client: TestClient) -> None:
    ack_holder = register(
        client,
        "node-holder",
        cached_shards=[REF_0.model_dump(mode="json")],
        peer_url="http://10.0.0.7:6000",
    ).json()
    ack_non = register(client, "node-non").json()
    ack_req = register(client, "node-req").json()

    # An OFFLINE / never-connected holder must NOT be offered as a peer source.
    r = client.get(
        "/model/peers/model-g/shard-0", headers=auth("node-req", ack_req["node_token"])
    )
    assert r.json()["peers"] == []

# Holder connected → it becomes a source; non-holder and requester excluded.
    with client.websocket_connect(
        f"/node/ws?node_id=node-holder&token={ack_holder['node_token']}"
    ):
        still_connected = client.get(
            "/model/peers/model-g/shard-0", headers=auth("node-req", ack_req["node_token"])
        ).json()
        # Reconnect the non-holder too: it still holds nothing, so it stays absent.
        with client.websocket_connect(
            f"/node/ws?node_id=node-non&token={ack_non['node_token']}"
        ):
            still = client.get(
                "/model/peers/model-g/shard-0", headers=auth("node-req", ack_req["node_token"])
            ).json()
    assert still_connected.get("model_id") == "model-g"
    assert still_connected["peers"] == ["http://10.0.0.7:6000"]
    assert still["peers"] == ["http://10.0.0.7:6000"]


def test_peers_excludes_requester_even_if_it_holds_the_shard(client: TestClient) -> None:
    ack = register(
        client,
        "node-req",
        cached_shards=[REF_0.model_dump(mode="json")],
        peer_url="http://10.0.0.9:7000",
    ).json()
    with client.websocket_connect(f"/node/ws?node_id=node-req&token={ack['node_token']}"):
        r = client.get(
            "/model/peers/model-g/shard-0", headers=auth("node-req", ack["node_token"])
        ).json()
    assert r["peers"] == []


def test_peers_requires_node_auth(client: TestClient) -> None:
    register(
        client,
        "node-a",
        cached_shards=[REF_0.model_dump(mode="json")],
        peer_url="http://10.0.0.7:6000",
    )
    assert client.get("/model/peers/model-g/shard-0").status_code == 401


# -- size probe (HEAD) ------------------------------------------------------------


def test_head_shard_reports_size(tmp_path) -> None:
    store_dir = tmp_path / "store"
    model_dir = store_dir / "model-g"
    model_dir.mkdir(parents=True)
    (model_dir / "shard-0.safetensors").write_bytes(b"x" * 5000)
    settings = make_settings(tmp_path, model_store_dir=str(store_dir))
    with make_client(settings) as client:
        ack = register(client, "node-a").json()
        h = client.head(
            "/model/shard/model-g/shard-0", headers=auth("node-a", ack["node_token"])
        )
        assert h.status_code == 200
        assert h.headers["content-length"] == "5000"
        assert h.headers["accept-ranges"] == "bytes"
        assert h.content == b""

        miss = client.head(
            "/model/shard/model-g/nope", headers=auth("node-a", ack["node_token"])
        )
        assert miss.status_code == 404


def test_head_shard_rejects_unsafe_ids(tmp_path) -> None:
    settings = make_settings(tmp_path, model_store_dir=str(tmp_path / "store"))
    with make_client(settings) as client:
        ack = register(client, "node-a").json()
        # The traversal path has extra segments → never reaches the handler.
        h = client.head(
            "/model/shard/..%2F..%2Fetc/shady", headers=auth("node-a", ack["node_token"])
        )
        assert h.status_code == 404
    # The handler guard itself rejects any unsafe id component.
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from dain_coordinator.api import _shard_path

    for bad in ("a/b", "..", ""):
        try:
            _shard_path(settings, bad, "shard-0")
        except StarletteHTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError(f"_shard_path accepted unsafe id {bad!r}")
