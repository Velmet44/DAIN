"""S2 checkpoint: registry state (records, tokens, history) survives restart."""

import dataclasses

from conftest import ADMIN_HEADERS, make_client, make_settings, register_payload, wait_for
from dain_common.schemas import Envelope, Heartbeat, MessageType

JOIN = "dain-dev-join-token"


def test_state_survives_restart(tmp_path) -> None:
    settings = make_settings(tmp_path)
    db = str(tmp_path / "coordinator.sqlite3")

    # First process: register + one heartbeat, then die.
    with make_client(dataclasses.replace(settings, db_path=db)) as client:
        ack = client.post("/node/register", json=register_payload("node-a", auth_token=JOIN)).json()
        assert ack["accepted"] is True
        envelope = Envelope.wrap(
            MessageType.HEARTBEAT, Heartbeat(node_id="node-a", seq=0), ts=1.75e9
        )
        with client.websocket_connect(f"/node/ws?node_id=node-a&token={ack['node_token']}") as ws:
            ws.send_text(envelope.model_dump_json())

    # Second process: same DB file, fresh app.
    with make_client(dataclasses.replace(settings, db_path=db)) as client:
        nodes = client.get("/admin/nodes", headers=ADMIN_HEADERS).json()
        assert [n["node_id"] for n in nodes] == ["node-a"]
        assert nodes[0]["last_seq"] == 0

        # The issued token still authenticates the WS connection.
        envelope = Envelope.wrap(
            MessageType.HEARTBEAT, Heartbeat(node_id="node-a", seq=1), ts=1.75e9 + 1
        )
        with client.websocket_connect(f"/node/ws?node_id=node-a&token={ack['node_token']}") as ws:
            ws.send_text(envelope.model_dump_json())

        # History from the first process is intact, including original registration.
        detail = client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()
        assert detail["history"][0]["to_state"] == "online"
        assert detail["history"][0]["reason"] == "registered"

        # The stale node (silent since before the restart) is demoted by the new
        # process's monitor — persistence and detection both work across restarts.
        def timed_out() -> bool:
            history = client.get("/admin/nodes/node-a", headers=ADMIN_HEADERS).json()["history"]
            return any(h["reason"] == "heartbeat_timeout" for h in history)

        assert wait_for(timed_out, timeout_s=3.0)


def test_create_app_is_lifecycle_safe(tmp_path) -> None:
    """Two create_app() cycles on the same path never corrupt the store."""
    settings = make_settings(tmp_path)
    for _ in range(2):
        with make_client(settings):  # opens + closes registry, runs monitor
            pass
