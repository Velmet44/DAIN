"""Admin controls: node evict/recover, job list/cancel, model delete/rescan, log tail."""

from __future__ import annotations

import json

from conftest import manifest, register_payload
from dain_common.schemas import ModelManifest
from fastapi.testclient import TestClient

from dain_coordinator.app import create_app
from dain_coordinator.settings import CoordinatorSettings

ADMIN = "admin-key-123"
API = "client-key-123"
JOIN = "abcdef12"


def _settings(tmp_path, **overrides) -> CoordinatorSettings:
    values = dict(
        db_path=str(tmp_path / "ctrl.sqlite3"),
        model_store_dir=str(tmp_path / "store"),
        join_token=JOIN,
        api_key=API,
        admin_api_key=ADMIN,
    )
    values.update(overrides)
    return CoordinatorSettings(**values)


def _admin(client: TestClient) -> None:
    client.headers.update({"X-Admin-Key": ADMIN})


def _register(client: TestClient, node_id: str) -> None:
    client.headers.update({"X-API-Key": API})
    resp = client.post(
        "/node/register", json=register_payload(node_id, auth_token=JOIN, manifest_model=manifest())
    )
    assert resp.json()["accepted"] is True


def test_evict_and_recover_node(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        _register(client, "node-1")
        _admin(client)
        nodes = client.get("/admin/nodes").json()
        assert [n["node_id"] for n in nodes] == ["node-1"]

        resp = client.post("/admin/nodes/node-1/offline")
        assert resp.status_code == 200
        assert resp.json()["state"] == "offline"
        assert client.get("/admin/nodes").json()[0]["state"] == "offline"
        assert client.post("/admin/nodes/node-1/offline").status_code == 409

        resp = client.post("/admin/nodes/node-1/online")
        assert resp.status_code == 200
        assert resp.json()["state"] == "online"
        assert client.get("/admin/nodes").json()[0]["state"] == "online"
        assert client.post("/admin/nodes/node-1/online").status_code == 409


def test_evict_unknown_node_409(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        _admin(client)
        assert client.post("/admin/nodes/nope/offline").status_code == 409


def _make_model(tmp_path) -> None:
    store = tmp_path / "store" / "m-1"
    store.mkdir(parents=True)
    (store / "manifest.json").write_text(
        json.dumps(manifest_model_dict("m-1")), encoding="utf-8"
    )
    (store / "model.bin").write_bytes(b"\x00" * 2048)


def manifest_model_dict(model_id: str) -> dict:
    """Build a real ModelManifest dict (byte-level dev-model shape)."""
    return ModelManifest(
        model_id=model_id,
        name="test model",
        layers=16,
        hidden=256,
        heads=8,
        kv_heads=8,
        intermediate=1024,
        vocab_size=256,
        eos_token_id=1,
    ).model_dump(mode="json")


def test_models_list_size_delete_rescan(tmp_path) -> None:
    _make_model(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        _admin(client)
        models = client.get("/admin/models").json()["models"]
        assert [m["model_id"] for m in models] == ["m-1"]
        assert models[0]["size_bytes"] >= 2048

        assert client.post("/admin/models/rescan").json()["ok"] is True

        resp = client.post("/admin/models/m-1/delete")
        assert resp.status_code == 200
        assert client.get("/admin/models").json()["models"] == []
        assert client.post("/admin/models/m-1/delete").status_code == 404


def test_model_path_traversal_rejected(tmp_path) -> None:
    _make_model(tmp_path)
    with TestClient(create_app(_settings(tmp_path))) as client:
        _admin(client)
        assert client.post("/admin/models/..%2Fsecret/delete").status_code in (400, 404)
        assert client.post("/admin/models/evil..name/delete").status_code == 400


def test_jobs_list_and_cancel(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        _admin(client)
        empty = client.get("/admin/jobs").json()
        assert empty["jobs"] == []
        assert empty["active"] == 0

        job = client.app.state.jobs.create(
            "m-1", "hi", {}, ModelManifest(**manifest_model_dict("m-1"))
        )
        resp = client.post(f"/admin/jobs/{job.job_id}/cancel")
        assert resp.status_code == 200
        assert client.get("/admin/jobs").json()["jobs"][0]["state"] == "failed"
        assert resp.status_code == 200
        assert client.post(f"/admin/jobs/{job.job_id}/cancel").status_code == 409
        assert client.post("/admin/jobs/does-not-exist/cancel").status_code == 404


def test_logs_tail(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        _admin(client)
        resp = client.get("/admin/logs?lines=5")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["logs"], list)
        assert len(body["logs"]) <= 5
