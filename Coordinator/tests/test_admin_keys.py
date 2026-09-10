"""Admin key management: rotate / reset the join token, API key, admin key.

Runtime rotation swaps the settings object live (auth deps read
``app.state.settings`` per request) and persists the three key fields into the
coordinator's ``config.json`` when one is loaded.
"""

from __future__ import annotations

import json

from conftest import register_payload
from fastapi.testclient import TestClient

from dain_coordinator.app import create_app
from dain_coordinator.config import config_path, load_config, write_default_config
from dain_coordinator.settings import (
    DEFAULT_ADMIN_API_KEY,
    DEFAULT_API_KEY,
    DEFAULT_JOIN_TOKEN,
    CoordinatorSettings,
)

JOIN = "abcdef12"
API = "client-key-123"
ADMIN = "admin-key-123"


def _settings(tmp_path) -> CoordinatorSettings:
    return CoordinatorSettings(
        db_path=str(tmp_path / "keys.sqlite3"),
        join_token=JOIN,
        api_key=API,
        admin_api_key=ADMIN,
    )


def _headers(key: str) -> dict:
    return {"X-Admin-Key": key}


def _admin(client: TestClient, key: str) -> TestClient:
    client.headers.update(_headers(key))
    return client


def test_get_keys_requires_admin(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/admin/settings/keys").status_code == 401
        state = client.get("/admin/settings/keys", headers=_headers(ADMIN)).json()
        assert state["join_token"] == JOIN
        assert state["api_key"] == API
        assert state["admin_api_key"] == ADMIN
        assert state["writable"] is False


def test_change_api_key_takes_effect(tmp_path) -> None:
    path = config_path(tmp_path)
    write_default_config(path)
    with TestClient(create_app(_settings(tmp_path), settings_path=str(path))) as client:
        _admin(client, ADMIN)
        res = client.put(
            "/admin/settings/keys", json={"api_key": "new-api-key-456"}
        )
        assert res.status_code == 200
        assert res.json()["api_key"] == "new-api-key-456"
        assert res.json()["writable"] is True

        # Old API key dies immediately; the new one works.
        assert client.get("/v1/nodes").status_code == 401
        assert (
            client.get("/v1/nodes", headers={"X-API-Key": "new-api-key-456"}).status_code == 200
        )


def test_change_admin_key_requires_new_admin_key(tmp_path) -> None:
    path = config_path(tmp_path)
    write_default_config(path)
    with TestClient(create_app(_settings(tmp_path), settings_path=str(path))) as client:
        _admin(client, ADMIN)
        res = client.put("/admin/settings/keys", json={"admin_api_key": "new-admin-key-789"})
        assert res.status_code == 200
        assert client.get("/admin/nodes").status_code == 401
        assert (
            client.get("/admin/nodes", headers=_headers("new-admin-key-789")).status_code == 200
        )


def test_join_token_rotation_gates_registration(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        _admin(client, ADMIN)
        payload = register_payload("node-1", auth_token=JOIN)
        assert client.post("/node/register", json=payload).json()["accepted"] is True

        client.put("/admin/settings/keys", json={"join_token": "new-join-token-9"})
        payload["auth_token"] = JOIN
        assert client.post("/node/register", json=payload).json()["accepted"] is False
        fresh = register_payload("node-2", auth_token="new-join-token-9")
        assert client.post("/node/register", json=fresh).json()["accepted"] is True


def test_unknown_field_rejected(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        _admin(client, ADMIN)
        assert (
            client.put("/admin/settings/keys", json={"db_path": "/tmp/x"}).status_code == 422
        )


def test_short_key_rejected(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        _admin(client, ADMIN)
        assert client.put("/admin/settings/keys", json={"api_key": "abc"}).status_code == 422
        assert client.put("/admin/settings/keys", json={"join_token": "a"}).status_code == 422


def test_reset_restores_defaults(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        _admin(client, ADMIN)
        res = client.put(
            "/admin/settings/keys",
            json={"join_token": "x" * 16, "api_key": "y" * 16, "admin_api_key": "z" * 16},
        )
        assert res.status_code == 200
        # The admin key rotated to "z"×16, so further admin calls need it.
        client.headers.update(_headers("z" * 16))
        res = client.put(
            "/admin/settings/keys",
            json={
                "reset_join_token": True,
                "reset_api_key": True,
                "reset_admin_api_key": True,
            },
        ).json()
        assert res["join_token"] == DEFAULT_JOIN_TOKEN
        assert res["api_key"] == DEFAULT_API_KEY
        assert res["admin_api_key"] == DEFAULT_ADMIN_API_KEY


def test_persisted_to_config_and_survives_restart(tmp_path) -> None:
    path = config_path(tmp_path)
    write_default_config(path)
    with TestClient(create_app(_settings(tmp_path), settings_path=str(path))) as client:
        _admin(client, ADMIN)
        client.put("/admin/settings/keys", json={"api_key": "persisted-key-999"})
    data = load_config(path)
    assert data["api_key"] == "persisted-key-999"

    reloaded = CoordinatorSettings.from_config(data, tmp_path)
    assert reloaded.api_key == "persisted-key-999"
    with TestClient(create_app(reloaded, settings_path=str(path))) as client2:
        assert (
            client2.get(
                "/v1/nodes", headers={"X-API-Key": "persisted-key-999"}
            ).status_code == 200
        )


def test_no_settings_path_is_in_memory_only(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        _admin(client, ADMIN)
        res = client.put("/admin/settings/keys", json={"api_key": "mem-only-key-777"})
        assert res.status_code == 200
        assert res.json()["writable"] is False
        assert (
            client.get("/v1/nodes", headers={"X-API-Key": "mem-only-key-777"}).status_code
            == 200
        )


def test_unchanged_payload_is_noop(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        _admin(client, ADMIN)
        state = client.get("/admin/settings/keys").json()
        res = client.put(
            "/admin/settings/keys",
            json={
                "join_token": state["join_token"],
                "api_key": state["api_key"],
                "admin_api_key": state["admin_api_key"],
            },
        )
        assert res.status_code == 200
        assert res.json() == state


def test_config_file_created_on_first_edit(tmp_path) -> None:
    path = config_path(tmp_path)
    assert not path.exists()
    with TestClient(create_app(_settings(tmp_path), settings_path=str(path))) as client:
        _admin(client, ADMIN)
        client.put("/admin/settings/keys", json={"api_key": "bootstrapped-123"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["api_key"] == "bootstrapped-123"
    assert data["admin_api_key"] == ADMIN  # settings used at runtime (not the default)
