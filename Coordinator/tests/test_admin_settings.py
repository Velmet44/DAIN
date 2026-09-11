"""Admin runtime settings (session 16): GET/PUT /admin/settings.

Live-editable settings swap the frozen settings object in place and persist
as-typed to config.json; changing model_store_dir rescans placements. Restart-
required fields (host/port/timing) are exposed read-only.
"""

from __future__ import annotations

import json

from dain_common.schemas import ModelManifest

from tests.conftest import ADMIN_HEADERS, make_client, make_settings


def _manifest(model_id: str) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        name="m",
        layers=2,
        hidden=8,
        heads=2,
        kv_heads=2,
        intermediate=8,
        vocab_size=16,
        eos_token_id=0,
    )


def test_get_settings_shape(tmp_path) -> None:
    with make_client(make_settings(tmp_path)) as client:
        r = client.get("/admin/settings", headers=ADMIN_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["settings"]["model_store_dir"]["type"] == "path"
        assert body["settings"]["min_score"]["type"] == "float"
        assert body["settings"]["queue_limit"]["value"] == 16
        assert body["writable"] is False  # no config file in the test fixture
        # Restart-required fields are exposed read-only.
        assert body["restart"]["host"] == "0.0.0.0"
        assert "port" in body["restart"]
        assert "model_store_dir" not in body["restart"]


def test_put_model_store_dir_applies_live_and_persists(tmp_path) -> None:
    # A second store containing one model the current store doesn't have.
    store2 = tmp_path / "other_store"
    (store2 / "other-model").mkdir(parents=True)
    (store2 / "other-model" / "manifest.json").write_text(
        _manifest("other-model").model_dump_json()
    )
    settings = make_settings(tmp_path, model_store_dir=str(tmp_path / "model_store"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_store_dir": str(tmp_path / "model_store")}))
    with make_client(settings) as client:
        app = client.app
        app.state.settings_path = str(config_path)  # enable persistence
        triggers: list[str] = []
        app.state.recompute_pool = triggers.append

        # The old store is empty.
        models = client.get("/v1/models", headers={"X-API-Key": "dain-dev-key"}).json()["models"]
        assert models == []

        r = client.put(
            "/admin/settings",
            headers=ADMIN_HEADERS,
            json={"model_store_dir": str(store2)},  # absolute; as-typed persistence below
        )
        assert r.status_code == 200
        body = r.json()
        assert body["applied"] == ["model_store_dir"]
        assert body["persisted"] is True
        # Live: /v1/models now serves the new store.
        models = client.get("/v1/models", headers={"X-API-Key": "dain-dev-key"}).json()["models"]
        assert body["settings"]["model_store_dir"]["value"].endswith("other_store")
        assert any(m["model_id"] == "other-model" for m in models)
        # Placement rescan fired.
        assert "admin_settings" in triggers
        # Persisted exactly as typed (no re-resolution → portable values stay).
        assert json.loads(config_path.read_text())["model_store_dir"] == str(store2)


def test_put_node_project_dir_validation(tmp_path) -> None:
    with make_client(make_settings(tmp_path)) as client:
        # Non-existent project → 400.
        r = client.put(
            "/admin/settings", headers=ADMIN_HEADERS, json={"node_project_dir": "no/such/dir"}
        )
        assert r.status_code == 400
        # Empty string is legal (= default sibling Node/).
        r = client.put("/admin/settings", headers=ADMIN_HEADERS, json={"node_project_dir": ""})
        assert r.status_code == 200
        assert r.json()["settings"]["node_project_dir"]["value"] == ""
        # An actual project dir passes.
        node = tmp_path / "NodeFake"
        node.mkdir()
        (node / "pyproject.toml").write_text("[project]\nname='x'\n")
        r = client.put(
            "/admin/settings", headers=ADMIN_HEADERS, json={"node_project_dir": str(node)}
        )
        assert r.status_code == 200


def test_put_numeric_bounds_and_live_swap(tmp_path) -> None:
    with make_client(make_settings(tmp_path)) as client:
        r = client.put("/admin/settings", headers=ADMIN_HEADERS, json={"queue_limit": 0})
        assert r.status_code == 422  # pydantic bounds
        r = client.put(
            "/admin/settings",
            headers=ADMIN_HEADERS,
            json={"min_score": 0.42, "job_timeout_s": 90.0, "max_completion_tokens": 256},
        )
        assert r.status_code == 200
        s = client.app.state.settings
        assert s.min_score == 0.42
        assert s.job_timeout_s == 90.0
        assert s.max_completion_tokens == 256


def test_put_unknown_and_noop(tmp_path) -> None:
    with make_client(make_settings(tmp_path)) as client:
        r = client.put("/admin/settings", headers=ADMIN_HEADERS, json={"host": "x"})
        assert r.status_code == 422  # extra=forbid: host is restart-only
        r = client.put("/admin/settings", headers=ADMIN_HEADERS, json={})
        assert r.status_code == 200
        assert "applied" not in r.json()  # nothing changed
