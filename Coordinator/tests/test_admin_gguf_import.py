"""Admin GGUF import endpoints (session 14).

The coordinator never runs torch itself: `/admin/models/import` shells out to
`uv run --project <Node> python -m dain_node.import_gguf`. Tests stub the
subprocess runner on `app.state.gguf_runner` and use a fake Node project dir.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import threading
import time

from dain_common.schemas import ModelManifest

from tests.conftest import ADMIN_HEADERS, make_client, make_settings


def _store(tmp_path: pathlib.Path, *, gguf: bool = False) -> pathlib.Path:
    store = tmp_path / "model_store"
    store.mkdir()
    (store / "my-model").mkdir()
    (store / "my-model" / "manifest.json").write_text("{}")
    if gguf:
        (store / "tiny.Q4_K_M.gguf").write_bytes(b"fake gguf bytes")
    return store


def _fake_node_project(tmp_path: pathlib.Path) -> pathlib.Path:
    node = tmp_path / "NodeFake"
    node.mkdir()
    (node / "pyproject.toml").write_text("[project]\nname='dain-node'\n")
    return node


def _manifest(model_id: str) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        name="imported",
        layers=2,
        hidden=8,
        heads=2,
        kv_heads=2,
        intermediate=8,
        vocab_size=16,
        eos_token_id=0,
    )


def test_imports_listing_classifies_status(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, gguf=True)
    settings = make_settings(
        tmp_path, model_store_dir=str(store), node_project_dir=str(_fake_node_project(tmp_path))
    )
    with make_client(settings) as client:
        app = client.app
        app.state.gguf_runner = None  # capability fields present, nothing running

        # Nothing imported yet -> pending.
        r = client.get("/admin/models/imports", headers=ADMIN_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["node_project"]
        assert body["busy"] is False
        assert body["imports"] == [
            {
                "filename": "tiny.Q4_K_M.gguf",
                "size_bytes": len(b"fake gguf bytes"),
                "model_id": None,
                "status": "pending",
                "imported_at": None,
            }
        ]

        # Simulate a finished import: marker + model dir present.
        (store / "dain-tiny").mkdir()
        (store / "dain-tiny" / "manifest.json").write_text(_manifest("dain-tiny").model_dump_json())
        (store / ".gguf-imports.json").write_text(
            json.dumps({"tiny.Q4_K_M.gguf": {"sha256": "a" * 64, "model_id": "dain-tiny"}})
        )
        body = client.get("/admin/models/imports", headers=ADMIN_HEADERS).json()
        assert body["imports"][0]["status"] == "imported"
        assert body["imports"][0]["model_id"] == "dain-tiny"

        # While a run is active the file shows `importing`.
        app.state.gguf_import = {"filename": None, "started_at": time.time()}
        body = client.get("/admin/models/imports", headers=ADMIN_HEADERS).json()
        assert body["busy"] is True
        assert body["imports"][0]["status"] == "importing"


def test_import_start_validation(tmp_path) -> None:
    store = _store(tmp_path)
    settings = make_settings(
        tmp_path, model_store_dir=str(store), node_project_dir=str(_fake_node_project(tmp_path))
    )
    with make_client(settings) as client:
        # Path traversal / non-gguf names are rejected before anything runs.
        r = client.post(
            "/admin/models/import",
            headers=ADMIN_HEADERS,
            json={"filename": "../model_store/tiny.gguf"},
        )
        assert r.status_code == 400
        r = client.post(
            "/admin/models/import", headers=ADMIN_HEADERS, json={"filename": "nope.gguf"}
        )
        assert r.status_code == 404
        # Unknown model store -> 404.
        settings2 = make_settings(
            tmp_path, model_store_dir=str(tmp_path / "missing"), node_project_dir=str(tmp_path)
        )
        with make_client(settings2) as c2:
            r2 = c2.post("/admin/models/import", headers=ADMIN_HEADERS, json={})
            assert r2.status_code == 404


def test_import_runs_converter_and_recomputes(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, gguf=True)
    settings = make_settings(
        tmp_path, model_store_dir=str(store), node_project_dir=str(_fake_node_project(tmp_path))
    )
    captured: dict = {}
    done = threading.Event()
    release = threading.Event()

    async def fake_runner(argv):
        captured["argv"] = argv
        done.set()
        await asyncio.to_thread(release.wait, 5)  # hold the "busy" window open
        return 0

    with make_client(settings) as client:
        app = client.app
        monkeypatch.setattr(app.state, "gguf_runner", fake_runner, raising=False)
        triggers: list[str] = []
        monkeypatch.setattr(app.state, "recompute_pool", triggers.append)

        r = client.post("/admin/models/import", headers=ADMIN_HEADERS, json={})
        assert r.status_code == 200
        assert r.json()["ok"] is True

        assert done.wait(5), "runner never invoked"
        argv = captured["argv"]
        assert argv[0].lower().endswith("uv.exe") or argv[0] == "uv"  # resolved via shutil.which
        assert argv[1:6] == ["run", "--project", str(tmp_path / "NodeFake"), "python", "-m"]
        assert "dain_node.import_gguf" in argv
        assert str(store) in argv
        assert "--dtype" in argv and "fp16" in argv

        # Busy while running...
        r = client.post("/admin/models/import", headers=ADMIN_HEADERS, json={})
        assert r.status_code == 409

        # ...then cleared after the fake run completes, and placements rescan.
        release.set()
        deadline = time.time() + 5
        while time.time() < deadline and app.state.gguf_import is not None:
            time.sleep(0.02)
        assert app.state.gguf_import is None
        assert app.state.gguf_import_error is None
        assert "gguf_import" in triggers


def test_import_reports_failure(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, gguf=True)
    settings = make_settings(
        tmp_path, model_store_dir=str(store), node_project_dir=str(_fake_node_project(tmp_path))
    )
    with make_client(settings) as client:
        app = client.app

        async def failing_runner(argv):
            return 1

        monkeypatch.setattr(app.state, "gguf_runner", failing_runner, raising=False)
        r = client.post("/admin/models/import", headers=ADMIN_HEADERS, json={})
        assert r.status_code == 200
        deadline = time.time() + 5
        while time.time() < deadline and app.state.gguf_import is not None:
            time.sleep(0.02)
        assert app.state.gguf_import is None
        assert app.state.gguf_import_error and "code 1" in app.state.gguf_import_error


def test_import_503_without_node_project(tmp_path) -> None:
    store = _store(tmp_path, gguf=True)
    settings = make_settings(
        tmp_path, model_store_dir=str(store), node_project_dir=str(tmp_path / "NoSuchNode")
    )
    with make_client(settings) as client:
        r = client.post("/admin/models/import", headers=ADMIN_HEADERS, json={})
        assert r.status_code == 503
        assert "Node project not found" in r.json()["detail"]
