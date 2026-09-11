"""Admin model-export endpoints (session N).

The coordinator never runs torch: `/admin/models/export` shells out to
`uv run --project <Node> python -m dain_node.model_export`. Tests stub the
subprocess runner on `app.state.model_export_runner` and use a fake Node
project dir, mirroring the GGUF-import tests.
"""

from __future__ import annotations

import json
import pathlib
import time

from dain_common.schemas import ModelManifest, QuantizationSpec

from tests.conftest import ADMIN_HEADERS, make_client, make_settings


def _store(tmp_path: pathlib.Path) -> pathlib.Path:
    store = tmp_path / "model_store"
    store.mkdir()
    return store


def _fake_node_project(tmp_path: pathlib.Path) -> pathlib.Path:
    node = tmp_path / "NodeFake"
    node.mkdir()
    (node / "pyproject.toml").write_text("[project]\nname='dain-node'\n")
    return node


def _quantized_manifest(model_id: str) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        name="quantized",
        layers=2,
        hidden=8,
        heads=2,
        kv_heads=2,
        intermediate=8,
        vocab_size=16,
        eos_token_id=0,
        format="torch_pt",
        quantization=QuantizationSpec(
            backend="torchao",
            scheme="int4_weight_only",
            bits=4,
            group_size=128,
            packing_layout="int4_cpu",
        ),
    )


def test_export_roots_listing(tmp_path) -> None:
    store = _store(tmp_path)
    root = tmp_path / "sources"
    root.mkdir()
    settings = make_settings(
        tmp_path,
        model_store_dir=str(store),
        node_project_dir=str(_fake_node_project(tmp_path)),
        export_roots=(str(root),),
    )
    with make_client(settings) as client:
        r = client.get("/admin/models/export/roots", headers=ADMIN_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["roots"] == [{"path": str(root), "exists": True}]
        assert body["busy"] is False


def test_export_validate_within_root(tmp_path) -> None:
    store = _store(tmp_path)
    root = tmp_path / "sources"
    src = root / "tiny-model"
    src.mkdir(parents=True)
    (src / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (src / "model.safetensors").write_bytes(b"x")
    settings = make_settings(
        tmp_path,
        model_store_dir=str(store),
        node_project_dir=str(_fake_node_project(tmp_path)),
        export_roots=(str(root),),
    )
    with make_client(settings) as client:
        r = client.post(
            "/admin/models/export/validate",
            headers=ADMIN_HEADERS,
            json={"source_dir": str(src), "model_id": "tiny-1"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["valid"] is True
        assert body["weight_count"] == 1
        assert body["has_fast_tokenizer"] is False


def test_export_validate_rejects_outside_root(tmp_path) -> None:
    store = _store(tmp_path)
    root = tmp_path / "sources"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    settings = make_settings(
        tmp_path,
        model_store_dir=str(store),
        node_project_dir=str(_fake_node_project(tmp_path)),
        export_roots=(str(root),),
    )
    with make_client(settings) as client:
        r = client.post(
            "/admin/models/export/validate",
            headers=ADMIN_HEADERS,
            json={"source_dir": str(outside), "model_id": "tiny-2"},
        )
        assert r.status_code == 403


def test_export_start_and_complete(tmp_path) -> None:
    store = _store(tmp_path)
    root = tmp_path / "sources"
    src = root / "tiny-model"
    src.mkdir(parents=True)
    (src / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (src / "model.safetensors").write_bytes(b"x")
    settings = make_settings(
        tmp_path,
        model_store_dir=str(store),
        node_project_dir=str(_fake_node_project(tmp_path)),
        export_roots=(str(root),),
    )
    with make_client(settings) as client:
        app = client.app

        async def fake_runner(argv):
            # emulate the successful exporter: publish the model dir + marker
            model_dir = store / "tiny-3"
            model_dir.mkdir()
            (model_dir / "manifest.json").write_text(
                _quantized_manifest("tiny-3").model_dump_json()
            )
            (store / ".model-exports.json").write_text(
                json.dumps({"tiny-3": {"model_id": "tiny-3"}})
            )
            return 0

        app.state.model_export_runner = fake_runner

        r = client.post(
            "/admin/models/export",
            headers=ADMIN_HEADERS,
            json={"source_dir": str(src), "model_id": "tiny-3"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["started"] == "tiny-3"

        # waits for the background task to finish
        deadline = time.time() + 3.0
        while time.time() < deadline and app.state.model_export is not None:
            time.sleep(0.02)
        assert app.state.model_export is None
        assert app.state.model_export_error is None


def test_export_rejects_when_busy(tmp_path) -> None:
    store = _store(tmp_path)
    root = tmp_path / "sources"
    src = root / "tiny-model"
    src.mkdir(parents=True)
    (src / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (src / "model.safetensors").write_bytes(b"x")
    settings = make_settings(
        tmp_path,
        model_store_dir=str(store),
        node_project_dir=str(_fake_node_project(tmp_path)),
        export_roots=(str(root),),
    )
    with make_client(settings) as client:
        client.app.state.model_export = {"model_id": "stuck", "started_at": time.time()}
        r = client.post(
            "/admin/models/export",
            headers=ADMIN_HEADERS,
            json={"source_dir": str(src), "model_id": "tiny-4"},
        )
        assert r.status_code == 409


def test_export_requires_roots(tmp_path) -> None:
    store = _store(tmp_path)
    settings = make_settings(
        tmp_path,
        model_store_dir=str(store),
        node_project_dir=str(_fake_node_project(tmp_path)),
        export_roots=(),
    )
    with make_client(settings) as client:
        r = client.post(
            "/admin/models/export",
            headers=ADMIN_HEADERS,
            json={"source_dir": str(tmp_path / "x"), "model_id": "tiny-5"},
        )
        assert r.status_code == 400


def test_public_models_include_quantization(tmp_path) -> None:
    store = _store(tmp_path)
    model_dir = store / "quant-model"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text(_quantized_manifest("quant-model").model_dump_json())
    (store / "legacy-model").mkdir()
    (store / "legacy-model" / "manifest.json").write_text(
        ModelManifest(
            model_id="legacy-model", name="legacy", layers=2, hidden=8, heads=2,
            kv_heads=2, intermediate=8, vocab_size=16, eos_token_id=0,
        ).model_dump_json()
    )
    settings = make_settings(tmp_path, model_store_dir=str(store))
    with make_client(settings) as client:
        r = client.get("/v1/models", headers={"X-API-Key": "dain-dev-key"})
        assert r.status_code == 200
        by_id = {m["model_id"]: m for m in r.json()["models"]}
        assert by_id["quant-model"]["format"] == "torch_pt"
        assert by_id["quant-model"]["quantization"]["bits"] == 4
        assert by_id["quant-model"]["quantization"]["packing_layout"] == "int4_cpu"
        assert by_id["legacy-model"]["quantization"] is None
