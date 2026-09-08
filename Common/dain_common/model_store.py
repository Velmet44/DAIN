"""Model-store manifest IO (spec §8/§11).

The only module in `dain_common` that touches the filesystem: manifest.json
reading/listing shared by the coordinator (serving) and the node (export).
Pure schema logic stays I/O-free per the package contract.
"""

from __future__ import annotations

import json
import os

from dain_common.schemas import ModelManifest


def load_manifest(store_dir: str, model_id: str) -> ModelManifest | None:
    path = os.path.join(store_dir, model_id, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return ModelManifest.model_validate(json.load(fh))


def list_models(store_dir: str) -> list[ModelManifest]:
    if not os.path.isdir(store_dir):
        return []
    models = []
    for entry in sorted(os.listdir(store_dir)):
        manifest = load_manifest(store_dir, entry)
        if manifest is not None:
            models.append(manifest)
    return models
