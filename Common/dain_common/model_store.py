"""Model-store manifest IO (spec §8/§11).

The only module in `dain_common` that touches the filesystem: manifest.json
reading/listing shared by the coordinator (serving) and the node (export).
Pure schema logic stays I/O-free per the package contract.
"""

from __future__ import annotations

import json
import os
import re

from dain_common.schemas import ModelManifest

# Model ids are filesystem components served over HTTP: restrict them to safe
# characters so a hostile id can never escape the store directory ("..", "/",
# absolute paths, shell metacharacters, …).
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def is_safe_model_id(model_id: str) -> bool:
    """True when *model_id* is a single safe directory component."""
    return bool(model_id) and bool(_SAFE_COMPONENT.match(model_id))


def load_manifest(store_dir: str, model_id: str) -> ModelManifest | None:
    if not is_safe_model_id(model_id):
        return None
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
        if not is_safe_model_id(entry):
            continue
        manifest = load_manifest(store_dir, entry)
        if manifest is not None:
            models.append(manifest)
    return models
