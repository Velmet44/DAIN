"""Coordinator config.json loader.

Like the node, the coordinator reads a ``config.json`` sitting next to its
package root (``Coordinator/`` in dev) instead of requiring env vars.  On the
first run a full default config is written so every option is discoverable and
editable.  Environment variables still win on a per-field basis when they are
set (launchers like ``Scripts/start-samepc.ps1`` keep their precedence).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dain_coordinator.settings import COORDINATOR_ENV, DEFAULT_CONFIG, CoordinatorSettings

log = logging.getLogger("dain.coordinator.config")


def find_base_dir() -> Path:
    """The directory that owns the coordinator's runtime files.

    Dev: the ``Coordinator/`` project root (one level above this package).
    Frozen (future packaging): the directory containing the executable.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def config_path(base_dir: Path | None = None) -> Path:
    return (base_dir or find_base_dir()) / "config.json"


def load_config(path: Path) -> dict:
    """Read *path*; missing or unparseable files yield an empty dict."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            log.warning("config_not_dict path=%s — ignoring", path)
            return {}
        return data
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("config_load_failed path=%s error=%s — using defaults", path, exc)
        return {}


def write_default_config(path: Path) -> None:
    """Write the full default config (used on first run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(DEFAULT_CONFIG, fh, indent=2)
        fh.write("\n")


def resolve_settings(
    base_dir: Path | None = None,
) -> tuple[CoordinatorSettings, Path, bool]:
    """Load effective settings: config.json (or defaults) overlaid by env.

    Returns ``(settings, config_path, config_created)`` where *config_created*
    is True when the default file was generated on this run.  Env vars
    override the file per-field via ``COORDINATOR_ENV`` so nothing yet set by a
    launcher is lost.
    """
    base = base_dir or find_base_dir()
    path = config_path(base)

    data = DEFAULT_CONFIG.copy()
    created = False
    if path.is_file():
        data.update(load_config(path))
    else:
        write_default_config(path)
        created = True
        log.info("config_created path=%s — edit it and restart", path)

    for field, env in COORDINATOR_ENV.items():
        if env in os.environ:
            data[field] = os.environ[env]

    return CoordinatorSettings.from_config(data, base), path, created
