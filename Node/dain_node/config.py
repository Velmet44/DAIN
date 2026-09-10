"""Config file loader and watcher for the standalone node distribution.

When distributed as a self-contained folder (DainNode.exe + config.json), the
node reads its settings from config.json next to the executable.  The watcher
detects edits and signals a restart so the node picks up new settings without
manual intervention.

In dev mode (``python -m dain_node``) there is no config.json and settings
come from environment variables as before — this module is a no-op.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("dain.node.config")

# Default config written into the distribution folder by build-node.ps1.
DEFAULT_CONFIG: dict = {
    "coord_url": "ws://localhost:8000",
    "join_token": "dain-dev-join-token",
    "node_id": "",
    "heartbeat_s": 5,
    "model": "",
    "cache_dir": "shard_cache",
    "state_path": "node_state.json",
    "net_bw_mbps": 100,
    "net_lat_ms": 50,
    "log_json": False,
}


def find_base_dir() -> Path:
    """Return the directory that owns the runtime files.

    * Frozen (PyInstaller) → directory containing the ``.exe``
    * Dev (``python -m``)   → the ``Node/`` project root
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # __file__ = …/Node/dain_node/config.py  → parent.parent = Node/
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict:
    """Read *config_path* and return its contents as a plain dict.

    Returns an empty dict when the file is missing or unparseable (the caller
    falls back to environment variables or hard-coded defaults).
    """
    if not config_path.is_file():
        return {}
    try:
        text = config_path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            log.warning("config_not_dict path=%s — ignoring", config_path)
            return {}
        log.info("config_loaded path=%s keys=%s", config_path, sorted(data))
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("config_load_failed path=%s error=%s — using defaults", config_path, exc)
        return {}


class ConfigWatcher:
    """Poll-based watcher for ``config.json`` modifications.

    Call :meth:`check` periodically (e.g. after the agent loop returns).  When
    the file's ``mtime`` changes the watcher signals a restart.
    """

    def __init__(self, config_path: Path) -> None:
        self._path = config_path
        self._last_mtime: float = self._current_mtime()

    def _current_mtime(self) -> float:
        try:
            return os.path.getmtime(self._path)
        except OSError:
            return 0.0

    def check(self) -> bool:
        """Return *True* if the config file was modified since the last check."""
        if not self._path.is_file():
            return False
        mtime = self._current_mtime()
        if mtime != self._last_mtime:
            self._last_mtime = mtime
            return True
        return False
