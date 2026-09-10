"""Entry point: ``python -m dain_node`` runs the agent until SIGINT/SIGTERM.

In standalone mode (``config.json`` exists next to the exe) the node watches
for config edits and automatically restarts with the new settings.  In dev
mode (no ``config.json``) behaviour is unchanged — settings come from
environment variables.
"""

import asyncio
import logging
from pathlib import Path

from dain_common.logging_setup import configure_logging

from dain_node.agent import run_agent
from dain_node.config import ConfigWatcher, find_base_dir, load_config
from dain_node.jobs import JobHandler
from dain_node.llm import ModelStoreClient
from dain_node.settings import NodeSettings


def _build_settings(config: dict, base_dir: Path) -> NodeSettings:
    """Create settings from config.json when present, else from env vars."""
    if config:
        return NodeSettings.from_config(config, base_dir)
    return NodeSettings.from_env()


def main() -> int:
    base_dir = find_base_dir()
    config_path = base_dir / "config.json"
    config = load_config(config_path)
    watcher = ConfigWatcher(config_path)

    # ── restart loop ──────────────────────────────────────────────────────
    # When the config file is edited on disk the agent shuts down and loops
    # back here to pick up the new settings.  A normal Ctrl+C / SIGTERM exits
    # immediately.
    while True:
        settings = _build_settings(config, base_dir)
        configure_logging(json_mode=settings.log_json, level=logging.INFO)
        log = logging.getLogger("dain.node")
        log.info(
            "node_start coord=%s model=%s config=%s",
            settings.coord_url,
            settings.model_id or "(none)",
            str(config_path) if config else "(env)",
        )

        store = ModelStoreClient(cache_dir=settings.model_cache_dir)
        handler = JobHandler(settings, store)

        # Block until the agent exits (Ctrl+C, SIGTERM, or transport failure).
        rc = asyncio.run(run_agent(settings, handler))

        # Check whether the config file changed while we were running.
        if watcher.check():
            log.info("config_changed — restarting with new settings")
            config = load_config(config_path)
            continue

        return rc


if __name__ == "__main__":
    raise SystemExit(main())
