"""Entry point: ``python -m dain_node`` runs the agent until SIGINT/SIGTERM.

In standalone mode (``config.json`` exists next to the exe) the node watches
for config edits and automatically restarts with the new settings.  In dev
mode (no ``config.json``) behaviour is unchanged — settings come from
environment variables.

Alongside the agent, the node runs a small HTTP peer server that serves its
cached shards to sibling nodes (P2P distribution): the agent advertises the
peer URL to the coordinator, which routes its peers' downloads to it.
"""

import asyncio
import logging
from pathlib import Path
from urllib.parse import urlparse

from dain_common.logging_setup import configure_logging

from dain_node.agent import run_agent
from dain_node.config import ConfigWatcher, find_base_dir, load_config, write_config
from dain_node.discovery import discover_coordinator
from dain_node.jobs import JobHandler
from dain_node.llm import ModelStoreClient
from dain_node.peer_server import PeerShardServer, resolve_lan_ip
from dain_node.settings import DEFAULT_JOIN_TOKEN, NodeSettings


def _build_settings(config: dict, base_dir: Path) -> NodeSettings:
    """Create settings from config.json when present, else from env vars."""
    if config:
        return NodeSettings.from_config(config, base_dir)
    return NodeSettings.from_env()


def _auto_discover(config: dict, config_path: Path) -> dict:
    """Fill an empty ``coord_url`` in config.json via LAN discovery.

    Only runs in standalone mode (config.json present) without an explicit
    ``coord_url``.  When a coordinator answers, the discovered URL is written
    into config.json atomically so later starts skip the probe entirely and
    the user can see exactly what the node connected to.
    """
    url = str(config.get("coord_url") or "").strip()
    if url:
        return config
    token = str(config.get("join_token") or DEFAULT_JOIN_TOKEN)
    log = logging.getLogger("dain.node")
    log.info("coord_url_empty — probing the LAN for a coordinator...")
    found: str | None = None
    try:
        found = asyncio.run(discover_coordinator(token))
    except Exception:
        log.exception("coordinator_discovery_error")
    if not found:
        log.warning(
            "coordinator_discovery_failed — set coord_url in %s to the "
            "coordinator's address",
            config_path,
        )
        return config
    config["coord_url"] = found
    write_config(config_path, config)
    log.info("coord_url_discovered_url=%s — saved to %s", found, config_path)
    return config


def main() -> int:
    base_dir = find_base_dir()
    config_path = base_dir / "config.json"
    config = load_config(config_path)
    if config:
        config = _auto_discover(config, config_path)
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
        store.set_peer_token(settings.join_token)
        handler = JobHandler(settings, store)
        peer_server = None
        peer_url = None
        if settings.peer_enabled:
            coord_host = urlparse(settings.coord_url).hostname
            advertise = resolve_lan_ip(coord_host or None)
            peer_server = PeerShardServer(
                settings.model_cache_dir,
                settings.peer_bind_host,
                settings.peer_port,
                advertise_host=advertise,
                join_token=settings.join_token,
            )

        async def _run(
            _peer_server=peer_server, _settings=settings, _handler=handler, _log=log
        ) -> int:
            nonlocal peer_url
            if _peer_server is not None:
                await _peer_server.start()
                peer_url = _peer_server.url()
                _log.info("peer_server_url=%s", peer_url)
            try:
                return await run_agent(_settings, _handler, peer_url=peer_url)
            finally:
                if _peer_server is not None:
                    await _peer_server.stop()

        # Block until the agent exits (Ctrl+C, SIGTERM, or transport failure).
        rc = asyncio.run(_run())

        # Check whether the config file changed while we were running.
        if watcher.check():
            log.info("config_changed — restarting with new settings")
            config = load_config(config_path)
            continue

        return rc


if __name__ == "__main__":
    raise SystemExit(main())
