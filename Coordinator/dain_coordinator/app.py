"""Coordinator ASGI application: node API + admin API + client inference API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pathlib
from collections.abc import AsyncIterator

from dain_common.logging_setup import configure_logging
from dain_common.model_store import list_models as shard_store_list
from dain_common.schemas import ModelManifest, NodeState
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from dain_coordinator.api import (
    admin_router,
    ledger_router,
    model_router,
    node_router,
    v1_router,
)
from dain_coordinator.connections import NodeConnections
from dain_coordinator.discovery import DiscoveryResponder
from dain_coordinator.faults import FaultManager
from dain_coordinator.jobs import ActivationRelay, JobTracker
from dain_coordinator.ledger import Ledger
from dain_coordinator.logs import attach_ring
from dain_coordinator.monitor import HeartbeatMonitor
from dain_coordinator.nodes import NodeService
from dain_coordinator.partition import PlacementRecorder, recompute_pool
from dain_coordinator.ratelimit import RateLimiter
from dain_coordinator.settings import CoordinatorSettings, harden_production_secrets
from dain_coordinator.store import SQLiteRegistry

log = logging.getLogger("dain.coordinator.app")

_UI_HTML: str | None = None
_CHAT_UI_HTML: str | None = None


def _admin_ui() -> str:
    """The self-contained admin page (no build step), read once at import."""
    global _UI_HTML
    if _UI_HTML is None:
        _UI_HTML = pathlib.Path(__file__).with_name("admin_ui.html").read_text(encoding="utf-8")
    return _UI_HTML


def _chat_ui() -> str:
    """The self-contained chat page (no build step), read once at import."""
    global _CHAT_UI_HTML
    if _CHAT_UI_HTML is None:
        _CHAT_UI_HTML = pathlib.Path(__file__).with_name("chat_ui.html").read_text(encoding="utf-8")
    return _CHAT_UI_HTML


async def _watchdog_loop(faults, settings_getter) -> None:
    """Background stage-watchdog task (spec §13 detection).

    Reads live settings through `settings_getter` so an admin PUT /settings
    applies to the loop instead of the startup snapshot.
    """
    while True:
        await asyncio.sleep(settings_getter().watchdog_tick_s)
        try:
            await faults.tick()
        except Exception:
            # The watchdog must survive any transient error.
            log.exception("watchdog_tick_failed")


def create_app(
    settings: CoordinatorSettings | None = None, settings_path: str | None = None
) -> FastAPI:
    if settings is None:
        settings = harden_production_secrets(CoordinatorSettings.from_env())
    configure_logging(json_mode=settings.log_json)
    attach_ring()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = SQLiteRegistry(settings.db_path)
        registry.open()
        connections = NodeConnections()
        service = NodeService(registry, settings, is_connected=connections.is_connected)
        jobs = JobTracker(max_jobs=settings.job_history_max, job_ttl_s=settings.job_ttl_s)
        relay = ActivationRelay(connections)
        placements = PlacementRecorder()
        faults = FaultManager(
            settings_getter=lambda: app.state.settings,
            jobs=jobs,
            connections=connections,
            registry=registry,
            service=service,
        )
        ledger = Ledger(settings, registry, jobs)

        def recompute_pool_events(trigger: str) -> None:
            """Spec §12: recompute placement on join/leave/DEGRADED transitions.

            Recomputes the plan for every model in the store (dev serves one);
            a store read failure must never break the triggering path.

            Reads live settings from `app.state.settings` (admin PUT /settings
            replaces it) rather than the startup snapshot captured at lifespan.
            """
            live = app.state.settings
            try:
                manifests: list[ModelManifest] = list(shard_store_list(live.model_store_dir))
            except OSError:
                return
            rows = [
                r
                for r in registry.list_nodes(NodeState.ONLINE)
                if connections.is_connected(r.node_id)
            ]
            for manifest in manifests:
                recompute_pool(
                    manifest,
                    rows,
                    layers_per_node_target=live.layers_per_node_target,
                    min_k=live.min_stages,
                    max_k=live.max_stages,
                    backup_count=live.backup_count,
                    min_nodes=live.min_nodes,
                    trigger=trigger,
                    recorder=placements,
                )

        connections.on_disconnect = lambda node_id: (
            faults.handle_node_lost(node_id),
            recompute_pool_events("leave"),
        )
        service.on_pool_change = lambda node_id, to_state: recompute_pool_events(
            "degraded" if to_state == NodeState.DEGRADED else "recovered"
        )
        service.on_node_lost = faults.handle_node_lost
        jobs.on_terminal = ledger.emit_job_terminal
        ledger.on_verification_flag = lambda node_id, note: service.apply_verification_penalty(
            node_id, note
        )
        app.state.registry = registry
        app.state.service = service
        app.state.connections = connections
        app.state.jobs = jobs
        app.state.relay = relay
        app.state.placements = placements
        app.state.faults = faults
        app.state.ledger = ledger
        app.state.rate_limiter = RateLimiter(settings.rate_limit_per_min)
        app.state.recompute_pool = recompute_pool_events
        monitor = HeartbeatMonitor(service)
        task = asyncio.create_task(monitor.run(), name="heartbeat-monitor")
        watchdog = asyncio.create_task(
            _watchdog_loop(faults, lambda: app.state.settings), name="stage-watchdog"
        )
        discovery: DiscoveryResponder | None
        if settings.discovery_enabled:
            discovery = DiscoveryResponder(
                settings.discovery_port, settings.join_token, settings.port
            )
        else:
            discovery = None
        app.state.discovery = discovery
        discovery_task: asyncio.Task[None] | None = None
        if discovery is not None:
            try:
                discovery.start()
            except OSError:
                log.warning(
                    "discovery_disabled bind_failed port=%d", settings.discovery_port
                )
                discovery = None
        if discovery is not None:
            discovery_task = asyncio.create_task(discovery.run(), name="lan-discovery")
        yield
        task.cancel()
        watchdog.cancel()
        if discovery_task is not None:
            discovery_task.cancel()
        if discovery is not None:
            discovery.close()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog
        if discovery_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await discovery_task
        registry.close()

    app = FastAPI(title="DAIN Coordinator", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.settings_path = settings_path
    app.state.discovery = None
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(node_router)
    app.include_router(admin_router)
    app.include_router(v1_router)
    app.include_router(model_router)
    app.include_router(ledger_router)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/admin", include_in_schema=False)
    @app.get("/admin/", include_in_schema=False)
    def admin_ui() -> HTMLResponse:
        """Served shell only — every data call the page makes is authed separately."""
        return HTMLResponse(_admin_ui(), headers={"Cache-Control": "no-store"})

    @app.get("/msg", include_in_schema=False)
    @app.get("/msg/", include_in_schema=False)
    def chat_ui() -> HTMLResponse:
        """Self-contained chat interface at /msg (no build step, no static hosting)."""
        return HTMLResponse(_chat_ui())

    return app
