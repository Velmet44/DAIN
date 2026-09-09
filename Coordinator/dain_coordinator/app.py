"""Coordinator ASGI application: node API + admin API + client inference API."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from dain_common.logging_setup import configure_logging
from dain_common.model_store import list_models as shard_store_list
from dain_common.schemas import ModelManifest, NodeState
from fastapi import FastAPI

from dain_coordinator.api import admin_router, model_router, node_router, v1_router
from dain_coordinator.connections import NodeConnections
from dain_coordinator.jobs import ActivationRelay, JobTracker
from dain_coordinator.monitor import HeartbeatMonitor
from dain_coordinator.nodes import NodeService
from dain_coordinator.partition import PlacementRecorder, recompute_pool
from dain_coordinator.settings import CoordinatorSettings
from dain_coordinator.store import SQLiteRegistry


def create_app(settings: CoordinatorSettings | None = None) -> FastAPI:
    settings = settings or CoordinatorSettings.from_env()
    configure_logging(json_mode=settings.log_json)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = SQLiteRegistry(settings.db_path)
        registry.open()
        service = NodeService(registry, settings)
        connections = NodeConnections()
        jobs = JobTracker()
        relay = ActivationRelay(connections)
        placements = PlacementRecorder()

        def recompute_pool_events(trigger: str) -> None:
            """Spec §12: recompute placement on join/leave/DEGRADED transitions.

            Recomputes the plan for every model in the store (dev serves one);
            a store read failure must never break the triggering path.
            """
            try:
                manifests: list[ModelManifest] = list(shard_store_list(settings.model_store_dir))
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
                    layers_per_node_target=settings.layers_per_node_target,
                    min_k=settings.min_stages,
                    max_k=settings.max_stages,
                    backup_count=settings.backup_count,
                    trigger=trigger,
                    recorder=placements,
                )

        connections.on_disconnect = lambda node_id: (
            jobs.fail_jobs_of_node(node_id, "node connection lost"),
            recompute_pool_events("leave"),
        )
        service.on_pool_change = lambda node_id, to_state: recompute_pool_events(
            "degraded" if to_state == NodeState.DEGRADED else "recovered"
        )
        app.state.registry = registry
        app.state.service = service
        app.state.connections = connections
        app.state.jobs = jobs
        app.state.relay = relay
        app.state.placements = placements
        app.state.recompute_pool = recompute_pool_events
        monitor = HeartbeatMonitor(service, settings)
        task = asyncio.create_task(monitor.run(), name="heartbeat-monitor")
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        registry.close()

    app = FastAPI(title="DAIN Coordinator", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(node_router)
    app.include_router(admin_router)
    app.include_router(v1_router)
    app.include_router(model_router)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
