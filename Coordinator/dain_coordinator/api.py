"""HTTP/WS surface (S2/S4): node registration + heartbeat WS, admin views,
client inference API (SSE), and the model store.

Auth model (spec §11/§16, MVP): registration requires the shared join token and
returns a per-node token; WS connections authenticate with `node_id` + that
token as query params; deregistration requires the node token; the /v1 client
API requires an API key; /model/* requires node credentials. Admin endpoints
are unauthenticated in dev (hardening lands with the public deployment, S9).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import secrets
import time
from typing import Literal

from dain_common.schemas import (
    ActivationRelayHeader,
    Envelope,
    GenerationParams,
    JobAssign,
    JobState,
    JobStatus,
    MessageType,
    ModelManifest,
    NodeState,
    Register,
    RegisterAck,
    TaskOutcome,
    TokenBatch,
)
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from dain_coordinator.jobs import ActivationRelay, JobTracker
from dain_coordinator.nodes import MessageOutcome, NodeService
from dain_coordinator.partition import plan_placement
from dain_coordinator.store import NodeRow

log = logging.getLogger("dain.coordinator.api")


def require_api_key(request: Request) -> None:
    settings = request.app.state.settings
    provided = request.headers.get("x-api-key")
    if provided is None:
        auth = request.headers.get("authorization", "")
        provided = auth.removeprefix("Bearer ").strip() or None
    if provided is None or not secrets.compare_digest(provided, settings.api_key):
        raise HTTPException(status_code=401, detail="invalid API key")


def require_node(request: Request) -> None:
    """Node-facing model-store auth: node_id + per-node token headers."""
    service: NodeService = request.app.state.service
    node_id = request.headers.get("x-node-id", "")
    token = request.headers.get("x-node-token", "")
    if not service.authenticate(node_id, token):
        raise HTTPException(status_code=401, detail="invalid node credentials")


node_router = APIRouter(prefix="/node", tags=["node"])
admin_router = APIRouter(prefix="/admin", tags=["admin"])
v1_router = APIRouter(prefix="/v1", tags=["client"], dependencies=[Depends(require_api_key)])
model_router = APIRouter(prefix="/model", tags=["model"], dependencies=[Depends(require_node)])
ledger_router = APIRouter(
    prefix="/ledger", tags=["ledger"], dependencies=[Depends(require_api_key)]
)


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=8192)
    max_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0, le=2)
    seed: int | None = None
    stream: bool = True


class DeregisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    auth_token: str


class StateChangeView(BaseModel):
    from_state: str | None
    to_state: str
    reason: str | None
    ts: float


class NodeView(BaseModel):
    node_id: str
    state: str
    connected: bool = False
    score: float | None
    score_components: dict[str, float]
    agent_version: str
    gpu: str | None
    vram_free_gb: float | None
    uptime_ratio: float
    failure_rate: float
    last_seq: int | None
    last_heartbeat: float | None
    registered_at: float


class NodeDetail(NodeView):
    history: list[StateChangeView]


def _view(row: NodeRow, *, connected: bool = False) -> NodeView:
    return NodeView(
        node_id=row.node_id,
        state=row.state.value,
        connected=connected,
        score=row.score,
        score_components=row.score_components,
        agent_version=row.agent_version,
        gpu=row.manifest.gpu.name if row.manifest.gpu else None,
        vram_free_gb=row.manifest.gpu.vram_free_gb if row.manifest.gpu else None,
        uptime_ratio=row.uptime_ratio,
        failure_rate=row.failure_rate,
        last_seq=row.last_seq,
        last_heartbeat=row.last_heartbeat,
        registered_at=row.registered_at,
    )


def _service(request: Request) -> NodeService:
    return request.app.state.service  # type: ignore[no-any-return]


# -- node API ------------------------------------------------------------------


@node_router.post("/register", response_model=RegisterAck)
def register(payload: Register, request: Request) -> RegisterAck:
    store_url = str(request.base_url).rstrip("/") + "/model"
    return _service(request).register(payload, model_store_url=store_url)


@node_router.post("/deregister")
def deregister(payload: DeregisterRequest, request: Request) -> dict[str, object]:
    result = _service(request).deregister(payload.node_id, payload.auth_token)
    if result.status_code == 404:
        raise HTTPException(status_code=404, detail=result.detail)
    if result.status_code == 403:
        raise HTTPException(status_code=403, detail=result.detail)
    return {"ok": result.ok, "detail": result.detail}


@node_router.websocket("/ws")
async def node_ws(websocket: WebSocket) -> None:
    service: NodeService = websocket.app.state.service  # type: ignore[assignment]
    connections = websocket.app.state.connections
    jobs: JobTracker = websocket.app.state.jobs
    relay: ActivationRelay = websocket.app.state.relay
    node_id = websocket.query_params.get("node_id", "")
    token = websocket.query_params.get("token", "")
    if not service.authenticate(node_id, token):
        await websocket.close(code=4401, reason="invalid node credentials")
        return
    await websocket.accept()
    connections.register(node_id, websocket)
    log.info("ws_connected node=%s", node_id)
    pending_activation: ActivationRelayHeader | None = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"] is not None:
                if pending_activation is None:
                    log.warning("bytes_without_header node=%s", node_id)
                    continue
                header, pending_activation = pending_activation, None
                await relay.route(header, message["bytes"], jobs)
                continue
            raw = message.get("text")
            if raw is None:
                continue
            outcome = service.handle_message(node_id, raw)
            if outcome == MessageOutcome.CLOSE_MALFORMED:
                await websocket.close(code=4400, reason="malformed message")
                return
            if outcome == MessageOutcome.CLOSE_IDENTITY:
                await websocket.close(code=4400, reason="node_id mismatch")
                return
            # Relay plumbing: an ACTIVATION_RELAY header announces the next bytes.
            if raw.strip().startswith("{"):
                try:
                    envelope = Envelope.model_validate_json(raw)
                except Exception:
                    continue
                if envelope.type == MessageType.ACTIVATION_RELAY:
                    pending_activation = ActivationRelayHeader.model_validate(envelope.payload)
                elif envelope.type == MessageType.TOKEN_BATCH:
                    jobs.on_token_batch(TokenBatch.model_validate(envelope.payload))
                elif envelope.type == MessageType.JOB_STATUS:
                    status = JobStatus.model_validate(envelope.payload)
                    jobs.on_job_status(
                        status.job_id,
                        status.stage_idx,
                        status.state,
                        status.tokens_done,
                        status.detail,
                    )
    except WebSocketDisconnect:
        log.info("ws_disconnected node=%s", node_id)
    finally:
        connections.unregister(node_id, websocket)


# -- admin API -----------------------------------------------------------------


@admin_router.get("/nodes", response_model=list[NodeView])
def list_nodes(request: Request, state: str | None = None) -> list[NodeView]:
    service = _service(request)
    connections = request.app.state.connections
    if state is not None:
        try:
            state_filter = NodeState(state)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown state {state!r}") from exc
        rows = service.registry.list_nodes(state_filter)
        return [_view(row, connected=connections.is_connected(row.node_id)) for row in rows]
    rows = service.registry.list_nodes()
    return [_view(row, connected=connections.is_connected(row.node_id)) for row in rows]


@admin_router.get("/nodes/{node_id}", response_model=NodeDetail)
def node_detail(node_id: str, request: Request) -> NodeDetail:
    service = _service(request)
    row = service.registry.get_node(node_id)
    if row is None:
        raise HTTPException(status_code=404, detail="unknown node")
    history = [
        StateChangeView(
            from_state=c.from_state.value if c.from_state is not None else None,
            to_state=c.to_state.value,
            reason=c.reason,
            ts=c.ts,
        )
        for c in service.registry.history(node_id)
    ]
    connected = request.app.state.connections.is_connected(row.node_id)
    return NodeDetail(**_view(row, connected=connected).model_dump(), history=history)


# -- client API (S4): SSE streaming completions ----------------------------------


def _sse(frame: dict) -> str:
    return f"data: {json.dumps(frame, separators=(',', ':'))}\n\n"


def _release(request: Request, node_ids: list[str]) -> None:
    """Return stage nodes from BUSY to ONLINE after a job reaches a terminal state."""
    service = _service(request)
    for node_id in node_ids:
        service.release_node(node_id)


@v1_router.get("/models")
def list_models(request: Request) -> dict:
    settings = request.app.state.settings
    models = shard_store_list(settings.model_store_dir)
    return {
        "models": [
            {"model_id": m.model_id, "name": m.name, "layers": m.layers, "hidden": m.hidden}
            for m in models
        ]
    }


@v1_router.post("/completions")
async def completions(payload: CompletionRequest, request: Request):
    settings = request.app.state.settings
    jobs: JobTracker = request.app.state.jobs
    connections = request.app.state.connections
    manifest = shard_store_load(settings.model_store_dir, payload.model_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"unknown model {payload.model_id!r}")

    active = sum(
        1 for j in jobs.jobs.values() if j.state not in (JobState.COMPLETED, JobState.FAILED)
    )
    api_key = request.headers.get("x-api-key") or "anonymous"
    per_key = sum(
        1
        for j in jobs.jobs.values()
        if j.api_key == api_key and j.state not in (JobState.COMPLETED, JobState.FAILED)
    )
    service = _service(request)
    rows = [
        r
        for r in service.registry.list_nodes(NodeState.ONLINE)
        if request.app.state.connections.is_connected(r.node_id)
    ]
    plan = plan_placement(
        manifest,
        rows,
        layers_per_node_target=settings.layers_per_node_target,
        backup_count=settings.backup_count,
    )
    if plan is None:
        raise HTTPException(
            status_code=429,
            detail="no connected node pool can host this model; retry later",
            headers={"Retry-After": "5"},
        )
    if active >= settings.queue_limit or per_key >= settings.max_concurrent_per_key:
        raise HTTPException(
            status_code=429,
            detail="server busy: queue limit reached; retry later",
            headers={"Retry-After": "5"},
        )
    stages = plan.stages
    # Exclusive execution per stage node (MVP has no intra-node batching): a node
    # marked BUSY is excluded from new placements until the job finishes.
    busy_nodes = [s.node_id for s in stages]
    for node_id in busy_nodes:
        service.mark_busy(node_id)

    record = jobs.create(
        payload.model_id,
        payload.prompt,
        {
            "max_tokens": payload.max_tokens,
            "temperature": payload.temperature,
            "seed": payload.seed,
        },
        manifest,
        api_key=api_key,
        backups=plan.backups,
    )
    params = GenerationParams(
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=1.0,
        seed=payload.seed,
    )
    dispatched = True
    for stage in stages:
        assign = JobAssign(
            job_id=record.job_id,
            model_id=payload.model_id,
            my_stage_idx=stage.stage_idx,
            stages=stages,
            prompt=payload.prompt if stage.stage_idx == 0 else None,
            params=params,
        )
        sent = await connections.send_envelope(
            stage.node_id, Envelope.wrap(MessageType.JOB_ASSIGN, assign, ts=time.time())
        )
        if not sent:
            dispatched = False
            break
    if not dispatched:
        for node_id in busy_nodes:
            service.release_node(node_id)
        jobs.fail_job(record.job_id, "stage node connection lost before dispatch")
        raise HTTPException(status_code=503, detail="stage node connection lost")
    jobs.mark_dispatched(record.job_id, stages[0].node_id, stages)
    queue = jobs.attach(record.job_id)

    async def event_stream():
        try:
            yield _sse({"job_id": record.job_id, "status": "dispatched"})
            deadline = asyncio.get_event_loop().time() + settings.job_timeout_s
            while True:
                try:
                    frame = await asyncio.wait_for(
                        queue.get(), timeout=max(0.1, deadline - asyncio.get_event_loop().time())
                    )
                except TimeoutError:
                    jobs.fail_job(record.job_id, "job timeout")
                    yield _sse({"type": "error", "job_id": record.job_id, "detail": "job timeout"})
                    break
                yield _sse(frame)
                if frame["type"] in ("final", "error"):
                    break
            yield "data: [DONE]\n\n"
        finally:
            _release(request, busy_nodes)

    if payload.stream:
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    # Non-streaming: drain to the final frame.
    text_parts: list[str] = []
    final: dict = {}
    deadline = asyncio.get_event_loop().time() + settings.job_timeout_s
    try:
        while True:
            try:
                frame = await asyncio.wait_for(
                    queue.get(), timeout=max(0.1, deadline - asyncio.get_event_loop().time())
                )
            except TimeoutError:
                jobs.fail_job(record.job_id, "job timeout")
                raise HTTPException(status_code=504, detail="job timeout") from None
            if frame["type"] == "token":
                text_parts.append(frame["token"])
            elif frame["type"] == "final":
                final = frame
                break
            elif frame["type"] == "error":
                raise HTTPException(status_code=502, detail=frame["detail"])
    finally:
        _release(request, busy_nodes)
    return {
        "job_id": record.job_id,
        "text": "".join(text_parts),
        "finish_reason": final.get("finish_reason"),
        "usage": final.get("usage"),
    }


@v1_router.get("/jobs/{job_id}")
def job_view(job_id: str, request: Request) -> dict:
    jobs: JobTracker = request.app.state.jobs
    view = jobs.view(job_id)
    if view is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return view


# -- ledger API (S8, spec §15) ---------------------------------------------------


def _ledger(request: Request):
    return request.app.state.ledger  # type: ignore[no-any-return]


@ledger_router.get("/node/{node_id}")
def ledger_node(node_id: str, request: Request) -> list[dict]:
    return _ledger(request).events_for_node(node_id)


@ledger_router.get("/summary")
def ledger_summary(request: Request, since: float | None = None) -> dict:
    events = _ledger(request).events_all(since=since)
    totals: dict[str, dict] = {}
    for event in events:
        bucket = totals.setdefault(
            event["node_id"],
            {
                "events": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "flops_est": 0.0,
                "compute_seconds": 0.0,
                "credits": 0.0,
                "success": 0,
                "retried_away": 0,
                "failed": 0,
                "flagged": 0,
            },
        )
        bucket["events"] += 1
        bucket["tokens_in"] += event["tokens_in"]
        bucket["tokens_out"] += event["tokens_out"]
        bucket["flops_est"] += event["flops_est"]
        bucket["compute_seconds"] += event["compute_seconds"]
        bucket["credits"] += event["credit"]
        bucket["success"] += event["outcome"] == TaskOutcome.SUCCESS.value
        bucket["retried_away"] += event["outcome"] == TaskOutcome.RETRIED_AWAY.value
        bucket["failed"] += event["outcome"] == TaskOutcome.FAILED.value
        if not event["verified"]:
            bucket["flagged"] += 1
    return {
        "generated_at": time.time(),
        "nodes": [
            {"node_id": nid, **bucket}
            for nid, bucket in sorted(totals.items())
        ],
        "totals": {
            "events": sum(b["events"] for b in totals.values()),
            "credits": round(sum(b["credits"] for b in totals.values()), 6),
            "tokens_out": sum(b["tokens_out"] for b in totals.values()),
            "flagged": sum(b["flagged"] for b in totals.values()),
        },
    }


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["csv", "json"] = "json"
    since: float | None = None


@ledger_router.post("/export")
def ledger_export(payload: ExportRequest, request: Request) -> Response:
    rows = _ledger(request).events_all(since=payload.since)
    if payload.format == "json":
        return Response(
            content=json.dumps(rows, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="ledger.json"'},
        )
    import csv
    import io

    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="ledger.csv"'},
    )


# -- model store (node-facing; spec §8/§11) ----------------------------------------


def _shard_path(settings, model_id: str, shard_id: str) -> str:
    # Path-traversal guard: ids are restricted to safe characters.
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not set(model_id) <= safe or not set(shard_id) <= safe:
        raise HTTPException(status_code=400, detail="invalid model or shard id")
    return os.path.join(settings.model_store_dir, model_id, f"{shard_id}.safetensors")


@model_router.get("/manifest/{model_id}")
def get_manifest(model_id: str, request: Request) -> ModelManifest:
    settings = request.app.state.settings
    manifest = shard_store_load(settings.model_store_dir, model_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="unknown model")
    return manifest


@model_router.get("/shard/{model_id}/{shard_id}")
def get_shard(model_id: str, shard_id: str, request: Request) -> Response:
    settings = request.app.state.settings
    path = pathlib.Path(_shard_path(settings, model_id, shard_id))
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown shard")
    return Response(
        content=path.read_bytes(),
        media_type="application/octet-stream",
    )


# -- shard store helpers -----------------------------------------------------------

from dain_common.model_store import list_models as shard_store_list  # noqa: E402
from dain_common.model_store import load_manifest as shard_store_load  # noqa: E402
