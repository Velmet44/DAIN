"""HTTP/WS surface (S2/S4/S9): node registration + heartbeat WS, admin views,
client inference API (SSE), and the model store.

Auth model (spec §11/§16): registration requires the shared join token and
returns a per-node token; WS connections authenticate with `node_id` + that
token as query params; deregistration requires the node token; the /v1 client
API requires an API key; /model/* requires node credentials; /admin/* requires
the admin API key (S9 hardening). Exception (session 15): same-machine
requests to /v1 and /admin are trusted without keys — guarded by loopback
peer + localhost Host (DNS-rebinding) + Origin (drive-by CSRF) checks, see
`_local_trusted`.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import os
import pathlib
import random
import secrets
import shutil
import time
from dataclasses import replace as dataclass_replace
from typing import Literal
from urllib.parse import urlparse

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
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from dain_coordinator.config import find_base_dir, persist_config
from dain_coordinator.jobs import ActivationRelay, JobTracker
from dain_coordinator.logs import snapshot as log_snapshot
from dain_coordinator.nodes import MessageOutcome, NodeService
from dain_coordinator.partition import event_view, plan_placement
from dain_coordinator.ratelimit import RateLimiter
from dain_coordinator.settings import (
    COORDINATOR_ENV,
    DEFAULT_ADMIN_API_KEY,
    DEFAULT_API_KEY,
    DEFAULT_JOIN_TOKEN,
)
from dain_coordinator.store import NodeRow

log = logging.getLogger("dain.coordinator.api")


def _loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # e.g. TestClient's "testclient"


def _local_trusted(request: Request) -> bool:
    """True for a same-machine request that a browser cannot forge.

    Keyless localhost admin (session 15): the admin page on 127.0.0.1 works
    without keys. Three checks make that safe:
    - the socket peer is loopback (direct local connection only);
    - the Host header is localhost (defeats DNS rebinding, where a remote
      site resolves a domain to 127.0.0.1 — the browser then sends *its*
      hostname as Host);
    - a cross-site drive-by POST from a visited web page always carries an
      Origin header, so a non-localhost Origin is rejected. Absent Origin
      (curl, the local SPA's same-origin fetches in some browsers) passes.
    """
    client = request.client
    if client is None or not _loopback(client.host):
        return False
    host = (request.headers.get("host") or "").lower().rsplit(":", 1)[0]
    if host not in ("127.0.0.1", "localhost", "[::1]", "::1"):
        return False
    origin = request.headers.get("origin")
    if origin:
        o = urlparse(origin)
        if (o.hostname or "") not in ("127.0.0.1", "localhost", "::1"):
            return False
    return True


def _provided_key(request: Request, header: str) -> str | None:
    provided = request.headers.get(header)
    if provided is None:
        auth = request.headers.get("authorization", "")
        provided = auth.removeprefix("Bearer ").strip() or None
    return provided or None  # an empty header value counts as "not provided"


def require_api_key(request: Request) -> None:
    settings = request.app.state.settings
    provided = _provided_key(request, "x-api-key")
    if provided is None and _local_trusted(request):
        return
    if provided is None or not secrets.compare_digest(
        provided.encode("utf-8"), settings.api_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid API key")


def require_node(request: Request) -> None:
    """Node-facing model-store auth: node_id + per-node token headers."""
    service: NodeService = request.app.state.service
    node_id = request.headers.get("x-node-id", "")
    token = request.headers.get("x-node-token", "")
    if not service.authenticate(node_id, token):
        raise HTTPException(status_code=401, detail="invalid node credentials")


def require_admin(request: Request) -> None:
    """Admin API auth: admin key, or keyless from a trusted localhost request."""
    settings = request.app.state.settings
    provided = _provided_key(request, "x-admin-key")
    if provided is None and _local_trusted(request):
        return
    if provided is None or not secrets.compare_digest(
        provided.encode("utf-8"), settings.admin_api_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid admin key")


node_router = APIRouter(prefix="/node", tags=["node"])
admin_router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])
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


# -- admin settings: keys & tokens --------------------------------------------------

KEY_FIELDS = ("join_token", "api_key", "admin_api_key")


class KeyState(BaseModel):
    join_token: str
    api_key: str
    admin_api_key: str
    defaults: dict[str, str]
    env_overrides: dict[str, str]
    writable: bool


class KeysUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    join_token: str | None = Field(default=None, min_length=8, max_length=512)
    api_key: str | None = Field(default=None, min_length=8, max_length=512)
    admin_api_key: str | None = Field(default=None, min_length=8, max_length=512)
    reset_join_token: bool = False
    reset_api_key: bool = False
    reset_admin_api_key: bool = False


def _key_defaults() -> dict[str, str]:
    return {
        "join_token": DEFAULT_JOIN_TOKEN,
        "api_key": DEFAULT_API_KEY,
        "admin_api_key": DEFAULT_ADMIN_API_KEY,
    }


def _key_state(request: Request) -> KeyState:
    settings = request.app.state.settings
    writable = bool(getattr(request.app.state, "settings_path", None))
    env_overrides = {
        field: os.environ[env_var]
        for field, env_var in COORDINATOR_ENV.items()
        if field in KEY_FIELDS and env_var in os.environ
    }
    return KeyState(
        join_token=settings.join_token,
        api_key=settings.api_key,
        admin_api_key=settings.admin_api_key,
        defaults=_key_defaults(),
        env_overrides=env_overrides,
        writable=writable,
    )


@admin_router.get("/settings/keys", response_model=KeyState)
def get_keys(request: Request) -> KeyState:
    return _key_state(request)


@admin_router.put("/settings/keys", response_model=KeyState)
def update_keys(payload: KeysUpdate, request: Request) -> KeyState:
    """Rotate any of the three admission keys at runtime.

    A provided field sets a new value; the matching ``reset_*`` flag restores
    the shipped default.  Changes apply immediately (existing node sessions
    keep their per-node tokens) and are persisted to the coordinator's
    ``config.json`` when one is loaded (env-var-provided keys still win at the
    next restart, flagged in the response).
    """
    settings = request.app.state.settings
    updates: dict[str, str] = {}
    for field in KEY_FIELDS:
        if getattr(payload, f"reset_{field}"):
            updates[field] = _key_defaults()[field]
        else:
            value = getattr(payload, field)
            if value is not None and value != getattr(settings, field):
                updates[field] = value
    if not updates:
        return _key_state(request)

    new_settings = dataclass_replace(settings, **updates)
    request.app.state.settings = new_settings
    request.app.state.service.settings = new_settings
    discovery = getattr(request.app.state, "discovery", None)
    if discovery is not None and "join_token" in updates:
        discovery.set_join_token(updates["join_token"])

    persisted = False
    path = getattr(request.app.state, "settings_path", None)
    if path:
        try:
            baseline = {field: getattr(settings, field) for field in KEY_FIELDS}
            persist_config(pathlib.Path(path), updates, baseline=baseline)
            persisted = True
        except OSError as exc:
            log.warning("keys_persist_failed path=%s error=%s", path, exc)

    for field in updates:
        log.info("admin_key_updated field=%s persisted=%s", field, persisted)
    return _key_state(request)


# -- admin controls: nodes, jobs, models, logs ------------------------------------


# Settings editable live from the admin page (session 16). Each entry: the
# field name → its input type; changing any of them swaps the live settings
# object and persists to config.json. Fields NOT listed here (host/port,
# heartbeat/watchdog timing, discovery, CORS, db_path) are built into the app,
# monitor loops, or middleware at startup — the UI shows them read-only as
# "restart required".
LIVE_SETTINGS: dict[str, str] = {
    "model_store_dir": "path",
    "node_project_dir": "path",
    "max_completion_tokens": "int",
    "queue_limit": "int",
    "max_concurrent_per_key": "int",
    "rate_limit_per_min": "int",
    "layers_per_node_target": "int",
    "backup_count": "int",
    "min_score": "float",
    "job_timeout_s": "float",
}
RESTART_FIELDS = (
    "host",
    "port",
    "db_path",
    "heartbeat_interval_s",
    "offline_after_missed",
    "monitor_tick_s",
    "min_stages",
    "max_stages",
    "watchdog_tick_s",
    "stage_deadline_min_s",
    "stage_deadline_max_s",
    "max_stage_attempts",
    "max_job_restarts",
    "min_nodes",
    "cors_origins",
    "discovery_enabled",
    "discovery_port",
)


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_store_dir: str | None = Field(default=None, min_length=1)
    node_project_dir: str | None = None  # empty string = default sibling Node/
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)
    queue_limit: int | None = Field(default=None, ge=1, le=4096)
    max_concurrent_per_key: int | None = Field(default=None, ge=1, le=256)
    rate_limit_per_min: int | None = Field(default=None, ge=0, le=100000)
    layers_per_node_target: int | None = Field(default=None, ge=1, le=128)
    backup_count: int | None = Field(default=None, ge=0, le=16)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    job_timeout_s: float | None = Field(default=None, ge=1.0, le=3600.0)


def _settings_state(request: Request) -> dict:
    settings = request.app.state.settings
    env_overridden = {
        field for field in LIVE_SETTINGS if COORDINATOR_ENV.get(field) in os.environ
    }
    fields = {
        field: {
            "value": getattr(settings, field),
            "type": kind,
            "env_overridden": field in env_overridden,
        }
        for field, kind in LIVE_SETTINGS.items()
    }
    return {
        "settings": fields,
        "restart": {field: getattr(settings, field) for field in RESTART_FIELDS},
        "writable": bool(getattr(request.app.state, "settings_path", None)),
    }


@admin_router.get("/settings")
def get_settings(request: Request) -> dict:
    """Live-editable settings + restart-required fields, for the admin UI."""
    return _settings_state(request)


@admin_router.put("/settings")
def update_settings(payload: SettingsUpdate, request: Request) -> dict:
    """Apply runtime settings from the admin page.

    Fields present in the body are validated, applied to the live settings
    object immediately, and persisted (as-typed, so relative paths stay
    portable) to ``config.json`` when one exists. ``model_store_dir`` triggers
    a placement rescan; ``rate_limit_per_min`` rebuilds the limiter. Path
    fields accept relative values resolved against the coordinator directory.
    """
    settings = request.app.state.settings
    base_dir = find_base_dir()
    changes: dict[str, object] = {}
    for field in LIVE_SETTINGS:
        value = getattr(payload, field)
        if value is None or value == getattr(settings, field):
            continue
        if field == "model_store_dir":
            store = pathlib.Path(str(value))
            resolved = store if store.is_absolute() else base_dir / store
            try:
                resolved.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise HTTPException(
                    status_code=400, detail=f"model_store_dir unusable: {exc}"
                ) from None
        elif field == "node_project_dir" and str(value):
            node = pathlib.Path(str(value))
            resolved = node if node.is_absolute() else base_dir / node
            if not (resolved / "pyproject.toml").is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"node_project_dir has no pyproject.toml: {resolved}",
                )
        changes[field] = value
    if not changes:
        return _settings_state(request)

    new_settings = dataclass_replace(settings, **changes)
    request.app.state.settings = new_settings
    request.app.state.service.settings = new_settings
    if "rate_limit_per_min" in changes:
        request.app.state.rate_limiter = RateLimiter(new_settings.rate_limit_per_min)
    if "model_store_dir" in changes:
        request.app.state.recompute_pool("admin_settings")

    persisted = False
    path = getattr(request.app.state, "settings_path", None)
    if path:
        try:
            persist_config(pathlib.Path(path), dict(changes))
            persisted = True
        except OSError as exc:
            log.warning("settings_persist_failed path=%s error=%s", path, exc)
    log.info("admin_settings_updated fields=%s persisted=%s", ",".join(changes), persisted)
    state = _settings_state(request)
    state["applied"] = sorted(changes)
    state["persisted"] = persisted
    return state


@admin_router.post("/nodes/{node_id}/offline")
def evict_node(node_id: str, request: Request) -> dict:
    """Force a node OFFLINE and fail its active jobs (admin eviction)."""
    service = _service(request)
    jobs: JobTracker = request.app.state.jobs
    failed = jobs.fail_jobs_of_node(node_id, "admin eviction")
    moved = service.transition(node_id, NodeState.OFFLINE, "admin_evict")
    if not moved:
        raise HTTPException(status_code=409, detail="no transition to OFFLINE")
    log.warning("admin_evict node=%s jobs_failed=%d", node_id, failed)
    return {"ok": True, "node_id": node_id, "state": NodeState.OFFLINE.value, "jobs_failed": failed}


@admin_router.post("/nodes/{node_id}/online")
def recover_node(node_id: str, request: Request) -> dict:
    """Force a node back ONLINE (admin recovery; real liveness still enforced)."""
    service = _service(request)
    moved = service.transition(node_id, NodeState.ONLINE, "admin_recover")
    if not moved:
        raise HTTPException(status_code=409, detail="no transition to ONLINE")
    log.warning("admin_recover node=%s", node_id)
    return {"ok": True, "node_id": node_id, "state": NodeState.ONLINE.value}


@admin_router.get("/jobs")
def admin_jobs(request: Request) -> dict:
    """Recent + active job list, newest first."""
    tracker: JobTracker = request.app.state.jobs
    recent = list(reversed(tracker._order))[:50]  # noqa: SLF001 — tracker is app-internal
    jobs = [tracker.view(jid) for jid in recent if tracker.get(jid) is not None]
    return {"jobs": [j for j in jobs if j is not None], "active": len(tracker.active_jobs())}


@admin_router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict:
    """Cancel an active job: release its stage nodes and mark it FAILED."""
    tracker: JobTracker = request.app.state.jobs
    job = tracker.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    if job.state in (JobState.COMPLETED, JobState.FAILED):
        raise HTTPException(status_code=409, detail="job already terminal")
    service = _service(request)
    for stage in job.stages:
        service.release_node(stage.node_id)
    tracker.fail_job(job_id, "cancelled by admin")
    log.warning("admin_cancel job=%s model=%s", job_id, job.model_id)
    return {"ok": True, "job_id": job_id}


@admin_router.get("/models")
def admin_models(request: Request) -> dict:
    """Model-store listing with on-disk sizes."""
    settings = request.app.state.settings
    models = []
    for manifest_model in shard_store_list(settings.model_store_dir):
        model_dir = pathlib.Path(settings.model_store_dir) / manifest_model.model_id
        size = (
            sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            if model_dir.is_dir()
            else 0
        )
        models.append(
            {
                "model_id": manifest_model.model_id,
                "name": manifest_model.name,
                "layers": manifest_model.layers,
                "hidden": manifest_model.hidden,
                "size_bytes": size,
            }
        )
    return {"models": models}


@admin_router.post("/models/rescan")
def rescan_models(request: Request) -> dict:
    """Recompute every model's placement plan immediately."""
    request.app.state.recompute_pool("admin_rescan")
    log.info("admin_rescan")
    return {"ok": True}


@admin_router.post("/models/{model_id}/delete")
def delete_model(model_id: str, request: Request) -> dict:
    """Delete a model from the coordinator's store and recompute placements."""
    settings = request.app.state.settings
    if not shard_store_safe(model_id):
        raise HTTPException(status_code=400, detail="invalid model id")
    root = pathlib.Path(settings.model_store_dir).resolve()
    target = (root / model_id).resolve()
    if root not in target.parents or not target.is_dir():
        raise HTTPException(status_code=404, detail="unknown model")
    shutil.rmtree(target)
    request.app.state.recompute_pool("admin_delete")
    log.warning("admin_model_deleted model=%s", model_id)
    return {"ok": True, "model_id": model_id}


@admin_router.get("/logs")
def admin_logs(request: Request, lines: int = 200) -> dict:
    """Tail of the in-process coordinator log ring."""
    lines = max(1, min(lines, 2000))
    return {"logs": log_snapshot(lines)}


# -- admin: GGUF import (session 14) ------------------------------------------------


class GgufImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str | None = Field(default=None, min_length=1)
    model_id: str | None = Field(default=None, min_length=1)
    tokenizer: str | None = Field(default=None, min_length=1)
    dtype: Literal["fp16", "fp32"] = "fp16"
    layers_per_shard: int = Field(default=4, ge=1, le=64)
    force: bool = False


def _resolve_node_project(settings) -> pathlib.Path | None:
    """The Node project whose converter the admin page shells out to (uv run)."""
    candidate = (
        pathlib.Path(settings.node_project_dir)
        if settings.node_project_dir
        else find_base_dir().parent / "Node"
    )
    return candidate if (candidate / "pyproject.toml").is_file() else None


def _gguf_marker(store: pathlib.Path) -> dict[str, dict]:
    """The converter's .gguf-imports.json (file -> sha256/model_id), if present."""
    try:
        with open(store / ".gguf-imports.json", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@admin_router.get("/models/imports")
def admin_gguf_imports(request: Request) -> dict:
    """GGUF files sitting in the model store + their import status."""
    settings = request.app.state.settings
    store = pathlib.Path(settings.model_store_dir)
    marker = _gguf_marker(store)
    active = getattr(request.app.state, "gguf_import", None)
    files = []
    for f in sorted(store.glob("*.gguf")) if store.is_dir() else []:
        entry = marker.get(f.name)
        imported = bool(
            entry and (store / str(entry.get("model_id", "?")) / "manifest.json").is_file()
        )
        if active is not None and active.get("filename") in (None, f.name):
            status = "importing"
        elif imported:
            status = "imported"
        else:
            status = "pending"
        files.append(
            {
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "model_id": entry.get("model_id") if entry else None,
                "status": status,
                "imported_at": entry.get("imported_at") if entry else None,
            }
        )
    node_dir = _resolve_node_project(settings)
    return {
        "imports": files,
        "busy": active is not None,
        "error": getattr(request.app.state, "gguf_import_error", None),
        "node_project": str(node_dir) if node_dir else None,
    }


@admin_router.post("/models/import")
async def admin_import_gguf(payload: GgufImportRequest, request: Request) -> dict:
    """Convert .gguf files in the model store via the Node converter.

    The coordinator stays lightweight (no torch): it runs
    `uv run --project <Node> python -m dain_node.import_gguf` as a subprocess
    and streams its output into the ring log. Only one import runs at a time.
    """
    settings = request.app.state.settings
    if getattr(request.app.state, "gguf_import", None) is not None:
        raise HTTPException(status_code=409, detail="a GGUF import is already running")
    store = pathlib.Path(settings.model_store_dir)
    if not store.is_dir():
        raise HTTPException(status_code=404, detail="model store directory not found")
    if payload.filename:
        if payload.filename != pathlib.Path(payload.filename).name or not (
            payload.filename.endswith(".gguf")
        ):
            raise HTTPException(
                status_code=400, detail="filename must be a .gguf directly inside the model store"
            )
        if not (store / payload.filename).is_file():
            raise HTTPException(
                status_code=404, detail=f"{payload.filename!r} not found in the model store"
            )
    node_dir = _resolve_node_project(settings)
    if node_dir is None:
        raise HTTPException(
            status_code=503,
            detail="Node project not found (set node_project_dir / DAIN_NODE_PROJECT_DIR); "
            "run Scripts/import-gguf.ps1 on a machine with the DAIN checkout instead",
        )
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        raise HTTPException(status_code=503, detail="uv not found on PATH")

    argv = [
        uv_bin,
        "run",
        "--project",
        str(node_dir),
        "python",
        "-m",
        "dain_node.import_gguf",
        str(store),
    ]
    if payload.filename:
        argv.append(payload.filename)
    if payload.model_id:
        argv += ["--model-id", payload.model_id]
    if payload.tokenizer:
        argv += ["--tokenizer", payload.tokenizer]
    if payload.force:
        argv.append("--force")
    argv += ["--dtype", payload.dtype, "--layers-per-shard", str(payload.layers_per_shard)]

    request.app.state.gguf_import = {"filename": payload.filename, "started_at": time.time()}
    asyncio.create_task(_run_gguf_import(request.app, argv))
    log.info(
        "admin_gguf_import_start file=%s node_project=%s", payload.filename or "all", node_dir
    )
    return {"ok": True, "started": payload.filename or "all pending", "node_project": str(node_dir)}


async def _run_gguf_import(app, argv: list[str]) -> None:
    try:
        runner = getattr(app.state, "gguf_runner", None)
        rc = await runner(argv) if runner is not None else _subprocess_import(argv)
        if rc == 0:
            app.state.gguf_import_error = None
            app.state.recompute_pool("gguf_import")
            log.info("gguf_import_done rc=0")
        else:
            app.state.gguf_import_error = f"converter exited with code {rc} (see logs)"
            log.error("gguf_import_failed rc=%d", rc)
    except Exception as exc:  # noqa: BLE001 — background task must never crash the loop
        app.state.gguf_import_error = str(exc)
        log.exception("gguf_import_error")
    finally:
        app.state.gguf_import = None


async def _subprocess_import(argv: list[str]) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None
    async for raw in proc.stdout:
        log.info("gguf_import %s", raw.decode("utf-8", "replace").rstrip())
    return await proc.wait()


# -- admin: model export (session N) ---------------------------------------------------------------


class ExportModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_dir: str
    model_id: str
    quantization: Literal["int4"] = "int4"
    group_size: int = Field(default=128, ge=1)
    activation_dtype: Literal["fp16", "bf16"] = "fp16"
    layers_per_shard: int = Field(default=4, ge=1, le=64)
    force: bool = False
    trust_remote_code: bool = False


class ExportValidateRequest(BaseModel):
    """`Detect` in the admin UI probes a directory before any export config is
    filled in, so only *source_dir* is required here — model_id is optional."""

    model_config = ConfigDict(extra="forbid")

    source_dir: str
    model_id: str | None = Field(default=None, min_length=1)


class ExportJobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    state: Literal["queued", "running", "completed", "failed"]
    source_dir: str
    model_id: str
    started_at: float | None = None
    completed_at: float | None = None
    progress: float | None = None
    output_model_id: str | None = None
    error: str | None = None
    logs: list[str] = []


def _export_marker(store: pathlib.Path) -> dict[str, dict]:
    """The exporter's .model-exports.json marker (model_id -> metadata), if present."""
    try:
        with open(store / ".model-exports.json", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _validate_export_source(source_dir: str, export_roots: tuple[str, ...]) -> pathlib.Path:
    """Resolve and path-check source_dir against approved export_roots."""
    src = pathlib.Path(source_dir).resolve()
    if not export_roots:
        raise HTTPException(
            status_code=400, detail="no export_roots configured; set DAIN_EXPORT_ROOTS"
        )
    allowed = False
    for root in export_roots:
        try:
            src.relative_to(pathlib.Path(root).resolve())
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=f"source_dir {src} is not under any configured export_root",
        )
    return src


def _active_export(request: Request) -> dict | None:
    return getattr(request.app.state, "model_export", None)


@admin_router.get("/models/export/roots")
def admin_export_roots(request: Request) -> dict:
    settings = request.app.state.settings
    roots = []
    for r in settings.export_roots:
        p = pathlib.Path(r)
        roots.append({
            "path": str(p),
            "exists": p.is_dir(),
        })
    return {"roots": roots, "busy": _active_export(request) is not None}


def _export_source_weights(src: pathlib.Path) -> list[pathlib.Path]:
    return sorted(src.glob("*.safetensors")) or sorted(src.glob("*.bin"))


def _export_source_size(src: pathlib.Path) -> int:
    return sum(p.stat().st_size for p in _export_source_weights(src))


def _reject_if_too_large(settings, src: pathlib.Path) -> int:
    """Shared size gate for validate + start; returns size_bytes."""
    size_bytes = _export_source_size(src)
    limit_gb = getattr(settings, "max_export_size_gb", 100.0)
    if size_bytes > limit_gb * 1e9:
        raise HTTPException(
            status_code=413,
            detail=(
                f"source too large: {size_bytes / 1e9:.1f} GB exceeds "
                f"max_export_size_gb ({limit_gb:.1f} GB)"
            ),
        )
    return size_bytes


@admin_router.post("/models/export/validate")
def admin_export_validate(payload: ExportValidateRequest, request: Request) -> dict:
    settings = request.app.state.settings
    src = _validate_export_source(payload.source_dir, settings.export_roots)
    info: dict = {"source_dir": str(src), "model_id": payload.model_id}
    if not src.is_dir():
        info["valid"] = False
        info["error"] = f"source directory {src} not found"
        return info
    config_file = src / "config.json"
    weights = _export_source_weights(src)
    if not config_file.is_file() or not weights:
        info["valid"] = False
        info["error"] = "missing config.json and/or model weights"
        return info
    size_bytes = sum(p.stat().st_size for p in weights)
    if size_bytes > settings.max_export_size_gb * 1e9:
        info["valid"] = False
        info["error"] = (
            f"source too large: {size_bytes / 1e9:.1f} GB exceeds "
            f"max_export_size_gb ({settings.max_export_size_gb:.1f} GB)"
        )
        return info
    info.update({
        "valid": True,
        "weight_count": len(weights),
        "size_bytes": size_bytes,
        "has_fast_tokenizer": (src / "tokenizer.json").is_file(),
    })
    return info


@admin_router.get("/models/export/exports")
def admin_model_exports(request: Request) -> dict:
    settings = request.app.state.settings
    store = pathlib.Path(settings.model_store_dir)
    marker = _export_marker(store)
    active = _active_export(request)
    files: list[dict] = []
    for model_dir in sorted(store.iterdir()) if store.is_dir() else []:
        if not model_dir.is_dir():
            continue
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        meta = marker.get(model_dir.name)
        files.append({
            "model_id": model_dir.name,
            "has_manifest": True,
            "exported": bool(meta),
            "active": active is not None and active.get("model_id") == model_dir.name,
        })
    return {
        "exports": files,
        "busy": active is not None,
        "error": getattr(request.app.state, "model_export_error", None),
        "node_project": (
            str(_resolve_node_project(settings))
            if _resolve_node_project(settings) else None
        ),
    }


@admin_router.post("/models/export")
async def admin_export_model(payload: ExportModelRequest, request: Request) -> dict:
    """Export a local HF model to DAIN INT4 shards via the Node exporter.

    Runs `uv run --project <Node> python -m dain_node.model_export` as a
    subprocess.  Only one export runs at a time.
    """
    settings = request.app.state.settings
    if _active_export(request) is not None:
        raise HTTPException(status_code=409, detail="a model export is already running")
    src = _validate_export_source(payload.source_dir, settings.export_roots)
    if not src.is_dir():
        raise HTTPException(status_code=404, detail=f"source directory {src} not found")
    _reject_if_too_large(settings, src)
    node_dir = _resolve_node_project(settings)
    if node_dir is None:
        raise HTTPException(
            status_code=503,
            detail="Node project not found (set node_project_dir / DAIN_NODE_PROJECT_DIR)",
        )
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        raise HTTPException(status_code=503, detail="uv not found on PATH")

    argv = [
        uv_bin, "run", "--project", str(node_dir),
        "python", "-m", "dain_node.model_export",
        "--source-dir", str(src),
        "--model-id", payload.model_id,
        "--output-store", str(pathlib.Path(settings.model_store_dir)),
        "--quantization", payload.quantization,
        "--group-size", str(payload.group_size),
        "--activation-dtype", payload.activation_dtype,
        "--layers-per-shard", str(payload.layers_per_shard),
        "--json-progress",
    ]
    if payload.force:
        argv.append("--force")
    if payload.trust_remote_code:
        argv.append("--trust-remote-code")

    request.app.state.model_export = {
        "model_id": payload.model_id,
        "started_at": time.time(),
    }
    request.app.state.model_export_error = None
    asyncio.create_task(_run_model_export(request.app, argv))
    log.info(
        "admin_export_start model_id=%s source=%s node_project=%s",
        payload.model_id, str(src), node_dir,
    )
    return {"ok": True, "started": payload.model_id, "node_project": str(node_dir)}


async def _run_model_export(app, argv: list[str]) -> None:
    settings = getattr(app.state, "settings", None)
    timeout = getattr(settings, "export_timeout_s", None)
    try:
        runner = getattr(app.state, "model_export_runner", None)
        if runner is not None:
            rc = await asyncio.wait_for(runner(argv), timeout=timeout)
        else:
            rc = await _subprocess_export(argv, timeout_s=timeout)
        if rc == 0:
            app.state.model_export_error = None
            app.state.recompute_pool("model_export")
            log.info("model_export_done rc=0")
        else:
            app.state.model_export_error = f"exporter exited with code {rc} (see logs)"
            log.error("model_export_failed rc=%d", rc)
    except TimeoutError:
        app.state.model_export_error = f"export timed out after {timeout}s"
        log.error("model_export_timeout timeout=%s", timeout)
    except Exception as exc:  # noqa: BLE001 — background task must never crash the loop
        app.state.model_export_error = str(exc)
        log.exception("model_export_error")
    finally:
        app.state.model_export = None


async def _subprocess_export(argv: list[str], timeout_s: float | None = None) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None
    try:
        if timeout_s is None:
            async for raw in proc.stdout:
                log.info("model_export %s", raw.decode("utf-8", "replace").rstrip())
            return await proc.wait()
        # Bound the whole run (including an exporter that stalls without
        # emitting output) and never leak the child on timeout.
        async with asyncio.timeout(timeout_s):
            async for raw in proc.stdout:
                log.info("model_export %s", raw.decode("utf-8", "replace").rstrip())
            return await proc.wait()
    except TimeoutError:
        log.error("model_export_proc_timeout timeout=%s", timeout_s)
        proc.terminate()
        with contextlib.suppress(ProcessLookupError):
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except TimeoutError:
                proc.kill()
                await proc.wait()
        raise
    except asyncio.CancelledError:
        # Coordinator shutdown / caller cancelled: take the child with us.
        proc.terminate()
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()
        raise


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
            {
                "model_id": m.model_id,
                "name": m.name,
                "layers": m.layers,
                "hidden": m.hidden,
                "format": getattr(m, "format", None),
                "architecture": getattr(m, "architecture", None),
                "quantization": {
                    "backend": m.quantization.backend,
                    "bits": m.quantization.bits,
                    "packing_layout": getattr(m.quantization, "packing_layout", None),
                } if getattr(m, "quantization", None) and m.quantization.is_quantized else None,
            }
            for m in models
        ]
    }


@v1_router.get("/nodes")
def list_public_nodes(request: Request) -> dict:
    """S9 dashboard feed (spec §12 observability): node rows, last placements,
    active job count — everything the browser dashboard needs."""
    service = _service(request)
    connections = request.app.state.connections
    rows = service.registry.list_nodes()
    placements = request.app.state.placements
    jobs: JobTracker = request.app.state.jobs
    active = sum(
        1 for j in jobs.jobs.values() if j.state not in (JobState.COMPLETED, JobState.FAILED)
    )
    return {
        "nodes": [
            _view(row, connected=connections.is_connected(row.node_id)).model_dump() for row in rows
        ],
        "placements": [event_view(e) for e in placements.events()[-16:]],
        "active_jobs": active,
    }


@v1_router.post("/completions")
async def completions(payload: CompletionRequest, request: Request):
    settings = request.app.state.settings
    jobs: JobTracker = request.app.state.jobs
    connections = request.app.state.connections
    limiter = request.app.state.rate_limiter
    client_ip = request.client.host if request.client else "unknown"
    if not limiter.allow(f"{client_ip}:{request.headers.get('x-api-key', '')}"):
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded; slow down",
            headers={"Retry-After": "60"},
        )
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
    queue = jobs.attach(record.job_id)
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

    async def event_stream():
        try:
            yield _sse({"job_id": record.job_id, "status": "dispatched"})
            deadline = asyncio.get_running_loop().time() + settings.job_timeout_s
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    jobs.fail_job(record.job_id, "job timeout")
                    yield _sse({"type": "error", "job_id": record.job_id, "detail": "job timeout"})
                    break
                try:
                    frame = await asyncio.wait_for(
                        queue.get(), timeout=min(10.0, remaining)
                    )
                except TimeoutError:
                    yield _sse({"type": "heartbeat", "job_id": record.job_id})
                    continue
                yield _sse(frame)
                if frame["type"] in ("final", "error"):
                    break
            yield "data: [DONE]\n\n"
        finally:
            if record.state not in (JobState.COMPLETED, JobState.FAILED):
                # Client went away mid-stream: reach a terminal state so the
                # ledger fires exactly once and the watchdog stops restarting
                # a job nobody is reading (nodes are freed by _release below).
                jobs.fail_job(record.job_id, "client disconnected")
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
    deadline = asyncio.get_running_loop().time() + settings.job_timeout_s
    try:
        while True:
            try:
                frame = await asyncio.wait_for(
                    queue.get(), timeout=max(0.1, deadline - asyncio.get_running_loop().time())
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
        if record.state not in (JobState.COMPLETED, JobState.FAILED):
            jobs.fail_job(record.job_id, "client disconnected")
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
        "nodes": [{"node_id": nid, **bucket} for nid, bucket in sorted(totals.items())],
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

#: Shard file extensions by ShardRef.format. Quantized shards are `torch_pt`
#: (TorchAO packing metadata survives only torch.save), fp16/fp32 stay
#: `safetensors`; None = legacy safetensors (plan §2.1).
_SHARD_EXTS = {
    None: ".safetensors",
    "safetensors": ".safetensors",
    "torch_pt": ".pt",
}


def _shard_ext(settings, model_id: str, shard_id: str) -> str:
    """Resolve *shard_id* against the manifest so quantized shards serve as .pt.

    A cold node first downloads from the coordinator: if the manifest records
    `format="torch_pt"` the bytes live at `layers_XX_YY.pt`, not
    `layers_XX_YY.safetensors` — guessing the extension always 404s.
    """
    manifest = shard_store_load(settings.model_store_dir, model_id)
    if manifest is not None:
        for shard in manifest.shards:
            if shard.shard_id == shard_id:
                return _SHARD_EXTS.get(shard.format, ".safetensors")
    # Unknown shard/model → fall back to the legacy extension, matching the
    # old single-format behavior; the existence check still 404s cleanly.
    return ".safetensors"


def _shard_path(settings, model_id: str, shard_id: str) -> str:
    # Path-traversal guard: mirror the store's own safe-component rule so ids
    # that pass manifest loading (dots, dashes, …) can never escape the store.
    if not shard_store_safe(model_id) or not shard_store_safe(shard_id):
        raise HTTPException(status_code=400, detail="invalid model or shard id")
    # Quantized models live as `.pt` — resolve the manifest's format so a cold
    # node requesting a quantized shard gets bytes instead of a 404.
    ext = _shard_ext(settings, model_id, shard_id)
    return os.path.join(settings.model_store_dir, model_id, f"{shard_id}{ext}")


def _model_dir(settings, model_id: str) -> str:
    if not shard_store_safe(model_id):
        raise HTTPException(status_code=400, detail="invalid model id")
    return os.path.join(settings.model_store_dir, model_id)


@model_router.get("/manifest/{model_id}")
def get_manifest(model_id: str, request: Request) -> ModelManifest:
    settings = request.app.state.settings
    manifest = shard_store_load(settings.model_store_dir, model_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="unknown model")
    return manifest


@model_router.get("/shard/{model_id}/{shard_id}")
def get_shard(model_id: str, shard_id: str, request: Request) -> FileResponse:
    settings = request.app.state.settings
    path = pathlib.Path(_shard_path(settings, model_id, shard_id))
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown shard")
    # Stream the file in chunks: multi-GB shards must not buffer in RAM
    # (path.read_bytes() would balloon coordinator memory per download).
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@model_router.head("/shard/{model_id}/{shard_id}")
def head_shard(model_id: str, shard_id: str, request: Request) -> Response:
    """Shard size probe: nodes use `Content-Length` to plan parallel ranges."""
    settings = request.app.state.settings
    path = pathlib.Path(_shard_path(settings, model_id, shard_id))
    if not path.exists():
        raise HTTPException(status_code=404, detail="unknown shard")
    response = Response()
    response.headers["Content-Length"] = str(path.stat().st_size)
    response.headers["Accept-Ranges"] = "bytes"
    return response


@model_router.get("/peers/{model_id}/{shard_id}")
def shard_peers(model_id: str, shard_id: str, request: Request) -> dict:
    """Nodes whose cache already holds this shard, as HTTP peer URLs.

    The requesting node downloads from peers first (each peer serves its own
    cached copy at byte-range granularity) and only falls back to this
    coordinator's store when no peer responds. Peers must be ONLINE *and*
    currently connected (a stale registered row has no live bytes).
    """
    service: NodeService = request.app.state.service
    connections = request.app.state.connections
    me = request.headers.get("x-node-id", "")
    peers = service.registry.peers_for_shard(model_id, shard_id, exclude=me)
    urls: list[str] = []
    for row in peers:
        # Only live, currently-connected peers are useful sources of bytes.
        if row.node_id == me:
            continue
        if row.state != NodeState.ONLINE or not connections.is_connected(row.node_id):
            continue
        if row.peer_url:
            urls.append(row.peer_url.rstrip("/"))
    random.shuffle(urls)
    return {"model_id": model_id, "shard_id": shard_id, "peers": urls[:6]}


@model_router.get("/tokenizer/{model_id}")
def get_tokenizer(model_id: str, request: Request) -> Response:
    """Serve the model's fast tokenizer.json so nodes can verify + cache it
    (spec §11); byte-level dev models have none → 404."""
    settings = request.app.state.settings
    manifest = shard_store_load(settings.model_store_dir, model_id)
    if manifest is None or manifest.tokenizer_file is None:
        raise HTTPException(status_code=404, detail="model has no tokenizer")
    file_name = os.path.basename(manifest.tokenizer_file)
    path = pathlib.Path(os.path.join(_model_dir(settings, model_id), file_name))
    if not path.exists():
        raise HTTPException(status_code=404, detail="tokenizer file missing")
    return FileResponse(path, media_type="application/json", filename=file_name)


# -- shard store helpers -----------------------------------------------------------

from dain_common.model_store import is_safe_model_id as shard_store_safe  # noqa: E402
from dain_common.model_store import list_models as shard_store_list  # noqa: E402
from dain_common.model_store import load_manifest as shard_store_load  # noqa: E402
