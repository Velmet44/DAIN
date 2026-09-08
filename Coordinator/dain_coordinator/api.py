"""HTTP/WS surface (S2): node registration + heartbeat WS + admin views.

Auth model (spec §11/§16, MVP): registration requires the shared join token and
returns a per-node token; WS connections authenticate with `node_id` + that
token as query params; deregistration requires the node token. Admin endpoints
are unauthenticated in dev (hardening lands with the public deployment, S9).
"""

from __future__ import annotations

import logging

from dain_common.schemas import Register, RegisterAck
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict

from dain_coordinator.nodes import MessageOutcome, NodeService
from dain_coordinator.store import NodeRow

log = logging.getLogger("dain.coordinator.api")

node_router = APIRouter(prefix="/node", tags=["node"])
admin_router = APIRouter(prefix="/admin", tags=["admin"])


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


def _view(row: NodeRow) -> NodeView:
    return NodeView(
        node_id=row.node_id,
        state=row.state.value,
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
    return _service(request).register(payload)


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
    node_id = websocket.query_params.get("node_id", "")
    token = websocket.query_params.get("token", "")
    if not service.authenticate(node_id, token):
        await websocket.close(code=4401, reason="invalid node credentials")
        return
    await websocket.accept()
    log.info("ws_connected node=%s", node_id)
    try:
        while True:
            raw = await websocket.receive_text()
            outcome = service.handle_message(node_id, raw)
            if outcome == MessageOutcome.CLOSE_MALFORMED:
                await websocket.close(code=4400, reason="malformed message")
                return
            if outcome == MessageOutcome.CLOSE_IDENTITY:
                await websocket.close(code=4400, reason="node_id mismatch")
                return
    except WebSocketDisconnect:
        log.info("ws_disconnected node=%s", node_id)


# -- admin API -----------------------------------------------------------------


@admin_router.get("/nodes", response_model=list[NodeView])
def list_nodes(request: Request, state: str | None = None) -> list[NodeView]:
    service = _service(request)
    if state is not None:
        from dain_common.schemas import NodeState

        try:
            state_filter = NodeState(state)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown state {state!r}") from exc
        return [_view(row) for row in service.registry.list_nodes(state_filter)]
    return [_view(row) for row in service.registry.list_nodes()]


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
    return NodeDetail(**_view(row).model_dump(), history=history)
