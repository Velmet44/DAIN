"""Live node WebSocket connections (S4): the coordinator's send side.

The node API accepts connections; this registry lets the rest of the system
dispatch JOB_ASSIGN envelopes and relay activation bytes to a connected node.
Per-connection locks serialize concurrent sends (uvicorn does not guarantee
frame atomicity across tasks).
"""

from __future__ import annotations

import asyncio
import logging

from dain_common.schemas import Envelope
from starlette.websockets import WebSocket

log = logging.getLogger("dain.coordinator.connections")


class NodeConnections:
    def __init__(self) -> None:
        self._conns: dict[str, WebSocket] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.on_disconnect = None  # set by the app: callable(node_id)

    def register(self, node_id: str, websocket: WebSocket) -> None:
        old = self._conns.get(node_id)
        if old is not None and old is not websocket:
            log.warning("connection_replaced node=%s", node_id)
        self._conns[node_id] = websocket
        self._locks.setdefault(node_id, asyncio.Lock())

    def unregister(self, node_id: str, websocket: WebSocket) -> None:
        if self._conns.get(node_id) is websocket:
            del self._conns[node_id]
            log.info("connection_gone node=%s", node_id)
            if self.on_disconnect is not None:
                self.on_disconnect(node_id)

    def is_connected(self, node_id: str) -> bool:
        return node_id in self._conns

    def connected_ids(self) -> list[str]:
        return sorted(self._conns)

    async def send_envelope(self, node_id: str, envelope: Envelope) -> bool:
        ws = self._conns.get(node_id)
        if ws is None:
            return False
        try:
            async with self._locks.setdefault(node_id, asyncio.Lock()):
                await ws.send_text(envelope.model_dump_json())
            return True
        except Exception as exc:  # noqa: BLE001 — transport failures are expected
            log.warning("send_failed node=%s err=%s", node_id, exc)
            return False

    async def send_bytes(self, node_id: str, data: bytes) -> bool:
        ws = self._conns.get(node_id)
        if ws is None:
            return False
        try:
            async with self._locks.setdefault(node_id, asyncio.Lock()):
                await ws.send_bytes(data)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("send_bytes_failed node=%s err=%s", node_id, exc)
            return False
