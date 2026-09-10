"""LAN discovery responder (UDP).

Replies **only** to token-authenticated ``discover`` probes with a unicast
``hello`` carrying the coordinator's WebSocket port; the node builds its URL
from the reply's source address.  Malformed datagrams, unknown ops and wrong
tokens are dropped silently (with a debug log).  The bind failure path raises
``OSError`` so ``create_app`` can disable discovery without killing the
coordinator.
"""

from __future__ import annotations

import asyncio
import logging
import socket

from dain_common.node_discovery import (
    MAX_DATAGRAM_BYTES,
    DiscoverRequest,
    make_hello_packet,
    token_matches,
)

log = logging.getLogger("dain.coordinator.discovery")


class DiscoveryResponder:
    """One UDP socket answering discovery probes on a single background task."""

    def __init__(self, port: int, join_token: str, ws_port: int) -> None:
        self._port = port
        self._join_token = join_token
        self._ws_port = ws_port
        self._sock: socket.socket | None = None

    def start(self) -> None:
        """Bind the non-blocking UDP socket (raises ``OSError`` if taken)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self._port))
        sock.setblocking(False)
        self._sock = sock
        log.info("discovery_listening port=%d ws_port=%d", self._port, self._ws_port)

    def set_join_token(self, join_token: str) -> None:
        """Rotate the admission token the responder authenticates probes with."""
        self._join_token = join_token
        log.info("discovery_join_token_rotated")

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    async def run(self) -> None:
        """Serve probes until cancelled (``asyncio.CancelledError`` propagates)."""
        assert self._sock is not None, "call start() before run()"
        loop = asyncio.get_running_loop()
        while True:
            data, addr = await loop.sock_recvfrom(self._sock, MAX_DATAGRAM_BYTES)
            req = DiscoverRequest.parse(data)
            if req is None:
                continue
            if not token_matches(self._join_token, req.join_token):
                log.debug("discovery_rejected source=%s", addr[0])
                continue
            await loop.sock_sendto(self._sock, make_hello_packet(self._ws_port), addr)
            log.info("discovery_request source=%s", addr[0])
