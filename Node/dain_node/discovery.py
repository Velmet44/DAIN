"""LAN coordinator discovery probe (node side).

Run once at first startup when ``config.json`` has no ``coord_url``: broadcast
a token-authenticated ``discover`` datagram, wait for the coordinator's
``hello``, and return ``ws://<reply-source-ip>:<port>``.  The reply's source
address is authoritative (it is the address that can actually reach this node),
so no IP ever has to be typed.  Network-contended LANs may need a larger
*timeout*; very quiet probes can be repeated.
"""

from __future__ import annotations

import asyncio
import logging
import socket

from dain_common.node_discovery import (
    DEFAULT_DISCOVERY_PORT,
    MAX_DATAGRAM_BYTES,
    HelloReply,
    broadcast_targets,
    make_discover_packet,
)

log = logging.getLogger("dain.node.discovery")


async def discover_coordinator(
    join_token: str,
    port: int = DEFAULT_DISCOVERY_PORT,
    timeout_s: float = 2.0,
    targets: list[str] | None = None,
) -> str | None:
    """Return ``ws://<ip>:<port>`` for the best coordinator that answered.

    *targets* overrides the broadcast set (used by tests to probe loopback).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setblocking(False)
    loop = asyncio.get_running_loop()
    packet = make_discover_packet(join_token)
    pending = list(targets if targets is not None else broadcast_targets())
    try:
        for target in pending:
            try:
                await loop.sock_sendto(sock, packet, (target, port))
            except OSError:
                log.debug("discovery_send_failed target=%s", target)

        best: HelloReply | None = None
        best_ip: str | None = None
        deadline = loop.time() + timeout_s
        while pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                data, addr = await asyncio.wait_for(
                    loop.sock_recvfrom(sock, MAX_DATAGRAM_BYTES), remaining
                )
            except TimeoutError:
                break
            except ConnectionResetError:
                # Windows surfaces ICMP "port unreachable" from silent hosts as
                # a reset on recvfrom(); not an error — keep listening.
                continue
            reply = HelloReply.parse(data)
            if reply is None:
                continue
            if best is None or reply.version > best.version:
                best = reply
                best_ip = addr[0]
                log.debug("discovery_reply ip=%s port=%d", addr[0], reply.port)
    finally:
        sock.close()

    if best is None or best_ip is None:
        log.info("coordinator_discovery_failed port=%d", port)
        return None
    url = f"ws://{best_ip}:{best.port}"
    log.info("coordinator_discovered url=%s", url)
    return url
