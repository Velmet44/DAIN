"""LAN coordinator discovery (stdlib-only, UDP).

The coordinator listens on a UDP port and answers token-authenticated
``discover`` probes.  A node sends the probe once at first startup, and treats
the *source address* of the reply as the coordinator's reachable LAN address —
so nobody has to type an IP into ``config.json`` by hand.  The reply's source
address is used (not anything the coordinator claims) so multi-homed
coordinators always advertise the address that can actually reach the node.

Protocol (v1, JSON datagrams, ``<= 512`` bytes):

* ``{"v": 1, "op": "discover", "k": "<join_token>"}``  — broadcast by the node
* ``{"v": 1, "op": "hello", "port": 8000}``            — unicast from coordinator

The node builds ``ws://<reply-source-ip>:<port>``.  The join token is a
pre-shared secret (config.json ships it); a coordinator only answers probes
that present the correct token.  Limited to one LAN subnet by design — a WAN
deployment disables discovery and sets ``coord_url`` explicitly.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass

DISCOVERY_VERSION = 1
DEFAULT_DISCOVERY_PORT = 8456
MAX_DATAGRAM_BYTES = 512

_LIMITED_BROADCAST = "255.255.255.255"


def make_discover_packet(join_token: str) -> bytes:
    """Serialise the node's one-shot discovery probe."""
    return json.dumps(
        {"v": DISCOVERY_VERSION, "op": "discover", "k": join_token},
        separators=(",", ":"),
    ).encode("utf-8")


def make_hello_packet(port: int) -> bytes:
    """Serialise the coordinator's unicast reply."""
    return json.dumps(
        {"v": DISCOVERY_VERSION, "op": "hello", "port": port},
        separators=(",", ":"),
    ).encode("utf-8")


def token_matches(expected: str, received: str) -> bool:
    """Timing-safe comparison of the pre-shared join token."""
    return secrets.compare_digest(expected.encode("utf-8"), received.encode("utf-8"))


@dataclass(frozen=True)
class DiscoverRequest:
    version: int
    join_token: str

    @classmethod
    def parse(cls, data: bytes) -> DiscoverRequest | None:
        """Reject malformed/oversized datagrams, unknown ops, and wrong protocol
        versions (silent)."""
        if not data or len(data) > MAX_DATAGRAM_BYTES:
            return None
        try:
            obj = json.loads(data)
            if obj.get("op") != "discover":
                return None
            token = obj.get("k")
            if not isinstance(token, str) or not token:
                return None
            if int(obj.get("v", 0)) != DISCOVERY_VERSION:
                return None
            return cls(version=DISCOVERY_VERSION, join_token=token)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class HelloReply:
    version: int
    port: int

    @classmethod
    def parse(cls, data: bytes) -> HelloReply | None:
        if not data or len(data) > MAX_DATAGRAM_BYTES:
            return None
        try:
            obj = json.loads(data)
            if obj.get("op") != "hello":
                return None
            port = int(obj.get("port", 0))
            if not (1 <= port <= 65535):
                return None
            if int(obj.get("v", 0)) != DISCOVERY_VERSION:
                return None
            return cls(version=DISCOVERY_VERSION, port=port)
        except (TypeError, ValueError):
            return None


def broadcast_targets() -> list[str]:
    """Addresses the node sends its probe to.

    Uses the IPv4 limited broadcast, which home routers / APs forward to every
    host on the same LAN subnet.  Cross-subnet or WAN setups must set
    ``coord_url`` explicitly (discovery is subnet-scoped by design).
    """
    return [_LIMITED_BROADCAST]
