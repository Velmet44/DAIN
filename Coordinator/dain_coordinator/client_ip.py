"""Real client-IP resolution behind reverse proxies (S24).

uvicorn's built-in `proxy_headers` rewriting takes the FIRST entry of
`X-Forwarded-For` — the entry the connecting client itself supplied — which
lets any remote caller spoof their IP (and, worse, spoof membership of the
tailnet range used for keyless admin). This middleware instead:

- keeps the RAW socket peer in `scope["client"]` (untouched), and
- exposes `request.state.client_ip`: the raw peer, or — only when the raw
  peer is a configured trusted proxy — the LAST (rightmost) XFF entry, i.e.
  the address our own proxy observed.

With a single trusted proxy layer (cloudflared / Caddy / nginx on localhost)
the rightmost entry is the one value the client cannot forge.
"""

from __future__ import annotations

import ipaddress

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


def trusted_proxy_ips(spec: str) -> frozenset[str]:
    """Parse DAIN_TRUSTED_PROXIES ("127.0.0.1,::1" or "*") into peer strings."""
    spec = (spec or "").strip()
    if not spec:
        return frozenset({"127.0.0.1"})
    if spec == "*":
        return frozenset({"*"})
    return frozenset(entry.strip() for entry in spec.split(",") if entry.strip())


def _peer_is_trusted(peer: str, trusted: frozenset[str]) -> bool:
    if "*" in trusted:
        return True
    if peer in trusted:
        return True
    try:
        peer_ip = ipaddress.ip_address(peer)
    except ValueError:
        return False
    for entry in trusted:
        try:
            if peer_ip in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


def _rightmost_xff(chain: str) -> str | None:
    entries = [entry.strip() for entry in chain.split(",") if entry.strip()]
    if not entries:
        return None
    last = entries[-1]
    # Strip a bracketed IPv6 port or plain port if present.
    if last.startswith("[") and "]" in last:
        return last[1 : last.index("]")]
    if last.count(":") == 1:
        host, _sep, port = last.partition(":")
        if port.isdigit():
            return host
    return last


class ClientIPMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, trusted_ips: frozenset[str]) -> None:
        super().__init__(app)
        self._trusted = trusted_ips

    async def dispatch(self, request: Request, call_next):
        peer = request.client.host if request.client else "unknown"
        client_ip = peer
        if _peer_is_trusted(peer, self._trusted):
            chain = request.headers.get("x-forwarded-for")
            if chain:
                forwarded = _rightmost_xff(chain)
                if forwarded:
                    client_ip = forwarded
        request.state.client_ip = client_ip
        request.state.raw_peer = peer
        return await call_next(request)
