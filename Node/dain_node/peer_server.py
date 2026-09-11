"""Node-side shard server for peer-to-peer distribution.

Every node that has shards cached can serve them to sibling nodes over the LAN
with byte-range (`Range`) support, so a downloading node can:
- resume an interrupted transfer mid-file (Phase 1 reliability), and
- pull different byte ranges of one shard from several peers in parallel
  (multi-source, effectively the LAN bandwidth ├ù peers rather than one stream).

Security model: the coordinator is the source of truth for *who* is allowed to
join and for *which* shards each node holds. Peer requests are authenticated
with the cluster's shared `join_token` (every node already holds it in its own
config, so it is the natural LAN trust boundary). The server only serves shard
files already verified into `cache_dir` ΓÇö path traversal is blocked by the same
safe-component rule the rest of the system uses.

Implementation: a tight asyncio HTTP/1.1 server (GET/HEAD only). It runs inside
the agent's event loop so it needs no extra package and one process. Requests
are capped and idle sockets are reaped so a misbehaving LAN client cannot pin
the node open.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import os
import socket
from urllib.parse import urlparse

from dain_common.model_store import is_safe_model_id

log = logging.getLogger("dain.node.peer")

_STREAM_CHUNK = 1 << 20  # 1 MiB disk reads per iteration
_HEADER_MAX = 64 * 1024  # an oversized header block is dropped


def _crc_bytes(s: str) -> bytes:
    return s.encode("utf-8")


def resolve_lan_ip(coord_host: str | None = None) -> str:
    """The LAN-facing IP this node should advertise as a peer.

    Opens a UDP socket toward the coordinator's host (no packets are sent) and
    reads back the local interface the kernel would use ΓÇö the address a sibling
    on the same network can actually reach. Falls back to the primary hostname
    resolution when the coordinator host has no route (e.g. tests).
    """
    target = (coord_host or "192.0.2.1", 9)
    if coord_host:
        with contextlib.suppress(OSError):
            host = coord_host.rsplit(":", 1)[0]
            candidate = socket.gethostbyname(host)  # verify it's resolvable/IP
            if candidate:
                target = (candidate, 9)
    with contextlib.suppress(OSError):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(target)
            addr = probe.getsockname()[0]
            if addr and not addr.startswith("127."):
                return addr
        finally:
            probe.close()
    with contextlib.suppress(OSError):
        fallback = socket.gethostbyname(socket.gethostname())
        if fallback and not fallback.startswith("127."):
            return fallback
    return "0.0.0.0"


class PeerShardServer:
    """Owns one listening socket serving `GET /peer/shard/<model>/<shard>`.

    Constructed by `__main__.py` and `start()`ed alongside the agent; the agent
    advertises `url()` in its `Register.peer_url` so the coordinator can route
    peer downloads.
    """

    def __init__(
        self,
        cache_dir: str,
        bind_host: str = "0.0.0.0",
        port: int = 0,
        *,
        advertise_host: str | None = None,
        join_token: str = "",
    ) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        self._bind_host = bind_host
        self.port = port
        # What peers actually reach: the LAN interface we advertise in Register,
        # NOT the bind address ("0.0.0.0" is not routable from a sibling).
        self._advertise = advertise_host or "127.0.0.1"
        self._join_token = join_token
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> PeerShardServer:
        """Bind the socket (port 0 ΓåÆ OS-assigned) and start accepting."""
        self._server = await asyncio.start_server(
            self._client, self._bind_host, self.port, limit=_HEADER_MAX
        )
        bound = self._server.sockets
        if bound:
            self.port = bound[0].getsockname()[1]
        log.info(
            "peer_server_listen bind=%s advertise=%s port=%d",
            self._bind_host,
            self._advertise,
            self.port,
        )
        return self

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
            log.info("peer_server_stopped")

    def url(self) -> str:
        return f"http://{self._advertise}:{self.port}"

    # -- request handling ----------------------------------------------------------

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await asyncio.wait_for(self._serve(reader, writer), timeout=30.0)
        except (TimeoutError, ConnectionError, OSError):
            pass
        except Exception:  # noqa: BLE001 ΓÇö a broken client must not kill the server
            log.exception("peer_request_failed")
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError) as exc:
            log.warning("peer_bad_headers err=%r", exc)
            return
        request_line, *header_lines = head.decode("utf-8", "replace").split("\r\n")
        parts = request_line.split(" ")
        if len(parts) < 3:
            return
        method, target, _proto = parts[0], parts[1], parts[2]
        headers: dict[str, str] = {}
        for line in header_lines:
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()

        accepted = self._client_ok(headers)
        if not accepted:
            await self._respond(writer, b"403 Forbidden", "text/plain", b"forbidden")
            return

        if method not in ("GET", "HEAD"):
            await self._respond(writer, b"405 Method Not Allowed", "text/plain", b"")
            return

        parsed = urlparse(target)
        path = parsed.path
        if not path.startswith("/peer/shard/"):
            await self._respond(writer, b"404 Not Found", "text/plain", b"")
            return
        parts = path.split("/")
        if len(parts) != 5 or parts[1:3] != ["peer", "shard"]:
            await self._respond(writer, b"404 Not Found", "text/plain", b"")
            return
        model_id, shard_id = parts[3], parts[4]
        if not is_safe_model_id(model_id) or not is_safe_model_id(shard_id):
            await self._respond(writer, b"400 Bad Request", "text/plain", b"bad id")
            return

        path = os.path.join(self.cache_dir, model_id, f"{shard_id}.safetensors")
        if not os.path.isfile(path):
            # Quantized shards ship as .pt files (TorchAO tensor subclasses).
            path = os.path.join(self.cache_dir, model_id, f"{shard_id}.pt")
        if not os.path.isfile(path):
            log.info("peer_miss model=%s shard=%s", model_id, shard_id)
            await self._respond(writer, b"404 Not Found", "text/plain", b"")
            return

        size = os.path.getsize(path)
        if size == 0:
            await self._respond(writer, b"500 Internal Server Error", "text/plain", b"empty")
            return

        start, end = _parse_range(headers.get("range", ""), size, 1 << 20)
        if method == "HEAD":
            return await self._respond_sized(writer, start, end, size)
        await self._stream(writer, path, start, end, size)

    def _client_ok(self, headers: dict[str, str]) -> bool:
        provided = headers.get("x-peer-token", "")
        return bool(self._join_token) and hmac.compare_digest(
            provided, self._join_token
        )

    async def _respond(
        self, writer: asyncio.StreamWriter, status: bytes, media: str, body: bytes
    ) -> None:
        lines = [
            b"HTTP/1.1 " + status,
            b"Content-Type: " + _crc_bytes(media),
            b"Content-Length: " + str(len(body)).encode(),
            b"Connection: close",
            b"",
            b"",
        ]
        writer.write(b"\r\n".join(lines) + body)
        await writer.drain()

    async def _respond_sized(
        self, writer: asyncio.StreamWriter, start: int, end: int, size: int
    ) -> None:
        partial = start != 0 or end != size - 1
        lines = [
            b"HTTP/1.1 " + (b"206 Partial Content" if partial else b"200 OK"),
            b"Accept-Ranges: bytes",
        ]
        length = end - start + 1
        if partial:
            lines.append(b"Content-Range: bytes " + f"{start}-{end}/{size}".encode())
        lines.append(b"Content-Length: " + str(length).encode())
        lines.append(b"Connection: close")
        lines.append(b"")
        lines.append(b"")
        writer.write(b"\r\n".join(lines))
        await writer.drain()

    async def _stream(
        self, writer: asyncio.StreamWriter, path: str, start: int, end: int, size: int
    ) -> None:
        partial = start != 0 or end != size - 1
        status = b"206 Partial Content" if partial else b"200 OK"
        length = end - start + 1
        head_lines = [
            b"HTTP/1.1 " + status,
            b"Accept-Ranges: bytes",
            b"Content-Type: application/octet-stream",
        ]
        if partial:
            head_lines.append(b"Content-Range: bytes " + f"{start}-{end}/{size}".encode())
        head_lines += [
            b"Content-Length: " + str(length).encode(),
            b"Connection: close",
        ]
        head = b"\r\n".join(head_lines) + b"\r\n\r\n"
        try:
            fh = open(path, "rb")
        except OSError:
            await self._respond(writer, b"500 Internal Server Error", "text/plain", b"")
            return
        try:
            fh.seek(start)
            remaining = length
            writer.write(head)
            await writer.drain()
            while remaining > 0:
                chunk = await asyncio.to_thread(fh.read, min(_STREAM_CHUNK, remaining))
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()
                remaining -= len(chunk)
        finally:
            fh.close()
            with contextlib.suppress(Exception):
                writer.write_eof()
            log.debug(
                "peer_served url=%s bytes=%d range=%d-%d",
                path,
                length,
                start,
                end,
            )


def _parse_range(value: str, size: int, default_chunk: int) -> tuple[int, int]:
    """Honor `Range: bytes=N-` / `bytes=N-M`; default is the whole file.

    Returns an inclusive (start, end) range. Malformed headers fall back to
    the full file so a brain-dead client still gets all the bytes.
    """
    if not value or not value.startswith("bytes="):
        return 0, size - 1
    spec = value[6:]
    try:
        if spec.startswith("-"):
            # RFC 7233 suffix form `bytes=-N`: the *last* N bytes.
            n = int(spec[1:])
            if n <= 0:
                return 0, size - 1
            return max(size - n, 0), size - 1
        raw_start, _, raw_end = spec.partition("-")
        start = int(raw_start)
        if raw_end:
            end = min(int(raw_end), size - 1)
        else:
            end = size - 1
        if start < 0:
            start = max(size + start, 0)
        if start >= size or end < start:
            return 0, size - 1
        return start, end
    except ValueError:
        return 0, size - 1
