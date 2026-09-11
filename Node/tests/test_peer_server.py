"""P2P shard server: serving, byte-range support, auth, and safety rules.

The peer server is the LAN face of a node: siblings pull cached shards from it
with optional `Range` headers (used for parallel chunk downloads + resume). It
must reject unauthenticated requesters and never escape the cache directory.

The server's asyncio loop lives on a background thread because Windows closes
proactor sockets when a loop exits; the requests run on the test's own loop.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from dain_node.peer_server import PeerShardServer, _parse_range

DATA = bytes(range(256)) * 4096  # deterministic 1 MiB blob
TOKEN = "lan-secret"


@pytest.fixture
def server(tmp_path):
    cache = tmp_path / "cache"
    model_dir = cache / "model-1"
    model_dir.mkdir(parents=True)
    (model_dir / "shard1.safetensors").write_bytes(DATA)
    svc = PeerShardServer(
        str(cache), bind_host="127.0.0.1", port=0, advertise_host="127.0.0.1", join_token=TOKEN
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="peer-test-loop", daemon=True)
    thread.start()
    try:
        asyncio.run_coroutine_threadsafe(svc.start(), loop).result(timeout=10)
        yield svc
    finally:
        asyncio.run_coroutine_threadsafe(svc.stop(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)


async def _get(url: str, *, headers=None):
    async with httpx.AsyncClient() as client:
        return await client.get(url, headers=headers)


async def _head(url: str, *, headers=None):
    async with httpx.AsyncClient() as client:
        return await client.head(url, headers=headers)


def test_parse_range_defaults_to_full_file() -> None:
    assert _parse_range("", 1000, 64) == (0, 999)
    assert _parse_range("bytes=0-9", 1000, 64) == (0, 9)
    assert _parse_range("bytes=500-", 1000, 64) == (500, 999)
    assert _parse_range("bytes=-25", 1000, 64) == (975, 999)
    assert _parse_range("bytes=10-5", 1000, 64) == (0, 999)  # malformed → full
    assert _parse_range("bytes=9999-", 1000, 64) == (0, 999)  # past EOF → full


def test_get_full_file(server) -> None:
    url = f"{server.url()}/peer/shard/model-1/shard1"
    r = asyncio.run(_get(url, headers={"X-Peer-Token": TOKEN}))
    assert r.status_code == 200
    assert r.content == DATA


def test_get_byte_range(server) -> None:
    url = f"{server.url()}/peer/shard/model-1/shard1"
    r = asyncio.run(
        _get(url, headers={"X-Peer-Token": TOKEN, "Range": "bytes=100-199"})
    )
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 100-199/{len(DATA)}"
    assert r.content == DATA[100:200]


def test_resume_range_open_ended(server) -> None:
    url = f"{server.url()}/peer/shard/model-1/shard1"
    r = asyncio.run(
        _get(url, headers={"X-Peer-Token": TOKEN, "Range": "bytes=1000-"})
    )
    assert r.status_code == 206
    assert r.content == DATA[1000:]


def test_head_reports_size(server) -> None:
    url = f"{server.url()}/peer/shard/model-1/shard1"
    r = asyncio.run(_head(url, headers={"X-Peer-Token": TOKEN}))
    assert r.status_code == 200
    assert r.headers["content-length"] == str(len(DATA))


def test_requires_peer_token(server) -> None:
    url = f"{server.url()}/peer/shard/model-1/shard1"
    r_missing = asyncio.run(_get(url))
    r_bad = asyncio.run(_get(url, headers={"X-Peer-Token": "wrong"}))
    assert r_missing.status_code == 403
    assert r_bad.status_code == 403


def test_unknown_shard_404(server) -> None:
    url = f"{server.url()}/peer/shard/model-1/nope"
    r = asyncio.run(_get(url, headers={"X-Peer-Token": TOKEN}))
    assert r.status_code == 404


def test_wrong_path_is_404(server) -> None:
    r = asyncio.run(_get(f"{server.url()}/other", headers={"X-Peer-Token": TOKEN}))
    assert r.status_code == 404


def test_invalid_ids_rejected(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    svc = PeerShardServer(
        str(cache), bind_host="127.0.0.1", port=0, advertise_host="127.0.0.1", join_token="s"
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="peer-test-loop", daemon=True)
    thread.start()
    try:
        asyncio.run_coroutine_threadsafe(svc.start(), loop).result(timeout=10)
        for target in ("bad id", "a/b", "..", ""):
            url = f"{svc.url()}/peer/shard/{target}/x"
            r = asyncio.run(_get(url, headers={"X-Peer-Token": "s"}))
            assert r.status_code in (400, 404)
    finally:
        asyncio.run_coroutine_threadsafe(svc.stop(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
