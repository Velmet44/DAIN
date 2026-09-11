"""LAN-fast shard download: parallel ranges, resume, peer-first sourcing.

The downloads are simulated against an in-memory HTTP origin (httpx
MockTransport + streaming responses) so no real network or coordinator is
needed:
- a shard > 8 MiB is pulled as parallel byte ranges,
- a range that dies mid-stream resumes from its on-disk partial, and
- peer URLs returned by the coordinator are tried before the coordinator.
"""

from __future__ import annotations

import asyncio
import hashlib
import os

import httpx
import pytest

from dain_node.llm import ModelStoreClient, _chunk_state_ok, _split_ranges, _write_chunk_state

MODEL = "model-g"
SHARD = "shard-0"
COORD_BASE = "http://coord:8000/model"
PEER_BASE = "http://peerhost:6000"

SIZE = 12 * 1024 * 1024  # > 8 MiB → parallel path kicks in
BLOCK = bytes((i * 7) & 0xFF for i in range(4096))
DATA = (BLOCK * (SIZE // len(BLOCK) + 1))[:SIZE]
HASH = hashlib.sha256(DATA).hexdigest()

RD = SIZE // 4  # first parallel range width when split in 4


class FlakyStream(httpx.AsyncByteStream):
    """Yields `body`, then (optionally) raises a transport error mid-stream."""

    def __init__(self, body: bytes, fail_after: int | None = None) -> None:
        self.body = body
        self.fail_after = fail_after
        self._pos = 0

    async def __aiter__(self):
        step = 64 * 1024
        while self._pos < len(self.body):
            if self.fail_after is not None and self._pos >= self.fail_after:
                raise httpx.NetworkError("simulated hotspot drop")
            chunk = self.body[self._pos : self._pos + step]
            self._pos += len(chunk)
            yield chunk


class FakeOrigin:
    """One HTTP origin speaking the coordinator + peer subset of the protocol."""

    def __init__(self, data: bytes = DATA) -> None:
        self.data = data
        self.requests: list[tuple[str, str, str, str]] = []  # (method, host, path, range)
        self.fail_exact: dict[tuple[str, str], int] = {}  # (path, range) → attempts to NACK
        self.fail_partial: dict[tuple[str, str], int] = {}  # (path, range) → then drop

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        host = request.url.host
        rng = request.headers.get("range", "")
        self.requests.append((request.method, host, path, rng))

        if request.method == "HEAD" and "/shard/" in path:
            return httpx.Response(
                200,
                headers={"Content-Length": str(len(self.data)), "Accept-Ranges": "bytes"},
            )
        if request.method == "GET" and path.endswith(f"/peers/{MODEL}/{SHARD}"):
            return httpx.Response(200, json={"peers": []})

        shard_paths = (f"/shard/{MODEL}/{SHARD}", f"/peer/shard/{MODEL}/{SHARD}")
        if not any(path.endswith(sp) for sp in shard_paths):
            return httpx.Response(404)

        key = (path, rng)
        if self.fail_exact.get(key, 0) > 0:
            self.fail_exact[key] -= 1
            raise httpx.NetworkError("simulated drop")
        if not rng:
            return httpx.Response(200, stream=FlakyStream(self.data))
        start, end = (int(v) for v in rng[6:].split("-"))
        body = self.data[start : end + 1]
        headers = {
            "Content-Range": f"bytes {start}-{end}/{len(self.data)}",
            "Content-Length": str(len(body)),
        }
        partial = self.fail_partial.get(key)
        if partial is not None and partial < len(body):
            self.fail_partial[key] = 0
            headers["Content-Length"] = str(partial)
            return httpx.Response(
                206,
                headers=headers,
                stream=FlakyStream(body, fail_after=partial),
            )
        return httpx.Response(206, headers=headers, stream=FlakyStream(body))


class PeeredOrigin(FakeOrigin):
    """Like FakeOrigin, but advertises a sibling node that holds the shard."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith(f"/peers/{MODEL}/{SHARD}"):
            return httpx.Response(200, json={"peers": [PEER_BASE]})
        return super().__call__(request)


class RangeIgnoringOrigin(FakeOrigin):
    """Serves the full body for every GET — Range headers are ignored (HTTP 200)."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        rng = request.headers.get("range", "")
        self.requests.append((request.method, request.url.host, path, rng))
        if request.method == "HEAD" and "/shard/" in path:
            return httpx.Response(
                200,
                headers={"Content-Length": str(len(self.data)), "Accept-Ranges": "bytes"},
            )
        if request.method == "GET" and path.endswith(f"/peers/{MODEL}/{SHARD}"):
            return httpx.Response(200, json={"peers": []})
        shard_paths = (f"/shard/{MODEL}/{SHARD}", f"/peer/shard/{MODEL}/{SHARD}")
        if not any(path.endswith(sp) for sp in shard_paths):
            return httpx.Response(404)
        return httpx.Response(200, stream=FlakyStream(self.data))


@pytest.fixture
def origin():
    return FakeOrigin()


def make_store(tmp_path, origin: FakeOrigin) -> ModelStoreClient:
    store = ModelStoreClient(cache_dir=str(tmp_path / "cache"))
    store.set_base_url(COORD_BASE)
    store.set_auth("node-a", "node-tok")
    store.set_peer_token("lan-secret")
    store._client = httpx.AsyncClient(
        transport=httpx.MockTransport(origin), trust_env=False, timeout=30.0
    )
    return store


async def close(store: ModelStoreClient | None) -> None:
    if store is not None:
        await store.close()


# -- pure helpers -----------------------------------------------------------------


def test_split_ranges_covers_total_exactly() -> None:
    assert _split_ranges(10_000, 4) == [(0, 2500), (2500, 5000), (5000, 7500), (7500, 10_000)]
    assert sum(b - a for a, b in _split_ranges(10_001, 4)) == 10_001
    assert _split_ranges(7, 4) == [(0, 2), (2, 4), (4, 6), (6, 7)]


def test_chunk_state_roundtrip(tmp_path) -> None:
    ranges = [(0, 100), (100, 200)]
    d = tmp_path / "d.chunks"
    _write_chunk_state(str(d), ranges)
    assert _chunk_state_ok(str(d), ranges)
    assert not _chunk_state_ok(str(d), [(0, 100), (100, 201)])  # resized shard → invalidate


# -- full download -----------------------------------------------------------------


def test_parallel_download_verifies_hash(tmp_path, origin) -> None:
    os.environ["DAIN_PARALLEL_CHUNKS"] = "4"
    os.environ["DAIN_DOWNLOAD_CHUNK"] = "65536"
    store = make_store(tmp_path, origin)
    try:
        path = asyncio.run(store.ensure_shard(MODEL, SHARD, HASH))
        assert open(path, "rb").read() == DATA
        # The 12 MiB shard was pulled as 4 concurrent ranges with Range headers.
        ranges = [r for (m, h, p, r) in origin.requests if m == "GET" and "bytes=" in r]
        assert len(ranges) == 4
        assert max(int(r[6:].split("-")[1]) for r in ranges) == SIZE - 1
        # Sidecar is written so later inventory doesn't re-hash.
        assert open(path + ".sha256").read() == HASH
    finally:
        asyncio.run(close(store))
        os.environ.pop("DAIN_PARALLEL_CHUNKS", None)
        os.environ.pop("DAIN_DOWNLOAD_CHUNK", None)


def test_small_shard_single_range(tmp_path) -> None:
    small = DATA[: 2 * 1024 * 1024]  # below the parallel threshold
    tiny_origin = FakeOrigin(data=small)
    os.environ["DAIN_PARALLEL_CHUNKS"] = "4"  # still ≤ 1 because the file is small
    store = make_store(tmp_path, tiny_origin)
    try:
        path = asyncio.run(
            store.ensure_shard(MODEL, SHARD, hashlib.sha256(small).hexdigest())
        )
        assert open(path, "rb").read() == small
        byte_gets = [r for (m, h, p, r) in tiny_origin.requests if m == "GET" and r]
        assert byte_gets == [f"bytes=0-{len(small) - 1}"]  # one ranged stream
    finally:
        asyncio.run(close(store))
        os.environ.pop("DAIN_PARALLEL_CHUNKS", None)


# -- resume ------------------------------------------------------------------------


def test_partial_range_failure_resumes_inside_one_run(tmp_path) -> None:
    """A range that drops mid-stream resumes from its on-disk partial."""
    origin = FakeOrigin()
    target = f"bytes=0-{RD - 1}"  # first of the 4 parallel ranges
    origin.fail_partial[(f"/model/shard/{MODEL}/{SHARD}", target)] = 1500
    os.environ["DAIN_PARALLEL_CHUNKS"] = "4"
    store = make_store(tmp_path, origin)
    try:
        path = asyncio.run(store.ensure_shard(MODEL, SHARD, HASH))
        assert open(path, "rb").read() == DATA
        # FlakyStream drops after the 64 KiB chunk that contained byte 1500,
        # and the retry must resume from there — never restart the range.
        resumes = [
            r
            for (m, h, p, r) in origin.requests
            if m == "GET" and r.startswith("bytes=65536-")
        ]
        assert resumes, "retry must resume from byte 65536, not restart"
    finally:
        asyncio.run(close(store))
        os.environ.pop("DAIN_PARALLEL_CHUNKS", None)


def test_failed_download_keeps_chunks_and_second_run_resumes(tmp_path) -> None:
    """Session crash keeps partials; the next ensure_shard continues mid-file."""
    origin = FakeOrigin()
    tgt = (f"/model/shard/{MODEL}/{SHARD}", f"bytes=0-{RD - 1}")
    origin.fail_exact[tgt] = 10_000  # hard-fail this range every attempt

    os.environ["DAIN_PARALLEL_CHUNKS"] = "4"
    store = make_store(tmp_path, origin)
    with pytest.raises(RuntimeError):
        asyncio.run(store.ensure_shard(MODEL, SHARD, HASH))
    asyncio.run(close(store))

    # Partials survived in the stable .chunks dir next to the final path.
    cache = store.cache_dir
    chunks = os.path.join(cache, MODEL, f"{SHARD}.safetensors.chunks")
    assert os.path.isdir(chunks)
    assert os.path.exists(os.path.join(chunks, "state.json"))
    assert os.path.exists(os.path.join(chunks, "c1.part"))

    # Second run re-fetches only the failed range; the rest come from partials.
    origin.requests.clear()
    origin.fail_exact.clear()
    store2 = make_store(tmp_path, origin)
    try:
        path2 = asyncio.run(store2.ensure_shard(MODEL, SHARD, HASH))
        assert open(path2, "rb").read() == DATA
        ranges = [r for (m, h, p, r) in origin.requests if m == "GET" and "bytes=" in r]
        assert ranges == [f"bytes=0-{RD - 1}"]  # only the empty range was fetched
    finally:
        asyncio.run(close(store2))
        os.environ.pop("DAIN_PARALLEL_CHUNKS", None)


# -- Range-ignoring servers -------------------------------------------------------


def test_range_ignoring_origin_falls_back_to_linear_get(tmp_path) -> None:
    """A source (coordinator or peer) that answers ranged GETs with a full-body
    200 must never be assembled chunk-by-chunk (byte misalignment); the whole
    shard is re-streamed sequentially instead."""
    origin = RangeIgnoringOrigin()
    os.environ["DAIN_PARALLEL_CHUNKS"] = "4"
    store = make_store(tmp_path, origin)
    try:
        path = asyncio.run(store.ensure_shard(MODEL, SHARD, HASH))
        assert open(path, "rb").read() == DATA
        gets = [(p, r) for (m, h, p, r) in origin.requests if m == "GET"]
        ranged = [r for p, r in gets if r.startswith("bytes=")]
        whole = [r for p, r in gets if not r]
        assert ranged, "ranged path must be attempted before the fallback"
        assert whole, "fallback must re-stream the full body un-ranged"
    finally:
        asyncio.run(close(store))
        os.environ.pop("DAIN_PARALLEL_CHUNKS", None)


# -- peer-first sourcing -----------------------------------------------------------


def test_download_uses_peers_before_coordinator(tmp_path) -> None:
    origin = PeeredOrigin()
    os.environ["DAIN_PARALLEL_CHUNKS"] = "4"
    os.environ["DAIN_DOWNLOAD_CHUNK"] = "65536"
    store = make_store(tmp_path, origin)
    try:
        path = asyncio.run(store.ensure_shard(MODEL, SHARD, HASH))
        assert open(path, "rb").read() == DATA
        peer_served = [
            p
            for (m, h, p, r) in origin.requests
            if h == "peerhost" and p.startswith("/peer/shard")
        ]
        assert peer_served, "ranged downloads should hit the advertised peer first"
    finally:
        asyncio.run(close(store))
        os.environ.pop("DAIN_PARALLEL_CHUNKS", None)
        os.environ.pop("DAIN_DOWNLOAD_CHUNK", None)


# -- cached inventory --------------------------------------------------------------


def test_cached_inventory_lists_verified_shards(tmp_path) -> None:
    store = ModelStoreClient(cache_dir=str(tmp_path / "cache"))
    model_dir = os.path.join(store.cache_dir, MODEL)
    os.makedirs(model_dir, exist_ok=True)
    shard_path = os.path.join(model_dir, f"{SHARD}.safetensors")
    with open(shard_path, "wb") as fh:
        fh.write(b"x" * 512)
    with open(shard_path + ".sha256", "w", encoding="ascii") as fh:
        fh.write(HASH)
    refs = store.cached_inventory()
    assert len(refs) == 1
    assert refs[0].model_id == MODEL
    assert refs[0].shard_id == SHARD
    assert refs[0].content_hash == HASH
    assert refs[0].size_bytes == 512
