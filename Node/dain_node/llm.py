"""Model store client + shard verification (spec §8/§11).

Shards are downloaded once per node into a local cache, sha256-verified against
the manifest, and only the shards a stage needs are fetched (network-level
distribution). Stage modules are built from transformers Llama classes with
explicit weight mapping — only assigned layers are ever instantiated.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import os
import secrets
import shutil
import time

import httpx
import torch
from dain_common.model_store import is_safe_model_id
from dain_common.schemas import ModelManifest, ShardRef
from safetensors.torch import load_file
from transformers.cache_utils import DynamicCache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from dain_node.byte_tokenizer import ByteTokenizer
from dain_node.hf_tokenizer import HFTokenizer

log = logging.getLogger("dain.node.llm")

_CHUNK = 1 << 20  # 1 MiB stream chunks

_HASH_SUFFIX = ".sha256"
_PART_SUFFIX = ".part"
_CHUNKS_DIR = ".chunks"
_PEER_CACHE_TTL = 5.0
_RANGE_RETRIES = 3


def _read_chunk_size() -> int:
    """Bigger read chunks keep the Python loop from stalling high-speed links.

    httpx's default 64 KiB frames cap a 1 Gbps transfer at a few hundred MiB/s
    because every byte goes through asyncio loop + sha256.update() per frame;
    8 MiB frames keep the CPU far ahead of the NIC.
    """
    try:
        limit = max(1, min(1 << 25, int(os.environ.get("DAIN_DOWNLOAD_CHUNK", "8388608"))))
    except ValueError:
        limit = 1 << 23
    return limit


def _read_parallel_chunks() -> int:
    """How many byte ranges one shard is split into for concurrent download.

    Each range is pulled over its own connection (from peers first, coordinator
    as the guaranteed last-resort source). LAN transfers thus run at roughly
    `parallel` × single-stream speed before the link saturates.
    """
    try:
        return max(1, min(16, int(os.environ.get("DAIN_PARALLEL_CHUNKS", "4"))))
    except ValueError:
        return 4


def _verbose_progress() -> bool:
    return os.environ.get("DAIN_VERBOSE_PROGRESS", "").lower() in ("1", "true", "yes")


def _sequential_shards() -> bool:
    return os.environ.get("DAIN_SEQUENTIAL_SHARDS", "").lower() in ("1", "true", "yes")


def _split_ranges(total: int, parts: int) -> list[tuple[int, int]]:
    """Partition [0, total) into `parts` contiguous ranges."""
    span = total // max(parts, 1)
    spans = total % max(parts, 1)
    ranges: list[tuple[int, int]] = []
    pos = 0
    for i in range(max(parts, 1)):
        length = span + (1 if i < spans else 0)
        end = pos + length if i < parts - 1 else total
        ranges.append((pos, end))
        pos = end
    return ranges


class _Progress:
    """Shared per-shard progress counter across concurrent range fetchers."""

    def __init__(self, model_id: str, shard_id: str, total: int) -> None:
        self.model_id = model_id
        self.shard_id = shard_id
        self.total = total
        self.done = 0
        self._last_pct = -1

    def add(self, n: int) -> None:
        self.done += n
        pct = int(100 * self.done / self.total) if self.total > 0 else 100
        if pct // 5 != self._last_pct // 5 or pct == 100:
            elapsed = max(time.monotonic() - self._t0, 1e-6)
            log.info(
                "shard_download model=%s shard=%s progress=%d%% bytes=%d/%d mbps=%.0f",
                self.model_id,
                self.shard_id,
                pct,
                self.done,
                self.total,
                self.done / elapsed / 1e6,
            )
            self._last_pct = pct

    _t0 = time.monotonic()


class ModelStoreClient:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        self.base_url: str | None = None
        self.auth: dict[str, str] = {}
        self.peer_token: str | None = None
        self._manifests: dict[str, ModelManifest] = {}
        self._client: httpx.AsyncClient | None = None
        self._shard_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._peer_cache: dict[tuple[str, str], tuple[tuple[str, ...], float]] = {}

    async def _http(self) -> httpx.AsyncClient:
        """One long-lived client for the whole node: connection reuse across
        shards is what makes multi-GB transfers fast (TCP window/latency
        amortized per stream, no TLS/connect per request). `trust_env=False` so
        a stray HTTP_PROXY on the host can never tunnel LAN transfers."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
                trust_env=False,
                limits=httpx.Limits(
                    max_connections=32, max_keepalive_connections=16, keepalive_expiry=15.0
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and self._client.is_closed:
            return
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def set_base_url(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def set_auth(self, node_id: str, node_token: str) -> None:
        self.auth = {"X-Node-Id": node_id, "X-Node-Token": node_token}

    def set_peer_token(self, join_token: str) -> None:
        self.peer_token = join_token

    def _cached_hash_ok(self, path: str, content_hash: str) -> bool:
        """Fast cache hit: trust a sidecar digest instead of re-hashing the file.

        Re-hashing a ~1 GB shard on every job costs ~seconds of disk+CPU per
        shard; the sidecar is written right after a verified download/verify, so
        it is only stale when the file is externally replaced.
        """
        sidecar = path + _HASH_SUFFIX
        try:
            with open(sidecar, encoding="ascii") as fh:
                return fh.read(64).strip() == content_hash
        except OSError:
            return False

    def _write_hash_sidecar(self, path: str, content_hash: str) -> None:
        with open(path + _HASH_SUFFIX, "w", encoding="ascii") as fh:
            fh.write(content_hash)

    async def _verify(self, path: str, content_hash: str) -> bool:
        if self._cached_hash_ok(path, content_hash):
            return True
        try:
            if await asyncio.to_thread(self._hash_file, path) == content_hash:
                self._write_hash_sidecar(path, content_hash)
                return True
        except OSError:
            pass
        return False

    # -- local inventory (peer advertisement) -------------------------------------

    def cached_inventory(self) -> tuple[ShardRef, ...]:
        """Every shard this node can currently serve to peers.

        Built from the local cache directory. Hashes come from the sidecars we
        write after every verified download; a file without a sidecar is hashed
        once and remembered (keyed by path) so the agent never re-hashes a
        multi-GB shard on every heartbeat. Only files the node itself verified
        are advertised — anything else could poison the peer network.
        """
        refs: list[ShardRef] = []
        hashes = getattr(self, "_inventory_hashes", None)
        if hashes is None:
            hashes = {}
            self._inventory_hashes = hashes
        if not os.path.isdir(self.cache_dir):
            return ()
        for model_id in sorted(os.listdir(self.cache_dir)):
            if not is_safe_model_id(model_id):
                continue
            model_dir = os.path.join(self.cache_dir, model_id)
            if not os.path.isdir(model_dir):
                continue
            for fname in sorted(os.listdir(model_dir)):
                if not fname.endswith(".safetensors"):
                    continue
                shard_path = os.path.join(model_dir, fname)
                if not os.path.isfile(shard_path):
                    continue
                shard_id = fname[: -len(".safetensors")]
                if not is_safe_model_id(shard_id):
                    continue
                content_hash = self._verifiable_hash(shard_path, hashes)
                if content_hash is None:
                    continue
                refs.append(
                    ShardRef(
                        model_id=model_id,
                        shard_id=shard_id,
                        content_hash=content_hash,
                        size_bytes=os.path.getsize(shard_path),
                    )
                )
        return tuple(refs)

    def _verifiable_hash(self, path: str, cache: dict[str, str]) -> str | None:
        sidecar = path + _HASH_SUFFIX
        try:
            with open(sidecar, encoding="ascii") as fh:
                digest = fh.read(64).strip()
                if digest:
                    return digest
        except OSError:
            pass
        # Sidecar-less file (pre-verified grace): cache by (path, size, mtime)
        # so a re-exported shard at the same path never advertises a stale hash.
        stat = None
        try:
            stat = os.stat(path)
        except OSError:
            return None
        key = (path, stat.st_size, stat.st_mtime_ns)
        digest = cache.get(key)
        if digest:
            return digest
        try:
            digest = self._hash_file(path)
        except OSError:
            return None
        cache[key] = digest
        return digest

    async def fetch_manifest(self, model_id: str, *, refresh: bool = False) -> ModelManifest:
        if not refresh and model_id in self._manifests:
            return self._manifests[model_id]
        if self.base_url is None:
            raise RuntimeError("model store base URL not set yet")
        client = await self._http()
        response = await client.get(f"{self.base_url}/manifest/{model_id}", headers=self.auth)
        response.raise_for_status()
        manifest = ModelManifest.model_validate(response.json())
        self._manifests[model_id] = manifest
        log.info(
            "manifest_refreshed model=%s layers=%d shards=%d refresh=%s",
            model_id,
            manifest.layers,
            len(manifest.shards),
            refresh,
        )
        return manifest

    # -- peer inventory -------------------------------------------------------------

    async def _query_peers(self, model_id: str, shard_id: str) -> tuple[str, ...]:
        """Which online nodes already hold this shard (coordinator-derived).

        "almost instant on LAN": if any sibling has the bytes, every download
        can come from peers at LAN speed instead of one coordinator stream.
        """
        key = (model_id, shard_id)
        now = time.monotonic()
        cached = self._peer_cache.get(key)
        if cached and now - cached[1] < _PEER_CACHE_TTL:
            return cached[0]
        peers: tuple[str, ...] = ()
        try:
            client = await self._http()
            response = await client.get(
                f"{self.base_url}/peers/{model_id}/{shard_id}", headers=self.auth, timeout=10.0
            )
            if response.status_code == 200:
                peers = tuple(response.json().get("peers", []))
        except Exception:  # noqa: BLE001 — peers are an optimization, never fatal
            log.debug("peer_query_failed model=%s shard=%s", model_id, shard_id)
        self._peer_cache[key] = (peers, now)
        if peers:
            log.info(
                "peer_hits model=%s shard=%s sources=%d", model_id, shard_id, len(peers)
            )
        return peers

    # -- shard download -----------------------------------------------------------

    async def ensure_shard(self, model_id: str, shard_id: str, content_hash: str) -> str:
        """Download + verify one shard; returns the cached file path.

        Faster than it used to be, three ways:
        - the byte stream is split into ranges pulled concurrently from peers
          (+ the coordinator as last resort);
        - a partly missing file resumes (per-range) instead of restarting;
        - the connection is reused across shards via one shared httpx client.
        """
        model_cache = os.path.join(self.cache_dir, model_id)
        path = os.path.join(model_cache, f"{shard_id}.safetensors")
        if os.path.exists(path):
            if await self._verify(path, content_hash):
                log.info("shard_hit model=%s shard=%s path=%s", model_id, shard_id, path)
                return path
        if self.base_url is None:
            raise RuntimeError("model store base URL not set yet")
        os.makedirs(model_cache, exist_ok=True)
        lock = self._shard_locks.setdefault((model_id, shard_id), asyncio.Lock())
        async with lock:
            # Another task may have finished the same shard while we waited.
            if os.path.exists(path) and await self._verify(path, content_hash):
                log.info("shard_hit_race model=%s shard=%s", model_id, shard_id)
                return path
            started = time.monotonic()
            # Stable per-shard chunk dir (keyed to the *final* path): a failed
            # session leaves its partials here so the next attempt resumes
            # mid-file instead of restarting. The assembled tmp is ephemeral.
            chunk_dir = path + _CHUNKS_DIR
            tmp = f"{path}.{secrets.token_hex(4)}.tmp"
            try:
                await self._download_shard(tmp, chunk_dir, model_id, shard_id, content_hash)
                if not await self._verify(tmp, content_hash):
                    raise RuntimeError(f"shard {shard_id} failed hash verification")
                os.replace(tmp, path)
                self._write_hash_sidecar(path, content_hash)
                # Success: sweep the per-chunk partials. On failure they stay in
                # place so the next attempt resumes mid-file instead of restarting.
                with contextlib.suppress(OSError, shutil.Error):
                    shutil.rmtree(chunk_dir, ignore_errors=True)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.remove(tmp)
                raise
        size = os.path.getsize(path)
        mbps = size / max(time.monotonic() - started, 1e-6) / 1e6
        log.info(
            "shard_cached model=%s shard=%s bytes=%d mbps=%.0f",
            model_id,
            shard_id,
            size,
            mbps,
        )
        return path

    async def _download_shard(
        self, dst: str, chunk_dir: str, model_id: str, shard_id: str, content_hash: str
    ) -> None:
        """Fill `dst` with the complete, verified shard bytes.

        Strategy: probe the shard's size, then fetch its byte ranges
        concurrently (each range over its own connection, peers first,
        coordinator last). Ranges land in `chunk_dir` (stable across attempts
        so a failed one resumes); on success the ranges are assembled into
        `dst` in order.
        """
        client = await self._http()
        total = await self._probe_size(client, model_id, shard_id)
        peers = await self._query_peers(model_id, shard_id)
        chunk_size = _read_chunk_size()
        parallel = _read_parallel_chunks() if total is not None and total > (8 << 20) else 1
        sources: list[tuple[str, bool]] = [(p, False) for p in peers] + [
            (self.base_url, True)
        ]

        if total is None:
            # No size probe (coordinator too old / error): plain linear GET.
            await self._linear_fallback(client, sources, model_id, shard_id, dst, chunk_size)
            return

        ranges = _split_ranges(total, parallel)
        if not _chunk_state_ok(chunk_dir, ranges):
            with contextlib.suppress(shutil.Error):
                shutil.rmtree(chunk_dir, ignore_errors=True)
        os.makedirs(chunk_dir, exist_ok=True)
        _write_chunk_state(chunk_dir, ranges)

        prog = _Progress(model_id, shard_id, total)
        write_lock = asyncio.Lock()
        fetchers = [
            asyncio.create_task(
                self._fetch_range(
                    client,
                    chunk_dir,
                    write_lock,
                    prog,
                    model_id,
                    shard_id,
                    [start, end],
                    sources,
                    chunk_size,
                    i,
                )
            )
            for i, (start, end) in enumerate(ranges)
        ]
        try:
            results = await asyncio.gather(*fetchers)
        finally:
            for task in fetchers:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*fetchers, return_exceptions=True)
        for ok, err in results:
            if not ok:
                raise RuntimeError(
                    f"shard {shard_id} range fetch failed: {err}"
                ) from err
        _assemble_chunks(chunk_dir, dst, total)

    async def _probe_size(
        self, client: httpx.AsyncClient, model_id: str, shard_id: str
    ) -> int | None:
        try:
            response = await client.head(
                f"{self.base_url}/shard/{model_id}/{shard_id}", headers=self.auth, timeout=10.0
            )
            response.raise_for_status()
            length = response.headers.get("content-length")
            return int(length) if length else None
        except Exception:  # noqa: BLE001
            return None

    async def _linear_fallback(
        self,
        client: httpx.AsyncClient,
        sources: list[tuple[str, bool]],
        model_id: str,
        shard_id: str,
        dst: str,
        chunk_size: int,
    ) -> None:
        """Single un-ranged stream (no size knowledge). Retries a few sources."""
        resume = os.path.getsize(dst) if os.path.exists(dst) else 0
        last_err: Exception | None = None
        for _attempt in range(_RANGE_RETRIES):
            for base, is_coord in sources:
                if resume:
                    log.info(
                        "shard_resume model=%s shard=%s at=%d", model_id, shard_id, resume
                    )
                try:
                    await self._stream_source(
                        client, dst, base, is_coord, model_id, shard_id, resume, chunk_size
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    with contextlib.suppress(OSError):
                        resume = os.path.getsize(dst)
        raise RuntimeError(f"shard {shard_id} stream failed: {last_err}") from last_err

    async def _stream_source(
        self,
        client: httpx.AsyncClient,
        dst: str,
        base: str,
        is_coord: bool,
        model_id: str,
        shard_id: str,
        resume: int,
        chunk_size: int,
    ) -> None:
        headers = {"Range": f"bytes={resume}-"} if resume else {}
        if is_coord:
            request_headers = {**self.auth, **headers}
            url = f"{base}/shard/{model_id}/{shard_id}"
        else:
            request_headers = {"X-Peer-Token": self.peer_token or "", **headers}
            url = f"{base}/peer/shard/{model_id}/{shard_id}"
        async with client.stream("GET", url, headers=request_headers) as response:
            response.raise_for_status()
            with open(dst, "ab") as fh:
                async for chunk in response.aiter_bytes(chunk_size):
                    fh.write(chunk)

    async def _fetch_range(
        self,
        client: httpx.AsyncClient,
        chunk_dir: str,
        write_lock: asyncio.Lock,
        prog: _Progress,
        model_id: str,
        shard_id: str,
        byte_range: list[int],
        sources: list[tuple[str, bool]],
        chunk_size: int,
        index: int,
    ) -> tuple[bool, Exception | None]:
        """Download one byte range into chunk_dir/c{index}.part.

        Resumes at the chunk's current on-disk size; tries each source in
        order (peers first) with a small backoff, so a dropping hotspot link
        never restarts the file from byte 0.
        """
        start, end = byte_range
        chunk_path = os.path.join(chunk_dir, f"c{index}.part")
        local = os.path.getsize(chunk_path) if os.path.exists(chunk_path) else 0
        if local >= end - start:
            prog.add(end - start)
            return True, None
        last_err: Exception | None = None
        for attempt in range(_RANGE_RETRIES):
            try:
                # Inner loop: each byte-range GET opens a fresh connection, so
                # a mid-stream drop just abandons one source and the *next*
                # request resumes from wherever we wrote to disk.
                for base, is_coord in sources:
                    while local < end - start:
                        headers = {"Range": f"bytes={start + local}-{end - 1}"}
                        if is_coord:
                            request_headers = {**self.auth, **headers}
                            url = f"{base}/shard/{model_id}/{shard_id}"
                        else:
                            request_headers = {
                                "X-Peer-Token": self.peer_token or "", **headers
                            }
                            url = f"{base}/peer/shard/{model_id}/{shard_id}"
                        async with client.stream(
                            "GET", url, headers=request_headers
                        ) as response:
                            response.raise_for_status()
                            async with write_lock:
                                with open(chunk_path, "ab") as fh:
                                    # aiter_raw yields each chunk the transport
                                    # delivers. aiter_bytes() would buffer into
                                    # an 8 MiB frame and *discard* the partial
                                    # on a mid-stream error, so a dropped
                                    # hotspot would restart from byte 0.
                                    async for chunk in response.aiter_raw():
                                        fh.write(chunk)
                                        local += len(chunk)
                                        prog.add(len(chunk))
                    if local >= end - start:
                        return True, None
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning(
                    "range_retry model=%s shard=%s range=%d-%d source=%s err=%s",
                    model_id,
                    shard_id,
                    start + local,
                    end,
                    base,
                    exc,
                )
                with contextlib.suppress(OSError):
                    local = os.path.getsize(chunk_path)
            await asyncio.sleep(0.4 * (attempt + 1))
        return local >= end - start, last_err

    async def ensure_tokenizer(
        self, model_id: str, tokenizer_file: str, tokenizer_hash: str
    ) -> str:
        """Download + verify the model's tokenizer.json; returns its cache path."""
        model_cache = os.path.join(self.cache_dir, model_id)
        file_name = os.path.basename(tokenizer_file)
        path = os.path.join(model_cache, file_name)
        if os.path.exists(path):
            if await self._verify(path, tokenizer_hash):
                log.info("tokenizer_hit model=%s file=%s path=%s", model_id, file_name, path)
                return path
        if self.base_url is None:
            raise RuntimeError("model store base URL not set yet")
        os.makedirs(model_cache, exist_ok=True)
        tmp = f"{path}.{secrets.token_hex(4)}.tmp"
        hasher = hashlib.sha256()
        chunk_size = _read_chunk_size()
        try:
            client = await self._http()
            async with client.stream(
                "GET", f"{self.base_url}/tokenizer/{model_id}", headers=self.auth
            ) as response:
                response.raise_for_status()
                with open(tmp, "wb") as fh:
                    async for chunk in response.aiter_bytes(chunk_size):
                        fh.write(chunk)
                        hasher.update(chunk)
            if hasher.hexdigest() != tokenizer_hash:
                raise RuntimeError(f"tokenizer {file_name} failed hash verification")
            os.replace(tmp, path)
            self._write_hash_sidecar(path, tokenizer_hash)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise
        log.info(
            "tokenizer_cached model=%s file=%s bytes=%d",
            model_id,
            file_name,
            os.path.getsize(path),
        )
        return path

    @staticmethod
    def _hash_file(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


def _chunk_state_ok(chunk_dir: str, ranges: list[tuple[int, int]]) -> bool:
    try:
        with open(os.path.join(chunk_dir, "state.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        return [tuple(r) for r in data.get("ranges", [])] == ranges
    except (OSError, ValueError, TypeError):
        return False


def _write_chunk_state(chunk_dir: str, ranges: list[tuple[int, int]]) -> None:
    os.makedirs(chunk_dir, exist_ok=True)
    with open(os.path.join(chunk_dir, "state.json"), "w", encoding="utf-8") as fh:
        json.dump({"ranges": [list(r) for r in ranges]}, fh)


def _assemble_chunks(chunk_dir: str, dst: str, total: int) -> None:
    """Concatenate c0, c1, … into `dst` (+ a length sanity check)."""
    with open(dst, "wb") as out:
        written = 0
        for fh_name in sorted(os.listdir(chunk_dir)):
            if not fh_name.startswith("c") or not fh_name.endswith(".part"):
                continue
            with open(os.path.join(chunk_dir, fh_name), "rb") as fh:
                shutil.copyfileobj(fh, out, length=1 << 20)
            written += os.path.getsize(os.path.join(chunk_dir, fh_name))
    if written != total:
        raise RuntimeError(f"assembled {written} bytes, expected {total}")


def causal_mask(dtype: torch.dtype, q_len: int, past_len: int) -> torch.Tensor:
    """Additive 4D causal mask [1, 1, q_len, past+q_len] — no HF internals."""
    total = past_len + q_len
    q_idx = torch.arange(past_len, total).unsqueeze(1)
    k_idx = torch.arange(total).unsqueeze(0)
    mask = torch.zeros(1, 1, q_len, total, dtype=dtype)
    mask.masked_fill_(k_idx > q_idx, torch.finfo(dtype).min)
    return mask


class StageModel:
    """Owns one contiguous layer range of a Llama model + its KV caches.

    Weight mapping is explicit: only the parameters of assigned layers (plus
    embed on the first stage, norm/lm_head on the last) are instantiated and
    loaded — memory is proportional to the shard, not the model.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        layer_start: int,
        layer_end: int,
        state: dict[str, torch.Tensor],
        *,
        device: str = "cpu",
        tokenizer_path: str | None = None,
    ) -> None:
        if not 0 <= layer_start <= layer_end < manifest.layers:
            raise ValueError(f"invalid layer range [{layer_start}, {layer_end}]")
        self.manifest = manifest
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.first = layer_start == 0
        self.last = layer_end == manifest.layers - 1
        cfg = LlamaConfig(
            vocab_size=manifest.vocab_size,
            hidden_size=manifest.hidden,
            intermediate_size=manifest.intermediate,
            num_hidden_layers=manifest.layers,
            num_attention_heads=manifest.heads,
            num_key_value_heads=manifest.kv_heads,
            rope_theta=manifest.rope_theta,
            tie_word_embeddings=False,
        )
        self.cfg = cfg
        self.device = device
        dtype_label = manifest.dtype or "fp32"
        dtype = torch.float16 if dtype_label == "fp16" else torch.float32
        self.dtype = dtype
        self.dtype_label = "fp16" if dtype is torch.float16 else "fp32"

        self.layers: list[LlamaDecoderLayer] = []
        for global_idx in range(layer_start, layer_end + 1):
            layer = LlamaDecoderLayer(cfg, layer_idx=global_idx).to(device=device, dtype=dtype)
            local_idx = global_idx - layer_start
            layer.self_attn.layer_idx = local_idx
            prefix = f"model.layers.{global_idx}."
            for name, param in layer.named_parameters():
                key = prefix + name
                if key not in state:
                    raise KeyError(f"shard missing weight {key}")
                with torch.no_grad():
                    param.copy_(state[key].to(dtype))
            layer.eval()
            self.layers.append(layer)
        self.rotary = LlamaRotaryEmbedding(config=cfg, device=device)
        self.embed = None
        if self.first:
            self.embed = torch.nn.Embedding(manifest.vocab_size, manifest.hidden).to(
                device=device, dtype=dtype
            )
            with torch.no_grad():
                self.embed.weight.copy_(state["model.embed_tokens.weight"].to(dtype))
        self.norm = None
        self.lm_head = None
        if self.last:
            self.norm = LlamaRMSNorm(manifest.hidden, eps=cfg.rms_norm_eps).to(
                device=device, dtype=dtype
            )
            self.lm_head = torch.nn.Linear(manifest.hidden, manifest.vocab_size, bias=False).to(
                device=device, dtype=dtype
            )
            with torch.no_grad():
                self.norm.weight.copy_(state["model.norm.weight"].to(dtype))
                self.lm_head.weight.copy_(state["lm_head.weight"].to(dtype))
        self.cache: DynamicCache | None = None
        if tokenizer_path is not None:
            self.tokenizer = HFTokenizer(tokenizer_path)
        else:
            self.tokenizer = ByteTokenizer(manifest.vocab_size, manifest.eos_token_id)

    # -- cache lifecycle ---------------------------------------------------------

    def begin_job(self) -> None:
        self.cache = DynamicCache()

    def end_job(self) -> None:
        self.cache = None

    def _past_len(self) -> int:
        return self.cache.get_seq_length() if self.cache is not None else 0

    # -- forward paths -------------------------------------------------------------

    def _run_layers(self, hidden: torch.Tensor) -> torch.Tensor:
        past = self._past_len()
        batch, seq, _ = hidden.shape
        cache_position = torch.arange(past, past + seq, device=hidden.device)
        position_ids = cache_position.unsqueeze(0)
        mask = causal_mask(hidden.dtype, seq, past)
        position_embeddings = self.rotary(hidden, position_ids)
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_value=self.cache,
                use_cache=self.cache is not None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        return hidden

    def forward_ids(self, input_ids: list[int]) -> torch.Tensor:
        """Entry stage: embed token ids → run layers → hidden [1, T, H]."""
        if not self.first:
            raise RuntimeError("forward_ids requires the first stage")
        hidden = self.embed(torch.tensor([input_ids], device=self.device, dtype=torch.long))
        return self._run_layers(hidden)

    def forward_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Middle/last stage: run layers on inbound activations."""
        return self._run_layers(hidden)

    def logits_from(self, hidden: torch.Tensor) -> torch.Tensor:
        """Last stage: final norm + lm_head → logits [1, T, vocab]."""
        if not self.last:
            raise RuntimeError("logits_from requires the last stage")
        return self.lm_head(self.norm(hidden))

    def next_token_logits(self, token_id: int) -> torch.Tensor:
        """Entry stage decode step: embed one token → logits for that position."""
        hidden = self.embed(torch.tensor([[token_id]], device=self.device, dtype=torch.long))
        hidden = self._run_layers(hidden)
        return self.lm_head(self.norm(hidden))[:, -1, :]

    def embed_one(self, token_id: int) -> torch.Tensor:
        """Distributed entry decode step: embed one token → hidden after layers."""
        hidden = self.embed(torch.tensor([[token_id]], device=self.device, dtype=torch.long))
        return self._run_layers(hidden)

    def next_token_logits_full(self, input_ids: list[int]) -> torch.Tensor:
        """Full-model step (single-stage): embed → layers → logits of last position."""
        hidden = self.embed(torch.tensor([input_ids], device=self.device, dtype=torch.long))
        hidden = self._run_layers(hidden)
        return self.lm_head(self.norm(hidden))[:, -1, :]


def sample_token(
    logits: torch.Tensor, temperature: float, generator: torch.Generator | None
) -> int:
    """Greedy at temperature 0; temperature-scaled multinomial otherwise."""
    flat = logits.reshape(-1, logits.shape[-1])[-1]
    if temperature <= 0:
        return int(torch.argmax(flat).item())
    probs = torch.softmax(flat / temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


async def fetch_stage(
    store: ModelStoreClient,
    manifest: ModelManifest,
    layer_start: int,
    layer_end: int,
) -> tuple[StageModel, dict[str, str]]:
    """Download the shards covering [layer_start, layer_end] and build the stage."""
    needed: list[tuple[str, str]] = []
    for shard in manifest.shards:
        if shard.layer_start is None or shard.layer_end is None:
            continue
        if shard.layer_start <= layer_end and shard.layer_end >= layer_start:
            needed.append((shard.shard_id, shard.content_hash))
    log.info(
        "fetch_stage model=%s layers=[%d,%d] needed_shards=%s",
        manifest.model_id,
        layer_start,
        layer_end,
        ",".join(shard_id for shard_id, _ in needed),
    )
    if _sequential_shards() and len(needed) > 1:
        # Constrained links (a laptop hotspot): concurrent full-shard download
        # streams contend and trip read timeouts. Fetch one at a time instead.
        log.info("fetch_stage sequential model=%s shards=%d", manifest.model_id, len(needed))
        paths = []
        for shard_id, content_hash in needed:
            paths.append(await store.ensure_shard(manifest.model_id, shard_id, content_hash))
    else:
        # Parallel downloads: each stage pulls several ~0.5–1 GB shards over LAN,
        # so serializing them would lock the stage behind ~N× the link time.
        paths = (
            await asyncio.gather(
                *(
                    store.ensure_shard(manifest.model_id, shard_id, content_hash)
                    for shard_id, content_hash in needed
                )
            )
            if needed
            else []
        )
    shard_paths = dict(zip((shard_id for shard_id, _ in needed), paths, strict=True))
    state: dict[str, torch.Tensor] = {}
    for path in paths:
        state.update(load_file(path))
    tokenizer_path: str | None = None
    if manifest.tokenizer_file is not None and manifest.tokenizer_hash is not None:
        tokenizer_path = await store.ensure_tokenizer(
            manifest.model_id, manifest.tokenizer_file, manifest.tokenizer_hash
        )
    loop = asyncio.get_running_loop()
    build = functools.partial(
        StageModel, manifest, layer_start, layer_end, state, tokenizer_path=tokenizer_path
    )
    stage = await loop.run_in_executor(None, build)
    return stage, shard_paths
