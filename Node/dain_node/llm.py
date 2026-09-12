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

#: Shard file extensions by ShardRef.format (plan §2.1). Quantized shards are
#: `torch_pt` because TorchAO's AffineQuantizedTensor packing metadata only
#: survives torch.save/torch.load; fp16/fp32 shards stay safetensors.
_SHARD_EXTS = {
    None: ".safetensors",
    "safetensors": ".safetensors",
    "torch_pt": ".pt",
}

_ACTIVATION_DTYPES_LABEL = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

#: Packing layouts recorded in QuantizationSpec (mirrors model_export.LAYOUT_*).
_LAYOUT_TENSOR_CORE_TILED = "tensor_core_tiled"
_LAYOUT_INT4_CPU = "int4_cpu"


def _activation_dtype(label: str) -> torch.dtype:
    return _ACTIVATION_DTYPES_LABEL.get(label, torch.float32)


def select_device(manifest: ModelManifest) -> str:
    """Choose the execution device for a manifest (CUDA when available).

    The packing layout dictates where a *quantized* model must run: TorchAO's
    ``TensorCoreTiledLayout`` weights only execute on CUDA (tinygemm), while
    ``Int4CPULayout`` runs on CPU — silently placing a tensor_core_tiled model
    on CPU (or int4_cpu on a GPU) is a runtime failure, not an optimization.
    Legacy fp16/fp32 manifests prefer CUDA when present and fall back to CPU.
    """
    quant = getattr(manifest, "quantization", None)
    if quant is not None and quant.is_quantized:
        layout = getattr(quant, "packing_layout", None)
        if layout == _LAYOUT_TENSOR_CORE_TILED:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"model {manifest.model_id} requires CUDA "
                    f"(packing_layout={layout!r}) but this node has no GPU"
                )
            return "cuda:0"
        return "cpu"  # int4_cpu (and legacy/auto layouts) execute on CPU
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# -- inference backend abstraction (S22) --------------------------------------


class InferenceBackend:
    """How a node materializes and runs one model family.

    The exporter is the compiler; the backend is the runtime. `is_quantized`
    tells the stage builder that weight parameters must be *assigned* (the
    packed AffineQuantizedTensor) instead of fp16/fp32-copied.
    """

    name: str = "fp16"
    is_quantized: bool = False

    def deserialize_shard(self, path: str) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class TorchFp16Backend(InferenceBackend):
    name = "torch_fp16"
    is_quantized = False

    def deserialize_shard(self, path: str) -> dict[str, torch.Tensor]:
        return load_file(path)


class TorchAOInt4Backend(InferenceBackend):
    name = "torchao_int4"
    is_quantized = True

    def deserialize_shard(self, path: str) -> dict[str, torch.Tensor]:
        # weights_only=False: the pickle materializes AffineQuantizedTensor
        # subclasses that carry the INT4 packing metadata (plan §2.1).
        return torch.load(path, weights_only=False)


def select_backend(manifest: ModelManifest) -> InferenceBackend:
    """Choose the execution backend for a manifest (exported vs legacy)."""
    quant = getattr(manifest, "quantization", None)
    if manifest.format == "torch_pt" or (
        quant is not None and getattr(quant, "is_quantized", False)
    ):
        return TorchAOInt4Backend()
    return TorchFp16Backend()


class RangeIgnoredError(RuntimeError):
    """Raised when a source returns HTTP 200 to a Range request (body not byte-aligned)."""


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
        self._t0 = time.monotonic()

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
                ext = next((e for e in (".safetensors", ".pt") if fname.endswith(e)), None)
                if ext is None:
                    continue
                shard_path = os.path.join(model_dir, fname)
                if not os.path.isfile(shard_path):
                    continue
                shard_id = fname[: -len(ext)]
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
                        format="torch_pt" if ext == ".pt" else None,
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

    async def ensure_shard(
        self,
        model_id: str,
        shard_id: str,
        content_hash: str,
        format: str | None = None,
    ) -> str:
        """Download + verify one shard; returns the cached file path.

        Faster than it used to be, three ways:
        - the byte stream is split into ranges pulled concurrently from peers
          (+ the coordinator as last resort);
        - a partly missing file resumes (per-range) instead of restarting;
        - the connection is reused across shards via one shared httpx client.
        Quantized shards (``format="torch_pt"``) land as ``.pt`` files; legacy
        shards as ``.safetensors`` — the extension is part of the address.
        """
        model_cache = os.path.join(self.cache_dir, model_id)
        ext = _SHARD_EXTS.get(format, ".safetensors")
        path = os.path.join(model_cache, f"{shard_id}{ext}")
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
                self._shard_locks.pop((model_id, shard_id), None)
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
                # Shard is now cached: drop the per-shard lock so the dict does
                # not grow with every model ever downloaded. Any task already
                # waiting on this (old) lock re-checks the cache and no-ops;
                # later arrivals take the fast cache-hit path at the top.
                self._shard_locks.pop((model_id, shard_id), None)
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
        fetchers = [
            asyncio.create_task(
                self._fetch_range(
                    client,
                    chunk_dir,
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
            try:
                results = await asyncio.gather(*fetchers)
            finally:
                for task in fetchers:
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.gather(*fetchers, return_exceptions=True)
            for ok, err in results:
                if not ok:
                    if all(
                        not ok and isinstance(e, RangeIgnoredError)
                        for ok, e in results
                    ):
                        raise RangeIgnoredError(
                            f"shard {shard_id}: every source ignored Range"
                        ) from err
                    raise RuntimeError(
                        f"shard {shard_id} range fetch failed: {err}"
                    ) from err
            _assemble_chunks(chunk_dir, dst, total)
        except RangeIgnoredError:
            # Every source ignored the Range header (full-body HTTP 200), so
            # ranged chunks would be misaligned — never assemble them whole-body.
            # Fall back to a single sequential stream instead.
            log.warning(
                "shard_range_fallback model=%s shard=%s — sources ignored Range; linear GET",
                model_id,
                shard_id,
            )
            with contextlib.suppress(OSError):
                os.remove(dst)
            await self._linear_fallback(client, sources, model_id, shard_id, dst, chunk_size)

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
            if resume and response.status_code != 206:
                # Resume sent a Range header; a full-body 200 would append at
                # the wrong offset. Skip the source (which the retry loop does).
                raise RangeIgnoredError(f"{base} ignored Range (HTTP {response.status_code})")
            with open(dst, "ab") as fh:
                async for chunk in response.aiter_bytes(chunk_size):
                    fh.write(chunk)

    async def _fetch_range(
        self,
        client: httpx.AsyncClient,
        chunk_dir: str,
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
        never restarts the file from byte 0. Each chunk owns its file, so the
        parallel fetchers never contend on a shared write cursor.
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
                            if response.status_code != 206:
                                # Range-ignoring server: a full-body 200 would
                                # land at the wrong offset on resume and corrupt
                                # the chunk. Skip the source; the caller's
                                # linear fallback covers these servers.
                                raise RangeIgnoredError(
                                    f"{base} ignored Range (HTTP {response.status_code})"
                                )
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


def _chunk_sort_key(fh_name: str) -> tuple[int, str]:
    """Order chunk files numerically (c2.part < c10.part), not lexicographically."""
    if fh_name.startswith("c") and fh_name.endswith(".part"):
        try:
            return int(fh_name[1 : -len(".part")]), fh_name
        except ValueError:
            pass
    return (2**31, fh_name)


def _assemble_chunks(chunk_dir: str, dst: str, total: int) -> None:
    """Concatenate c0, c1, … into `dst` (+ a length sanity check)."""
    with open(dst, "wb") as out:
        written = 0
        for fh_name in sorted(os.listdir(chunk_dir), key=_chunk_sort_key):
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
        backend: InferenceBackend | None = None,
    ) -> None:
        if not 0 <= layer_start <= layer_end < manifest.layers:
            raise ValueError(f"invalid layer range [{layer_start}, {layer_end}]")
        self.manifest = manifest
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.first = layer_start == 0
        self.last = layer_end == manifest.layers - 1
        architecture_config = manifest.architecture_config
        cfg = LlamaConfig(
            vocab_size=manifest.vocab_size,
            hidden_size=manifest.hidden,
            intermediate_size=manifest.intermediate,
            num_hidden_layers=manifest.layers,
            num_attention_heads=manifest.heads,
            num_key_value_heads=manifest.kv_heads,
            hidden_act=architecture_config.get("hidden_act", "silu"),
            max_position_embeddings=architecture_config.get("max_position_embeddings", 2048),
            rms_norm_eps=architecture_config.get("rms_norm_eps", 1e-6),
            rope_theta=manifest.rope_theta,
            rope_scaling=architecture_config.get("rope_scaling"),
            attention_bias=architecture_config.get("attention_bias", False),
            attention_dropout=architecture_config.get("attention_dropout", 0.0),
            mlp_bias=architecture_config.get("mlp_bias", False),
            head_dim=architecture_config.get("head_dim"),
            tie_word_embeddings=architecture_config.get("tie_word_embeddings", False),
        )
        self.cfg = cfg
        self.device = device
        self.backend = backend or select_backend(manifest)
        quant = getattr(manifest, "quantization", None)
        dtype_label = (
            quant.activation_dtype
            if self.backend.is_quantized and quant is not None
            else (manifest.dtype or "fp32")
        )
        dtype = _activation_dtype(dtype_label)
        self.dtype = dtype
        self.dtype_label = dtype_label

        self.layers: list[LlamaDecoderLayer] = []
        for global_idx in range(layer_start, layer_end + 1):
            layer = LlamaDecoderLayer(cfg, layer_idx=global_idx).to(device=device, dtype=dtype)
            local_idx = global_idx - layer_start
            layer.self_attn.layer_idx = local_idx
            prefix = f"model.layers.{global_idx}."
            with torch.no_grad():
                if self.backend.is_quantized:
                    # Quantized projections arrive as AffineQuantizedTensor
                    # parameters: assign them in place (copy_ cannot handle the
                    # packed subclass) and move each AQT to the stage device —
                    # a tensor_core_tiled model must land on CUDA, int4_cpu on
                    # CPU; `.to()` on an AQT transfers the packed payload.
                    subset = {
                        name: (
                            state[prefix + name]
                            if device == "cpu"
                            else state[prefix + name].to(device)
                        )
                        for name, _param in layer.named_parameters()
                        if prefix + name in state
                    }
                    missing = [n for n, _ in layer.named_parameters() if prefix + n not in state]
                    if missing:
                        raise KeyError(f"shard missing weights {missing!r}")
                    layer.load_state_dict(subset, strict=False, assign=True)
                else:
                    for name, param in layer.named_parameters():
                        key = prefix + name
                        if key not in state:
                            raise KeyError(f"shard missing weight {key}")
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
                if self.backend.is_quantized:
                    self.embed.weight = torch.nn.Parameter(
                        state["model.embed_tokens.weight"].to(device), requires_grad=False
                    )
                else:
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
                if self.backend.is_quantized:
                    self.norm.weight = torch.nn.Parameter(
                        state["model.norm.weight"].to(device), requires_grad=False
                    )
                    self.lm_head.weight = torch.nn.Parameter(
                        state["lm_head.weight"].to(device), requires_grad=False
                    )
                else:
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

    @torch.inference_mode()
    def forward_ids(self, input_ids: list[int]) -> torch.Tensor:
        """Entry stage: embed token ids → run layers → hidden [1, T, H]."""
        if not self.first:
            raise RuntimeError("forward_ids requires the first stage")
        hidden = self.embed(torch.tensor([input_ids], device=self.device, dtype=torch.long))
        return self._run_layers(hidden)

    @torch.inference_mode()
    def forward_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Middle/last stage: run layers on inbound activations."""
        return self._run_layers(hidden)

    @torch.inference_mode()
    def logits_from(self, hidden: torch.Tensor) -> torch.Tensor:
        """Last stage: final norm + lm_head → logits [1, T, vocab]."""
        if not self.last:
            raise RuntimeError("logits_from requires the last stage")
        return self.lm_head(self.norm(hidden))

    @torch.inference_mode()
    def next_token_logits(self, token_id: int) -> torch.Tensor:
        """Entry stage decode step: embed one token → logits for that position."""
        hidden = self.embed(torch.tensor([[token_id]], device=self.device, dtype=torch.long))
        hidden = self._run_layers(hidden)
        return self.lm_head(self.norm(hidden))[:, -1, :]

    @torch.inference_mode()
    def embed_one(self, token_id: int) -> torch.Tensor:
        """Distributed entry decode step: embed one token → hidden after layers."""
        hidden = self.embed(torch.tensor([[token_id]], device=self.device, dtype=torch.long))
        return self._run_layers(hidden)

    @torch.inference_mode()
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
    backend = select_backend(manifest)
    needed: list[tuple[str, str, str | None]] = []
    for shard in manifest.shards:
        if shard.layer_start is None or shard.layer_end is None:
            continue
        if shard.layer_start <= layer_end and shard.layer_end >= layer_start:
            needed.append((shard.shard_id, shard.content_hash, shard.format))
    log.info(
        "fetch_stage model=%s layers=[%d,%d] backend=%s needed_shards=%s",
        manifest.model_id,
        layer_start,
        layer_end,
        backend.name,
        ",".join(shard_id for shard_id, _, _ in needed),
    )
    if _sequential_shards() and len(needed) > 1:
        # Constrained links (a laptop hotspot): concurrent full-shard download
        # streams contend and trip read timeouts. Fetch one at a time instead.
        log.info("fetch_stage sequential model=%s shards=%d", manifest.model_id, len(needed))
        paths = []
        for shard_id, content_hash, fmt in needed:
            paths.append(await store.ensure_shard(manifest.model_id, shard_id, content_hash, fmt))
    else:
        # Parallel downloads: each stage pulls several ~0.5–1 GB shards over LAN,
        # so serializing them would lock the stage behind ~N× the link time.
        paths = (
            await asyncio.gather(
                *(
                    store.ensure_shard(manifest.model_id, shard_id, content_hash, fmt)
                    for shard_id, content_hash, fmt in needed
                )
            )
            if needed
            else []
        )
    shard_paths = dict(zip((shard_id for shard_id, _, _ in needed), paths, strict=True))
    # deserialize_shard (torch.load / load_file on multi-GB shards) is CPU+disk
    # bound: run it off the event loop so heartbeats/streams never stall while
    # a stage is being fetched (same class of fix as inference runs).
    loop = asyncio.get_running_loop()
    states = await asyncio.gather(
        *(loop.run_in_executor(None, backend.deserialize_shard, path) for path in paths)
    )
    state: dict[str, torch.Tensor] = {}
    for tensors in states:
        state.update(tensors)
    tokenizer_path: str | None = None
    if manifest.tokenizer_file is not None and manifest.tokenizer_hash is not None:
        tokenizer_path = await store.ensure_tokenizer(
            manifest.model_id, manifest.tokenizer_file, manifest.tokenizer_hash
        )
    build = functools.partial(
        StageModel,
        manifest,
        layer_start,
        layer_end,
        state,
        device=select_device(manifest),
        tokenizer_path=tokenizer_path,
        backend=backend,
    )
    stage = await loop.run_in_executor(None, build)
    return stage, shard_paths
