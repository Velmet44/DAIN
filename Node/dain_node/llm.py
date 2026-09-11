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
import logging
import os
import secrets
import time

import httpx
import torch
from dain_common.schemas import ModelManifest
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


class ModelStoreClient:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        self.base_url: str | None = None
        self.auth: dict[str, str] = {}
        self._manifests: dict[str, ModelManifest] = {}

    def set_base_url(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def set_auth(self, node_id: str, node_token: str) -> None:
        self.auth = {"X-Node-Id": node_id, "X-Node-Token": node_token}

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

    async def fetch_manifest(self, model_id: str, *, refresh: bool = False) -> ModelManifest:
        if not refresh and model_id in self._manifests:
            return self._manifests[model_id]
        if self.base_url is None:
            raise RuntimeError("model store base URL not set yet")
        async with httpx.AsyncClient(timeout=15.0) as client:
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

    async def ensure_shard(self, model_id: str, shard_id: str, content_hash: str) -> str:
        """Download + verify one shard; returns the cached file path.

        Streams the bytes to a temp file while hashing them incrementally, so a
        multi-GB shard is neither double-loaded into RAM nor hashed separately
        after download (the old path buffered `response.content` *and* hashed
        the whole buffer again).
        """
        model_cache = os.path.join(self.cache_dir, model_id)
        path = os.path.join(model_cache, f"{shard_id}.safetensors")
        if os.path.exists(path):
            if self._cached_hash_ok(path, content_hash):
                log.info("shard_hit model=%s shard=%s path=%s", model_id, shard_id, path)
                return path
            # One-time migration: a pre-sidecar cache has no .sha256 yet. Verify
            # in full once, write the sidecar, and avoid re-downloading ~1 GB.
            try:
                if self._hash_file(path) == content_hash:
                    self._write_hash_sidecar(path, content_hash)
                    log.info("shard_hit_verified model=%s shard=%s path=%s", model_id, shard_id, path)
                    return path
            except OSError:
                pass
        if self.base_url is None:
            raise RuntimeError("model store base URL not set yet")
        os.makedirs(model_cache, exist_ok=True)
        # Unique temp name: two jobs can download the same missing shard
        # concurrently (both are spawned tasks), and a shared ".tmp" would let
        # their writes interleave into one corrupt file.
        tmp = f"{path}.{secrets.token_hex(4)}.tmp"
        hasher = hashlib.sha256()
        started = time.monotonic()
        chunk_size = _read_chunk_size()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "GET", f"{self.base_url}/shard/{model_id}/{shard_id}", headers=self.auth
                ) as response:
                    response.raise_for_status()
                    expected = int(response.headers.get("content-length", "0") or 0)
                    received = 0
                    last_pct = -1
                    with open(tmp, "wb") as fh:
                        async for chunk in response.aiter_bytes(chunk_size):
                            fh.write(chunk)
                            hasher.update(chunk)
                            received += len(chunk)
                            if expected > 0:
                                pct = int(100 * received / expected)
                                if pct // 5 != last_pct // 5 or pct == 100:
                                    elapsed = max(time.monotonic() - started, 1e-6)
                                    mbps = received / elapsed / 1e6
                                    log.info(
                                        "shard_download model=%s shard=%s progress=%d%% "
                                        "bytes=%d/%d mbps=%.0f",
                                        model_id,
                                        shard_id,
                                        pct,
                                        received,
                                        expected,
                                        mbps,
                                    )
                                    last_pct = pct
                            else:
                                log.info(
                                    "shard_download model=%s shard=%s bytes=%d",
                                    model_id,
                                    shard_id,
                                    received,
                                )
            if hasher.hexdigest() != content_hash:
                raise RuntimeError(f"shard {shard_id} failed hash verification")
            os.replace(tmp, path)
            self._write_hash_sidecar(path, content_hash)
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

    @staticmethod
    def _hash_file(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    async def ensure_tokenizer(
        self, model_id: str, tokenizer_file: str, tokenizer_hash: str
    ) -> str:
        """Download + verify the model's tokenizer.json; returns its cache path."""
        model_cache = os.path.join(self.cache_dir, model_id)
        file_name = os.path.basename(tokenizer_file)
        path = os.path.join(model_cache, file_name)
        if os.path.exists(path):
            if self._cached_hash_ok(path, tokenizer_hash):
                log.info("tokenizer_hit model=%s file=%s path=%s", model_id, file_name, path)
                return path
            try:
                if self._hash_file(path) == tokenizer_hash:
                    self._write_hash_sidecar(path, tokenizer_hash)
                    log.info("tokenizer_hit_verified model=%s file=%s path=%s", model_id, file_name, path)
                    return path
            except OSError:
                pass
        if self.base_url is None:
            raise RuntimeError("model store base URL not set yet")
        os.makedirs(model_cache, exist_ok=True)
        tmp = f"{path}.{secrets.token_hex(4)}.tmp"
        hasher = hashlib.sha256()
        chunk_size = _read_chunk_size()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
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
    # Parallel downloads: each stage pulls several ~0.5–1 GB shards over LAN, so
    # serialize them would lock the stage behind ~N× the link time. gather() lets
    # the streams run concurrently (the coordinator is async and serves them).
    paths = (
        await asyncio.gather(
            *(store.ensure_shard(manifest.model_id, shard_id, content_hash) for shard_id, content_hash in needed)
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
