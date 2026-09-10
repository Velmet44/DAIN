"""Model export (S4): seeded tiny Llama + real Hugging Face Llama → per-layer
safetensors shards.

`dain-tiny-16L` is a real Llama-family transformer (through transformers'
LlamaForCausalLM) with a byte-level tokenizer, generated deterministically from
a fixed seed. Every node rebuilds identical weights, so the whole distributed
pipeline is hermetic and parity-checkable without network or gated weights
(spec §18 decision; TinyLlama/Qwen run through the same code path on real nodes).

`export_hf_model` adds the real-model path: any Llama-family checkpoint from the
Hugging Face Hub (e.g. Llama-3.2-3B, Llama-3.2-1B, TinyLlama-1.1B) is sharded the
same way, in fp32 or fp16. Its fast tokenizer (`tokenizer.json`) is stored next
to the shards and served by the coordinator; the manifest records the digest so
nodes verify the download (spec §11). Qwen2-family models are not supported yet
— the node kernels are Llama-only (see Docs/stages.md S10).
"""

from __future__ import annotations

import hashlib
import os

import torch
from dain_common.model_store import load_manifest  # noqa: F401 (re-exported)
from dain_common.schemas import ModelManifest, ShardRef
from safetensors.torch import save_file
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

DEV_MODEL_ID = "dain-tiny-16L"
DTYPE = torch.float32


def _dtype_for(label: str) -> torch.dtype:
    if label == "fp16":
        return torch.float16
    return torch.float32


def tiny_config(layers: int = 16, hidden: int = 64) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=256,
        hidden_size=hidden,
        intermediate_size=172,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        eos_token_id=0,
        bos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
    )


def _write_shards(
    out_dir: str,
    *,
    model_id: str,
    name: str,
    config: LlamaConfig,
    state: dict[str, torch.Tensor],
    layers_per_shard: int,
    dtype_label: str,
    tokenizer_bytes: bytes | None = None,
    tokenizer_hash: str | None = None,
) -> ModelManifest:
    """Emit per-layer safetensors shards + manifest (+ tokenizer.json)."""
    dtype = _dtype_for(dtype_label)
    layers = config.num_hidden_layers
    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    shards: list[ShardRef] = []
    starts = list(range(0, layers, layers_per_shard))
    for shard_idx, start in enumerate(starts):
        end = min(start + layers_per_shard, layers) - 1
        shard_id = f"layers_{start:02d}_{end:02d}"
        tensors: dict[str, torch.Tensor] = {}
        if shard_idx == 0:
            tensors["model.embed_tokens.weight"] = (
                state["model.embed_tokens.weight"].to(dtype).contiguous()
            )
        for i in range(start, end + 1):
            prefix = f"model.layers.{i}."
            for key, tensor in state.items():
                if key.startswith(prefix):
                    tensors[key] = tensor.to(dtype).contiguous()
        if shard_idx == len(starts) - 1:
            tensors["model.norm.weight"] = state["model.norm.weight"].to(dtype).contiguous()
            tensors["lm_head.weight"] = state["lm_head.weight"].to(dtype).contiguous()

        path = os.path.join(model_dir, f"{shard_id}.safetensors")
        save_file(tensors, path)
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        shards.append(
            ShardRef(
                model_id=model_id,
                shard_id=shard_id,
                content_hash=digest,
                layer_start=start,
                layer_end=end,
                size_bytes=os.path.getsize(path),
            )
        )

    if tokenizer_bytes is not None:
        with open(os.path.join(model_dir, "tokenizer.json"), "wb") as fh:
            fh.write(tokenizer_bytes)

    manifest = ModelManifest(
        model_id=model_id,
        name=name,
        layers=layers,
        hidden=config.hidden_size,
        heads=config.num_attention_heads,
        kv_heads=config.num_key_value_heads,
        intermediate=config.intermediate_size,
        vocab_size=config.vocab_size,
        eos_token_id=config.eos_token_id,
        rope_theta=config.rope_theta,
        dtype=dtype_label,
        tokenizer_file="tokenizer.json" if tokenizer_bytes is not None else None,
        tokenizer_hash=tokenizer_hash,
        shards=tuple(shards),
    )
    with open(os.path.join(model_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(manifest.model_dump_json(indent=2))
    return manifest


def export_tiny_llama(
    out_dir: str,
    *,
    model_id: str = DEV_MODEL_ID,
    layers: int = 16,
    layers_per_shard: int = 4,
    seed: int = 1,
) -> ModelManifest:
    """Export the seeded model as contiguous per-layer shards + manifest.json."""
    torch.manual_seed(seed)
    config = tiny_config(layers=layers)
    model = LlamaForCausalLM(config)
    model.eval()
    return _write_shards(
        out_dir,
        model_id=model_id,
        name=f"DAIN Tiny Llama ({layers}L, seeded)",
        config=config,
        state=model.state_dict(),
        layers_per_shard=layers_per_shard,
        dtype_label="fp32",
    )


def export_hf_model(
    out_dir: str,
    *,
    model_id: str,
    name: str | None = None,
    hf_name: str | None = None,
    model: LlamaForCausalLM | None = None,
    tokenizer: object | None = None,
    layers_per_shard: int = 4,
    dtype: str = "fp16",
) -> ModelManifest:
    """Export a real Llama-family checkpoint to the same shard format.

    Pass `model`/`tokenizer` directly for hermetic tests, or leave them None and
    give `hf_name` to pull both from the Hugging Face Hub. The model is sharded
    in the requested dtype; the tokenizer `tokenizer.json` is embedded in the
    model dir and its sha256 recorded in the manifest so nodes verify it.
    """
    if model is None:
        if not hf_name:
            raise ValueError("provide either `model` or `hf_name`")
        model = LlamaForCausalLM.from_pretrained(hf_name, torch_dtype=torch.float32)
    model.eval()
    config = model.config
    model_type = getattr(config, "model_type", None)
    if model_type != "llama":
        raise ValueError(
            f"export_hf_model currently supports Llama-family models only (got {model_type!r})"
        )

    if tokenizer is None:
        if not hf_name:
            raise ValueError("provide either `tokenizer` or `hf_name`")
        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(hf_name, use_fast=True)
    tokenizer_bytes = tokenizer.backend_tokenizer.to_str().encode("utf-8")
    tokenizer_hash = hashlib.sha256(tokenizer_bytes).hexdigest()

    state: dict[str, torch.Tensor] = dict(model.state_dict())
    # Tied embeddings (Llama 1B/3B tie by default): state_dict holds one shared
    # tensor under both keys, but synthesize a copy defensively so the last
    # shard always carries lm_head.
    if "lm_head.weight" not in state and "model.embed_tokens.weight" in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"].detach().clone()
    if "lm_head.weight" not in state:
        raise ValueError("checkpoint has no lm_head.weight")

    return _write_shards(
        out_dir,
        model_id=model_id,
        name=name or f"HF {model_id} ({model_type})",
        config=config,
        state=state,
        layers_per_shard=layers_per_shard,
        dtype_label=dtype,
        tokenizer_bytes=tokenizer_bytes,
        tokenizer_hash=tokenizer_hash,
    )
