"""Dev model export (S4): seeded tiny Llama → per-layer safetensors shards.

`dain-tiny-16L` is a real Llama-family transformer (through transformers'
LlamaForCausalLM) with a byte-level tokenizer, generated deterministically from
a fixed seed. Every node rebuilds identical weights, so the whole distributed
pipeline is hermetic and parity-checkable without network or gated weights
(spec §18 decision; TinyLlama/Qwen run through the same code path on real nodes).
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
    state = model.state_dict()

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
            tensors["model.embed_tokens.weight"] = state["model.embed_tokens.weight"].contiguous()
        for i in range(start, end + 1):
            prefix = f"model.layers.{i}."
            for key, tensor in state.items():
                if key.startswith(prefix):
                    tensors[key] = tensor.contiguous()
        if shard_idx == len(starts) - 1:
            tensors["model.norm.weight"] = state["model.norm.weight"].contiguous()
            tensors["lm_head.weight"] = state["lm_head.weight"].contiguous()

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

    manifest = ModelManifest(
        model_id=model_id,
        name=f"DAIN Tiny Llama ({layers}L, seeded)",
        layers=layers,
        hidden=config.hidden_size,
        heads=config.num_attention_heads,
        kv_heads=config.num_key_value_heads,
        intermediate=config.intermediate_size,
        vocab_size=config.vocab_size,
        eos_token_id=config.eos_token_id,
        rope_theta=config.rope_theta,
        dtype="fp32",
        shards=tuple(shards),
    )
    with open(os.path.join(model_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(manifest.model_dump_json(indent=2))
    return manifest
