"""Storage-INT4 track: packed-int4 safetensors shards, fp16 runtime.

Covers the pack/unpack math, the state-dict transform, the export writer, the
node-side backend selection, and an end-to-end StageModel forward on a
dequantized tiny model. The tiny config uses hidden 64 / intermediate 128 so
the projections divide evenly by the 32-wide quantization group.
"""

import os

import torch
from dain_common.schemas import ModelManifest, QuantizationSpec, ShardRef
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from dain_node.llm import StageModel, StorageInt4Backend, select_backend, select_device
from dain_node.model_export import export_shards_storage_int4
from dain_node.shard_export import tiny_config
from dain_node.storage_int4 import (
    dequantize_state_dict,
    is_quantizable_key,
    pack_int4,
    quantize_state_dict_storage_int4,
    unpack_int4,
)

GROUP = 32
_PROJS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _tiny_state(layers: int = 4):
    torch.manual_seed(7)
    config = tiny_config(layers=layers)
    config.intermediate_size = 128  # divisible by GROUP (172 is not)
    model = LlamaForCausalLM(config)
    model.eval()
    state = dict(model.state_dict())
    if "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"].detach().clone()
    return state


# -- pack/unpack math -----------------------------------------------------------


def test_pack_round_trip_within_half_scale() -> None:
    w = torch.randn(8, 256) * 0.05
    packed, scales = pack_int4(w, group_size=128)
    assert packed.dtype == torch.uint8 and packed.shape == (8, 128)
    assert scales.dtype == torch.float16 and scales.shape == (8, 2)
    rebuilt = unpack_int4(packed, scales, group_size=128, out_dtype=torch.float16)
    # Per-weight error is bounded by half the group scale (RTN rounding).
    worst = scales.float().max().item() * 0.5 + 1e-6
    assert (rebuilt.float() - w).abs().max().item() <= worst


def test_pack_shape_guards() -> None:
    assert pack_int4(torch.randn(4, 5), 4) is None  # odd input dim
    assert pack_int4(torch.randn(4, 10), 8) is None  # in % group != 0
    assert pack_int4(torch.randn(4), 8) is None  # not 2-D
    packed, scales = pack_int4(torch.randn(4, 16), 16)
    assert packed.shape == (4, 8) and scales.shape == (4, 1)


def test_is_quantizable_key() -> None:
    assert is_quantizable_key("model.layers.3.self_attn.q_proj.weight")
    assert is_quantizable_key("model.layers.11.mlp.down_proj.weight")
    assert not is_quantizable_key("model.embed_tokens.weight")
    assert not is_quantizable_key("model.norm.weight")
    assert not is_quantizable_key("lm_head.weight")
    assert not is_quantizable_key("model.layers.3.self_attn.q_proj.bias")


# -- state-dict transform -------------------------------------------------------


def test_quantize_state_dict_packs_projections_only() -> None:
    state = _tiny_state(layers=2)
    packed_state = quantize_state_dict_storage_int4(state, group_size=GROUP)

    # All 7 projections per layer packed; embed/norm/lm_head untouched.
    for layer in range(2):
        for proj in _PROJS:
            key = f"model.layers.{layer}.{proj}.weight"
            assert key + ".q4p" in packed_state and key + ".q4s" in packed_state
            assert key not in packed_state
    assert "model.embed_tokens.weight" in packed_state
    assert "model.norm.weight" in packed_state
    assert "lm_head.weight" in packed_state

    # Round trip via the node-side dequantizer: original names, quantized
    # tensors rebuilt in fp16, pass-through tensors untouched.
    rebuilt = dequantize_state_dict(packed_state, group_size=GROUP)
    assert set(rebuilt) == set(state)
    for layer in range(2):
        for proj in _PROJS:
            key = f"model.layers.{layer}.{proj}.weight"
            assert rebuilt[key].dtype == torch.float16, key
            assert (rebuilt[key].float() - state[key].float()).abs().max().item() < 0.05
    for key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        assert rebuilt[key].dtype == state[key].dtype
        assert torch.equal(rebuilt[key], state[key])


# -- export + backend selection + StageModel forward ------------------------------


def test_export_storage_int4_end_to_end(tmp_path) -> None:
    state = _tiny_state(layers=4)
    config = tiny_config(layers=4)
    config.intermediate_size = 128

    manifest = export_shards_storage_int4(
        str(tmp_path / "model"),
        model_id="tiny-int4s",
        name="tiny storage",
        config=config,
        state=quantize_state_dict_storage_int4(state, group_size=GROUP),
        layers_per_shard=2,
        tokenizer_bytes=b"{}",
        tokenizer_hash="0" * 64,
        group_size=GROUP,
    )
    assert manifest.format == "safetensors"
    assert manifest.dtype == "fp16"
    assert manifest.quantization is not None
    assert manifest.quantization.backend == "storage_int4"
    assert manifest.quantization.is_quantized is True
    assert manifest.quantization.packing_layout == "storage"
    assert all(shard.format == "safetensors" for shard in manifest.shards)

    # Packed shards must be markedly smaller than the fp16 equivalents
    # (~4.2 bits/weight on the projections) while carrying the same tensors.
    packed_bytes = sum(s.size_bytes for s in manifest.shards)
    fp16_bytes = sum(t.numel() * t.element_size() for t in state.values())
    assert packed_bytes < fp16_bytes * 0.45

    # Node-side: backend selection and dequantizing load.
    backend = select_backend(manifest)
    assert isinstance(backend, StorageInt4Backend)
    assert backend.is_quantized is False  # StageModel takes the fp16 branch
    assert backend.group_size == GROUP

    shard_path = os.path.join(
        str(tmp_path / "model"), f"{manifest.shards[0].shard_id}.safetensors"
    )
    tensors = backend.deserialize_shard(shard_path)
    assert "model.embed_tokens.weight" in tensors
    for layer in range(2):
        for proj in _PROJS:
            key = f"model.layers.{layer}.{proj}.weight"
            assert key in tensors and tensors[key].dtype == torch.float16
            assert (tensors[key].float() - state[key].float()).abs().max().item() < 0.05

    # A dequantized stage actually runs.
    stage = StageModel(
        manifest,
        0,
        1,
        tensors,
        device=select_device(manifest),
        tokenizer_path=None,
        backend=backend,
    )
    stage.begin_job()
    hidden = stage.forward_ids([1, 2, 3])
    assert hidden.shape == (1, 3, manifest.hidden)
    assert torch.isfinite(hidden).all()
    stage.end_job()


def test_select_backend_and_device_for_storage_int4() -> None:
    manifest = ModelManifest(
        model_id="m",
        name="m",
        layers=2,
        hidden=8,
        heads=2,
        kv_heads=2,
        intermediate=16,
        vocab_size=16,
        eos_token_id=0,
        shards=(
            ShardRef(
                model_id="m",
                shard_id="layers_00_01",
                content_hash="a" * 64,
                layer_start=0,
                layer_end=1,
                size_bytes=10,
                format="safetensors",
            ),
        ),
        dtype="fp16",
        format="safetensors",
        quantization=QuantizationSpec(
            backend="storage_int4",
            scheme="rtn_g128",
            bits=4,
            group_size=GROUP,
            coverage="selected",
            packing_layout="storage",
        ),
    )
    assert isinstance(select_backend(manifest), StorageInt4Backend)
    # Legacy device decision: CUDA when present, CPU otherwise (NOT the
    # torchao layout pinning).
    expected = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert select_device(manifest) == expected
