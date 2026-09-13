"""Storage INT4: small shard files, fp16 runtime (plan §2.1 storage track).

Quantization here is a *serialization* format, not an execution format: the
exporter round-to-nearest quantizes the linear projections to 4-bit groups,
packs two nibbles per byte, and writes plain ``.safetensors`` shards. The node
dequantizes back to fp16 once while loading a shard (``StorageInt4Backend`` in
``llm.py``) and executes the exact fp16 kernels of the un-quantized path — so
transfer/download shrink ~4x at zero runtime-speed cost, with no TorchAO and no
special hardware requirement.

Tensor contract (manifest is the source of truth for ``group_size``):
- ``<orig>.weight``                    → NOT present for quantized tensors
- ``<orig>.weight.q4p``  (uint8)       → packed nibbles, shape [out, in/2];
  low nibble = first element of each pair, high nibble = second
- ``<orig>.weight.q4s``  (fp16)        → per-group scales [out, in/group_size]

Each nibble stores ``round(w/scale) + 8`` (0..15, symmetric int4 [-8, 7]).
Embeddings, norms, and lm_head are never quantized (same ``coverage="selected"``
rule as the TorchAO runtime track).
"""

from __future__ import annotations

import torch

STORAGE_INT4_BACKEND = "storage_int4"
STORAGE_INT4_SCHEME = "rtn_g128"
PACKING_LAYOUT_STORAGE = "storage"
STORAGE_INT4_PACKING_VERSION = "1"

_QPACK_SUFFIX = ".q4p"
_SCALE_SUFFIX = ".q4s"

#: Projection names eligible for storage quantization (mirrors
#: model_export._LLAMA_QUANT_PROJS): the seven linear projections of a Llama
#: decoder layer. Embeddings, norms and lm_head stay at runtime precision.
QUANTIZED_PROJ_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)

_RANGE_MAX = 7  # symmetric int4: round(w/scale) clamped to [-8, 7]
_NIBBLE_BIAS = 8  # stored value = q + 8, so nibbles are 0..15


def is_quantizable_key(key: str) -> bool:
    """True for ``model.layers.<N>.<self_attn|mlp>.<proj>.weight`` keys."""
    parts = key.split(".")
    return (
        len(parts) == 6
        and parts[0] == "model"
        and parts[1] == "layers"
        and parts[2].isdigit()
        and parts[3] in ("self_attn", "mlp")
        and parts[4] in QUANTIZED_PROJ_NAMES
        and parts[5] == "weight"
    )


def pack_int4(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Quantize one 2-D ``[out, in]`` weight to packed int4 + per-group scales.

    Returns ``(packed uint8 [out, in/2], scales fp16 [out, in/group_size])`` or
    None when the shape is unsuitable (non-2-D, odd input dim, or input dim not
    divisible by ``group_size``) — the caller then keeps the tensor in fp16.
    """
    if weight.dim() != 2:
        return None
    out_f, in_f = weight.shape
    if in_f % 2 != 0 or in_f % group_size != 0 or group_size > in_f:
        return None
    groups = in_f // group_size
    wf = weight.detach().float().reshape(out_f, groups, group_size)
    scale = wf.abs().amax(dim=-1, keepdim=True) / float(_RANGE_MAX)
    scale = torch.clamp(scale, min=1e-12)
    q = torch.round(wf / scale).clamp_(-_NIBBLE_BIAS, _RANGE_MAX)
    nibbles = (q + _NIBBLE_BIAS).to(torch.uint8).reshape(out_f, in_f // 2, 2)
    packed = (nibbles[..., 0] | (nibbles[..., 1] << 4)).contiguous()
    scales = scale.squeeze(-1).to(torch.float16).contiguous()
    return packed, scales


def quantize_state_dict_storage_int4(
    state: dict[str, torch.Tensor], group_size: int = 128
) -> dict[str, torch.Tensor]:
    """Return a copy of ``state`` with quantizable weights packed for storage.

    Non-quantizable tensors (and quantizable ones whose shape does not fit the
    group rule) pass through unchanged.
    """
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if is_quantizable_key(key):
            packed = pack_int4(tensor, group_size)
            if packed is not None:
                packed_t, scales = packed
                out[key + _QPACK_SUFFIX] = packed_t
                out[key + _SCALE_SUFFIX] = scales
                continue
        out[key] = tensor
    return out


def unpack_int4(
    packed: torch.Tensor, scales: torch.Tensor, group_size: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """Inverse of ``pack_int4``: rebuild the fp16 (or other dtype) weight."""
    out_f, half = packed.shape
    p = packed.to(torch.int16)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    q = torch.stack((lo, hi), dim=-1).reshape(out_f, half * 2)
    q = (q - _NIBBLE_BIAS).reshape(out_f, -1, group_size).float()
    w = q * scales.float().unsqueeze(-1)
    return w.reshape(out_f, half * 2).to(out_dtype).contiguous()


def dequantize_state_dict(
    state: dict[str, torch.Tensor], group_size: int = 128, out_dtype: torch.dtype = torch.float16
) -> dict[str, torch.Tensor]:
    """Materialize packed storage tensors back to runtime fp16 under the
    original names; everything else passes through untouched."""
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.endswith(_QPACK_SUFFIX):
            base = key[: -len(_QPACK_SUFFIX)]
            scales = state.get(base + _SCALE_SUFFIX)
            if scales is None:
                raise KeyError(f"packed tensor {key!r} has no scales sidecar")
            out[base] = unpack_int4(tensor, scales, group_size, out_dtype)
        elif key.endswith(_SCALE_SUFFIX):
            continue  # consumed by its packed sibling
        else:
            out[key] = tensor
    return out


def has_packed_tensors(state: dict[str, torch.Tensor]) -> bool:
    return any(key.endswith(_QPACK_SUFFIX) for key in state)
