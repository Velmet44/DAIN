"""Model export pipeline (session N): local HuggingFace Safetensors → DAIN
TorchAO-INT4 shards.

The exporter is the *model compiler* of the cluster (architecture decision in
Docs/plan-model-export-pipeline.md §2): quantization happens once here, never at
node start-up. Nodes only ever execute the packed artifacts.

Flow::

    validate_source() -> detect_model() -> build_and_quantize()
        -> export_shards() -> atomic publish

Two serial sizes ship on one format axis:
- legacy/real-model fp16/fp32 keeps using safetensors (``.safetensors``);
- quantized state dicts carry TorchAO ``AffineQuantizedTensor`` tensor
  subclasses whose packing metadata ONLY survives ``torch.save``/``torch.load``,
  so quantized shards are ``.pt`` files (plan §2.1). ``safetensors.save_file``
  cannot represent the subclass and silently drops the quantization.

Adapter pattern: a registered ``ModelAdapter`` declares an architecture family
(today: ``llama``). Only decoder-only causal LMs with a known adapter are
supported; anything else fails with an actionable ``ValueError``.

This module runs as a subprocess (``python -m dain_node.model_export``) so the
Coordinator never imports torch/torchao. With ``--json-progress`` it prints
newline-delimited JSON events the Coordinator streams into the admin job log.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from dain_common.schemas import ModelManifest, QuantizationSpec, ShardRef
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dain_node.shard_export import _dtype_for

log = logging.getLogger("dain.node.model_export")

MARKER_FILE = ".model-exports.json"

#: Adapter-filtered quantization targets for Llama-family. Embeddings, norms and
#: the output head stay in higher precision so semantic loss stays bounded.
_LLAMA_QUANT_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

LAYOUT_CUDA = "tensor_core_tiled"
LAYOUT_CPU = "int4_cpu"


# -- adapter registry ---------------------------------------------------------


class ModelAdapter:
    """Declares support for one HF architecture family."""

    architecture: str = ""
    display_name: str = ""

    def is_supported(self, config) -> bool:
        raise NotImplementedError

    def quantizable_layers(self, model) -> dict[str, nn.Linear]:
        """FQN → Linear for the projections eligible for INT4 (adapter-filtered).

        Used both to collect the torchao filter set and (indirectly) the FQN
        names in manifests. Embeddings, norms, and the output head are never
        quantized.
        """
        raise NotImplementedError

    def validate_model(self, model) -> None:
        """Raise ValueError when a loaded model isn't actually usable."""


class LlamaAdapter(ModelAdapter):
    architecture = "llama"
    display_name = "Llama-family (decoder-only)"

    def is_supported(self, config) -> bool:
        return (
            getattr(config, "model_type", None) == "llama"
            and not bool(getattr(config, "is_encoder_decoder", False))
        )

    def quantizable_layers(self, model) -> dict[str, nn.Linear]:
        layers: dict[str, nn.Linear] = {}
        for fqn, mod in dict(model.named_modules()).items():
            if not isinstance(mod, nn.Linear):
                continue
            if fqn.rsplit(".", 1)[-1] in _LLAMA_QUANT_PROJS:
                layers[fqn] = mod
        return layers

    def validate_model(self, model) -> None:
        if model.__class__.__name__ != "LlamaForCausalLM":
            raise ValueError(
                f"adapter 'llama' requires LlamaForCausalLM (got {model.__class__.__name__})"
            )


ADAPTERS: dict[str, type[ModelAdapter]] = {}


def register_adapter(adapter: type[ModelAdapter]) -> None:
    ADAPTERS[adapter.architecture] = adapter


register_adapter(LlamaAdapter)


# -- source validation --------------------------------------------------------


def validate_source(source_dir: str) -> dict:
    """Check the directory looks like a local HF model (config + weights).

    Returns a summary dict; raises ValueError with an actionable message on
    missing core files.
    """
    src = Path(source_dir)
    if not src.is_dir():
        raise ValueError(f"source directory {source_dir} does not exist")

    config_file = src / "config.json"
    if not config_file.is_file():
        raise ValueError(f"{source_dir} has no config.json — not a HuggingFace model directory")

    weights = sorted(src.glob("*.safetensors")) or sorted(src.glob("*.bin"))
    if not weights:
        raise ValueError(f"{source_dir} has no model weights (expected *.safetensors or *.bin)")

    net_bytes = sum(p.stat().st_size for p in weights)
    config = AutoConfig.from_pretrained(str(src), trust_remote_code=False)
    tok_file = src / "tokenizer.json"
    return {
        "config_file": str(config_file),
        "weight_files": [p.name for p in weights],
        "weight_count": len(weights),
        "size_bytes": net_bytes,
        "architecture": getattr(config, "model_type", None),
        "num_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "has_fast_tokenizer": tok_file.is_file(),
        "can_build_tokenizer": tok_file.is_file() or (src / "tokenizer_config.json").is_file(),
    }


def detect_model(config: AutoConfig) -> ModelAdapter:
    """Match the config against the adapter registry; raises on unsupported."""
    adapter_cls = ADAPTERS.get(getattr(config, "model_type", None))
    if adapter_cls is None:
        raise ValueError(
            f"architecture {getattr(config, 'model_type', '?')!r} is not supported. "
            "DAIN currently supports: " + ", ".join(sorted(ADAPTERS))
        )
    adapter = adapter_cls()
    if not adapter.is_supported(config):
        raise ValueError(
            f"architecture {getattr(config, 'model_type', '?')!r} is not a supported "
            "decoder-only causal LM for this adapter"
        )
    return adapter


# -- quantization -------------------------------------------------------------


def pick_layout() -> str:
    """torchao CPU vs GPU tensor layout for this *export* machine (plan §2.1).

    CUDA exporters target the tinygemm ``TensorCoreTiledLayout``; CPU export
    machines use ``Int4CPULayout`` (``_weight_int4pack_mm_for_cpu``). The layout
    is recorded in the manifest so the scheduler only places the model on nodes
    that can execute it.
    """
    try:
        if torch.cuda.is_available():
            return LAYOUT_CUDA
    except Exception:  # noqa: BLE001 — device probing must never break export
        pass
    return LAYOUT_CPU


def _layout_kwargs(layout: str) -> dict:
    if layout == LAYOUT_CUDA:
        from torchao.dtypes import TensorCoreTiledLayout

        return {"layout": TensorCoreTiledLayout(inner_k_tiles=8)}
    if layout == LAYOUT_CPU:
        from torchao.dtypes import Int4CPULayout

        return {"layout": Int4CPULayout()}
    return {}


def build_and_quantize(
    src: str,
    config,
    adapter: ModelAdapter,
    *,
    activation_dtype: str,
    group_size: int,
    trust_remote_code: bool,
    dry_run: bool = False,
) -> tuple[nn.Module, str]:
    """Load the HF model, verify it with the adapter, then TorchAO-INT4 quantize."""
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    dtype = _dtype_for(activation_dtype)
    log.info(
        "export_load source=%s dtype=%s trust_remote_code=%s",
        src,
        activation_dtype,
        trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    adapter.validate_model(model)
    model.eval()

    layer_specs = adapter.quantizable_layers(model)
    if not layer_specs:
        raise ValueError(
            f"adapter {adapter.architecture} found no quantizable linear layers in the model"
        )

    layout = pick_layout()
    log.info(
        "export_quantize scheme=int4_groupwise group_size=%d layout=%s layers=%d",
        group_size,
        layout,
        len(layer_specs),
    )
    if dry_run:
        log.info("dry_run — skipping quantize_ call")
        return model, layout

    target_fqns = set(layer_specs)
    quantize_(
        model,
        Int4WeightOnlyConfig(
            group_size=group_size,
            set_inductor_config=False,
            **_layout_kwargs(layout),
        ),
        # filter_fn(module, fqn): quantize only the adapter-listed projections.
        filter_fn=lambda module, fqn: isinstance(module, nn.Linear) and fqn in target_fqns,
    )
    return model, layout


# -- shard serialization ------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def export_shards(
    out_model_dir: str,
    *,
    model_id: str,
    name: str,
    config,
    model: nn.Module,
    layers_per_shard: int,
    dtype_label: str,
    tokenizer_bytes: bytes | None,
    tokenizer_hash: str | None,
    quantization: QuantizationSpec,
    adapter: ModelAdapter,
    base_model_id: str | None,
    source_config_hash: str,
) -> ModelManifest:
    """Serialize the quantized state dict as per-layer ``.pt`` DAIN shards.

    Every shard is a ``torch.save(state_dict_slice)`` of the per-layer parameter
    group, with the first shard carrying the embedding and the last shard the
    norm + output projection — matching the safetensors layout so the nodes'
    ``ShardRef.layer_start/end`` overlap logic is unchanged (plan §3).
    """
    layers = config.num_hidden_layers
    os.makedirs(out_model_dir, exist_ok=True)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if "lm_head.weight" not in state and "model.embed_tokens.weight" in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()

    shards: list[ShardRef] = []
    starts = list(range(0, layers, layers_per_shard))
    for shard_idx, start in enumerate(starts):
        end = min(start + layers_per_shard, layers) - 1
        shard_id = f"layers_{start:02d}_{end:02d}"
        tensors: dict[str, torch.Tensor] = {}
        if shard_idx == 0:
            tensors["model.embed_tokens.weight"] = state["model.embed_tokens.weight"]
        for i in range(start, end + 1):
            prefix = f"model.layers.{i}."
            for key, tensor in state.items():
                if key.startswith(prefix):
                    tensors[key] = tensor
        if shard_idx == len(starts) - 1:
            tensors["model.norm.weight"] = state["model.norm.weight"]
            tensors["lm_head.weight"] = state["lm_head.weight"]

        path = os.path.join(out_model_dir, f"{shard_id}.pt")
        torch.save(tensors, path)
        digest = _sha256_file(Path(path))
        shards.append(
            ShardRef(
                model_id=model_id,
                shard_id=shard_id,
                content_hash=digest,
                layer_start=start,
                layer_end=end,
                size_bytes=os.path.getsize(path),
                format="torch_pt",
            )
        )

    if tokenizer_bytes is not None:
        with open(os.path.join(out_model_dir, "tokenizer.json"), "wb") as fh:
            fh.write(tokenizer_bytes)

    manifest = ModelManifest(
        model_id=model_id,
        name=name,
        layers=layers,
        hidden=config.hidden_size,
        heads=config.num_attention_heads,
        kv_heads=config.num_key_value_heads,
        intermediate=config.intermediate_size,
        vocab_size=getattr(config, "vocab_size", 0),
        eos_token_id=getattr(config, "eos_token_id", 0),
        rope_theta=getattr(config, "rope_theta", 10000.0),
        dtype=dtype_label,
        tokenizer_file="tokenizer.json" if tokenizer_bytes is not None else None,
        tokenizer_hash=tokenizer_hash,
        shards=tuple(shards),
        format="torch_pt",
        quantization=quantization,
        base_model_id=base_model_id,
        architecture=adapter.architecture,
        adapter_id=adapter.architecture,
        architecture_config={
            "hidden_act": getattr(config, "hidden_act", "silu"),
            "max_position_embeddings": getattr(config, "max_position_embeddings", 2048),
            "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),
            "rope_scaling": getattr(config, "rope_scaling", None),
            "attention_bias": getattr(config, "attention_bias", False),
            "attention_dropout": getattr(config, "attention_dropout", 0.0),
            "mlp_bias": getattr(config, "mlp_bias", False),
            "head_dim": getattr(config, "head_dim", None),
            "tie_word_embeddings": getattr(config, "tie_word_embeddings", False),
        },
        artifact_version=1,
        source_config_hash=source_config_hash,
    )
    with open(os.path.join(out_model_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(manifest.model_dump_json(indent=2))
    return manifest


def _read_tokenizer(src: Path) -> tuple[bytes | None, str | None]:
    """Fast tokenizer.json (or build one from tokenizer files) → bytes + sha256."""
    tok_file = src / "tokenizer.json"
    if tok_file.is_file():
        data = tok_file.read_bytes()
        return data, hashlib.sha256(data).hexdigest()
    try:
        tok = AutoTokenizer.from_pretrained(str(src), use_fast=True)
    except Exception as exc:  # noqa: BLE001 — surfaced as a hard export failure below
        log.warning("tokenizer_build_failed source=%s err=%s", src, exc)
        return None, None
    if tok is None or tok.backend_tokenizer is None:
        return None, None
    data = tok.backend_tokenizer.to_str().encode("utf-8")
    return data, hashlib.sha256(data).hexdigest()


def _check_tokenizer_vocab(src: Path, config) -> None:
    """A tokenizer whose vocab overruns the model's embedding is a silent
    ``index out of range`` at first embed — refuse the export instead."""
    try:
        tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    except Exception:
        return
    model_vocab = getattr(config, "vocab_size", None)
    if isinstance(model_vocab, int) and tok.get_vocab_size() > model_vocab:
        raise ValueError(
            f"tokenizer vocab ({tok.get_vocab_size()}) exceeds model vocab_size "
            f"({model_vocab}) in {src}; fix the tokenizer/config before exporting"
        )


def _sha256_config(source_dir: str) -> str:
    try:
        with open(os.path.join(source_dir, "config.json"), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


def _torchao_version() -> str:
    try:
        import torchao

        return getattr(torchao, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def _emit_json(**fields) -> None:
    print(json.dumps(fields, separators=(",", ":")), flush=True)


# -- full export --------------------------------------------------------------


def export_model(cfg) -> ModelManifest:
    """Run the complete export pipeline; returns the published manifest.

    The model is staged under a sibling directory and atomically moved into
    place on success, so a crash mid-export never leaves a half-written model
    and never replaces an already-published one.
    """
    store = Path(cfg.output_store)
    store.mkdir(parents=True, exist_ok=True)

    summary = validate_source(cfg.source_dir)
    if cfg.json_progress:
        _emit_json(event="detected", **summary)

    config = AutoConfig.from_pretrained(str(Path(cfg.source_dir)), trust_remote_code=False)
    adapter = detect_model(config)
    log.info(
        "export_start source=%s model=%s arch=%s adapter=%s layers=%s",
        cfg.source_dir,
        cfg.model_id,
        summary.get("architecture"),
        adapter.architecture,
        summary.get("num_layers"),
    )

    quant_spec = QuantizationSpec(
        backend="torchao",
        scheme="int4_weight_only",
        bits=4,
        group_size=cfg.group_size,
        activation_dtype=cfg.activation_dtype,
        packing_version="1",
        quantizer_version=_torchao_version(),
        coverage="selected",
    )

    final_dir = store / cfg.model_id
    if final_dir.exists() and any(final_dir.iterdir()) and not cfg.force:
        raise FileExistsError(
            f"model {cfg.model_id} already exists in {store}; pass --force to overwrite"
        )

    tokenizer_bytes, tokenizer_hash = _read_tokenizer(Path(cfg.source_dir))
    if not cfg.dry_run and tokenizer_bytes is None:
        # A real model without its tokenizer would run under ByteTokenizer and
        # emit garbage tokens while *looking* healthy — refuse the export.
        raise ValueError(
            "no usable tokenizer found — real-model exports require tokenizer.json "
            "(or tokenizer_config.json + vocab files that HF can build a fast "
            "tokenizer from); refusing so nodes never silently fall back to a "
            "byte-level vocabulary"
        )
    _check_tokenizer_vocab(Path(cfg.source_dir), config)
    config_hash = _sha256_config(cfg.source_dir)
    base_model_id = getattr(config, "_name_or_path", None) or cfg.model_id

    layout = pick_layout()
    quant_spec = QuantizationSpec.model_validate(
        {**quant_spec.model_dump(), "packing_layout": layout}
    )

    manifest = None
    if not cfg.dry_run:
        staging = store / f".export-{cfg.model_id}-{secrets.token_hex(4)}"
        model_dir = staging / cfg.model_id
        try:
            model, _layout = build_and_quantize(
                cfg.source_dir,
                config,
                adapter,
                activation_dtype=cfg.activation_dtype,
                group_size=cfg.group_size,
                trust_remote_code=cfg.trust_remote_code,
            )
            if cfg.json_progress:
                _emit_json(event="quantized", layout=_layout)
            manifest = export_shards(
                str(model_dir),
                model_id=cfg.model_id,
                name=f"{adapter.display_name} INT4-{cfg.activation_dtype}",
                config=config,
                model=model,
                layers_per_shard=cfg.layers_per_shard,
                dtype_label=cfg.activation_dtype,
                tokenizer_bytes=tokenizer_bytes,
                tokenizer_hash=tokenizer_hash,
                quantization=quant_spec,
                adapter=adapter,
                base_model_id=base_model_id,
                source_config_hash=config_hash,
            )
            if cfg.json_progress:
                _emit_json(
                    event="written",
                    model_id=cfg.model_id,
                    layout=layout,
                    shards=len(manifest.shards),
                    bytes=sum(s.size_bytes for s in manifest.shards),
                )
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        # Publish atomically: replace the previous model only under --force.
        if cfg.force and final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        os.replace(model_dir, final_dir)
        with contextlib.suppress(OSError):
            shutil.rmtree(staging, ignore_errors=True)
        _record_export(cfg.source_dir, cfg.model_id, cfg)
    else:
        # Dry-run: still exercise build_and_quantize() for validation.
        model, layout = build_and_quantize(
            cfg.source_dir,
            config,
            adapter,
            activation_dtype=cfg.activation_dtype,
            group_size=cfg.group_size,
            trust_remote_code=cfg.trust_remote_code,
            dry_run=True,
        )
        manifest = export_shards(
            str(store / f".dry-run-{cfg.model_id}"),
            model_id=cfg.model_id,
            name=f"dry-run {cfg.model_id}",
            config=config,
            model=model,
            layers_per_shard=cfg.layers_per_shard,
            dtype_label=cfg.activation_dtype,
            tokenizer_bytes=None,
            tokenizer_hash=None,
            quantization=quant_spec,
            adapter=adapter,
            base_model_id=base_model_id,
            source_config_hash=config_hash,
        )
        with contextlib.suppress(OSError):
            shutil.rmtree(store / f".dry-run-{cfg.model_id}", ignore_errors=True)

    if cfg.json_progress:
        _emit_json(event="done", model_id=cfg.model_id, dry_run=cfg.dry_run)
    log.info(
        "export_done model=%s shards=%d bytes=%d",
        cfg.model_id,
        len(manifest.shards),
        sum(s.size_bytes for s in manifest.shards),
    )
    return manifest


def _record_export(source_dir: str, model_id: str, cfg) -> None:
    marker_path = Path(cfg.output_store) / MARKER_FILE
    marker: dict = {}
    if marker_path.exists():
        with open(marker_path, encoding="utf-8-sig") as fh:
            try:
                marker = json.load(fh)
            except ValueError:
                marker = {}
        if not isinstance(marker, dict):
            marker = {}
    marker[model_id] = {
        "source_dir": source_dir,
        "exported_at": time.time(),
        "quantization": cfg.quantization,
        "group_size": cfg.group_size,
        "activation_dtype": cfg.activation_dtype,
    }
    tmp = marker_path.with_name(f"{marker_path.name}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(marker, fh, indent=2)
    tmp.replace(marker_path)


# -- CLI ----------------------------------------------------------------------


#: Model ids become store directory names: the admin API only enforces
#: min_length=1, so a caller-supplied id must be sanitized here (matching the
#: import_gguf/delete guards) or ``../../x`` writes outside the model store.
_SAFE_ID = re.compile(r"[^a-z0-9_-]+")


def _sanitize_model_id(value: str) -> str:
    cleaned = _SAFE_ID.sub("-", value.lower()).strip("-")
    if not cleaned:
        raise ValueError(f"model id {value!r} contains no usable filename characters")
    return cleaned


class ExportConfig:
    """Validated export parameters (CLI flags or Coordinator POST body)."""

    def __init__(
        self,
        *,
        source_dir: str,
        model_id: str,
        output_store: str,
        quantization: str = "int4",
        group_size: int = 128,
        activation_dtype: str = "fp16",
        layers_per_shard: int = 4,
        force: bool = False,
        trust_remote_code: bool = False,
        dry_run: bool = False,
        json_progress: bool = False,
    ) -> None:
        self.source_dir = source_dir
        self.model_id = _sanitize_model_id(model_id)
        self.output_store = output_store
        self.quantization = quantization
        self.group_size = group_size
        self.activation_dtype = activation_dtype
        self.layers_per_shard = layers_per_shard
        self.force = force
        self.trust_remote_code = trust_remote_code
        self.dry_run = dry_run
        self.json_progress = json_progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dain_node.model_export",
        description="Export a local HuggingFace model into quantized DAIN shards.",
    )
    parser.add_argument("--source-dir", required=True, help="local HF model directory")
    parser.add_argument("--model-id", required=True, help="model id for the model store")
    parser.add_argument("--output-store", required=True, help="model store directory")
    parser.add_argument("--quantization", choices=["int4"], default="int4")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--activation-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--layers-per-shard", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="overwrite an existing model")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate + detect only")
    parser.add_argument("--json-progress", action="store_true", help="emit NDJSON events")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = ExportConfig(
        source_dir=args.source_dir,
        model_id=args.model_id,
        output_store=args.output_store,
        quantization=args.quantization,
        group_size=args.group_size,
        activation_dtype=args.activation_dtype,
        layers_per_shard=args.layers_per_shard,
        force=args.force,
        trust_remote_code=args.trust_remote_code,
        dry_run=args.dry_run,
        json_progress=args.json_progress,
    )
    try:
        manifest = export_model(cfg)
    except (ValueError, FileExistsError, OSError) as exc:
        if args.json_progress:
            _emit_json(event="error", error=str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"detected: {manifest.architecture or '?'} -> would quantize INT4")
        return 0
    size = sum(s.size_bytes for s in manifest.shards)
    print(
        f"exported: {manifest.model_id} ({len(manifest.shards)} shards, "
        f"{size / 1024 / 1024:.1f} MiB, INT4-{manifest.dtype})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
