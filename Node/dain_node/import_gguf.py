"""GGUF import: convert a Llama-family GGUF into DAIN's sharded safetensors format.

Drop ``*.gguf`` files into the model store and convert with::

    python -m dain_node.import_gguf <model_store> [file.gguf] [options]

(``Scripts/import-gguf.ps1`` and the coordinator admin page run this exact
command through uv). Weights are **dequantized to fp32/fp16** — DAIN executes
safetensors, not GGUF quants — then sharded exactly like ``export_hf_model``,
so a Q4_K_M 7B (~4 GB file) becomes ~14 GB of fp16 shards spread across nodes.
Only Llama-architecture GGUFs are supported; the node kernels are Llama-only
(same restriction as ``export_hf_model``).

Tokenizers: the embedded GGUF vocab is converted to an HF fast tokenizer when
possible (Llama-3-style BPE works; some SPM files do not). When conversion
fails, pass ``--tokenizer <hf-repo>`` to source it from the Hub instead.

Idempotency: ``<model_store>/.gguf-imports.json`` records each file's sha256
and the model_id it produced. A file is skipped while its model dir still has
a manifest *and* the recorded sha matches; deleting the model dir (or
replacing the file) re-imports it on the next run. ``--force`` overrides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import secrets
import sys
import time
from pathlib import Path

import torch
from dain_common.schemas import ModelManifest
from gguf import GGUFReader
from transformers import AutoTokenizer
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from dain_node.shard_export import _write_shards

log = logging.getLogger("dain.node.import_gguf")

MARKER_FILE = ".gguf-imports.json"
_SAFE_ID = re.compile(r"[^a-z0-9_-]+")
# Buffers llama.cpp writes that HF models recompute from config (rotary inv
# freq caches) — present in real Llama-3.x GGUFs, not parameters; dropped
# before load instead of failing the strict tensor-mapping check.
_GGUF_IGNORE_TENSORS = frozenset({"rope_freqs.weight", "rotary_emb.inv_freq"})


# -- gguf metadata ------------------------------------------------------------


def _gguf_str(reader: GGUFReader, key: str) -> str | None:
    field = reader.get_field(key)
    if field is None:
        return None
    try:
        return bytes(field.parts[field.data[0]]).decode("utf-8", "replace").strip("\x00 ")
    except (IndexError, TypeError, ValueError):
        return None


def _derive_model_id(reader: GGUFReader, gguf_path: Path, override: str | None = None) -> str:
    """A filesystem-safe model id from GGUF metadata or the file name.

    Restricted to ``[a-z0-9_-]`` so imported models stay deletable through the
    admin API (which enforces the same charset).
    """
    source = (
        override
        or _gguf_str(reader, "general.name")
        or _gguf_str(reader, "general.basename")
        or gguf_path.stem
    )
    model_id = _SAFE_ID.sub("-", source.lower()).strip("-")
    return model_id or "imported-model"


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# -- import marker ------------------------------------------------------------


def _marker_path(store: Path) -> Path:
    return store / MARKER_FILE


def load_marker(store: str | Path) -> dict[str, dict]:
    try:
        with open(_marker_path(Path(store)), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_marker(store: Path, marker: dict[str, dict]) -> None:
    final = _marker_path(store)
    tmp = final.with_name(f"{final.name}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(marker, fh, indent=2)
    tmp.replace(final)


# -- converter ----------------------------------------------------------------


def import_gguf(
    out_dir: str | Path,
    gguf_path: str | Path,
    *,
    model_id: str | None = None,
    layers_per_shard: int = 4,
    dtype: str = "fp16",
    tokenizer: object | None = None,
    tokenizer_ref: str | None = None,
    force: bool = False,
) -> ModelManifest | None:
    """Import one GGUF; returns the manifest, or None when already imported."""
    gguf = Path(gguf_path)
    store = Path(out_dir)

    reader = GGUFReader(str(gguf))
    arch = _gguf_str(reader, "general.architecture")
    if arch != "llama":
        raise ValueError(
            f"GGUF architecture {arch!r} is not supported — DAIN's kernels are "
            "Llama-family only (see Docs/stages.md S10)"
        )
    derived = model_id or _derive_model_id(reader, gguf)

    sha = _sha256_file(gguf)
    marker = load_marker(store)
    entry = marker.get(gguf.name)
    manifest_path = store / derived / "manifest.json"
    if not force and entry and entry.get("sha256") == sha and manifest_path.exists():
        log.info("gguf_skip file=%s model=%s (already imported)", gguf.name, derived)
        return None

    log.info(
        "gguf_import_start file=%s model=%s bytes=%d dtype=%s",
        gguf.name,
        derived,
        gguf.stat().st_size,
        dtype,
    )
    model = _load_llama_from_gguf(gguf)
    config = model.config
    if getattr(config, "model_type", None) != "llama":
        raise ValueError(
            f"GGUF produced a {getattr(config, 'model_type', '?')!r} model "
            "— only Llama is supported"
        )

    if tokenizer is None:
        if tokenizer_ref:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, use_fast=True)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(gguf.parent), gguf_file=str(gguf), use_fast=True
                )
            except Exception as exc:  # noqa: BLE001 — report every failure mode the same way
                raise ValueError(
                    "could not build a tokenizer from the GGUF metadata "
                    f"({exc}); pass --tokenizer <hf-repo> to source one from the Hub"
                ) from exc
    tokenizer_bytes = tokenizer.backend_tokenizer.to_str().encode("utf-8")  # type: ignore[attr-defined]
    tokenizer_hash = hashlib.sha256(tokenizer_bytes).hexdigest()

    state: dict[str, torch.Tensor] = dict(model.state_dict())
    # Tied embeddings: synthesize lm_head so the last shard always carries it
    # (same treatment as export_hf_model).
    if "lm_head.weight" not in state and "model.embed_tokens.weight" in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"].detach().clone()
    if "lm_head.weight" not in state:
        raise ValueError("GGUF contains no output weights (lm_head)")

    manifest = _write_shards(
        str(store),
        model_id=derived,
        name=_gguf_str(reader, "general.name") or derived,
        config=config,
        state=state,
        layers_per_shard=layers_per_shard,
        dtype_label=dtype,
        tokenizer_bytes=tokenizer_bytes,
        tokenizer_hash=tokenizer_hash,
    )

    marker[gguf.name] = {
        "sha256": sha,
        "model_id": derived,
        "imported_at": time.time(),
    }
    _save_marker(store, marker)
    log.info(
        "gguf_import_done file=%s model=%s shards=%d",
        gguf.name,
        derived,
        len(manifest.shards),
    )
    return manifest


def _load_llama_from_gguf(gguf: Path) -> LlamaForCausalLM:
    """Dequantize the GGUF into a LlamaForCausalLM.

    Deliberately not ``from_pretrained(..., gguf_file=...)``: that path goes
    through accelerate's meta-device init, which segfaults on the pinned
    torch-CPU/Windows stack (verified against 4.46.3). ``load_gguf_checkpoint``
    + an explicit ``load_state_dict(assign=True)`` does the same job without
    accelerate and without copying the (potentially multi-GB) tensors.
    """
    parsed = load_gguf_checkpoint(str(gguf), return_tensors=True)
    raw = {k: v for k, v in parsed["config"].items() if k != "_model_name_or_path"}
    config = LlamaConfig.from_dict(raw)
    model = LlamaForCausalLM(config)
    tensors = {k: v for k, v in parsed["tensors"].items() if k not in _GGUF_IGNORE_TENSORS}
    missing, unexpected = model.load_state_dict(tensors, strict=False, assign=True)
    # A missing lm_head is legitimate (tied embeddings — the caller synthesizes
    # it or relies on tying); anything else means the GGUF is incomplete.
    if unexpected or [k for k in missing if k != "lm_head.weight"]:
        raise ValueError(
            f"GGUF tensors do not map onto a Llama model "
            f"(missing={missing}, unexpected={unexpected})"
        )
    model.eval()
    return model


# -- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dain_node.import_gguf",
        description="Import Llama-family GGUF files into the DAIN model store.",
    )
    parser.add_argument("store", help="model store directory (e.g. ../model_store)")
    parser.add_argument(
        "gguf_file", nargs="?", help="single .gguf file (default: all in the store)"
    )
    parser.add_argument("--model-id", help="override the derived model id")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--layers-per-shard", type=int, default=4)
    parser.add_argument(
        "--tokenizer",
        help="HF repo providing a fast tokenizer, when the GGUF's embedded one cannot be converted",
    )
    parser.add_argument("--force", action="store_true", help="re-import even if already imported")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    store = Path(args.store)
    files = [Path(args.gguf_file)] if args.gguf_file else sorted(store.glob("*.gguf"))
    if not files:
        print(f"no .gguf files found in {store}")
        return 0

    failed = 0
    for gguf in files:
        if not gguf.is_file():
            print(f"error: {gguf} not found")
            failed += 1
            continue
        try:
            manifest = import_gguf(
                store,
                gguf,
                model_id=args.model_id,
                layers_per_shard=args.layers_per_shard,
                dtype=args.dtype,
                tokenizer_ref=args.tokenizer,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001 — CLI reports and continues with the rest
            print(f"error: {gguf.name}: {exc}")
            failed += 1
            continue
        if manifest is None:
            print(f"skip: {gguf.name} (already imported)")
        else:
            size = sum(s.size_bytes for s in manifest.shards)
            print(
                f"imported: {manifest.model_id} ({len(manifest.shards)} shards, "
                f"{size / 1024 / 1024:.1f} MiB, {manifest.dtype})"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
