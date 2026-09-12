"""Quantized-stage smoke tests (StageModel + backend abstraction, llm.py S22)."""

import pytest
import torch
from dain_common.schemas import ModelManifest, QuantizationSpec
from transformers import LlamaConfig, LlamaForCausalLM

from dain_node.llm import (
    StageModel,
    TorchAOInt4Backend,
    TorchFp16Backend,
    select_backend,
    select_device,
)
from dain_node.model_export import ExportConfig, export_model


def _tiny_manifest(tmp_path_factory, *, layers=8, hidden=128, inter=256):
    """Return (manifest, store_dir) for a hermetic INT4-exported tiny Llama."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    src = tmp_path_factory.mktemp("src")
    corpus = ["hello world", "the quick brown fox"] * 4
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=128,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer)
    real_vocab = len(tok.get_vocab())
    pt = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    cfg = LlamaConfig(
        vocab_size=real_vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=pt.convert_tokens_to_ids("</s>"),
        bos_token_id=pt.convert_tokens_to_ids("<s>"),
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).eval()
    model.save_pretrained(str(src))
    pt.save_pretrained(str(src))
    store = tmp_path_factory.mktemp("store")
    manifest = export_model(ExportConfig(
        source_dir=str(src),
        model_id="quant-smoke",
        output_store=str(store),
        group_size=128,
        activation_dtype="fp16",
        layers_per_shard=4,
    ))
    return manifest, store


@pytest.fixture(scope="module")
def quant_exported(tmp_path_factory):
    return _tiny_manifest(tmp_path_factory)


# ---------------------------------------------------------------------------
# device selection (real CUDA branches)
# ---------------------------------------------------------------------------


class TestSelectDevice:
    """validate the CUDA/CPU split of select_device.

    A real GPU is not a test dependency: the CUDA branch is exercised by
    monkeypatching ``torch.cuda.is_available`` — exactly the code path a
    CUDA-equipped node runs at stage build time.
    """

    @staticmethod
    def _manifest(**overrides) -> ModelManifest:
        base = dict(
            model_id="m",
            name="m",
            layers=4,
            hidden=8,
            heads=2,
            kv_heads=2,
            intermediate=16,
            vocab_size=16,
            eos_token_id=0,
        )
        base.update(overrides)
        return ModelManifest(**base)

    @staticmethod
    def _quant(layout: str) -> ModelManifest:
        return TestSelectDevice._manifest(
            format="torch_pt",
            quantization=QuantizationSpec(
                backend="torchao",
                scheme="int4_weight_only",
                bits=4,
                group_size=128,
                packing_layout=layout,
            ),
        )

    def test_legacy_cuda_when_available(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert select_device(self._manifest()) == "cuda:0"

    def test_legacy_cpu_when_without_cuda(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert select_device(self._manifest()) == "cpu"

    def test_tensor_core_tiled_requires_cuda(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="requires CUDA"):
            select_device(self._quant("tensor_core_tiled"))

    def test_tensor_core_tiled_lands_on_cuda(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert select_device(self._quant("tensor_core_tiled")) == "cuda:0"

    def test_int4_cpu_stays_on_cpu_even_with_cuda(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert select_device(self._quant("int4_cpu")) == "cpu"


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------

class TestSelectBackend:
    def test_quantized_manifest_selects_torchao(self, quant_exported):
        manifest, _ = quant_exported
        backend = select_backend(manifest)
        assert isinstance(backend, TorchAOInt4Backend)
        assert backend.is_quantized

    def test_legacy_manifest_selects_fp16(self):
        m = ModelManifest(
            model_id="x", name="x", layers=1, hidden=8, heads=1,
            kv_heads=1, intermediate=16, vocab_size=10, eos_token_id=0,
            rope_theta=1.0, shards=(),
        )
        backend = select_backend(m)
        assert isinstance(backend, TorchFp16Backend)
        assert not backend.is_quantized


def _load_quant_state(manifest, store):
    backend = select_backend(manifest)
    state: dict = {}
    for shard in manifest.shards:
        path = str(store / manifest.model_id / f"{shard.shard_id}.pt")
        state.update(backend.deserialize_shard(path))
    return backend, state


# ---------------------------------------------------------------------------
# shard deserialization
# ---------------------------------------------------------------------------

class TestShardDeserialize:
    def test_loads_aqts(self, quant_exported):
        manifest, store = quant_exported
        backend, state = _load_quant_state(manifest, store)
        aqt_count = sum(
            "AffineQuantizedTensor" in type(v).__name__
            for v in state.values()
        )
        assert aqt_count == 8 * 7


# ---------------------------------------------------------------------------
# StageModel quantized forward
# ---------------------------------------------------------------------------

class TestQuantizedStageModel:
    def test_forward_finite(self, quant_exported):
        manifest, store = quant_exported
        backend, state = _load_quant_state(manifest, store)
        tok_path = str(store / manifest.model_id / "tokenizer.json")
        stage = StageModel(
            manifest, 0, manifest.layers - 1, state,
            tokenizer_path=tok_path, backend=backend,
        )
        stage.begin_job()
        logits = stage.next_token_logits_full([75, 72, 79, 79, 82, 224, 90, 82])
        assert bool(torch.isfinite(logits).all())

    def test_dtype_label(self, quant_exported):
        manifest, store = quant_exported
        backend, state = _load_quant_state(manifest, store)
        tok_path = str(store / manifest.model_id / "tokenizer.json")
        stage = StageModel(
            manifest, 0, manifest.layers - 1, state,
            tokenizer_path=tok_path, backend=backend,
        )
        assert stage.dtype_label == "fp16"

    def test_full_layers(self, quant_exported):
        manifest, store = quant_exported
        backend, state = _load_quant_state(manifest, store)
        tok_path = str(store / manifest.model_id / "tokenizer.json")
        stage = StageModel(
            manifest, 0, manifest.layers - 1, state,
            tokenizer_path=tok_path, backend=backend,
        )
        assert stage.first is True
        assert stage.last is True
        assert len(stage.layers) == 8
