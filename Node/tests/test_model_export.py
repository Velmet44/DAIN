"""Smoke tests for the TorchAO INT4 model-export pipeline (model_export.py).

These fixtures build a *tiny* Llama (8 layers, hidden=128) with a real BPE
tokenizer whose vocab matches the model, so the full encode → quantize → save
→ load → StageModel → forward path is exercised end-to-end.
"""

import json

import pytest
import torch
from dain_common.schemas import ModelManifest
from transformers import LlamaConfig, LlamaForCausalLM

from dain_node.model_export import (
    LAYOUT_CPU,
    ExportConfig,
    _check_tokenizer_vocab,
    detect_model,
    export_model,
    pick_layout,
    validate_source,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def src_dir(tmp_path):
    """Populate a HuggingFace-style source directory (real BPE, matching vocab)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    src = tmp_path / "src"
    src.mkdir()
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
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=8,
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
    return src, cfg, real_vocab


@pytest.fixture()
def store(tmp_path):
    return tmp_path / "store"


# ---------------------------------------------------------------------------
# validate_source / detect_model
# ---------------------------------------------------------------------------

class TestValidateSource:
    def test_missing_config(self, tmp_path):
        with pytest.raises(ValueError, match="no config.json"):
            validate_source(str(tmp_path))

    def test_missing_weights(self, tmp_path):
        (tmp_path / "config.json").write_text('{"model_type":"llama"}')
        with pytest.raises(ValueError, match="no model weights"):
            validate_source(str(tmp_path))


class TestDetectModel:
    def test_returns_manifest_builder(self, src_dir, store):
        from transformers import AutoConfig
        src, cfg, vocab = src_dir
        config = AutoConfig.from_pretrained(str(src), trust_remote_code=False)
        adapter = detect_model(config)
        assert adapter.architecture == "llama"

    def test_layout(self):
        layout = pick_layout()
        assert layout in (LAYOUT_CPU, "tensor_core_tiled")


# ---------------------------------------------------------------------------
# export_model
# ---------------------------------------------------------------------------

class TestExportModel:
    def test_basic_export_fields(self, src_dir, store):
        src, cfg, vocab = src_dir
        manifest = export_model(ExportConfig(
            source_dir=str(src),
            model_id="t1",
            output_store=str(store),
            group_size=128,
            activation_dtype="fp16",
            layers_per_shard=4,
        ))
        assert manifest.model_id == "t1"
        assert manifest.format == "torch_pt"
        assert manifest.architecture == "llama"
        assert manifest.layers == 8
        assert manifest.vocab_size == vocab
        assert manifest.quantization is not None
        assert manifest.quantization.is_quantized
        assert manifest.quantization.backend == "torchao"
        assert manifest.quantization.bits == 4
        assert manifest.quantization.group_size == 128
        assert manifest.quantization.packing_layout in (LAYOUT_CPU, "tensor_core_tiled")
        assert manifest.artifact_version == 1

    def test_manifest_json_written(self, src_dir, store):
        src, cfg, vocab = src_dir
        export_model(ExportConfig(
            source_dir=str(src),
            model_id="t2",
            output_store=str(store),
            group_size=128,
            activation_dtype="fp16",
        ))
        manifest_path = store / "t2" / "manifest.json"
        assert manifest_path.exists()
        loaded = ModelManifest.model_validate(json.loads(manifest_path.read_text()))
        assert loaded.model_id == "t2"
        assert loaded.format == "torch_pt"

    def test_shards_are_pt(self, src_dir, store):
        src, cfg, vocab = src_dir
        manifest = export_model(ExportConfig(
            source_dir=str(src),
            model_id="t3",
            output_store=str(store),
            group_size=128,
            activation_dtype="fp16",
            layers_per_shard=8,
        ))
        assert len(manifest.shards) == 1
        shard_path = store / "t3" / f"{manifest.shards[0].shard_id}.pt"
        assert shard_path.exists()

    def test_shards_contain_aqts(self, src_dir, store):
        src, cfg, vocab = src_dir
        manifest = export_model(ExportConfig(
            source_dir=str(src),
            model_id="t4",
            output_store=str(store),
            group_size=128,
            activation_dtype="fp16",
        ))
        state = {}
        for shard in manifest.shards:
            state.update(torch.load(store / "t4" / f"{shard.shard_id}.pt", weights_only=False))
        aqt_count = sum("AffineQuantizedTensor" in type(v).__name__ for v in state.values())
        assert aqt_count == 8 * 7  # 7 quantizable projections per Llama decoder layer

    def test_idempotency(self, src_dir, store):
        src, cfg, vocab = src_dir
        export_model(ExportConfig(
            source_dir=str(src),
            model_id="t5",
            output_store=str(store),
            group_size=128,
            activation_dtype="fp16",
        ))
        with pytest.raises(FileExistsError):
            export_model(ExportConfig(
                source_dir=str(src),
                model_id="t5",
                output_store=str(store),
                group_size=128,
                activation_dtype="fp16",
            ))

    def test_force_overwrite(self, src_dir, store):
        src, cfg, vocab = src_dir
        export_model(ExportConfig(
            source_dir=str(src),
            model_id="t6",
            output_store=str(store),
            group_size=128,
            activation_dtype="fp16",
        ))
        manifest = export_model(ExportConfig(
            source_dir=str(src),
            model_id="t6",
            output_store=str(store),
            group_size=128,
            activation_dtype="fp16",
            force=True,
        ))
        assert len(manifest.shards) == 2  # 8 layers, 4 per shard

    def test_export_refuses_real_model_without_tokenizer(self, tmp_path, store):
        """A real model without any tokenizer must fail, not publish a manifest
        whose nodes would silently run a byte-level vocabulary."""
        src = tmp_path / "src-notok"
        src.mkdir()
        (src / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (src / "model.safetensors").write_bytes(b"x" * 8)
        with pytest.raises(ValueError, match="tokenizer"):
            export_model(ExportConfig(
                source_dir=str(src),
                model_id="t7",
                output_store=str(store),
                group_size=128,
                activation_dtype="fp16",
            ))
        assert not (store / "t7").exists()  # nothing partially published


# ---------------------------------------------------------------------------
# _check_tokenizer_vocab
# ---------------------------------------------------------------------------

class TestCheckTokenizerVocab:
    def test_raises_on_mismatch(self, src_dir, store):
        src, cfg, vocab = src_dir
        bad_cfg = LlamaConfig(
            vocab_size=10,  # far too small for real tokenizer
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        with pytest.raises(ValueError, match="exceeds model vocab_size"):
            _check_tokenizer_vocab(src, bad_cfg)

    def test_passes_when_tokenizer_absent(self, tmp_path):
        _check_tokenizer_vocab(tmp_path, LlamaConfig(vocab_size=32))
