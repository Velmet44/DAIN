"""S18: real-model export path — HF Llama + fast tokenizer → fp16 shards.

Hermetic: the Llama model and BPE tokenizer are built locally (no hub), then run
through the same `export_hf_model` → `StageModel` path a real deploy uses.
"""

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from dain_node.hf_tokenizer import HFTokenizer
from dain_node.llm import StageModel
from dain_node.shard_export import export_hf_model, tiny_config

CORPUS = [
    "hello world hello world",
    "the quick brown fox jumps over the lazy dog",
    "once upon a time in a land far far away",
    "pickle jar on the highest shelf of the pantry",
] * 8


def build_bpe_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=96,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer=trainer)
    return tok


def hf_export(tmp_path, *, layers: int = 12) -> tuple[str, dict, object]:
    """Export a local 12L Llama with a trained BPE tokenizer; returns store dir,
    the FAQ Tokenizer, and the PreTrainedTokenizerFast (for reference encode)."""
    tok = build_bpe_tokenizer()
    cfg = tiny_config(layers=layers)
    cfg.vocab_size = tok.get_vocab_size()
    cfg.eos_token_id = tok.token_to_id("</s>")
    cfg.bos_token_id = tok.token_to_id("<s>")
    torch.manual_seed(7)
    model = LlamaForCausalLM(cfg)
    model.eval()
    pt = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    store = tmp_path / "hstore"
    manifest = export_hf_model(
        str(store), model_id="llama-hf-test", model=model, tokenizer=pt, dtype="fp16"
    )
    return str(store), manifest, pt


def test_export_hf_model_writes_fp16_tokenized_shards(tmp_path) -> None:
    store, manifest, _pt = hf_export(tmp_path)
    assert manifest.dtype == "fp16"
    assert manifest.layers == 12
    assert manifest.tokenizer_file == "tokenizer.json"
    assert len(manifest.tokenizer_hash) == 64
    assert len(manifest.shards) == 3  # 12 layers / 4 per shard
    assert (tmp_path / "hstore" / "llama-hf-test" / "tokenizer.json").exists()


def test_hf_tokenizer_round_trip_and_feed(tmp_path) -> None:
    store, manifest, pt = hf_export(tmp_path)
    path = f"{store}/{manifest.model_id}/tokenizer.json"
    tok = HFTokenizer(path)
    text = "hello world"
    ids = pt.encode(text)
    assert tok.encode(text) == ids
    assert tok.decode(ids) == text
    fragment = tok.feed(ids[0])  # per-token fragments must be strings
    assert isinstance(fragment, str)


def test_stage_model_loads_fp16_with_hf_tokenizer(tmp_path) -> None:
    store, manifest, pt = hf_export(tmp_path)
    shards = {
        s.shard_id: load_file(f"{store}/{manifest.model_id}/{s.shard_id}.safetensors")
        for s in manifest.shards
    }
    state: dict[str, torch.Tensor] = {}
    for shard in shards.values():
        state.update(shard)
    path = f"{store}/{manifest.model_id}/tokenizer.json"
    stage = StageModel(manifest, 0, manifest.layers - 1, state, tokenizer_path=path)
    assert stage.dtype is torch.float16
    assert stage.dtype_label == "fp16"
    assert stage.embed.weight.dtype == torch.float16
    assert stage.lm_head.weight.dtype == torch.float16

    stage.begin_job()
    ids = pt.encode("hello world")
    with torch.no_grad():
        logits = stage.next_token_logits_full(ids)
    assert logits.dtype == torch.float16
    assert torch.isfinite(logits).all()
    token = int(torch.argmax(logits[0]).item())
    assert 0 <= token < manifest.vocab_size


def test_export_hf_model_rejects_non_llama(tmp_path) -> None:
    tok = build_bpe_tokenizer()
    cfg = tiny_config(layers=4)
    cfg.vocab_size = tok.get_vocab_size()
    model = LlamaForCausalLM(cfg)
    model.config.model_type = "qwen2"  # simulate an unsupported architecture
    pt = PreTrainedTokenizerFast(tokenizer_object=tok)
    try:
        export_hf_model(str(tmp_path), model_id="qx", model=model, tokenizer=pt)
    except ValueError as exc:
        assert "Llama" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-Llama model")
