"""GGUF import (session 14): llama.cpp-layout GGUF -> DAIN shards.

Hermetic: a seeded tiny Llama is written into a real GGUF file (llama.cpp
tensor naming + the q/k rope half-split permute, GQA-aware for k) and run
through `import_gguf`, exactly what a user-dropped ``*.gguf`` goes through.
"""

import json
import shutil

import pytest
import torch
from dain_common.schemas import ModelManifest
from gguf import GGUFWriter
from safetensors.torch import load_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from dain_node.import_gguf import MARKER_FILE, import_gguf, load_marker, main
from dain_node.shard_export import tiny_config

CORPUS = ["hello world hello world", "the quick brown fox jumps over the lazy dog"] * 8


def build_bpe_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=96,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer)
    return tok


def _permute_qk(w: torch.Tensor, n_head: int) -> torch.Tensor:
    """llama.cpp write-side rope permute (inverse of transformers' reverse_permute)."""
    dim = w.shape[0] // n_head // 2
    return (
        w.reshape(n_head, 2, dim, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape).contiguous()
    )


def write_tiny_gguf(path, *, layers: int = 8, arch: str = "llama") -> tuple[dict, object]:
    """Write a seeded tiny Llama as a GGUF the way llama.cpp does."""
    torch.manual_seed(3)
    tok = build_bpe_tokenizer()
    cfg = tiny_config(layers=layers)
    cfg.vocab_size = tok.get_vocab_size()
    cfg.eos_token_id = tok.token_to_id("</s>")
    cfg.bos_token_id = tok.token_to_id("<s>")
    model = LlamaForCausalLM(cfg)
    model.eval()
    state = model.state_dict()

    heads, kv_heads, hidden = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size
    w = GGUFWriter(str(path), arch)
    w.add_string("general.name", "DAIN Test Tiny")
    if arch == "llama":
        w.add_uint32("llama.context_length", cfg.max_position_embeddings)
        w.add_uint32("llama.embedding_length", hidden)
        w.add_uint32("llama.block_count", cfg.num_hidden_layers)
        w.add_uint32("llama.feed_forward_length", cfg.intermediate_size)
        w.add_uint32("llama.attention.head_count", heads)
        w.add_uint32("llama.attention.head_count_kv", kv_heads)
        w.add_uint32("llama.rope.dimension_count", hidden // heads)
        w.add_float32("llama.rope.freq_base", cfg.rope_theta)
        w.add_uint32("llama.vocab_size", cfg.vocab_size)
        w.add_uint32("tokenizer.ggml.eos_token_id", cfg.eos_token_id)
        w.add_uint32("tokenizer.ggml.bos_token_id", cfg.bos_token_id)
        # Embedded gpt2-style BPE (vocab + merges from the trained tokenizer's
        # JSON) so the import exercises the offline embedded-tokenizer path.
        tj = json.loads(tok.to_str())["model"]
        vocab_items = sorted(tj["vocab"].items(), key=lambda kv: kv[1])
        specials = {"<unk>", "<s>", "</s>", "<pad>"}
        w.add_string("tokenizer.ggml.model", "gpt2")
        w.add_array("tokenizer.ggml.tokens", [t for t, _ in vocab_items])
        w.add_array(
            "tokenizer.ggml.token_type", [1 if t in specials else 0 for t, _ in vocab_items]
        )
        w.add_array("tokenizer.ggml.scores", [0.0] * len(vocab_items))
        w.add_array("tokenizer.ggml.merges", list(tj["merges"]))

    def t(name: str, tensor: torch.Tensor) -> None:
        w.add_tensor(name, tensor.detach().numpy())

    t("token_embd.weight", state["model.embed_tokens.weight"])
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        g = f"blk.{i}."
        t(g + "attn_norm.weight", state[p + "input_layernorm.weight"])
        t(g + "attn_q.weight", _permute_qk(state[p + "self_attn.q_proj.weight"], heads))
        t(g + "attn_k.weight", _permute_qk(state[p + "self_attn.k_proj.weight"], kv_heads))
        t(g + "attn_v.weight", state[p + "self_attn.v_proj.weight"])
        t(g + "attn_output.weight", state[p + "self_attn.o_proj.weight"])
        t(g + "ffn_norm.weight", state[p + "post_attention_layernorm.weight"])
        t(g + "ffn_gate.weight", state[p + "mlp.gate_proj.weight"])
        t(g + "ffn_up.weight", state[p + "mlp.up_proj.weight"])
        t(g + "ffn_down.weight", state[p + "mlp.down_proj.weight"])
    t("output_norm.weight", state["model.norm.weight"])
    t("output.weight", state["lm_head.weight"])
    # Real Llama-3.x GGUFs carry this rotary-inv-freq cache buffer; the
    # converter must drop it instead of failing the tensor-mapping check.
    head_dim = hidden // heads
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    w.add_tensor("rope_freqs.weight", inv_freq.numpy())

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    pt = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    return state, pt


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory):
    """One GGUF written per session; tests copy it into their own fresh store."""
    path = tmp_path_factory.mktemp("ggufsrc") / "dain-test-tiny.Q8_0.gguf"
    state, pt = write_tiny_gguf(path)
    return path, state, pt


@pytest.fixture
def store_with_gguf(tmp_path, tiny_gguf):
    """A fresh model store containing one copy of the session GGUF."""
    gguf, state, pt = tiny_gguf
    shutil.copy(gguf, tmp_path / gguf.name)
    return tmp_path, tmp_path / gguf.name, state, pt


def test_import_round_trip(store_with_gguf) -> None:
    store, gguf, state, pt = store_with_gguf
    manifest = import_gguf(store, gguf, tokenizer=pt, dtype="fp32")
    assert manifest is not None
    assert manifest.model_id == "dain-test-tiny"  # derived from general.name
    assert manifest.layers == 8
    assert manifest.tokenizer_file == "tokenizer.json"
    assert manifest.tokenizer_hash

    # Weights survived the GGUF round trip (fp32 => exact).
    manifest_path = store / manifest.model_id / "manifest.json"
    on_disk = ModelManifest.model_validate_json(manifest_path.read_text())
    assert len(on_disk.shards) == 2  # 8 layers / 4 per shard
    tensors = {}
    for shard in on_disk.shards:
        tensors.update(load_file(str(store / manifest.model_id / f"{shard.shard_id}.safetensors")))
    torch.testing.assert_close(
        tensors["model.embed_tokens.weight"], state["model.embed_tokens.weight"]
    )
    torch.testing.assert_close(
        tensors["model.layers.0.self_attn.q_proj.weight"],
        state["model.layers.0.self_attn.q_proj.weight"],
    )
    torch.testing.assert_close(
        tensors["model.layers.0.self_attn.k_proj.weight"],
        state["model.layers.0.self_attn.k_proj.weight"],
    )  # GQA permute round trip
    torch.testing.assert_close(tensors["lm_head.weight"], state["lm_head.weight"])


def test_import_is_idempotent(store_with_gguf) -> None:
    store, gguf, state, pt = store_with_gguf
    assert import_gguf(store, gguf, tokenizer=pt, dtype="fp32") is not None
    marker = load_marker(store)
    entry = marker[gguf.name]
    assert entry["model_id"] == "dain-test-tiny"
    assert len(entry["sha256"]) == 64
    # Second run: skipped.
    assert import_gguf(store, gguf, tokenizer=pt, dtype="fp32") is None
    # Deleting the model dir re-imports; a changed file (force) re-imports too.
    shutil.rmtree(store / "dain-test-tiny")
    assert import_gguf(store, gguf, tokenizer=pt, dtype="fp32") is not None
    assert import_gguf(store, gguf, tokenizer=pt, dtype="fp32", force=True) is not None


def test_rejects_non_llama_architecture(tmp_path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    gguf = store / "qwen.gguf"
    write_tiny_gguf(gguf, arch="qwen2")
    with pytest.raises(ValueError, match="Llama-family"):
        import_gguf(store, gguf, tokenizer=object())


def test_cli_imports_all_pending(store_with_gguf, capsys) -> None:
    store, gguf, state, pt = store_with_gguf
    rc = main([str(store), "--dtype", "fp32"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "imported: dain-test-tiny" in out
    marker = json.loads((store / MARKER_FILE).read_text())
    assert gguf.name in marker
    # Second run: nothing to do.
    rc = main([str(store)])
    assert rc == 0
    assert "skip" in capsys.readouterr().out
