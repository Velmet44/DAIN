"""S4: StageModel parity against a full HF Llama reference + deterministic greedy.

The exported shards are seeded, so these tests are hermetic and reproducible:
same seed → same weights → same logits, forever.
"""

import asyncio

import pytest
import torch
from safetensors.torch import load_file
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from dain_node.llm import StageModel
from dain_node.shard_export import export_tiny_llama, tiny_config

PROMPT_IDS = [79, 115, 115, 105, 112, 33]  # arbitrary fixed byte ids
STEPS = 8


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    store = tmp_path_factory.mktemp("store")
    manifest = export_tiny_llama(str(store))
    shards = {
        s.shard_id: load_file(str(store / manifest.model_id / f"{s.shard_id}.safetensors"))
        for s in manifest.shards
    }
    return manifest, shards


def full_reference(manifest) -> LlamaForCausalLM:
    torch.manual_seed(1)
    config = tiny_config(layers=manifest.layers)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def greedy_reference(
    model: LlamaForCausalLM, ids: list[int], steps: int
) -> tuple[list[int], torch.Tensor]:
    """Cache-free greedy reference: recompute the full sequence every step."""
    current = list(ids)
    first_logits = None
    for step in range(steps):
        with torch.no_grad():
            logits = model(torch.tensor([current])).logits[0, -1, :]
        if step == 0:
            first_logits = logits.clone()
        token = int(torch.argmax(logits).item())
        current.append(token)
    return current[len(ids) :], first_logits


def test_stage_full_model_logits_parity(exported) -> None:
    manifest, shards = exported
    state: dict[str, torch.Tensor] = {}
    for shard in shards.values():
        state.update(shard)
    stage = StageModel(manifest, 0, manifest.layers - 1, state)
    stage.begin_job()
    with torch.no_grad():
        ours = stage.next_token_logits_full(PROMPT_IDS)
    reference = full_reference(manifest)
    with torch.no_grad():
        expected = reference(torch.tensor([PROMPT_IDS])).logits[0, -1, :]
    max_diff = float((ours[0] - expected).abs().max())
    assert max_diff < 1e-3, f"logits max diff {max_diff}"


def test_greedy_generation_is_deterministic(exported) -> None:
    manifest, shards = exported
    state: dict[str, torch.Tensor] = {}
    for shard in shards.values():
        state.update(shard)
    tokens_one = asyncio.run(_generate(manifest, shards, STEPS))
    tokens_two = asyncio.run(_generate(manifest, shards, STEPS))
    assert tokens_one == tokens_two and len(tokens_one) == STEPS


async def _generate(manifest, shards, steps: int) -> list[int]:
    state: dict[str, torch.Tensor] = {}
    for shard in shards.values():
        state.update(shard)
    stage = StageModel(manifest, 0, manifest.layers - 1, state)
    stage.begin_job()
    current = list(PROMPT_IDS)
    out = []
    for _ in range(steps):
        logits = stage.next_token_logits_full(current)
        token = int(torch.argmax(logits[0]).item())
        out.append(token)
        current = [token]
    return out


def test_pipeline_stage_logits_parity(exported) -> None:
    """The S5 core, proven at unit level: 4 stages × 4 layers ≡ full model."""
    manifest, shards = exported
    state: dict[str, torch.Tensor] = {}
    for shard in shards.values():
        state.update(shard)
    stages = [
        StageModel(manifest, start, min(start + 3, manifest.layers - 1), state)
        for start in range(0, manifest.layers, 4)
    ]
    for stage in stages:
        stage.begin_job()

    # Prefill through the chain: every stage runs its layers, last adds norm+head.
    hidden = stages[0].forward_ids(PROMPT_IDS)
    for stage in stages[1:]:
        hidden = stage.forward_hidden(hidden)
    ours = stages[-1].logits_from(hidden)[0, -1, :]
    reference = full_reference(manifest)
    with torch.no_grad():
        expected = reference(torch.tensor([PROMPT_IDS])).logits[0, -1, :]
    max_diff = float((ours - expected).abs().max())
    assert max_diff < 1e-3, f"pipeline logits max diff {max_diff}"


def test_pipeline_greedy_matches_full_reference(exported) -> None:
    """Greedy token streams must be identical across the pipeline split."""
    manifest, shards = exported
    state: dict[str, torch.Tensor] = {}
    for shard in shards.values():
        state.update(shard)
    stages = [
        StageModel(manifest, start, min(start + 3, manifest.layers - 1), state)
        for start in range(0, manifest.layers, 4)
    ]
    for stage in stages:
        stage.begin_job()

    reference_model = full_reference(manifest)
    reference_tokens, _ = greedy_reference(reference_model, PROMPT_IDS, STEPS)

    distributed_tokens = []
    hidden = stages[0].forward_ids(PROMPT_IDS)
    for _ in range(STEPS):
        for stage in stages[1:]:
            hidden = stage.forward_hidden(hidden)
        logits = stages[-1].logits_from(hidden)[0, -1, :]
        token = int(torch.argmax(logits).item())
        distributed_tokens.append(token)
        hidden = stages[0].embed_one(token)
    assert distributed_tokens == reference_tokens
