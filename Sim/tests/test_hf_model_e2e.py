"""S18 checkpoint: real-model e2e — 2 agents run a fp16 HF-exported Llama.

Proves the new real-model path end to end without the hub: a small Llama + BPE
tokenizer is exported via `export_hf_model` (fp16), two agents split it (12
layers → 2 stages × 6), and a streamed completion is verified against a greedy
HF reference decoded with the same tokenizer.
"""

import asyncio
import os
import subprocess
import sys
import time

import httpx
import torch
from dain_common.schemas import NodeState
from dain_coordinator.settings import CoordinatorSettings
from dain_node.shard_export import export_hf_model, tiny_config
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from dain_sim.dev import ADMIN_HEADERS, ADMIN_KEY, API_KEY, JOIN_TOKEN
from dain_sim.server import start_server, stop_server

MODEL_ID = "hf-e2e-llama"
PROMPT = "hello world"
MAX_TOKENS = 12

CORPUS = [
    "hello world hello world",
    "the quick brown fox jumps over the lazy dog",
    "once upon a time in a land far far away",
] * 10


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


def hf_reference_text(tok: Tokenizer, prompt: str, steps: int) -> str:
    torch.manual_seed(7)
    cfg = tiny_config(layers=12)
    cfg.vocab_size = tok.get_vocab_size()
    cfg.eos_token_id = tok.token_to_id("</s>")
    model = LlamaForCausalLM(cfg).to(dtype=torch.float16).eval()
    current = tok.encode(prompt).ids
    out = []
    with torch.no_grad():
        for _ in range(steps):
            token = int(torch.argmax(model(torch.tensor([current])).logits[0, -1]).item())
            out.append(token)
            current.append(token)
    return tok.decode(out, skip_special_tokens=True)


def spawn_agent(server_port: int, workdir, node_id: str) -> subprocess.Popen:
    env = dict(
        os.environ,
        DAIN_COORD_URL=f"ws://127.0.0.1:{server_port}",
        DAIN_JOIN_TOKEN=JOIN_TOKEN,
        DAIN_NODE_ID=node_id,
        DAIN_HEARTBEAT_S="0.5",
        DAIN_NODE_STATE_PATH=str(workdir / f"{node_id}_state.json"),
        DAIN_MODEL=MODEL_ID,
        DAIN_MODEL_CACHE=str(workdir / f"{node_id}_cache"),
    )
    return subprocess.Popen([sys.executable, "-m", "dain_node"], cwd=workdir, env=env)


async def wait_connected(
    client: httpx.AsyncClient, server, count: int, timeout_s: float = 40.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        listing = (await client.get(f"{server.base_url}/admin/nodes", headers=ADMIN_HEADERS)).json()
        connected = sum(1 for n in listing if n.get("connected"))
        if connected >= count:
            return
        await asyncio.sleep(0.25)
    raise AssertionError(f"only {connected}/{count} nodes with live WS in time")


async def wait_online(
    client: httpx.AsyncClient, server, count: int, timeout_s: float = 40.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        listing = (await client.get(f"{server.base_url}/admin/nodes", headers=ADMIN_HEADERS)).json()
        online = sum(1 for n in listing if n["state"] == NodeState.ONLINE.value)
        if online >= count:
            return
        await asyncio.sleep(0.25)
    raise AssertionError(f"only {online}/{count} nodes ONLINE in time")


def test_hf_model_e2e_two_agents(tmp_path) -> None:
    tok = build_bpe_tokenizer()
    expected_text = hf_reference_text(tok, PROMPT, MAX_TOKENS)
    assert expected_text, "reference generation must produce text"

    async def main() -> None:
        store_dir = tmp_path / "model_store"
        cfg = tiny_config(layers=12)
        cfg.vocab_size = tok.get_vocab_size()
        cfg.eos_token_id = tok.token_to_id("</s>")
        torch.manual_seed(7)
        model = LlamaForCausalLM(cfg).eval()
        pt = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
        export_hf_model(
            str(store_dir), model_id=MODEL_ID, model=model, tokenizer=pt, dtype="fp16"
        )

        settings = CoordinatorSettings(
            db_path=str(tmp_path / "coordinator.sqlite3"),
            model_store_dir=str(store_dir),
            heartbeat_interval_s=0.5,
            offline_after_missed=3,
            monitor_tick_s=0.25,
            api_key=API_KEY,
            admin_api_key=ADMIN_KEY,
            join_token=JOIN_TOKEN,
            job_timeout_s=90.0,
            layers_per_node_target=6,  # 12 layers → 2 stages
        )
        server = await start_server(settings)
        procs: list[subprocess.Popen] = []
        workdir = tmp_path / "agents"
        workdir.mkdir()
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                procs.append(spawn_agent(server.port, workdir, "hf-node-0"))
                procs.append(spawn_agent(server.port, workdir, "hf-node-1"))
                await wait_online(client, server, 2)
                await wait_connected(client, server, 2)

                frames = []
                job_id = None
                async with client.stream(
                    "POST",
                    f"{server.base_url}/v1/completions",
                    headers={"X-API-Key": API_KEY},
                    json={
                        "model_id": MODEL_ID,
                        "prompt": PROMPT,
                        "max_tokens": MAX_TOKENS,
                        "stream": True,
                    },
                ) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        import json

                        frame = json.loads(line[len("data: ") :])
                        job_id = job_id or frame.get("job_id")
                        frames.append(frame)
                text = "".join(f["token"] for f in frames if f.get("type") == "token")

                assert text == expected_text, (
                    f"distributed HF stream mismatch:\n{text!r}\n{expected_text!r}"
                )

                view = (
                    await client.get(
                        f"{server.base_url}/v1/jobs/{job_id}", headers={"X-API-Key": API_KEY}
                    )
                ).json()
                assert len(view["stages"]) == 2, view["stages"]
                assert [s["layer_start"] for s in view["stages"]] == [0, 6]
                assert all(s["latency_ms"] is not None for s in view["stages"])
        finally:
            for proc in procs:
                proc.terminate()
            await asyncio.to_thread(lambda: [p.wait() for p in procs])
            await stop_server(server)

    asyncio.run(main())
