"""JobHandler unit tests (S4/S7/S13): stage execution, token streaming, S7 ack +
early-activation buffering, and stage-retry replay — against the real exported
shards, so verification exercises actual weights (hermetic, seeded).

Covers the previously-untested `dain_node/jobs.py` module: single-stage entry,
distributed entry ↔ sampling loop, last-stage decode, buffered replay for a
reassigned stage, transactional S7 ack before a fire-and-forget step, and
graceful shutdown of in-flight generation.
"""

import asyncio
import struct
import time
from pathlib import Path

import pytest
import torch
from dain_common.schemas import (
    ActivationRelayHeader,
    Envelope,
    GenerationParams,
    JobAssign,
    JobState,
    MessageType,
    StageAssignment,
    StageRetry,
    parse_payload,
)
from safetensors.torch import load_file

from dain_node.jobs import JobHandler
from dain_node.llm import StageModel
from dain_node.settings import NodeSettings
from dain_node.shard_export import export_tiny_llama


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    store_dir = tmp_path_factory.mktemp("store")
    manifest = export_tiny_llama(str(store_dir))
    shards = {
        s.shard_id: load_file(str(store_dir / manifest.model_id / f"{s.shard_id}.safetensors"))
        for s in manifest.shards
    }
    return store_dir, manifest, shards


class FakeStore:
    """Minimal model-store client: serves the exported manifest, no HTTP."""

    def __init__(self, manifest) -> None:
        self.manifest = manifest
        self.base = None

    async def fetch_manifest(self, model_id: str, *, refresh: bool = False) -> object:
        assert model_id == self.manifest.model_id
        return self.manifest

    def set_base_url(self, base_url: str) -> None:
        self.base = base_url

    async def close(self) -> None:
        """Result store teardown hook; the fake keeps no resources."""
        return None


class BoomStore(FakeStore):
    async def fetch_manifest(self, model_id: str, *, refresh: bool = False) -> object:
        raise RuntimeError("boom")


def make_fake_fetch(store_dir: Path, manifest):
    """Build the same stage a real fetch_stage would, straight from the export."""
    _state: dict[str, torch.Tensor] = {}

    def state() -> dict[str, torch.Tensor]:
        if not _state:
            for shard in manifest.shards:
                _state.update(
                    load_file(str(store_dir / manifest.model_id / f"{shard.shard_id}.safetensors"))
                )
        return _state

    async def fake(store, model_manifest, layer_start: int, layer_end: int):
        return StageModel(model_manifest, layer_start, layer_end, state()), {}

    return fake


def make_assign(
    job_id: str,
    manifest,
    layer_ranges: list[tuple[int, int]],
    my_stage_idx: int,
    *,
    prompt: str | None,
    max_tokens: int = 5,
) -> JobAssign:
    stages = tuple(
        StageAssignment(
            stage_idx=i, node_id="node-test", shard_id="x", layer_start=a, layer_end=b
        )
        for i, (a, b) in enumerate(layer_ranges)
    )
    return JobAssign(
        job_id=job_id,
        model_id=manifest.model_id,
        my_stage_idx=my_stage_idx,
        stages=stages,
        prompt=prompt,
        params=GenerationParams(max_tokens=max_tokens, temperature=0),
    )


async def wait_until(pred, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"condition not met within {timeout_s}s")


def make_handler(store) -> tuple[JobHandler, list[Envelope], list[bytes]]:
    envelopes: list[Envelope] = []
    raw: list[bytes] = []

    async def record_env(env: Envelope) -> None:
        envelopes.append(env)

    async def record_bytes(data: bytes) -> None:
        raw.append(data)

    handler = JobHandler(NodeSettings(job_timeout_s=30.0), store)
    handler.bind(record_env, record_bytes)
    return handler, envelopes, raw


def events(envelopes: list[Envelope], msg_type: MessageType) -> list:
    return [parse_payload(e) for e in envelopes if e.type == msg_type]


def final_token_batch(envelopes: list[Envelope]):
    for e in envelopes:
        if e.type == MessageType.TOKEN_BATCH and parse_payload(e).is_final:
            return parse_payload(e)
    return None


def hidden_payload(manifest, seq: int) -> tuple[ActivationRelayHeader, bytes]:
    tensor = torch.randn(1, 1, manifest.hidden)  # plausible forward_ids([tok]) shape
    data = tensor.to(torch.float32).contiguous().numpy().tobytes()
    header = ActivationRelayHeader(
        job_id="j000",
        stage_idx=0,  # from the (entry) upstream stage
        attempt=0,
        seq=seq,
        dtype="fp32",
        role="hidden",
        shape=(1, 1, manifest.hidden),
        n_bytes=len(data),
        is_final=False,
    )
    return header, data


async def feed(handler: JobHandler, header: ActivationRelayHeader, data: bytes) -> None:
    await handler.on_activation_header(header)
    await handler.on_payload(data)


def test_single_stage_streams_tokens_to_final(exported, monkeypatch) -> None:
    async def main() -> None:
        store_dir, manifest, _shards = exported
        monkeypatch.setattr("dain_node.jobs.fetch_stage", make_fake_fetch(store_dir, manifest))
        handler, envelopes, _raw = make_handler(FakeStore(manifest))

        assign = make_assign(
            "job-single", manifest, [(0, manifest.layers - 1)], 0, prompt="hi", max_tokens=5
        )
        await handler.on_job_assign(assign)
        rt = handler.jobs["job-single"]
        await asyncio.wait_for(rt.task, timeout=60)

        batches = events(envelopes, MessageType.TOKEN_BATCH)
        statuses = events(envelopes, MessageType.JOB_STATUS)
        final = [b for b in batches if b.is_final]
        assert final, "expected a final TOKEN_BATCH"
        assert final[0].finish_reason in ("length", "eos")
        assert final[0].tokens == ()
        # Non-final token frames must precede the terminal one.
        first_final = batches.index(final[0])
        assert any(not b.is_final for b in batches[:first_final])
        assert any(s.state == JobState.COMPLETED for s in statuses), statuses
        assert rt.generated >= 1
        assert handler._background == set()

    asyncio.run(main())


def test_distributed_entry_consumes_sampled_tokens(exported, monkeypatch) -> None:
    async def main() -> None:
        store_dir, manifest, _shards = exported
        monkeypatch.setattr("dain_node.jobs.fetch_stage", make_fake_fetch(store_dir, manifest))
        handler, envelopes, _raw = make_handler(FakeStore(manifest))

        assign = make_assign(
            "job-entry", manifest, [(0, 3), (4, manifest.layers - 1)], 0, prompt="hi", max_tokens=5
        )
        await handler.on_job_assign(assign)
        rt = handler.jobs["job-entry"]
        await wait_until(lambda: any(e.type == MessageType.ACTIVATION_RELAY for e in envelopes))

        # The sampling stage echoes 3 tokens back, then signals final.
        for seq in range(3):
            hdr = ActivationRelayHeader(
                job_id="job-entry", stage_idx=1, attempt=0, seq=seq,
                dtype="int64", role="sampled_token", shape=(1,), n_bytes=8, is_final=False,
            )
            await feed(handler, hdr, struct.pack("<q", 42 + seq))
        fin = ActivationRelayHeader(
            job_id="job-entry", stage_idx=1, attempt=0, seq=3,
            dtype="int64", role="sampled_token", shape=(1,), n_bytes=8, is_final=True,
        )
        await feed(handler, fin, struct.pack("<q", 45))
        await asyncio.wait_for(rt.task, timeout=60)

        statuses = events(envelopes, MessageType.JOB_STATUS)
        assert any(s.state == JobState.COMPLETED for s in statuses), statuses
        assert rt.generated == 3  # one per non-final sampled token

    asyncio.run(main())


def test_last_stage_ack_then_decode_to_final(exported, monkeypatch) -> None:
    async def main() -> None:
        store_dir, manifest, _shards = exported
        torch.manual_seed(0)
        monkeypatch.setattr("dain_node.jobs.fetch_stage", make_fake_fetch(store_dir, manifest))
        handler, envelopes, _raw = make_handler(FakeStore(manifest))

        assign = make_assign(
            "job-last", manifest, [(0, 3), (4, manifest.layers - 1)], 1, prompt=None, max_tokens=5
        )
        await handler.on_job_assign(assign)
        rt = handler.jobs["job-last"]
        await asyncio.wait_for(rt.task, timeout=60)  # stage bootstrapped

        # Drive decode like the entry stage would: one hidden activation per
        # generated token until the last stage reports the job finished.
        for seq in range(5):
            hdr, data = hidden_payload(manifest, seq)
            hdr = hdr.model_copy(update={"job_id": "job-last"})
            before = rt.generated
            await feed(handler, hdr, data)
            await wait_until(
                lambda before=before: final_token_batch(envelopes) is not None
                or rt.generated > before
            )
            if final_token_batch(envelopes) is not None:
                break
        if final_token_batch(envelopes) is None:
            await wait_until(lambda: final_token_batch(envelopes) is not None)

        statuses = events(envelopes, MessageType.JOB_STATUS)
        # S7: every inbound step is acknowledged before its fire-and-forget runner.
        assert any(
            s.detail == "step_ack" and s.state == JobState.RUNNING for s in statuses
        ), statuses
        final = final_token_batch(envelopes)
        assert final is not None and final.finish_reason in ("length", "eos")
        sampled = events(envelopes, MessageType.ACTIVATION_RELAY)
        token_back = [h for h in sampled if h.role == "sampled_token"]
        assert any(h.is_final for h in token_back)
        assert any(s.state == JobState.COMPLETED for s in statuses), statuses
        assert handler._background == set()

    asyncio.run(main())


def test_early_activation_buffered_and_replayed(exported, monkeypatch) -> None:
    async def main() -> None:
        store_dir, manifest, _shards = exported
        torch.manual_seed(0)
        monkeypatch.setattr("dain_node.jobs.fetch_stage", make_fake_fetch(store_dir, manifest))
        handler, envelopes, _raw = make_handler(FakeStore(manifest))
        job_id = "job-early"

        # Three activation chunks arrive before the replacement stage exists.
        for seq in range(3):
            hdr, data = hidden_payload(manifest, seq)
            hdr = hdr.model_copy(update={"job_id": job_id})
            await feed(handler, hdr, data)
        assert handler._early[job_id] and len(handler._early[job_id]) == 3

        assign = make_assign(
            job_id, manifest, [(0, 3), (4, manifest.layers - 1)], 1, prompt=None, max_tokens=5
        )
        await handler.on_job_assign(assign)
        rt = handler.jobs[job_id]
        await asyncio.wait_for(rt.task, timeout=60)  # bootstrap replays the buffer

        assert job_id not in handler._early
        assert rt.generated == 3, "replay must resume from exactly where it stopped"
        assert rt.generated <= 5
        assert final_token_batch(envelopes) is None, "3/5 steps must not have finished the job"

    asyncio.run(main())


def test_stage_retry_replays_buffered_activation(exported, monkeypatch) -> None:
    async def main() -> None:
        store_dir, manifest, _shards = exported
        monkeypatch.setattr("dain_node.jobs.fetch_stage", make_fake_fetch(store_dir, manifest))
        handler, envelopes, raw = make_handler(FakeStore(manifest))

        assign = make_assign(
            "job-retry", manifest, [(0, 3), (4, manifest.layers - 1)], 0, prompt="hi", max_tokens=5
        )
        await handler.on_job_assign(assign)
        rt = handler.jobs["job-retry"]
        await wait_until(lambda: rt.buffered is not None)

        relayed_before = sum(1 for e in envelopes if e.type == MessageType.ACTIVATION_RELAY)
        bytes_before = sum(len(b) for b in raw)

        await handler.on_stage_retry(
            StageRetry(job_id="job-retry", stage_idx=1, attempt=1, reason="watchdog")
        )

        relays = events(envelopes, MessageType.ACTIVATION_RELAY)
        assert len(relays) == relayed_before + 1, "retry must re-emit the buffered activation"
        replayed = relays[-1]
        assert replayed.attempt == 1 and replayed.is_final is False
        assert sum(len(b) for b in raw) == bytes_before + len(rt.buffered[1]), (
            "payload bytes re-sent"
        )

        # Retries of a non-adjacent stage (this one included) must not emit.
        count = len(events(envelopes, MessageType.ACTIVATION_RELAY))
        await handler.on_stage_retry(
            StageRetry(job_id="job-retry", stage_idx=0, attempt=1, reason="watchdog")
        )
        assert len(events(envelopes, MessageType.ACTIVATION_RELAY)) == count

    asyncio.run(main())


def test_setup_failure_emits_error_final(exported, monkeypatch) -> None:
    async def main() -> None:
        store_dir, manifest, _shards = exported
        monkeypatch.setattr("dain_node.jobs.fetch_stage", make_fake_fetch(store_dir, manifest))
        handler, envelopes, _raw = make_handler(BoomStore(manifest))

        assign = make_assign(
            "job-boom", manifest, [(0, manifest.layers - 1)], 0, prompt="hi", max_tokens=5
        )
        await handler.on_job_assign(assign)

        error = [
            b
            for b in events(envelopes, MessageType.TOKEN_BATCH)
            if b.finish_reason == "error"
        ]
        assert error and "boom" in (error[0].detail or "")
        assert "job-boom" not in handler.jobs

    asyncio.run(main())


def test_shutdown_cancels_inflight_generation(exported, monkeypatch) -> None:
    async def main() -> None:
        store_dir, manifest, _shards = exported
        monkeypatch.setattr("dain_node.jobs.fetch_stage", make_fake_fetch(store_dir, manifest))
        handler, _envelopes, _raw = make_handler(FakeStore(manifest))

        assign = make_assign(
            "job-hang", manifest, [(0, 3), (4, manifest.layers - 1)], 0, prompt="hi", max_tokens=5
        )
        await handler.on_job_assign(assign)
        rt = handler.jobs["job-hang"]
        await wait_until(lambda: rt.task is not None and not rt.task.done())

        await handler.shutdown()
        assert rt.task.cancelled()
        assert handler.jobs == {} and handler._stages == {} and handler._background == set()

    asyncio.run(main())
