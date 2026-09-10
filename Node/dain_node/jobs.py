"""Job execution handler (S4): JOB_ASSIGN → stage runner → TOKEN_BATCH stream.

Protocol invariant: **only the sampling stage emits TOKEN_BATCH frames** (the
final one carries the finish_reason); the entry stage owns the generation loop
and reports lifecycle via JOB_STATUS. Distributed decode: the last stage sends
each sampled token back to the entry stage as an int64 `sampled_token`
activation, which the entry embeds for the next step. Every inbound step is
acked via JOB_STATUS so senders can buffer unacked activations for replay (S7).
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import logging
import struct
import time
from dataclasses import dataclass

import torch
from dain_common.schemas import (
    ActivationRelayHeader,
    Envelope,
    JobAssign,
    JobState,
    JobStatus,
    MessageType,
    ModelManifest,
    StageRetry,
    TokenBatch,
)

from dain_node.llm import ModelStoreClient, StageModel, fetch_stage, sample_token
from dain_node.settings import NodeSettings

log = logging.getLogger("dain.node.jobs")

_INT64 = struct.Struct("<q")


class ByteStreamer:
    """Incremental UTF-8 decode: multi-byte characters survive byte-granular streaming."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, token_id: int) -> str:
        return self._decoder.decode(bytes([token_id & 0xFF]))


@dataclass
class JobRuntime:
    job: JobAssign
    manifest: ModelManifest
    layer_start: int
    layer_end: int
    first: bool
    last: bool
    stage: StageModel | None
    inbound: asyncio.Queue
    streamer: ByteStreamer
    task: asyncio.Task | None = None
    generated: int = 0
    finished: bool = False
    attempt: int = 0
    started_at: float = 0.0
    buffered: tuple[ActivationRelayHeader, bytes] | None = None


class JobHandler:
    """Agent-side job execution. Transport is injected (the agent owns the WS)."""

    def __init__(self, settings: NodeSettings, store: ModelStoreClient) -> None:
        torch.set_grad_enabled(False)  # inference-only agent
        self.settings = settings
        self.store = store
        self.send_envelope = None  # bound by the agent
        self.send_bytes = None
        self._stages: dict[tuple[str, int, int], StageModel] = {}
        self.jobs: dict[str, JobRuntime] = {}
        self._pending_header: ActivationRelayHeader | None = None
        self._send_lock = asyncio.Lock()
        self.node_lock = asyncio.Lock()
        # Activations arriving before the stage runtime is built (entry may prefill
        # immediately) are buffered here and drained in on_job_assign.
        self._early: dict[str, list[tuple[ActivationRelayHeader, bytes]]] = {}

    # -- wiring ------------------------------------------------------------------

    def bind(self, send_envelope, send_bytes) -> None:
        self.send_envelope = send_envelope
        self.send_bytes = send_bytes

    def set_store_base(self, base_url: str) -> None:
        self.store.set_base_url(base_url)

    async def warmup(self) -> None:
        if self.settings.model_id:
            try:
                await self.store.fetch_manifest(self.settings.model_id)
            except Exception as exc:  # noqa: BLE001 — the model is optional at startup
                log.warning("warmup_manifest_failed model=%s err=%s", self.settings.model_id, exc)

    async def shutdown(self) -> None:
        for rt in self.jobs.values():
            if rt.task is not None:
                rt.task.cancel()
        self._stages.clear()
        self.jobs.clear()

    # -- transport helpers -----------------------------------------------------------

    async def _emit(self, envelope: Envelope) -> None:
        async with self._send_lock:
            await self.send_envelope(envelope)

    async def _emit_bytes(self, data: bytes) -> None:
        async with self._send_lock:
            await self.send_bytes(data)

    async def _status(self, rt: JobRuntime, state: JobState, detail: str | None = None) -> None:
        await self._emit(
            Envelope.wrap(
                MessageType.JOB_STATUS,
                JobStatus(
                    job_id=rt.job.job_id,
                    stage_idx=rt.job.my_stage_idx,
                    attempt=rt.attempt,
                    state=state,
                    tokens_done=rt.generated,
                    detail=detail,
                ),
                ts=time.time(),
            )
        )

    async def _send_tokens(
        self, rt: JobRuntime, tokens: list[str], *, is_final: bool, finish_reason: str | None
    ) -> None:
        await self._emit(
            Envelope.wrap(
                MessageType.TOKEN_BATCH,
                TokenBatch(
                    job_id=rt.job.job_id,
                    tokens=tuple(tokens),
                    is_final=is_final,
                    finish_reason=finish_reason,  # type: ignore[arg-type]
                ),
                ts=time.time(),
            )
        )

    async def _send_activation(
        self,
        rt: JobRuntime,
        *,
        role: str,
        tensor_bytes: bytes,
        shape: tuple[int, ...],
        dtype: str,
        is_final: bool,
    ) -> None:
        header = ActivationRelayHeader(
            job_id=rt.job.job_id,
            stage_idx=rt.job.my_stage_idx,
            attempt=rt.attempt,
            seq=rt.generated,
            dtype=dtype,  # type: ignore[arg-type]
            role=role,  # type: ignore[arg-type]
            shape=shape,
            n_bytes=len(tensor_bytes),
            is_final=is_final,
        )
        # Buffer only downstream (hidden) activations for S7 replay. Upstream
        # `sampled_token` returns must NOT clobber this: the replay to a
        # reassigned downstream stage must always be the hidden activation it
        # was last asked to process, never a token flowing back toward entry.
        if role == "hidden":
            rt.buffered = (header, tensor_bytes)  # held until acked (S7 replay)
        await self._emit(Envelope.wrap(MessageType.ACTIVATION_RELAY, header, ts=time.time()))
        await self._emit_bytes(tensor_bytes)

    # -- entry points (called by the agent's receive loop) -----------------------------

    async def on_job_assign(self, job: JobAssign) -> None:
        try:
            manifest = await self.store.fetch_manifest(job.model_id)
        except Exception as exc:  # noqa: BLE001 — report setup failures to the client
            log.error("job_setup_failed job=%s err=%s", job.job_id, exc)
            await self._emit(
                Envelope.wrap(
                    MessageType.TOKEN_BATCH,
                    TokenBatch(
                        job_id=job.job_id,
                        tokens=(),
                        is_final=True,
                        finish_reason="error",
                        detail=f"model setup failed: {exc}",
                    ),
                    ts=time.time(),
                )
            )
            return
        mine = job.stages[job.my_stage_idx]
        previous = self.jobs.get(job.job_id)
        attempt = previous.attempt + 1 if previous is not None else 0
        if previous is not None and previous.task is not None:
            previous.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await previous.task
        rt = JobRuntime(
            job=job,
            manifest=manifest,
            layer_start=mine.layer_start,
            layer_end=mine.layer_end,
            first=mine.layer_start == 0,
            last=mine.layer_end == manifest.layers - 1,
            stage=None,
            inbound=asyncio.Queue(maxsize=256),
            streamer=ByteStreamer(),
            attempt=attempt,
            started_at=time.time(),
        )
        self.jobs[job.job_id] = rt
        buffered = self._early.pop(job.job_id, [])
        if job.my_stage_idx == 0:
            for header, payload in buffered:
                await rt.inbound.put((header, payload))
            rt.task = asyncio.create_task(self._run_entry(rt), name=f"entry-{job.job_id}")
        else:
            # Non-entry stage: activations that arrived before this runtime was
            # created (a reassigned stage resuming a stream, S7) were queued in
            # `_early`. Build the stage, then feed each through the normal step
            # path so a replacement picks up exactly where the failed stage stopped.
            rt.task = asyncio.create_task(
                self._bootstrap_stage(rt, buffered), name=f"stage-{job.job_id}"
            )

    async def on_activation_header(self, header: ActivationRelayHeader) -> None:
        self._pending_header = header

    async def on_payload(self, payload: bytes) -> None:
        header = self._pending_header
        self._pending_header = None
        if header is None:
            log.warning("payload_without_header bytes=%d", len(payload))
            return
        if len(payload) != header.n_bytes:
            log.warning(
                "payload_size_mismatch job=%s got=%d want=%d",
                header.job_id,
                len(payload),
                header.n_bytes,
            )
            return
        rt = self.jobs.get(header.job_id)
        if rt is None:
            if header.role == "hidden":
                self._early.setdefault(header.job_id, []).append((header, payload))
            else:
                log.warning("activation_for_unknown_job job=%s", header.job_id)
            return
        if rt.first and header.role == "sampled_token":
            await rt.inbound.put((header, payload))
            return
        # Acknowledge receipt so the sender can drop its replay buffer (S7).
        await self._status(rt, JobState.RUNNING, detail="step_ack")
        asyncio.create_task(self._run_step(rt, header, payload), name=f"step-{header.job_id}")

    async def on_stage_retry(self, retry: StageRetry) -> None:
        """Coordinator asks us (the failed stage's *upstream*) to replay the
        activation we are still buffering, so the replacement stage can resume
        from the completed prefix (spec §13). We are the upstream iff
        `my_stage_idx == retry.stage_idx - 1`.

        Re-sending increments the header's `attempt` to match the retry counter
        and routes through the normal relay path, which now points at the
        reassigned node.
        """
        rt = self.jobs.get(retry.job_id)
        if rt is None or rt.buffered is None:
            log.warning("stage_retry_ignored job=%s retry_stage=%d", retry.job_id, retry.stage_idx)
            return
        if not retry.stage_idx or rt.job.my_stage_idx != retry.stage_idx - 1:
            log.info(
                "stage_retry_not_upstream job=%s my_stage=%d retry_stage=%d",
                retry.job_id,
                rt.job.my_stage_idx,
                retry.stage_idx,
            )
            return
        header, payload = rt.buffered
        replayed = ActivationRelayHeader(
            job_id=header.job_id,
            stage_idx=header.stage_idx,
            attempt=retry.attempt,
            seq=header.seq,
            dtype=header.dtype,
            role=header.role,
            shape=header.shape,
            n_bytes=header.n_bytes,
            is_final=header.is_final,
        )
        log.info(
            "stage_replay job=%s stage=%d attempt=%d bytes=%d",
            retry.job_id,
            header.stage_idx,
            retry.attempt,
            len(payload),
        )
        await self._emit(Envelope.wrap(MessageType.ACTIVATION_RELAY, replayed, ts=time.time()))
        await self._emit_bytes(payload)

    # -- stage execution --------------------------------------------------------------

    async def _build_stage(self, rt: JobRuntime) -> StageModel:
        key = (rt.job.model_id, rt.layer_start, rt.layer_end)
        stage = self._stages.get(key)
        if stage is None:
            stage, _paths = await fetch_stage(self.store, rt.manifest, rt.layer_start, rt.layer_end)
            self._stages[key] = stage
        rt.stage = stage
        stage.begin_job()
        return stage

    async def _bootstrap_stage(self, rt: JobRuntime, buffered: list) -> None:
        """Build a reassigned stage, then replay any activations that arrived
        before this runtime existed (S7: a replacement resumes mid-stream)."""
        try:
            await self._build_stage(rt)
            for header, payload in buffered:
                await self._run_step(rt, header, payload)
        except Exception:  # noqa: BLE001 — surface the failure to the client
            log.exception("stage_bootstrap_failed job=%s", rt.job.job_id)
            if rt.last:
                with contextlib.suppress(Exception):
                    await self._send_tokens(rt, [], is_final=True, finish_reason="error")

    async def _run_entry(self, rt: JobRuntime) -> None:
        """Entry stage: owns the generation loop for the whole job."""
        job = rt.job
        try:
            stage = await self._build_stage(rt)
            async with self.node_lock:  # exclusive: one generation per node
                await self._status(rt, JobState.RUNNING, detail="entry_ready")
                ids = stage.tokenizer.encode(job.prompt or "")
                generator = (
                    torch.Generator().manual_seed(job.params.seed)
                    if job.params.seed is not None
                    else None
                )
                if rt.first and rt.last:
                    await self._run_single_stage(rt, stage, ids, generator)
                else:
                    await self._run_distributed_entry(rt, stage, ids, generator)
                await self._status(rt, JobState.COMPLETED, detail=f"generated={rt.generated}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — every failure must reach the client
            log.exception("entry_failed job=%s", job.job_id)
            rt.finished = True
            with contextlib.suppress(Exception):
                await self._send_tokens(rt, [], is_final=True, finish_reason="error")
                await self._status(rt, JobState.FAILED, detail=str(exc)[:200])
        finally:
            if rt.stage is not None:
                rt.stage.end_job()

    async def _run_single_stage(
        self, rt: JobRuntime, stage: StageModel, ids: list[int], generator
    ) -> None:
        current: list[int] = ids if ids else [1]  # bos fallback for empty prompts
        finish_reason: str | None = None
        log.info(
            "generation_start job=%s prompt_tokens=%d temp=%.2f",
            rt.job.job_id,
            len(current),
            rt.job.params.temperature,
        )
        while True:
            logits = stage.next_token_logits_full(current)
            token_id = sample_token(logits, rt.job.params.temperature, generator)
            current = [token_id]
            if token_id == stage.manifest.eos_token_id:
                finish_reason = "eos"
                break
            rt.generated += 1
            chars = rt.streamer.feed(token_id)
            if chars:
                await self._send_tokens(rt, [chars], is_final=False, finish_reason=None)
            if rt.generated >= rt.job.params.max_tokens:
                finish_reason = "length"
                break
        log.info(
            "generation_done job=%s tokens=%d reason=%s",
            rt.job.job_id,
            rt.generated,
            finish_reason or "length",
        )
        await self._send_tokens(rt, [], is_final=True, finish_reason=finish_reason or "length")

    async def _run_distributed_entry(
        self, rt: JobRuntime, stage: StageModel, ids: list[int], generator
    ) -> None:
        hidden = stage.forward_ids(ids)
        await self._send_activation(
            rt,
            role="hidden",
            tensor_bytes=hidden.contiguous().numpy().tobytes(),
            shape=tuple(hidden.shape),
            dtype="fp32",
            is_final=False,
        )
        while True:
            header, payload = await asyncio.wait_for(
                rt.inbound.get(), timeout=self.settings.job_timeout_s
            )
            if header.is_final:
                break  # the sampling stage already emitted the final TOKEN_BATCH
            token_id = _INT64.unpack(payload)[0]
            rt.generated += 1
            hidden = stage.embed_one(token_id)
            await self._send_activation(
                rt,
                role="hidden",
                tensor_bytes=hidden.contiguous().numpy().tobytes(),
                shape=tuple(hidden.shape),
                dtype="fp32",
                is_final=False,
            )

    async def _run_step(
        self, rt: JobRuntime, header: ActivationRelayHeader, payload: bytes
    ) -> None:
        """Non-entry stage step: forward → relay onward, or sample → stream."""
        try:
            async with self.node_lock:
                stage = rt.stage or await self._build_stage(rt)
                hidden = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(
                    tuple(header.shape)
                )
                out = stage.forward_hidden(hidden)
                if rt.last:
                    logits = stage.logits_from(out)[:, -1, :]
                    generator = (
                        torch.Generator().manual_seed((rt.job.params.seed or 0) + rt.generated)
                        if rt.job.params.seed is not None
                        else None
                    )
                    token_id = sample_token(logits, rt.job.params.temperature, generator)
                    is_eos = token_id == stage.manifest.eos_token_id
                    rt.generated += 1
                    finished = is_eos or rt.generated >= rt.job.params.max_tokens
                    finish_reason = "eos" if is_eos else "length" if finished else None
                    if not is_eos:
                        chars = rt.streamer.feed(token_id)
                        if chars:
                            await self._send_tokens(rt, [chars], is_final=False, finish_reason=None)
                    await self._send_tokens(
                        rt,
                        [],
                        is_final=finished,
                        finish_reason=finish_reason if finished else None,
                    )
                    await self._send_activation(
                        rt,
                        role="sampled_token",
                        tensor_bytes=_INT64.pack(token_id),
                        shape=(1,),
                        dtype="int64",
                        is_final=finished,
                    )
                    if finished:
                        log.info(
                            "generation_done job=%s tokens=%d reason=%s",
                            rt.job.job_id,
                            rt.generated,
                            finish_reason,
                        )
                        await self._status(
                            rt, JobState.COMPLETED, detail=f"generated={rt.generated}"
                        )
                else:
                    await self._send_activation(
                        rt,
                        role="hidden",
                        tensor_bytes=out.contiguous().numpy().tobytes(),
                        shape=tuple(out.shape),
                        dtype="fp32",
                        is_final=header.is_final,
                    )
        except Exception:  # noqa: BLE001
            log.exception("step_failed job=%s stage=%d", rt.job.job_id, rt.job.my_stage_idx)
            if rt.last:
                with contextlib.suppress(Exception):
                    await self._send_tokens(rt, [], is_final=True, finish_reason="error")
