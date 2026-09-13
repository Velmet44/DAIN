"""Background model provisioning (S22a/S22b, plan §3.1).

The coordinator assigns each node a desired model portfolio (`ASSIGNMENT`
envelopes). The Provisioner reconciles that portfolio in the background:

    assigned -> downloading (verified shard fetch, mirrors -> peers -> coord)
             -> files_ready (derived-fp16 cache materialized for storage-INT4)
             -> warm (stage built inside the warm budget) -> warm probe
             -> serving-eligible (`MODEL_STATUS state="warm"` with the probe's
               measured tok/s and resident RSS)

A node restart never re-downloads verified shards and never re-dequantizes a
valid derived cache; the probe runs one tiny generation so the first real
request skips kernel warm-up and a broken build never serves traffic.
"""

from __future__ import annotations

import asyncio
import logging
import time

from dain_common.schemas import Envelope, MessageType, ModelAssignment, ModelStatus
from transformers.cache_utils import DynamicCache

from dain_node.llm import ModelStoreClient, sample_token
from dain_node.settings import NodeSettings

log = logging.getLogger("dain.node.provisioner")


class Provisioner:
    """Reconciles the desired model portfolio and reports MODEL_STATUS."""

    def __init__(
        self,
        settings: NodeSettings,
        store: ModelStoreClient,
        handler,  # JobHandler (avoids an import cycle)
        *,
        node_id_fn,
    ) -> None:
        self._settings = settings
        self._store = store
        self._handler = handler
        self._node_id_fn = node_id_fn
        self._desired: dict[str, ModelAssignment] = {}
        self._last_status: dict[str, ModelStatus] = {}
        self._wake = asyncio.Event()

    # -- desired state (called from the agent receive loop) --------------------------

    def set_desired(self, assignment: ModelAssignment) -> None:
        if assignment.action == "revoke":
            self._desired.pop(assignment.model_id, None)
            self._handler.drop_warm_stages(assignment.model_id)
            self._report(
                ModelStatus(
                    node_id=self._node_id_fn(),
                    model_id=assignment.model_id,
                    state="revoked",
                )
            )
        else:
            self._desired[assignment.model_id] = assignment
        self._wake.set()

    # -- loop ----------------------------------------------------------------------

    async def run(self) -> None:
        """Reconcile on demand; re-report statuses every 30 s (a lost status
        frame after reconnect must self-heal)."""
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=30.0)
            except TimeoutError:
                pass
            self._wake.clear()
            try:
                await self._reconcile()
            except Exception:  # noqa: BLE001 — the loop must never die
                log.exception("provisioner_reconcile_failed")
            self._report_all()

    async def _reconcile(self) -> None:
        for model_id, assignment in list(self._desired.items()):
            if model_id not in self._desired:  # revoked mid-loop
                continue
            await self._ensure(model_id, assignment)

    async def _ensure(self, model_id: str, assignment: ModelAssignment) -> None:
        try:
            manifest = await self._store.fetch_manifest(model_id, refresh=True)
            if assignment.mode == "replica":
                layer_start, layer_end = 0, manifest.layers - 1
            else:
                layer_start = assignment.layer_start or 0
                layer_end = assignment.layer_end or manifest.layers - 1

            needed = [
                s
                for s in manifest.shards
                if s.layer_start is not None
                and s.layer_start <= layer_end
                and s.layer_end >= layer_start
            ]
            for idx, shard in enumerate(needed):
                self._report(
                    ModelStatus(
                        node_id=self._node_id_fn(),
                        model_id=model_id,
                        state="downloading",
                        progress=idx / max(len(needed), 1),
                    )
                )
                await self._store.ensure_shard(
                    model_id,
                    shard.shard_id,
                    shard.content_hash,
                    shard.format,
                    mirror_urls=getattr(manifest, "mirror_urls", ()) or (),
                )
            if manifest.tokenizer_file is not None and manifest.tokenizer_hash is not None:
                await self._store.ensure_tokenizer(
                    model_id, manifest.tokenizer_file, manifest.tokenizer_hash
                )
            self._report(
                ModelStatus(
                    node_id=self._node_id_fn(),
                    model_id=model_id,
                    state="files_ready",
                    progress=1.0,
                )
            )

            if assignment.mode == "replica" and self._handler._warm_mode != "files":
                stage = await self._handler.ensure_warm_stage(
                    manifest, layer_start, layer_end
                )
                if stage is None:
                    log.info(
                        "provision_files_only model=%s (warm budget exhausted)",
                        model_id,
                    )
                    return
                toks_s = await self._probe(stage)
                self._report(
                    ModelStatus(
                        node_id=self._node_id_fn(),
                        model_id=model_id,
                        state="warm",
                        progress=1.0,
                        toks_s=toks_s,
                        rss_gb=round(self._handler._stage_rss_gb(stage), 3),
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — report, never crash the loop
            log.exception("provision_failed model=%s", model_id)
            self._report(
                ModelStatus(
                    node_id=self._node_id_fn(),
                    model_id=model_id,
                    state="error",
                    detail=str(exc)[:200],
                )
            )

    async def _probe(self, stage) -> float:
        """Two-token warm probe: kernel warm-up + measured decode throughput."""
        async with self._handler.node_lock:
            cache = DynamicCache()
            ids = stage.tokenizer.encode("Hello")
            t0 = time.monotonic()
            logits = await asyncio.to_thread(stage.next_token_logits_full, ids, cache)
            token = sample_token(logits, 0.0, None)
            logits = await asyncio.to_thread(stage.next_token_logits_full, [token], cache)
            sample_token(logits, 0.0, None)
            elapsed = time.monotonic() - t0
            return round(2.0 / max(elapsed, 1e-6), 2)

    # -- reporting ------------------------------------------------------------------

    def _report(self, status: ModelStatus) -> None:
        self._last_status[status.model_id] = status
        envelope = Envelope.wrap(MessageType.MODEL_STATUS, status, ts=time.time())
        try:
            self._handler._spawn_report(envelope)
        except RuntimeError:
            log.debug("model_status_deferred model=%s (no loop)", status.model_id)

    def _report_all(self) -> None:
        """Re-send every current status (transport loss self-healing)."""
        for status in list(self._last_status.values()):
            envelope = Envelope.wrap(MessageType.MODEL_STATUS, status, ts=time.time())
            try:
                self._handler._spawn_report(envelope)
            except RuntimeError:
                return
