"""Node agent (S3): register → connect → heartbeat → reconnect → deregister.

Lifecycle per spec §6/§10/§11:
- outbound-only connections (persistent WSS to the coordinator; NAT-friendly);
- registration over REST (join token first time, per-node token afterwards;
  a rejected node token falls back to a fresh join — node_id and history persist);
- heartbeats with seq + psutil/CUDA metrics every acked interval;
- auto-reconnect with capped exponential backoff on any transport failure;
- graceful shutdown (SIGINT/SIGTERM/SIGBREAK) → deregister → exit 0.

Jobs are NOT handled yet (S4); incoming messages are parsed and deferred.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import time

import httpx
import psutil
import websockets
from dain_common.schemas import (
    Envelope,
    Heartbeat,
    JobAssign,
    MessageType,
    MetricsReport,
    Register,
    RegisterAck,
    parse_payload,
)
from websockets.exceptions import ConnectionClosed, WebSocketException

from dain_node.capabilities import probe
from dain_node.executor import Executor
from dain_node.identity import IdentityState
from dain_node.settings import NodeSettings

log = logging.getLogger("dain.node.agent")


def next_backoff(current_s: float, min_s: float, max_s: float) -> float:
    """Capped exponential backoff: 0.5 → 1 → 2 → 4 → 8 (max)."""
    return min(max(current_s * 2.0, min_s), max_s)


class StopGuard:
    """Signal → asyncio.Event bridge for graceful shutdown (SIGINT/SIGTERM/SIGBREAK)."""

    def __init__(self) -> None:
        self.event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def install(self) -> None:
        self._loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGBREAK):
            if sig is None:
                continue
            with contextlib.suppress(OSError, ValueError, NotImplementedError):
                signal.signal(sig, self._handle)

    def _handle(self, signum: int, _frame: object) -> None:
        log.info("signal_received signal=%s", signum)
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self.event.set)


class NodeAgent:
    def __init__(self, settings: NodeSettings, executor: Executor) -> None:
        self.settings = settings
        self.executor = executor
        self.identity: IdentityState = IdentityState.load(
            settings.state_path
        ) or IdentityState.create(settings.state_path, settings.node_id)
        self.heartbeat_interval_s = settings.heartbeat_interval_s
        self._seq = 0
        psutil.cpu_percent(interval=None)  # prime the non-blocking sampler

    # -- registration (REST) ----------------------------------------------------

    async def register(self, client: httpx.AsyncClient) -> RegisterAck:
        payload = Register(
            node_id=self.identity.node_id,
            auth_token=self.identity.node_token or self.settings.join_token,
            manifest=probe(self.settings),
            agent_version="0.1.0",
        )
        response = await client.post(
            f"{self.settings.http_base_url}/node/register",
            json=payload.model_dump(mode="json"),
        )
        response.raise_for_status()
        ack = RegisterAck.model_validate(response.json())
        if not ack.accepted:
            if ack.reason == "invalid node token" and self.identity.node_token is not None:
                # Coordinator lost our token (e.g. fresh DB): re-join under the same
                # node_id — history survives because it is keyed by node_id.
                log.warning("token_rejected node=%s re-joining", self.identity.node_id)
                self.identity.node_token = None
                return await self.register(client)
            raise RuntimeError(f"registration rejected: {ack.reason}")
        if ack.node_token and ack.node_token != self.identity.node_token:
            self.identity.node_token = ack.node_token
            self.identity.save(self.settings.state_path)
        self.heartbeat_interval_s = ack.heartbeat_interval_s
        log.info(
            "registered node=%s interval=%.1fs", self.identity.node_id, self.heartbeat_interval_s
        )
        return ack

    async def deregister(self, client: httpx.AsyncClient) -> None:
        if self.identity.node_token is None:
            return
        with contextlib.suppress(Exception):
            await client.post(
                f"{self.settings.http_base_url}/node/deregister",
                json={"node_id": self.identity.node_id, "auth_token": self.identity.node_token},
            )
        log.info("deregistered node=%s", self.identity.node_id)

    # -- metrics -----------------------------------------------------------------

    def _metrics(self) -> MetricsReport:
        vram_free = None
        with contextlib.suppress(ImportError):
            import torch

            if torch.cuda.is_available():
                free_b, _total_b = torch.cuda.mem_get_info(0)
                vram_free = free_b / 1e9
        return MetricsReport(
            gpu_util_pct=None,
            vram_free_gb=vram_free,
            cpu_util_pct=psutil.cpu_percent(interval=None),
            net_bw_mbps=self.settings.net_bw_mbps,
        )

    # -- sessions ------------------------------------------------------------------

    async def run(self, stop_event: asyncio.Event) -> None:
        """Reconnect loop: each session = register + WS heartbeat exchange."""
        backoff = self.settings.reconnect_min_s
        while not stop_event.is_set():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await self.register(client)
                await self._session(stop_event)
                backoff = self.settings.reconnect_min_s  # clean session end
            except asyncio.CancelledError:
                raise
            except (
                OSError,
                ConnectionClosed,
                WebSocketException,
                RuntimeError,
                httpx.HTTPError,
            ) as exc:
                if stop_event.is_set():
                    break
                log.warning(
                    "session_failed node=%s err=%s retry_in=%.1fs",
                    self.identity.node_id,
                    exc,
                    backoff,
                )
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                backoff = next_backoff(
                    backoff, self.settings.reconnect_min_s, self.settings.reconnect_max_s
                )
            except Exception:
                log.exception("unexpected_agent_error node=%s", self.identity.node_id)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=backoff)

    async def _session(self, stop_event: asyncio.Event) -> None:
        url = (
            f"{self.settings.ws_base_url}/node/ws"
            f"?node_id={self.identity.node_id}&token={self.identity.node_token}"
        )
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            log.info("ws_connected node=%s", self.identity.node_id)
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(ws, stop_event), name="heartbeat-loop"
            )
            receive_task = asyncio.create_task(self._receive_loop(ws), name="receive-loop")
            stop_task = asyncio.create_task(stop_event.wait(), name="stop-wait")
            # Wait for either the coordinator to end the session or a shutdown signal;
            # a silent server must never block graceful shutdown.
            done, pending = await asyncio.wait(
                {receive_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
            # Surface a receive failure (connection lost) so run() reconnects.
            for task in done:
                task.result()
        log.info("ws_closed node=%s", self.identity.node_id)

    async def _receive_loop(self, ws) -> None:
        async for raw in ws:
            self._handle_server_message(raw)

    async def _heartbeat_loop(self, ws, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            envelope = Envelope.wrap(
                MessageType.HEARTBEAT,
                Heartbeat(node_id=self.identity.node_id, seq=self._seq, metrics=self._metrics()),
                ts=time.time(),
            )
            await ws.send(envelope.model_dump_json())
            self._seq += 1
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=self.heartbeat_interval_s)

    def _handle_server_message(self, raw: str | bytes) -> None:
        try:
            envelope = Envelope.model_validate_json(raw)
            payload = parse_payload(envelope)
        except Exception:
            log.warning("malformed_server_message node=%s", self.identity.node_id)
            return
        if envelope.type == MessageType.JOB_ASSIGN:
            assert isinstance(payload, JobAssign)
            log.info(
                "job_assign_deferred node=%s job=%s (S4)", self.identity.node_id, payload.job_id
            )
        else:
            log.info("server_message node=%s type=%s", self.identity.node_id, envelope.type.value)


async def run_agent(settings: NodeSettings, executor: Executor) -> int:
    stop = StopGuard()
    stop.install()
    agent = NodeAgent(settings, executor)
    await agent.executor.warmup()
    try:
        await agent.run(stop.event)
    finally:
        await agent.executor.shutdown()
        async with httpx.AsyncClient(timeout=3.0) as client:
            await agent.deregister(client)
    log.info("agent_stopped node=%s", agent.identity.node_id)
    return 0
