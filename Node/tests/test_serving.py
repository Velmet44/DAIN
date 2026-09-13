"""S22 node-side: provisioner flow, warm stages, concurrent sessions, direct relay."""

import asyncio
import json
import time
from pathlib import Path

import pytest
import torch
from dain_common.schemas import (
    Envelope,
    GenerationParams,
    JobAssign,
    MessageType,
    ModelAssignment,
    ModelStatus,
    StageAssignment,
    parse_payload,
)
from safetensors.torch import load_file

from dain_node.jobs import JobHandler
from dain_node.llm import StageModel
from dain_node.peer_server import PeerShardServer
from dain_node.provisioner import Provisioner
from dain_node.settings import NodeSettings
from dain_node.shard_export import export_tiny_llama


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    store_dir = tmp_path_factory.mktemp("store")
    manifest = export_tiny_llama(str(store_dir))
    return store_dir, manifest


def _load_state(store_dir: Path, manifest) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for shard in manifest.shards:
        state.update(
            load_file(str(Path(store_dir) / manifest.model_id / f"{shard.shard_id}.safetensors"))
        )
    return state


def _fake_fetch(store_dir: Path, manifest):
    state: dict[str, torch.Tensor] = {}

    def _state():
        if not state:
            state.update(_load_state(store_dir, manifest))
        return state

    async def fake(store, m, layer_start: int, layer_end: int):
        return StageModel(m, layer_start, layer_end, _state()), {}

    return fake


class FakeStore:
    def __init__(self, manifest, store_dir=None):
        self.manifest = manifest
        self.store_dir = store_dir
        self.base = None
        self.peer_token = "tok"
        self.ensure_calls = 0
        self.posts: list[tuple[str, dict, bytes]] = []

    async def fetch_manifest(self, model_id: str, *, refresh: bool = False):
        return self.manifest

    async def ensure_shard(self, model_id, shard_id, content_hash, fmt=None, mirror_urls=()):
        self.ensure_calls += 1
        return f"cached/{model_id}/{shard_id}"

    async def ensure_tokenizer(self, model_id, f, h):
        return f"cached/{model_id}/tokenizer.json"

    async def post_bytes(self, url, *, headers, content, timeout_s=10.0):
        self.posts.append((url, headers, content))
        return 0

    def set_base_url(self, base_url):
        self.base = base_url

    async def close(self):
        return None


class RecordingHandler(JobHandler):
    """JobHandler with recorded status reports and bound envelope capture."""

    def __init__(self, settings, store):
        super().__init__(settings, store)
        self.statuses: list[ModelStatus] = []
        self.envelopes: list[Envelope] = []

        async def record_env(env: Envelope) -> None:
            self.envelopes.append(env)

        async def record_bytes(data: bytes) -> None:
            return None

        self.bind(record_env, record_bytes)

    def _spawn_report(self, envelope: Envelope) -> None:
        self.statuses.append(parse_payload(envelope))


def make_settings(**kw) -> NodeSettings:
    return NodeSettings(job_timeout_s=30.0, **kw)


def _replica_assign(job_id: str, manifest, node_id: str = "node-x") -> JobAssign:
    return JobAssign(
        job_id=job_id,
        model_id=manifest.model_id,
        my_stage_idx=0,
        stages=(
            StageAssignment(
                stage_idx=0,
                node_id=node_id,
                shard_id="layers_00_15",
                layer_start=0,
                layer_end=manifest.layers - 1,
            ),
        ),
        prompt="hi",
        params=GenerationParams(max_tokens=3, temperature=0),
    )


# -- provisioner flow -------------------------------------------------------------


def test_provisioner_reports_files_ready_and_warm(exported) -> None:
    async def main():
        store_dir, manifest = exported
        store = FakeStore(manifest, store_dir)
        handler = RecordingHandler(make_settings(warm_mode="resident"), store)
        monkey_stage = StageModel(
            manifest,
            0,
            manifest.layers - 1,
            _load_state(store_dir, manifest),
            device="cpu",
            tokenizer_path=None,
        )
        seen: list[tuple[int, int]] = []

        async def fake_ensure_warm(manifest_, ls, le):
            seen.append((ls, le))
            return monkey_stage

        handler.ensure_warm_stage = fake_ensure_warm  # type: ignore[method-assign]
        provisioner = Provisioner(
            make_settings(), store, handler, node_id_fn=lambda: "node-x"
        )
        provisioner.set_desired(
            ModelAssignment(model_id=manifest.model_id, action="ensure", mode="replica")
        )
        await provisioner._reconcile()

        states = [s.state for s in handler.statuses if s.model_id == manifest.model_id]
        assert states[0] == "downloading"
        assert "files_ready" in states
        warm = [s for s in handler.statuses if s.state == "warm"]
        assert len(warm) == 1 and warm[0].toks_s and warm[0].toks_s > 0
        assert seen == [(0, manifest.layers - 1)]

    asyncio.run(main())


def test_provisioner_pipeline_mode_reports_files_ready_only(exported) -> None:
    async def main():
        store_dir, manifest = exported
        store = FakeStore(manifest, store_dir)
        handler = RecordingHandler(make_settings(warm_mode="resident"), store)
        provisioner = Provisioner(
            make_settings(), store, handler, node_id_fn=lambda: "node-x"
        )
        provisioner.set_desired(
            ModelAssignment(
                model_id=manifest.model_id,
                action="ensure",
                mode="pipeline",
                layer_start=0,
                layer_end=3,
            )
        )
        await provisioner._reconcile()
        states = [s.state for s in handler.statuses]
        assert "files_ready" in states and "warm" not in states

    asyncio.run(main())


def test_provisioner_revocation(exported) -> None:
    async def main():
        store_dir, manifest = exported
        store = FakeStore(manifest, store_dir)
        handler = RecordingHandler(make_settings(), store)
        dropped: list[str] = []
        handler.drop_warm_stages = lambda model_id: dropped.append(model_id)  # type: ignore[method-assign]
        provisioner = Provisioner(
            make_settings(), store, handler, node_id_fn=lambda: "node-x"
        )
        provisioner.set_desired(
            ModelAssignment(model_id=manifest.model_id, action="revoke", mode="replica")
        )
        await provisioner._reconcile()
        assert dropped == [manifest.model_id]
        assert handler.statuses[-1].state == "revoked"

    asyncio.run(main())


def test_provisioner_error_reported(exported) -> None:
    async def main():
        store_dir, manifest = exported

        class BoomStore(FakeStore):
            async def fetch_manifest(self, model_id, *, refresh=False):
                raise RuntimeError("store unreachable")

        store = BoomStore(manifest, store_dir)
        handler = RecordingHandler(make_settings(), store)
        provisioner = Provisioner(
            make_settings(), store, handler, node_id_fn=lambda: "node-x"
        )
        provisioner.set_desired(
            ModelAssignment(model_id=manifest.model_id, action="ensure", mode="replica")
        )
        await provisioner._reconcile()
        error = [s for s in handler.statuses if s.state == "error"]
        assert len(error) == 1 and "store unreachable" in (error[0].detail or "")

    asyncio.run(main())


# -- warm stage budget / LRU -------------------------------------------------------


def test_warm_stage_budget(exported, monkeypatch) -> None:
    async def main():
        store_dir, manifest = exported
        store = FakeStore(manifest, store_dir)
        monkeypatch.setattr("dain_node.jobs.fetch_stage", _fake_fetch(store_dir, manifest))

        tiny_budget = RecordingHandler(
            make_settings(warm_budget_gb=0.001, warm_models=2, warm_mode="resident"),
            store,
        )
        monkeypatch.setattr(
            "dain_node.jobs.fetch_stage", _fake_fetch(store_dir, manifest)
        )
        # The tiny model (~1.2 MB fp32) exceeds a 1 MB budget -> no warm stage.
        assert (
            await tiny_budget.ensure_warm_stage(manifest, 0, manifest.layers - 1) is None
        )

        roomy = RecordingHandler(
            make_settings(warm_budget_gb=8.0, warm_models=1, warm_mode="resident"), store
        )
        stage = await roomy.ensure_warm_stage(manifest, 0, manifest.layers - 1)
        assert stage is not None
        again = await roomy.ensure_warm_stage(manifest, 0, manifest.layers - 1)
        assert again is stage  # cached hit

    asyncio.run(main())


# -- concurrent replica sessions (S22d) ----------------------------------------------


def test_concurrent_replica_sessions_interleave(exported, monkeypatch) -> None:
    async def main():
        store_dir, manifest = exported
        store = FakeStore(manifest, store_dir)
        handler = RecordingHandler(
            make_settings(max_sessions=4, warm_budget_gb=8.0), store
        )
        monkeypatch.setattr(
            "dain_node.jobs.fetch_stage", _fake_fetch(Path(store_dir), manifest)
        )

        def final_payload(job_id: str, final: bool):
            for env in reversed(handler.envelopes):
                if env.type == MessageType.TOKEN_BATCH:
                    payload = parse_payload(env)
                    if payload.job_id == job_id and payload.is_final == final:
                        return payload
            return None

        async def run_job(job_id: str):
            await handler.on_job_assign(_replica_assign(job_id, manifest))
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                batch = final_payload(job_id, final=True)
                if batch is not None:
                    return batch
                await asyncio.sleep(0.02)
            raise AssertionError(f"{job_id} never finished")

        results = await asyncio.gather(run_job("j-aa"), run_job("j-bb"))
        # Both replica sessions completed concurrently on the shared stage.
        assert all(r.finish_reason in ("length", "eos") for r in results)
        assert handler.active_sessions == 0
        assert (manifest.model_id, 0, manifest.layers - 1) in handler._stages

    asyncio.run(main())


# -- direct activation relay round trip (S22e) ---------------------------------------


def test_peer_server_activation_round_trip(exported) -> None:
    async def main():
        store_dir, manifest = exported
        server = PeerShardServer(
            str(store_dir), "127.0.0.1", 0, advertise_host="127.0.0.1", join_token="tok"
        )
        received: list[tuple[dict, bytes]] = []

        async def receiver(header_data: dict, payload: bytes) -> None:
            received.append((header_data, payload))

        server.activation_receiver = receiver
        await server.start()
        try:
            header = {
                "job_id": "j123",
                "stage_idx": 0,
                "attempt": 0,
                "seq": 1,
                "dtype": "fp16",
                "role": "hidden",
                "shape": [1, 1, 8],
                "n_bytes": 16,
                "is_final": False,
            }
            payload = b"\x00" * 16

            async def post(token: str) -> bytes:
                reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
                body = (
                    f"POST /peer/activation HTTP/1.1\r\n"
                    f"X-Peer-Token: {token}\r\n"
                    f"X-Activation: {json.dumps(header)}\r\n"
                    f"Content-Length: {len(payload)}\r\n\r\n"
                ).encode() + payload
                writer.write(body)
                await writer.drain()
                status_line = await reader.readline()
                writer.close()
                return status_line

            assert b"200" in await post("tok")
            await asyncio.sleep(0.05)
            assert len(received) == 1
            assert received[0][0]["job_id"] == "j123" and received[0][1] == payload

            assert b"403" in await post("wrong")
            await asyncio.sleep(0.05)
            assert len(received) == 1
        finally:
            await server.stop()

    asyncio.run(main())
