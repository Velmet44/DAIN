"""S1 checkpoint: protocol envelope + payload schemas (spec §10–§11)."""

import pytest
from pydantic import ValidationError

from dain_common import (
    PAYLOAD_TYPES,
    PROTOCOL_VERSION,
    ActivationRelayHeader,
    CapabilityManifest,
    CPUInfo,
    Envelope,
    GPUInfo,
    Heartbeat,
    JobAssign,
    JobState,
    LedgerEvent,
    MessageType,
    NetInfo,
    NodeState,
    Register,
    StageAssignment,
    TaskOutcome,
    parse_payload,
)

# -- helpers -----------------------------------------------------------------


def gpu(**overrides) -> GPUInfo:
    values = {
        "name": "RTX 3060",
        "vram_total_gb": 12.0,
        "vram_free_gb": 11.0,
        "tflops_claimed": 51.0,
    }
    values.update(overrides)
    return GPUInfo(**values)


def manifest(**overrides) -> CapabilityManifest:
    values = {
        "gpu": gpu(),
        "cpu": CPUInfo(cores=8, ram_total_gb=32.0, ram_free_gb=16.0),
        "net": NetInfo(bw_mbps=300.0, lat_ms_p95=25.0),
    }
    values.update(overrides)
    return CapabilityManifest(**values)


# -- envelope ----------------------------------------------------------------


def test_envelope_round_trip_and_payload_dispatch() -> None:
    hb = Heartbeat(node_id="node-alpha", seq=7)
    env = Envelope.wrap(MessageType.HEARTBEAT, hb, ts=1.75e9)
    assert env.v == PROTOCOL_VERSION

    restored = Envelope.model_validate_json(env.model_dump_json())
    assert restored == env
    parsed = parse_payload(restored)
    assert isinstance(parsed, Heartbeat)
    assert parsed.seq == 7
    assert parsed.node_id == "node-alpha"


def test_every_message_family_has_a_payload_model() -> None:
    assert set(PAYLOAD_TYPES) == set(MessageType)
    assert len(PAYLOAD_TYPES) == 9


def test_envelope_rejects_wrong_version() -> None:
    with pytest.raises(ValidationError, match="unsupported protocol version"):
        Envelope.model_validate(
            {"v": PROTOCOL_VERSION + 1, "type": "heartbeat", "ts": 1.0, "payload": {}}
        )


def test_envelope_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        Envelope.model_validate(
            {"v": PROTOCOL_VERSION, "type": "hyperspace_jump", "ts": 1.0, "payload": {}}
        )


def test_envelope_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Envelope.model_validate(
            {"v": PROTOCOL_VERSION, "type": "heartbeat", "ts": 1.0, "payload": {}, "evil": True}
        )


# -- state enums mirror the spec state machines ------------------------------


def test_state_machines() -> None:
    assert [s.value for s in NodeState] == ["online", "busy", "degraded", "offline"]
    assert [s.value for s in JobState] == [
        "queued",
        "dispatched",
        "running",
        "streaming",
        "completed",
        "failed",
        "retrying",
    ]
    assert [s.value for s in TaskOutcome] == ["success", "retried_away", "failed"]


# -- capability manifest validators (plausibility, spec §6.2) ----------------


def test_manifest_rejects_free_vram_above_total() -> None:
    with pytest.raises(ValidationError, match="vram_free_gb"):
        manifest(gpu=gpu(vram_free_gb=13.0))


def test_manifest_rejects_free_ram_above_total() -> None:
    with pytest.raises(ValidationError, match="ram_free_gb"):
        manifest(cpu=CPUInfo(cores=8, ram_total_gb=32.0, ram_free_gb=33.0))


def test_manifest_allows_cpu_only_node() -> None:
    m = manifest(gpu=None)
    assert m.gpu is None


def test_gpu_rejects_absurd_tflops_claim() -> None:
    with pytest.raises(ValidationError):
        gpu(tflops_claimed=99_999.0)


# -- job graph validators (spec §9.4) ----------------------------------------


def stages() -> tuple[StageAssignment, ...]:
    return (
        StageAssignment(stage_idx=0, node_id="node-a", shard_id="s0", layer_start=0, layer_end=7),
        StageAssignment(stage_idx=1, node_id="node-b", shard_id="s1", layer_start=8, layer_end=15),
    )


def test_job_assign_accepts_sane_graph() -> None:
    job = JobAssign(
        job_id="job-0001", model_id="tinyllama-1.1b", my_stage_idx=0, stages=stages(), prompt="hi"
    )
    assert job.stages[1].node_id == "node-b"


def test_job_assign_rejects_prompt_on_non_entry_stage() -> None:
    with pytest.raises(ValidationError, match="entry stage"):
        JobAssign(
            job_id="job-0001",
            model_id="tinyllama-1.1b",
            my_stage_idx=1,
            stages=stages(),
            prompt="hi",
        )


def test_job_assign_rejects_duplicate_stage_idx() -> None:
    dup = (
        StageAssignment(stage_idx=0, node_id="node-a", shard_id="s0", layer_start=0, layer_end=7),
        StageAssignment(stage_idx=0, node_id="node-b", shard_id="s1", layer_start=8, layer_end=15),
    )
    with pytest.raises(ValidationError, match="duplicate stage_idx"):
        JobAssign(job_id="job-0001", model_id="tinyllama-1.1b", my_stage_idx=0, stages=dup)


def test_job_assign_rejects_non_contiguous_graph() -> None:
    gap = (
        stages()[0],
        StageAssignment(stage_idx=2, node_id="node-b", shard_id="s1", layer_start=8, layer_end=15),
    )
    with pytest.raises(ValidationError, match="contiguous"):
        JobAssign(job_id="job-0001", model_id="tinyllama-1.1b", my_stage_idx=0, stages=gap)


def test_job_assign_rejects_descending_layers() -> None:
    reversed_stages = tuple(reversed(stages()))
    with pytest.raises(ValidationError, match="ascending"):
        JobAssign(
            job_id="job-0001", model_id="tinyllama-1.1b", my_stage_idx=0, stages=reversed_stages
        )


# -- registration + ledger ---------------------------------------------------


def test_register_round_trip() -> None:
    reg = Register(node_id="node-alpha", auth_token="tok_12345678", manifest=manifest())
    env = Envelope.wrap(MessageType.REGISTER, reg, ts=1.75e9)
    parsed = parse_payload(Envelope.model_validate_json(env.model_dump_json()))
    assert isinstance(parsed, Register)
    assert parsed.manifest.gpu is not None
    assert parsed.manifest.gpu.name == "RTX 3060"


def test_register_rejects_short_token() -> None:
    with pytest.raises(ValidationError):
        Register(node_id="node-alpha", auth_token="short", manifest=manifest())


def test_ledger_event_idempotency_key() -> None:
    event = LedgerEvent(
        node_id="node-alpha",
        job_id="job-0001",
        model_id="qwen2.5-7b",
        stage_idx=3,
        attempt=2,
        partition_id="layers_12_18",
        tokens_in=100,
        tokens_out=0,
        flops_est=3.8e11,
        compute_seconds=2.0,
        outcome=TaskOutcome.SUCCESS,
        ts=1.75e9,
    )
    assert event.idempotency_key == ("job-0001", 3, 2)


def test_ledger_event_rejects_negative_flops() -> None:
    with pytest.raises(ValidationError):
        LedgerEvent(
            node_id="node-alpha",
            job_id="job-0001",
            model_id="qwen2.5-7b",
            stage_idx=0,
            attempt=0,
            partition_id="layers_0_3",
            tokens_in=1,
            tokens_out=0,
            flops_est=-1.0,
            compute_seconds=0.1,
            outcome=TaskOutcome.FAILED,
            ts=1.75e9,
        )


def test_activation_relay_header_defaults() -> None:
    header = ActivationRelayHeader(
        job_id="job-0001", stage_idx=1, attempt=0, seq=0, shape=(4, 3584), n_bytes=57344
    )
    assert header.dtype == "fp32"
    assert header.is_final is False
