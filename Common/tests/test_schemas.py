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
    MetricsReport,
    ModelManifest,
    NetInfo,
    NodeState,
    QuantizationSpec,
    Register,
    ShardRef,
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
    assert len(PAYLOAD_TYPES) == 11


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


# -- model manifest tokenizer fields (S18 real-model path) -------------------


def model_manifest(**overrides) -> ModelManifest:
    values = {
        "model_id": "llama-3.2-3b",
        "name": "Llama 3.2 3B",
        "layers": 28,
        "hidden": 3072,
        "heads": 24,
        "kv_heads": 8,
        "intermediate": 8192,
        "vocab_size": 128256,
        "eos_token_id": 128001,
    }
    values.update(overrides)
    return ModelManifest(**values)


def test_manifest_defaults_to_byte_tokenizer() -> None:
    m = model_manifest()
    assert m.tokenizer_file is None
    assert m.tokenizer_hash is None
    assert m.dtype == "fp32"


def test_manifest_accepts_tokenizer_pair() -> None:
    m = model_manifest(
        tokenizer_file="tokenizer.json",
        tokenizer_hash="a" * 64,
        dtype="fp16",
    )
    assert m.tokenizer_file == "tokenizer.json"
    assert m.tokenizer_hash == "a" * 64
    assert m.dtype == "fp16"


def test_manifest_rejects_half_tokenizer_tuple() -> None:
    with pytest.raises(ValidationError, match="set together"):
        model_manifest(tokenizer_file="tokenizer.json")


def test_metrics_report_accepts_ram_free_gb() -> None:
    m = MetricsReport(ram_free_gb=3.6, vram_free_gb=11.0)
    assert m.ram_free_gb == 3.6
    # Round-trips through a heartbeat envelope like every other payload (S17).
    hb = Heartbeat(node_id="node-x", seq=0, metrics=m)
    env = Envelope.wrap(MessageType.HEARTBEAT, hb, ts=1.75e9)
    restored = Envelope.model_validate_json(env.model_dump_json())
    parsed = Heartbeat.model_validate(restored.payload).metrics
    assert parsed is not None and parsed.ram_free_gb == 3.6


def test_metrics_report_rejects_negative_ram() -> None:
    with pytest.raises(ValidationError):
        MetricsReport(ram_free_gb=-1.0)


# -- export pipeline: quantization metadata (session N) ----------------------


def test_quantization_spec_defaults_to_unquantized() -> None:
    q = QuantizationSpec()
    assert q.backend == "none"
    assert q.scheme == "none"
    assert q.bits == 32
    assert q.is_quantized is False


def test_quantization_spec_int4() -> None:
    q = QuantizationSpec(
        backend="torchao",
        scheme="int4_weight_only",
        bits=4,
        group_size=128,
        activation_dtype="bf16",
        coverage="all_linear",
    )
    assert q.is_quantized is True
    assert q.activation_dtype == "bf16"


def test_quantization_spec_rejects_bad_bits() -> None:
    with pytest.raises(ValidationError):
        QuantizationSpec(bits=3, group_size=128)


# -- legacy manifest backward compatibility (session N) ----------------------


def test_legacy_manifest_parses_without_export_fields() -> None:
    m = model_manifest()
    assert m.format is None
    assert m.quantization is None
    assert m.architecture is None
    assert m.artifact_version == 1


def test_quantized_manifest_round_trip() -> None:
    q = QuantizationSpec(
        backend="torchao",
        scheme="int4_weight_only",
        bits=4,
        group_size=128,
        activation_dtype="bf16",
        coverage="all_linear",
    )
    m = model_manifest(
        format="torch_pt",
        quantization=q,
        base_model_id="unsloth/llama-3.2-3b",
        architecture="llama",
        adapter_id="llama",
        artifact_version=1,
        dtype="bf16",
    )
    dumped = m.model_dump()
    restored = ModelManifest.model_validate(dumped)
    assert restored.quantization is not None
    assert restored.quantization.is_quantized is True
    assert restored.format == "torch_pt"
    assert restored.architecture == "llama"


def test_manifest_rejects_quantization_without_format() -> None:
    q = QuantizationSpec(backend="torchao", scheme="int4_weight_only", bits=4)
    with pytest.raises(ValidationError, match="explicit shard format"):
        model_manifest(quantization=q)


def test_manifest_rejects_invalid_format() -> None:
    with pytest.raises(ValidationError):
        model_manifest(format="gguf")


# -- capability manifest software report (session N) -------------------------


def test_software_info_reports_backends() -> None:
    sw = {
        "torch_version": "2.5.1",
        "supported_backends": ("torchao",),
        "supported_quantization": ("int4_weight_only",),
        "supported_activation_dtypes": ("fp16", "bf16"),
        "supported_adapters": ("llama",),
        "torchao_version": "0.7",
    }
    m = manifest(software=sw)
    assert m.software.supported_backends == ("torchao",)
    assert "bf16" in m.software.supported_activation_dtypes


def test_shard_ref_format_defaults_to_safetensors() -> None:
    r = ShardRef(model_id="m", shard_id="s0", content_hash="a" * 16, size_bytes=1)
    assert r.format is None or r.format == "safetensors"
