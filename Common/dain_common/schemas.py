"""Wire protocol schemas (spec §10–§11).

Every message on the wire is a JSON `Envelope`: `{v, type, ts, payload}` where
`payload` is one of the per-family payload models below. `parse_payload()` maps
an envelope's `type` to the right payload model and validates it.

Design constraints (spec §10): heartbeats ≤ 1 KiB; all messages versioned via
`Envelope.v`; extra fields rejected (`extra="forbid"`) so schema drift fails
loudly; job operations carry `(job_id, stage_idx, attempt)` so retries are
idempotent (spec §13).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROTOCOL_VERSION = 1


class MessageType(StrEnum):
    """Node↔coordinator message families (spec §10)."""

    REGISTER = "register"
    REGISTER_ACK = "register_ack"
    HEARTBEAT = "heartbeat"
    METRICS_REPORT = "metrics_report"
    JOB_ASSIGN = "job_assign"
    JOB_STATUS = "job_status"
    ACTIVATION_RELAY = "activation_relay"
    TOKEN_BATCH = "token_batch"
    STAGE_RETRY = "stage_retry"
    LEDGER_EVENT = "ledger_event"
    SHARD_MANIFEST = "shard_manifest"


class NodeState(StrEnum):
    """Coordinator-side node lifecycle (spec §6)."""

    ONLINE = "online"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class JobState(StrEnum):
    """Per-request job lifecycle (spec §5)."""

    QUEUED = "queued"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class TaskOutcome(StrEnum):
    """Ledger outcome classes (spec §15)."""

    SUCCESS = "success"
    RETRIED_AWAY = "retried_away"
    FAILED = "failed"


class _Model(BaseModel):
    """Base: immutable, no unknown fields, strict enums on dump/load."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Hardware / capability manifests (spec §6.2)
# ---------------------------------------------------------------------------


class GPUInfo(_Model):
    name: str = Field(min_length=1, max_length=128)
    count: int = Field(default=1, ge=1, le=64)
    vram_total_gb: float = Field(gt=0, le=8192)
    vram_free_gb: float = Field(ge=0)
    # Claimed by the node; scoring treats it as unverified (§7/§11)
    tflops_claimed: float = Field(gt=0, le=10_000)
    mem_bw_gbs: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _free_within_total(self) -> GPUInfo:
        if self.vram_free_gb > self.vram_total_gb:
            raise ValueError("vram_free_gb cannot exceed vram_total_gb")
        return self


class CPUInfo(_Model):
    cores: int = Field(ge=1, le=16384)
    ram_total_gb: float = Field(gt=0, le=1_048_576)
    ram_free_gb: float = Field(ge=0)

    @model_validator(mode="after")
    def _free_within_total(self) -> CPUInfo:
        if self.ram_free_gb > self.ram_total_gb:
            raise ValueError("ram_free_gb cannot exceed ram_total_gb")
        return self


class NetInfo(_Model):
    bw_mbps: float = Field(gt=0, le=1_000_000)
    lat_ms_p95: float = Field(gt=0, le=10_000)


class SoftwareInfo(_Model):
    runtime: str = "python"
    agent_version: str = "0.1.0"
    torch_version: str | None = None
    os_name: str | None = None
    # Export/quantization capability report (added with the export pipeline).
    # A node that does not list a backend here cannot host stages for models
    # that require it; the scheduler filters on these fields (§7, session N).
    supported_backends: tuple[str, ...] = ()
    supported_quantization: tuple[str, ...] = ()
    supported_activation_dtypes: tuple[str, ...] = ("fp16",)
    supported_adapters: tuple[str, ...] = ()
    torchao_version: str | None = None
    # Packing layouts the node's torchao installation can execute
    # ("tensor_core_tiled" needs CUDA+tinygemm; "int4_cpu" runs on CPU).
    supported_packing_layouts: tuple[str, ...] = ()


class PowerInfo(_Model):
    idle_watts: float | None = Field(default=None, ge=0)
    tdp_watts: float | None = Field(default=None, gt=0)


class CapabilityManifest(_Model):
    """Reported once at registration; `gpu=None` means a CPU-only node."""

    gpu: GPUInfo | None = None
    cpu: CPUInfo
    net: NetInfo
    software: SoftwareInfo = SoftwareInfo()
    power: PowerInfo | None = None


# ---------------------------------------------------------------------------
# Shards (spec §8, §11)
# ---------------------------------------------------------------------------

#: Serialized weight tensor container. `safetensors` keeps fp16/fp32 shards
#: (legacy, byte-portable); `torch_pt` carries TorchAO tensor subclasses whose
#: packing metadata only `torch.load` reproduces (quantized shards).
SHARD_FORMATS = ("safetensors", "torch_pt")


class QuantizationSpec(_Model):
    """Describes how a model's weights were quantized at export time.

    Quantization happens once in the exporter; node start-up never quantizes.
    `backend="none"`/`scheme="none"` describe un-quantized (legacy) models.
    """

    backend: str = "none"
    scheme: str = "none"
    bits: int = Field(default=32, ge=2, le=32)
    group_size: int = Field(default=128, ge=1)
    activation_dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
    # Packing/serialization format version of the exporter that produced it.
    packing_version: str = "1"
    # Library versions at export time; informational, never enforced.
    quantizer_version: str = ""
    # Which layers were quantized: "all_linear" | "selected" | "none".
    coverage: str = "none"
    # TorchAO tensor-subclass layout the weights were packed with. "auto"
    # (default) lets the exporter pick: TensorCoreTiledLayout on CUDA, else
    # Int4CPULayout. The scheduler must only place a model on nodes whose
    # SoftwareInfo lists the recorded layout.
    packing_layout: str = "auto"

    @field_validator("bits")
    @classmethod
    def _bits_power_of_two(cls, value: int) -> int:
        if value not in {2, 4, 8, 16, 32}:
            raise ValueError("bits must be 2, 4, 8, 16, or 32")
        return value

    @property
    def is_quantized(self) -> bool:
        return self.backend != "none" and self.scheme != "none"


class ShardRef(_Model):
    model_id: str = Field(min_length=1)
    shard_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=8, description="Hex digest of the shard file")
    layer_start: int | None = Field(default=None, ge=0)
    layer_end: int | None = Field(default=None, ge=0, description="Inclusive upper bound")
    size_bytes: int = Field(ge=0)
    # None = "safetensors" (backward compatible). Quantized shards are
    # `torch_pt` because TorchAO's AffineQuantizedTensor packing metadata only
    # survives `torch.save`/`torch.load` (plan §2.1).
    format: str | None = Field(default=None, pattern="^(safetensors|torch_pt)$")

    @model_validator(mode="after")
    def _layers_ordered(self) -> ShardRef:
        if (
            self.layer_start is not None
            and self.layer_end is not None
            and self.layer_end < self.layer_start
        ):
            raise ValueError("layer_end must be >= layer_start")
        return self


class ShardManifest(_Model):
    model_id: str = Field(min_length=1)
    shards: tuple[ShardRef, ...] = ()
    total_size_bytes: int = Field(ge=0)


class ModelManifest(_Model):
    """Served by the model store; nodes build their stage runners from it.

    The dev model (`dain-tiny-16L`) is a seeded Llama-family transformer with a
    byte-level tokenizer (vocab 256), so the whole pipeline is hermetic — no
    network or gated weights at test time (spec §18).
    """

    model_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    layers: int = Field(gt=0)
    hidden: int = Field(gt=0)
    heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    intermediate: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    eos_token_id: int = Field(ge=0)
    rope_theta: float = Field(default=10000.0, gt=0)
    dtype: str = Field(default="fp32")
    # Real-model deployments carry an HF fast tokenizer served by the model
    # store; `tokenizer_file=None` means the dev byte-level tokenizer (vocab
    # 256). `tokenizer_hash` lets nodes verify the downloaded file (spec §11).
    tokenizer_file: str | None = Field(default=None, min_length=1)
    tokenizer_hash: str | None = Field(default=None, min_length=8)
    shards: tuple[ShardRef, ...] = ()
    # New in the export pipeline. All optional so legacy fp16/fp32 manifests
    # (produced before the exporter existed) parse unchanged:
    #   format=None            -> legacy safetensors fp16/fp32
    #   quantization=None      -> un-quantized
    format: str | None = Field(default=None, pattern="^(safetensors|torch_pt)$")
    quantization: QuantizationSpec | None = None
    base_model_id: str | None = Field(default=None, min_length=1)
    architecture: str | None = Field(default=None, min_length=1)
    adapter_id: str | None = Field(default=None, min_length=1)
    architecture_config: dict[str, Any] = Field(default_factory=dict)
    # Monotonic artifact-format version; bump when shard layout changes shape.
    artifact_version: int = Field(default=1, ge=1)
    source_config_hash: str | None = Field(default=None, min_length=8)

    @model_validator(mode="after")
    def _tokenizer_consistent(self) -> ModelManifest:
        if (self.tokenizer_file is None) != (self.tokenizer_hash is None):
            raise ValueError("tokenizer_file and tokenizer_hash must be set together")
        if self.format is None and self.quantization is not None:
            raise ValueError("quantization requires an explicit shard format")
        return self


# ---------------------------------------------------------------------------
# Message payloads
# ---------------------------------------------------------------------------


class Register(_Model):
    node_id: str = Field(min_length=3, max_length=64)
    auth_token: str = Field(min_length=8, max_length=512)
    manifest: CapabilityManifest
    agent_version: str = "0.1.0"
    cached_shards: tuple[ShardRef, ...] = ()
    peer_url: str | None = Field(default=None, max_length=256)


class RegisterAck(_Model):
    accepted: bool
    heartbeat_interval_s: float = Field(default=5.0, gt=0)
    # Issued at first registration; the node persists it and authenticates WS
    # connections with it (spec §11).
    node_token: str | None = Field(default=None, max_length=512)
    assigned_shards: tuple[ShardRef, ...] = ()
    model_store_url: str | None = None
    reason: str | None = None


class MetricsReport(_Model):
    gpu_util_pct: float | None = Field(default=None, ge=0, le=100)
    vram_free_gb: float | None = Field(default=None, ge=0)
    # Live free system RAM (spec §12 capacity): refreshed every heartbeat so a
    # node's placement capacity never goes stale with its registration-day
    # snapshot (session 17).
    ram_free_gb: float | None = Field(default=None, ge=0)
    cpu_util_pct: float | None = Field(default=None, ge=0, le=100)
    net_bw_mbps: float | None = Field(default=None, ge=0)
    temp_c: float | None = Field(default=None, ge=-50, le=200)
    power_w: float | None = Field(default=None, ge=0)


class Heartbeat(_Model):
    node_id: str = Field(min_length=3, max_length=64)
    seq: int = Field(ge=0)
    metrics: MetricsReport | None = None
    # Advertised shard inventory. `None` = "unchanged" (a regular silent beat),
    # so the coordinator only rewrites a node's inventory when the node sends a
    # value; an explicit empty tuple revokes shards the node no longer holds.
    cached_shards: tuple[ShardRef, ...] | None = None


class StageAssignment(_Model):
    stage_idx: int = Field(ge=0)
    node_id: str = Field(min_length=3)
    shard_id: str = Field(min_length=1)
    layer_start: int = Field(ge=0)
    layer_end: int = Field(ge=0, description="Inclusive upper bound")
    # S11 (MoE) skeleton: an expert-set stage hosts these expert indices for the
    # layers in [layer_start, layer_end] instead of all layers in the range.
    expert_ids: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _layers_ordered(self) -> StageAssignment:
        if self.layer_end < self.layer_start:
            raise ValueError("layer_end must be >= layer_start")
        return self


class GenerationParams(_Model):
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: int | None = None


class JobAssign(_Model):
    """Dispatched to the entry node; carries the full stage graph (spec §9.4)."""

    job_id: str = Field(min_length=4, max_length=64)
    model_id: str = Field(min_length=1)
    my_stage_idx: int = Field(ge=0)
    stages: tuple[StageAssignment, ...] = ()
    prompt: str | None = Field(default=None, max_length=131072)
    params: GenerationParams = GenerationParams()

    @model_validator(mode="after")
    def _graph_sane(self) -> JobAssign:
        idxs = [s.stage_idx for s in self.stages]
        if len(idxs) != len(set(idxs)):
            raise ValueError("duplicate stage_idx in stage graph")
        if sorted(idxs) != list(range(len(idxs))):
            raise ValueError("stage graph must be contiguous from 0")
        if self.prompt is not None and self.my_stage_idx != 0:
            raise ValueError("prompt may only be assigned to the entry stage (stage_idx=0)")
        for a, b in zip(self.stages, self.stages[1:], strict=False):
            if b.layer_start < a.layer_start:
                raise ValueError("stage graph layers must be in ascending order")
        return self


class JobStatus(_Model):
    job_id: str = Field(min_length=4, max_length=64)
    stage_idx: int = Field(ge=0)
    attempt: int = Field(ge=0)
    state: JobState
    tokens_done: int = Field(default=0, ge=0)
    detail: str | None = Field(default=None, max_length=2048)


class ActivationRelayHeader(_Model):
    """Header of an activation chunk; bytes follow out-of-band (spec §10).

    `role="hidden"` carries stage activations toward the next stage;
    `role="sampled_token"` carries the last stage's sampled token id (int64
    bytes) back to the entry stage, which embeds it for the next decode step.
    `is_final` on a sampled_token means generation finished for the job.
    """

    job_id: str = Field(min_length=4, max_length=64)
    stage_idx: int = Field(ge=0)
    attempt: int = Field(ge=0)
    seq: int = Field(ge=0)
    dtype: Literal["fp32", "fp16", "bf16", "int64"] = "fp32"
    role: Literal["hidden", "sampled_token"] = "hidden"
    shape: tuple[int, ...] = ()
    n_bytes: int = Field(ge=0)
    is_final: bool = False


class TokenBatch(_Model):
    """Final-stage → coordinator → client token stream (spec §9.6)."""

    job_id: str = Field(min_length=4, max_length=64)
    tokens: tuple[str, ...] = ()
    is_final: bool = False
    finish_reason: Literal["length", "eos", "error"] | None = None
    detail: str | None = None


class StageRetry(_Model):
    """Coordinator → node control message (spec §13 retry, S7).

    Sent to the stage *upstream* of a failed stage: re-send the activation you
    are still buffering for `job_id` so the replacement stage (or a same-node
    retry, when `target_node_id` is the receiver itself) can resume from the
    completed prefix. Dedup on the receiving side is by `seq` — replays of an
    activation the stage already processed are dropped there.
    """

    job_id: str = Field(min_length=4, max_length=64)
    stage_idx: int = Field(ge=0, description="Index of the retried (downstream) stage")
    attempt: int = Field(ge=0, description="Retry attempt of the downstream stage")
    reason: Literal["watchdog", "node_lost", "reassign"] = "watchdog"


class LedgerEvent(_Model):
    """Append-only accounting record (spec §15). Idempotency key:
    `(job_id, stage_idx, attempt)` — retries never double-count."""

    node_id: str = Field(min_length=3)
    job_id: str = Field(min_length=4, max_length=64)
    model_id: str = Field(min_length=1)
    stage_idx: int = Field(ge=0)
    attempt: int = Field(ge=0)
    partition_id: str = Field(min_length=1)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    flops_est: float = Field(ge=0, description="Coordinator-derived, never node-claimed (§15)")
    compute_seconds: float = Field(ge=0)
    gpu_util_avg: float | None = Field(default=None, ge=0, le=100)
    cpu_util_avg: float | None = Field(default=None, ge=0, le=100)
    energy_kwh_est: float | None = Field(default=None, ge=0)
    outcome: TaskOutcome
    ts: float = Field(gt=0)

    @property
    def idempotency_key(self) -> tuple[str, int, int]:
        return (self.job_id, self.stage_idx, self.attempt)


# ---------------------------------------------------------------------------
# Envelope + payload dispatch
# ---------------------------------------------------------------------------

PAYLOAD_TYPES: dict[MessageType, type[BaseModel]] = {
    MessageType.REGISTER: Register,
    MessageType.REGISTER_ACK: RegisterAck,
    MessageType.HEARTBEAT: Heartbeat,
    MessageType.METRICS_REPORT: MetricsReport,
    MessageType.JOB_ASSIGN: JobAssign,
    MessageType.JOB_STATUS: JobStatus,
    MessageType.ACTIVATION_RELAY: ActivationRelayHeader,
    MessageType.TOKEN_BATCH: TokenBatch,
    MessageType.STAGE_RETRY: StageRetry,
    MessageType.LEDGER_EVENT: LedgerEvent,
    MessageType.SHARD_MANIFEST: ShardManifest,
}


class Envelope(_Model):
    v: int = PROTOCOL_VERSION
    type: MessageType
    ts: float = Field(gt=0, description="Unix epoch seconds")
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("v")
    @classmethod
    def _version_supported(cls, value: int) -> int:
        if value != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version {value}; expected {PROTOCOL_VERSION}")
        return value

    @classmethod
    def wrap(cls, msg_type: MessageType, payload: BaseModel, ts: float) -> Envelope:
        return cls(type=msg_type, ts=ts, payload=payload.model_dump(mode="json"))


def parse_payload(envelope: Envelope) -> BaseModel:
    """Validate `envelope.payload` against the model for `envelope.type`."""
    return PAYLOAD_TYPES[envelope.type].model_validate(envelope.payload)
