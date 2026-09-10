"""DAIN common: shared, dependency-light building blocks.

Modules:
- schemas:    wire protocol envelope + message payloads (spec §10–§11)
- config:     scoring/protocol configuration models (spec §7/§10)
- scoring:    node scoring math (spec §7), pure functions
- accounting: model registry, FLOP estimation, credit function (spec §15)

Nothing in this package performs I/O; it is imported by the coordinator, the
node agent, the simulator and the tests.
"""

from dain_common.accounting import (
    MODELS,
    OUTCOME_FACTOR,
    CreditWeights,
    ModelSpec,
    credit,
    estimate_prompt_tokens,
    flops_for_stages,
    flops_per_token,
    get_model,
)
from dain_common.config import (
    DEFAULT_HEARTBEAT_INTERVAL_S,
    DEFAULT_OFFLINE_AFTER_MISSED,
    ProtocolConfig,
    ScoringConfig,
    ScoringRefs,
    ScoringWeights,
)
from dain_common.schemas import (
    PAYLOAD_TYPES,
    PROTOCOL_VERSION,
    ActivationRelayHeader,
    CapabilityManifest,
    CPUInfo,
    Envelope,
    GenerationParams,
    GPUInfo,
    Heartbeat,
    JobAssign,
    JobState,
    JobStatus,
    LedgerEvent,
    MessageType,
    MetricsReport,
    ModelManifest,
    NetInfo,
    NodeState,
    PowerInfo,
    Register,
    RegisterAck,
    ShardManifest,
    ShardRef,
    SoftwareInfo,
    StageAssignment,
    TaskOutcome,
    parse_payload,
)
from dain_common.scoring import (
    WEIGHT_KEYS,
    NodeReputation,
    ScoreResult,
    ewma,
    is_feasible,
    norm,
    norm_inv,
    score_node,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL_S",
    "DEFAULT_OFFLINE_AFTER_MISSED",
    "PROTOCOL_VERSION",
    "WEIGHT_KEYS",
    "ActivationRelayHeader",
    "CPUInfo",
    "CapabilityManifest",
    "CreditWeights",
    "Envelope",
    "GenerationParams",
    "GPUInfo",
    "Heartbeat",
    "JobAssign",
    "JobState",
    "JobStatus",
    "LedgerEvent",
    "MessageType",
    "MetricsReport",
    "ModelManifest",
    "ModelSpec",
    "MODELS",
    "NetInfo",
    "NodeReputation",
    "NodeState",
    "OUTCOME_FACTOR",
    "PAYLOAD_TYPES",
    "PowerInfo",
    "ProtocolConfig",
    "Register",
    "RegisterAck",
    "ScoreResult",
    "ScoringConfig",
    "ScoringRefs",
    "ScoringWeights",
    "ShardManifest",
    "ShardRef",
    "SoftwareInfo",
    "StageAssignment",
    "TaskOutcome",
    "TokenBatch",
    "Envelope",
    "credit",
    "estimate_prompt_tokens",
    "ewma",
    "flops_for_stages",
    "flops_per_token",
    "get_model",
    "is_feasible",
    "norm",
    "norm_inv",
    "parse_payload",
    "score_node",
]
