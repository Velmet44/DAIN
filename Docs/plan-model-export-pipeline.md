# Implementation Plan: Model Export Pipeline with TorchAO INT4 Quantization

## 1. Executive Summary

This plan adds a complete model-export pipeline to DAIN that converts local HuggingFace-style Safetensors models into quantized (TorchAO INT4 weight-only) DAIN shards. The pipeline is exposed via CLI script, Coordinator API, and Web UI. It preserves existing fp16/fp32 paths and introduces a backend abstraction for extensibility.

## 2. Architecture Decisions

### 2.1 Serialization Format Decision

**Choice: `torch.save`/`torch.load` for quantized shards; `safetensors` for fp16/fp32 shards.**

Rationale:
- TorchAO uses tensor subclasses (`AffineQuantizedTensor`) that carry packing metadata, scales, zero_points, and shape info internally.
- `torch.save(state_dict)` preserves the full tensor subclass through serialization — no manual unpacking/repacking required.
- `safetensors` does NOT natively support arbitrary tensor subclasses — it only handles plain PyTorch tensors. Attempting to save TorchAO quantized tensors via `safetensors.save_file()` either fails or silently loses the quantization metadata.
- At load time, `torch.load(path, weights_only=False)` reconstructs the `AffineQuantizedTensor` automatically.
- The existing `load_file()` path from safetensors continues to work for fp16/fp32 shards — zero disruption.
- The shard file extension will be `.pt` for quantized shards and `.safetensors` for fp16/fp32 shards, making the format visually obvious.

Trade-off acknowledged: `.pt` files are less portable across non-PyTorch ecosystems than safetensors. This is acceptable because INT4 quantized execution inherently requires PyTorch + TorchAO, so there is no non-PyTorch consumer.

### 2.2 Adapter Architecture Decision

**Choice: A lightweight adapter registry, not a runtime architecture detection system.**

The exporter classifies inputs into three categories:
- **Supported**: decoder-only causal LM, locally loadable, has a registered adapter, compatible quantization target
- **Detectable but unsupported**: encoder-decoder, multimodal, MoE without adapter, custom remote code
- **Invalid**: missing config/weights/tokenizer, corrupt tensors, config/weight mismatch

The adapter is a thin class that maps weight tensor names to quantizable Linear layers and specifies which layers to quantize vs. keep in higher precision. For Llama-family, this is straightforward. Future architectures (Qwen2, Mistral, etc.) add new adapter classes.

### 2.3 Coordinator Subprocess Model

**Choice: Coordinator spawns exporter as a subprocess, same pattern as GGUF import.**

The Coordinator never imports torch/torchao. It shells out to `uv run --project <Node> python -m dain_node.model_export` and streams output into the ring log. This preserves the Coordinator as a lightweight FastAPI process.

## 3. File-by-File Implementation Plan

### Phase 1: Common Schema Extensions

#### `Common/dain_common/schemas.py` — Extend ModelManifest, add QuantizationSpec, extend CapabilityManifest

**New model: `QuantizationSpec`** (frozen, extra="forbid"):
```python
class QuantizationSpec(_Model):
    backend: str = "none"           # "none" | "torchao"
    scheme: str = "none"            # "none" | "int4_weight_only"
    bits: int = Field(default=32, ge=2, le=32)
    group_size: int = Field(default=128, ge=1)
    activation_dtype: str = "fp16"  # "fp16" | "bf16" | "fp32"
    packing_version: str = "1"      # serialization format version
    quantizer_version: str = ""     # torchao version used
    coverage: str = "all_linear"    # "all_linear" | "selected"
```

**Extend `ModelManifest`** with new optional fields (backward-compatible):
```python
# New fields — None means "legacy fp16/fp32" (backward compatible)
format: str | None = Field(default=None)          # "safetensors" | "torch_pt" | None
quantization: QuantizationSpec | None = None
base_model_id: str | None = None                  # original HF model name
architecture: str | None = None                   # "llama", "qwen2", etc.
adapter_id: str | None = None                     # registered adapter name
artifact_version: int = Field(default=1)          # for future format changes
source_config_hash: str | None = None             # sha256 of source config.json
```

When `format=None` and `quantization=None`, the manifest is a legacy fp16/fp32 model — fully backward compatible. Existing code that reads manifests without these fields continues to work because all new fields have defaults.

**Extend `ShardRef`** with optional format field:
```python
format: str | None = Field(default=None)  # None = "safetensors" (backward compat)
```

**Extend `SoftwareInfo`** in `CapabilityManifest`:
```python
supported_backends: tuple[str, ...] = ()     # ("torchao",)
supported_quantization: tuple[str, ...] = () # ("int4_weight_only",)
torchao_version: str | None = None
supported_adapters: tuple[str, ...] = ()     # ("llama",)
supported_activation_dtypes: tuple[str, ...] = ("fp16",)  # ("fp16", "bf16")
```

#### `Common/dain_common/model_store.py` — No changes needed

The existing `load_manifest` / `list_models` / `is_safe_model_id` functions work unchanged — they read `manifest.json` which is format-agnostic.

### Phase 2: Node Exporter Module

#### `Node/dain_node/model_export.py` — NEW: The core exporter

This is the main new module. Structure:

```
model_export.py
├── AdapterRegistry          # Registry of model adapters
├── ModelAdapter (base)      # Abstract adapter interface
├── LlamaAdapter             # Llama-family adapter
├── ExportConfig             # Pydantic model for export parameters
├── ExportResult             # Pydantic model for export results
├── validate_source()        # Validate source directory
├── detect_model()           # Load config, detect architecture
├── quantize_model()         # Apply TorchAO INT4 quantization
├── export_shards()          # Serialize quantized model as DAIN shards
├── export_model()           # Full pipeline: validate → detect → quantize → shard → manifest
└── main()                   # CLI entry point
```

**Key classes and functions:**

1. **`AdapterRegistry`**: Dict mapping architecture names to adapter classes. Auto-registers on import.

2. **`ModelAdapter` (base class)**:
   - `architecture: str` — name
   - `config_class` — HF config class to load
   - `is_supported(config) -> bool` — validate config is compatible
   - `get_linear_layers(model) -> list[str]` — FQN of linear layers eligible for quantization
   - `get_preserve_layers(model) -> list[str]` — layers to keep in higher precision (embed, norm, lm_head)

3. **`LlamaAdapter(ModelAdapter)`**:
   - `architecture = "llama"`
   - `config_class = LlamaConfig`
   - Checks for causal decoder-only architecture
   - Linear layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
   - Preserved: `embed_tokens`, `norm`, `lm_head`

4. **`ExportConfig`** (Pydantic BaseModel):
   - `source_dir: str`
   - `model_id: str`
   - `output_store: str`
   - `quantization: Literal["int4"] = "int4"`
   - `group_size: int = 128`
   - `activation_dtype: Literal["fp16", "bf16"] = "fp16"`
   - `layers_per_shard: int = 4`
   - `force: bool = False`
   - `trust_remote_code: bool = False`
   - `dry_run: bool = False`
   - `json_progress: bool = False`

5. **`validate_source(source_dir) -> dict`**: 
   - Check config.json exists and is valid
   - Check model weights exist (*.safetensors or *.bin)
   - Check tokenizer files exist (tokenizer.json preferred)
   - Return validation result with detected info

6. **`detect_model(config_path) -> tuple[AutoConfig, str]`**:
   - Load AutoConfig
   - Detect architecture
   - Select adapter from registry
   - Return config and adapter

7. **`quantize_model(model, config, export_config) -> nn.Module`**:
   - Build the Transformers model from config
   - Load weights from source safetensors
   - Apply `torchao.quantize_(model, Int4WeightOnlyConfig(group_size=group_size))`
   - Return quantized model

8. **`export_shards(quantized_model, ...) -> list[ShardRef]`**:
   - Partition model into per-layer groups
   - For each group, extract state dict slice
   - Save as `.pt` file (torch.save preserves AffineQuantizedTensor)
   - Compute SHA-256 hash
   - Return ShardRef list

9. **`export_model(config: ExportConfig) -> ModelManifest`**:
   - Full pipeline orchestration
   - Writes to a temp dir first, then atomic rename on success
   - Only publishes manifest after all artifacts are verified
   - Returns the completed manifest

10. **`main(argv)`**: CLI entry point with argparse, matching the specified flags.

#### `Node/dain_node/shard_export.py` — Extend existing

Add a new function `export_hf_model_torchao()` alongside the existing `export_hf_model()`:
- Takes the same inputs plus quantization config
- Calls into model_export.py's quantization pipeline
- Returns a manifest with the new QuantizationSpec fields populated

The existing `export_hf_model()` and `export_tiny_llama()` remain UNCHANGED.

#### `Node/dain_node/llm.py` — Backend abstraction + quantized loading

**New: `InferenceBackend` base class** with abstract methods:
```python
class InferenceBackend:
    def load_stage(self, manifest, layer_start, layer_end, state_dict, **kwargs) -> StageModel
    def deserialize_shard(self, path: str) -> dict[str, torch.Tensor]
    def capability_report() -> dict
```

**New: `TorchFp16Backend(InferenceBackend)`** — wraps current `StageModel` logic

**New: `TorchAOInt4Backend(InferenceBackend)`**:
- `deserialize_shard()` uses `torch.load(path, weights_only=False)` to load `.pt` shards containing AffineQuantizedTensors
- `load_stage()` constructs layers, applies quantized weights without `.to(dtype)` on packed tensors
- Preserves KV-cache behavior (activations remain fp16/bf16)
- Reports memory estimates for quantized layers

**Modify `StageModel.__init__`**: Accept a `backend` parameter. When backend is `torchao_int4`, skip `param.copy_(state[key].to(dtype))` for quantized weight keys and instead use `param.data = state[key]` (no dtype conversion on packed tensors).

**Modify `fetch_stage()`**: Check manifest format to determine which backend to use. Dispatch to the appropriate `InferenceBackend.deserialize_shard()`.

**Extend `_ACTIVATION_DTYPES`** in `jobs.py`: Add `"bf16": torch.bfloat16`.

### Phase 3: Node Runtime Updates

#### `Node/dain_node/capabilities.py` — Report TorchAO support

Add to the `probe()` function:
- Detect if `torchao` is importable
- If so, get version and add to `SoftwareInfo.supported_backends`
- Add supported quantization schemes
- Add supported adapters (from adapter registry)

#### `Node/dain_node/settings.py` — Export settings

Add optional export-related settings:
- `export_work_dir: str = "export_work"` — temp dir for export staging
- These are not required for the node agent runtime, only for the exporter subprocess.

### Phase 4: Coordinator Updates

#### `Coordinator/dain_coordinator/settings.py` — Export configuration

Add new fields to `CoordinatorSettings`:
```python
export_work_dir: str = ""          # staging directory for exports
export_roots: tuple[str, ...] = ()  # approved source directories
max_export_size_gb: float = 100.0
export_timeout_s: float = 3600.0
```

Add to `COORDINATOR_ENV` mapping and `DEFAULT_CONFIG`.

#### `Coordinator/dain_coordinator/api.py` — Export job API

**New request/response models:**
```python
class ExportModelRequest(BaseModel):
    source_dir: str
    model_id: str
    quantization: Literal["int4"] = "int4"
    group_size: int = 128
    activation_dtype: Literal["fp16", "bf16"] = "fp16"
    layers_per_shard: int = 4
    force: bool = False
    trust_remote_code: bool = False

class ExportJobStatus(BaseModel):
    job_id: str
    state: Literal["queued", "running", "completed", "failed"]
    source_dir: str
    model_id: str
    started_at: float | None = None
    completed_at: float | None = None
    progress: float | None = None  # 0-100
    output_model_id: str | None = None
    error: str | None = None
    logs: list[str] = []
```

**New endpoints on `admin_router`:**
- `POST /admin/models/export` — Start export job (validates paths against export_roots)
- `GET /admin/models/export/{job_id}` — Get job status
- `GET /admin/models/export/{job_id}/logs` — Get job logs
- `POST /admin/models/export/{job_id}/cancel` — Cancel job (best-effort)
- `GET /admin/models/export/roots` — List approved source directories
- `POST /admin/models/export/validate` — Validate a source directory without exporting

**Path security:**
- `source_dir` must resolve under one of the configured `export_roots`
- Reject `..`, symlinks, junctions, absolute path escape
- The directory is on the Coordinator machine, not the browser

**Job management:**
- One export job per source directory at a time (lock by source_dir)
- Job state persisted in-memory (same pattern as `gguf_import`)
- Subprocess runs `uv run --project <Node> python -m dain_node.model_export`
- stdout/stderr streamed to ring log
- On success: `recompute_pool()` to discover the new model variant
- On failure: clear error state, allow retry

**Extend `GET /admin/models`** to include quantization info per model.

**Extend `GET /v1/models`** to include quantization and backend info.

#### `Coordinator/dain_coordinator/partition.py` — Backend-aware placement

Extend `plan_placement()`:
- Check manifest's `quantization` field
- Filter nodes by `supported_backends` and `supported_quantization` in their `SoftwareInfo`
- A node that doesn't support torchao cannot be placed on a torchao-int4 model stage
- fp16/fp32 models continue to place on any compatible node

### Phase 5: CLI Script

#### `Scripts/export-model.ps1` — NEW

Portable PowerShell script, following the `import-gguf.ps1` pattern:

```powershell
param(
    [Parameter(Mandatory=$true)]
    [string]$SourceDir,
    [Parameter(Mandatory=$true)]
    [string]$ModelId,
    [string]$ModelStore = "",
    [ValidateSet("int4")]
    [string]$Quantization = "int4",
    [int]$GroupSize = 128,
    [ValidateSet("fp16", "bf16")]
    [string]$ActivationDtype = "fp16",
    [int]$LayersPerShard = 4,
    [switch]$Force,
    [switch]$DryRun
)
```

Runs: `uv run --project <Node> python -m dain_node.model_export --source-dir ... --model-id ...`

### Phase 6: Web UI Updates

#### `Coordinator/dain_coordinator/admin_ui.html` — Export UI card

Add a new "Model Export" card in the admin page:
- Source directory input (with browse from approved roots)
- Auto-detect button (calls `/admin/models/export/validate`)
- Shows detected: architecture, source dtype, estimated size, tokenizer status
- Model ID input (pre-filled from detection)
- Quantization selector (INT4)
- Group size selector (128)
- Activation dtype selector (fp16/bf16)
- Shard size selector
- Export button → starts job
- Progress bar + live log streaming
- Cancel button
- Result display: published model ID, shard count, total size

Update the existing "Models" card to show quantization and backend info per model.

### Phase 7: Testing

#### `Common/tests/test_schemas.py` — Extend

- Test QuantizationSpec validation
- Test backward-compatible old manifests (no new fields)
- Test invalid quantization metadata
- Test CapabilityManifest with new fields

#### `Node/tests/test_model_export.py` — NEW

- `test_validate_source_missing_config` — fails cleanly
- `test_validate_source_missing_weights` — fails cleanly
- `test_validate_source_missing_tokenizer` — warns but allows (byte tokenizer fallback)
- `test_detect_model_llama` — detects LlamaAdapter
- `test_detect_model_unsupported` — rejects non-Llama
- `test_export_tiny_llama_int4` — full export of the tiny model
- `test_export_produces_valid_manifest` — manifest has all required fields
- `test_export_shard_hashes_verified` — SHA-256 hashes match
- `test_export_atomic_failure` — no partial model_store on failure
- `test_export_idempotent` — re-export with same params works
- `test_export_force_overwrites` — --force replaces existing model

#### `Node/tests/test_quantized_stage.py` — NEW

- `test_load_quantized_shard` — torch.load reconstructs AffineQuantizedTensor
- `test_quantized_stage_forward` — forward pass produces finite logits
- `test_quantized_vs_fp16_output` — within tolerance (INT4 is lossy, ~2-5% deviation expected)
- `test_quantized_weights_not_expanded` — weight.data is still a quantized tensor subclass, not fp16
- `test_unsupported_backend_rejection` — node rejects model with unknown backend

#### `Coordinator/tests/test_export_api.py` — NEW

- `test_start_export_job` — 202 accepted
- `test_export_status_polling` — status transitions
- `test_export_logs` — log endpoint returns content
- `test_concurrent_duplicate_export` — 409 conflict
- `test_export_invalid_path` — 400 bad path
- `test_export_path_escape` — 400 rejected
- `test_model_visible_after_export` — model appears in list only after success
- `test_scheduler_filters_incompatible_nodes` — quantized model not placed on non-torchao nodes

#### `Sim/tests/test_quantized_e2e.py` — NEW (if torch available)

- Two-node quantized model execution
- Quantized shard download and peer transfer
- Existing fp16/fp32 flow remains working

### Phase 8: Documentation & Logging

#### `README.md` — Extend with:
1. Export workflow (CLI + UI)
2. INT4 limitations
3. Supported architectures
4. Hardware requirements
5. Storage estimates

#### `Docs/log.md` — Add new session entry

#### `.gitignore` — Add:
```
export_work/
export_staging/
```

## 4. Implementation Order

| Step | Phase | Files | Estimated Complexity |
|------|-------|-------|---------------------|
| 1 | Common schemas | `schemas.py`, test | Low |
| 2 | Node adapter registry | `model_export.py` (registry + LlamaAdapter) | Medium |
| 3 | Node exporter core | `model_export.py` (validate, detect, quantize, shard) | High |
| 4 | Node backend abstraction | `llm.py` (InferenceBackend, TorchAOInt4Backend) | High |
| 5 | Node capabilities | `capabilities.py` | Low |
| 6 | Coordinator export API | `api.py`, `settings.py` | Medium |
| 7 | Coordinator placement | `partition.py` | Low |
| 8 | CLI script | `export-model.ps1` | Low |
| 9 | Web UI | `admin_ui.html` | Medium |
| 10 | Tests | All test files | Medium |
| 11 | Documentation | README, log.md | Low |
| 12 | Git commit & push | — | Low |

## 5. Key Risks & Mitigations

1. **TorchAO version pinning**: Pin `torchao>=0.7,<1` in Node's pyproject.toml. The INT4WeightOnlyConfig API is stable across 0.7+.

2. **TorchAO CPU quantization**: INT4WeightOnly may require CUDA for optimal kernels. On CPU, the quantized tensor still works but is slower. The exporter should work on CPU; execution benefits from CUDA.

3. **Backward compatibility**: All manifest extensions use default values. Existing code that reads manifests without the new fields continues to work. The `format=None` path means "legacy safetensors."

4. **Subprocess isolation**: The exporter runs in a subprocess with its own torch/torchao environment. If it crashes, the coordinator is unaffected. No partial artifacts are left in model_store (atomic write pattern).

5. **Portability**: No hardcoded paths. All paths resolved via `pathlib` and configuration. Export roots are configurable. The `.pt` shard format is PyTorch-specific, but quantized execution inherently requires PyTorch.

## 6. Dependencies Added

### `Node/pyproject.toml`:
```toml
"torchao>=0.7,<1",
```

### `Coordinator/pyproject.toml`:
No new dependencies — the coordinator shells out to the Node project.

### `Common/pyproject.toml`:
No new dependencies.

## 7. What This Plan Does NOT Do

- Does not support MoE architectures (future adapter)
- Does not support encoder-decoder models (future adapter)
- Does not support models requiring custom remote code
- Does not support INT8 or FP8 quantization (future extension, same framework)
- Does not change the activation relay format (hidden states remain fp16/bf16)
- Does not quantize during node startup (export-only)
- Does not copy source HF directory into model_store
- Does not add torch/torchao dependencies to the Coordinator package
