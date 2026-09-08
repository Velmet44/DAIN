# DAIN — Decentralized AI Inference Network

**Technical Specification (Skeletal v0.2)**

Status: Draft. This document is a foundation to be expanded into a full architecture and
implementation specification. Items not yet decided are marked **TBD**.

v0.2 changes: locked the reference technology stack (§21) and reference models (§18);
added the latency envelope and feasibility analysis to §9 (first-order estimates from a
hardware model); clarified the NAT-friendly connection model (§10).

---

## 1. Project Overview

DAIN is a decentralized AI inference network in which a lightweight **coordinator** (a single
VPS in the initial deployment) orchestrates inference of a large AI model — potentially a
Mixture-of-Experts (MoE) model — across a pool of 8–16 independent, heterogeneous compute
nodes (typically GPU-equipped machines contributed by independent providers).

The coordinator performs **no primary AI computation**. Its responsibilities are
orchestration, scheduling, routing, monitoring, failure handling, accounting, and system
management. Model execution is partitioned across nodes: transformer layers/stages, MoE
experts, or capability-weighted partitions, depending on model architecture.

```
                ┌────────────────────────────┐
   Client ────► │        Coordinator         │ ◄──── Client
                │  (VPS, orchestration only) │
                │  registry · scheduler ·    │
                │  router · monitor · ledger │
                └─────┬───────┬───────┬──────┘
                      │       │       │   (control plane: HTTPS/WSS)
              ┌───────▼──┐ ┌──▼─────┐ ┌▼────────┐
              │  Node 1  │◄┼────────┼┤  Node 2  │    data plane:
              │ layers   │ ├──────► │ │ layers  │    node↔node activation
              │ 0..7     │ │        │ │ 8..15   │    transfer (gRPC/WebRTC TBD)
              └──────────┘ ┌▼───────┐ └─────────┘
                           │ Node N │
                           │ experts│
                           └────────┘
```

**Control plane** (client↔coordinator, node↔coordinator): registration, heartbeats,
scheduling, job tracking, accounting.
**Data plane** (node↔node, or relayed via coordinator in MVP): transfer of prompts and
intermediate activations, streaming of output tokens.

---

## 2. Goals

1. Run a single large model distributed across 8–16 heterogeneous nodes under one coordinator.
2. Intelligent, capability-aware partitioning of the model (layers, stages, and/or experts).
3. Normalized, configurable node scoring and top-K node selection.
4. Pipelined, streaming inference with concurrent requests.
5. Fault tolerance: heartbeat-based failure detection, stage-level retry, reassignment to
   backup nodes, degraded-mode operation.
6. Measurable-work-based compute accounting with a payment-agnostic ledger API.
7. A quantified experimental investigation of whether distributed use of existing hardware
   reduces demand for new centralized AI infrastructure.
8. A coordinator design that can later be replicated for high availability (no permanent
   single point of failure).

## 3. Non-Goals (for the MVP)

- No blockchain, crypto payments, or on-chain settlement (ledger only; settlement is out of scope).
- No training or fine-tuning; inference only.
- No strict latency SLOs (best-effort interactive latencies only).
- No Byzantine-robust consensus, zero-knowledge verification of node computation, or
  result-cryptographic attestation (reputation + spot-check verification only, see §16).
- No automatic elastic autoscaling of the node pool beyond join/leave handling.
- No multi-tenant isolation guarantees beyond API authentication.
- Not attempting to solve general distributed-deep-learning research problems
  (e.g., optimal auto-parallelism search); partitioning is static/semi-static.

---

## 4. System Architecture

### Components

| Component | Count (MVP) | Role |
|---|---|---|
| Coordinator | 1 VPS | Control plane: registry, scoring, scheduling, routing, monitoring, ledger, API |
| Compute nodes | 8–16 | Data plane: execute assigned model partitions |
| Model store | 1 (object storage or coordinator-hosted) | Sharded model weights |
| Client | n | Submits inference requests via coordinator API |

### Logical planes

```
┌───────── Client API (REST/gRPC, SSE or WebSocket streaming) ─────────┐
│                                                                      │
│  ┌────────────── COORDINATOR ──────────────┐                         │
│  │ Registry │ Scorer │ Scheduler │ Router  │  Job Tracker            │
│  │ Health Monitor │ Ledger │ Metrics │ Auth │                         │
│  └───────┬────────────────────────────────┘                         │
│          │  Node API (WSS/HTTPS: register, heartbeat, assign, report)│
│   ┌──────┼──────────────────────────────┐                            │
│   │      ▼          data plane           │                           │
│  Node A ─ activations ─► Node B ─► … ─► Node Z ─ tokens ─► client   │
└──────────────────────────────────────────────────────────────────────┘
```

- MVP data-plane topology: linear pipeline (stage *i* sends activations to stage *i+1*).
  Mesh/parallel (tensor-parallel) topologies and direct node↔node P2P links are future work;
  MVP may relay activations through the coordinator at a throughput cost (TBD).
- Node-to-node links carry only activations/KV-cache references, never weights.

## 5. Coordinator

Responsibilities:

- **Node registration & authentication** (§11): issue node identities and tokens.
- **Capability discovery**: store per-node hardware/software capability manifests.
- **Node scoring** (§7): compute and cache rolling node scores.
- **Scheduling** (§12): partition placement, top-K selection, backup assignment.
- **Model partition management**: maintain the partition table
  `(model_id, partition_id, layer_range | expert_ids) → node_id`.
- **Request routing**: accept client prompts, resolve execution graph, dispatch to entry node.
- **Job tracking**: per-request state machine (QUEUED → DISPATCHED → RUNNING → STREAMING →
  COMPLETED | FAILED | RETRYING), stage-level progress.
- **Health monitoring** (§13): heartbeats, timeout detection, node state machine.
- **Failure recovery** (§13): retry, reassignment, degraded mode.
- **Accounting** (§15): ledger of measured work per node.
- **Metrics** (§14) and **API access** (client API + node API + admin API).

HA path (post-MVP, design constraint now): all coordinator state (registry, partition table,
job tracker, ledger) lives in a replicated store (e.g., Postgres/Raft KV — TBD) so a second
coordinator instance can take over; API endpoints are stateless wrappers.

## 6. Compute Nodes

Node agent responsibilities:

1. Register with coordinator; authenticate with issued token (mTLS or bearer — TBD).
2. Report a **capability manifest**:
   - GPU: model, count, VRAM total/free, measured TFLOPS (FP16/INT8), memory bandwidth;
   - CPU: cores, ISA extensions; RAM total/free;
   - Network: measured bandwidth and latency to coordinator (and sampled peers);
   - Software: runtime, supported kernels/quantizations;
   - Optional: measured idle/system power draw for energy accounting.
3. Download/cache assigned model shards (weights fetched from model store, content-hashed).
4. Execute assigned inference workloads (layer range or expert set).
5. Stream intermediate activations to the next node; final node streams tokens to client
   (via coordinator relay in MVP).
6. Report per-job progress (tokens, layers completed, utilization).
7. Send heartbeats (interval: 5 s default; configurable) with health/resource payload.
8. Maintain secure channels to coordinator (TLS, pinned CA).

Heterogeneity: nodes may differ in VRAM, throughput, and bandwidth; the scheduler gates
placement on node score (§7) and sizes partitions proportional to each node's measured
throughput so slow nodes get fewer layers/experts (§8.1, §12).
A node must be able to load its assigned shard into free VRAM/RAM (with quantization
fallback, TBD).

### Node state machine (coordinator view)

```
                 register/score ≥ threshold
   (new) ───────────────► ONLINE ◄───────────────┐
                            │  │                  │ recovery
        assigned workload   │  │ heartbeat lost /  │ (heartbeat ok,
               ▼            │  │ errors/overload   │ score ok)
            BUSY ◄──────────┘  ▼                   │
              │  job done       DEGRADED ──────────┤
              └────────────► ONLINE                │
                             │  no heartbeat for   │
                             │  offline_timeout    │ re-register
                             ▼                     │
                          OFFLINE ─────────────────┘
```

Transitions and coordinator reactions:

| Transition | Trigger | Coordinator action |
|---|---|---|
| → ONLINE | registration, score ≥ `min_score` | Add to candidate pool; eligible for placement |
| ONLINE → BUSY | workload assigned | Mark capacity used; exclude from new placements |
| BUSY → ONLINE | job completed/failed | Release capacity; update score with new stats |
| → DEGRADED | heartbeat alive but errors/latency/thermal/load exceed thresholds, or score < `min_score` | Stop new placements on node; drain: let running stages finish, reassign queued work |
| DEGRADED → ONLINE | metrics recover | Re-admit to pool |
| → OFFLINE | no heartbeat for `offline_timeout` (default 15 s = 3 missed), or explicit deregister | Evict from pool; reassign its partitions to backups (§13) |

## 7. Node Scoring Algorithm

Each node reports metrics; the coordinator computes a normalized score in **[0, 1]**.
Each raw metric *m* is normalized against a reference value `m_ref` (the best observed in
the pool, or a configured baseline) with saturation:

```
norm(m) = m / (m + m_ref)          # saturating normalization, ∈ (0, 1)
```

Metrics where lower is better (latency, utilization, failure rate) use
`norm_inv(m) = 1 - m/(m + m_ref)`.

### Score

```
S(node) = Π penalty_j · Σ_i  w_i · n_i          with  Σ w_i = 1

  n_i ∈ {
    norm(FLOPS_measured)        # benchmarked throughput (FP16 or quantized)
    norm(VRAM_free)
    norm(MEM_BW)
    norm(NET_BW)
    norm_inv(NET_LAT_p95)
    norm_inv(utilization)       # current GPU load (lower better)
    uptime_ratio                # EWMA of availability, α TBD
    norm_inv(failure_rate)      # failed stages / total stages (EWMA)
    norm(energy_efficiency)     # TFLOPS/W, optional
  }

  penalty_j ∈ { 0 if VRAM_free < shard_min,      # hard infeasibility
                0.5 if uptime < 0.8, ... }       # soft penalties, TBD
```

Defaults (configurable via `scoring.weights`): throughput 0.30, free VRAM 0.20,
network 0.15 (bandwidth+latency), reliability 0.15 (uptime+failure rate),
memory bandwidth 0.10, load 0.05, energy 0.05. Weights are a coordinator config map,
hot-reloadable; `m_ref` values are recomputed on pool changes with hysteresis to avoid
score flapping (EWMA, half-life 5 min, TBD).

Score is recomputed on every metrics report (heartbeat) and cached; ranking is a simple
sort. Selection: filter feasible (penalty = 0 excluded, required VRAM fits), sort by score,
take top K where K ∈ [8, 16] per §12.

## 8. Model Distribution

Partition strategies (per model architecture):

1. **Layer/stage pipelining** (default, dense models): split transformer layers into
   contiguous ranges; each node holds one or more ranges. Sizes are chosen to **equalize
   per-stage latency**, i.e. proportional to each node's *measured inference throughput*
   on its partition (`layers_i ∝ T_i`, min 1 layer), not to the overall node score — the
   score gates admission (feasibility, top-K selection), while stage sizing targets the
   pipeline bottleneck. Simulation on a heterogeneous 12-node pool (TFLOPS 20–320) shows
   latency-proportional splitting yields ~40% higher steady-state throughput than
   score-proportional splitting and ~2.8× over equal splitting, because pipeline throughput
   is `1 / max(stage_latency)` and score compresses throughput differences. Integer-layer
   granularity limits balance on small models; use finer-grained partitions (attention/MLP
   sub-stages) when `L / K < 3` (TBD).
2. **Expert placement** (MoE): each node hosts a subset of experts. Experts are sized to fit
   node VRAM; hot experts (higher routing frequency, measured online) may be replicated onto
   spare capacity (semi-static rebalancing, TBD).
3. **Capability-weighted assignment**: quantized/heavier shards to high-VRAM nodes; CPU-only
   nodes get embedding, LM-head, or attention-light stages if feasible (TBD).
4. **Dynamic per-request selection**: MVP uses a standing placement, refreshed only when
   nodes join/leave or degrade ("semi-static"). Full dynamic per-job placement is future work.

MVP: one model, contiguous layer-range partitioning, semi-static.

## 9. Distributed Inference Pipeline

Per request:

1. Client → coordinator: `{model_id, prompt, params, stream:true}`.
2. Coordinator resolves the model's execution graph (ordered partition chain or
   layer+expert DAG) from the partition table.
3. Verifies assigned nodes are ONLINE/BUSY-but-healthy; else triggers §13.
4. Dispatches job descriptor to entry node: `{job_id, graph, partition_endpoints, prompt}`.
5. Entry node tokenizes; activations flow node → node along the pipeline:
   `h_i = stage_i(h_{i-1})`; for MoE, activations are routed to expert-hosting nodes per
   router logits, with results reduced (sum) at the reducer node (TBD which).
6. Final node samples tokens; tokens stream back (final node → coordinator → client via
   SSE/WS relay).
7. Coordinator tracks stage-level progress; on completion, records accounting events (§15).

**Pipelining**: while node *i* processes request *r*, node *i−1* already processes *r+1*.
The pipeline depth equals the number of stages; steady-state throughput approaches
`1 / max(stage_latency)`. Micro-batching (packing concurrent requests into batches of 2–8,
TBD) improves utilization. KV-cache for long generations stays on the node owning its
layers (MVP constraint: no cross-node KV migration).

```
time ►  Node A: |r1| |r2| |r3| |r4| ...
        Node B:     |r1| |r2| |r3| ...
        Node C:         |r1| |r2| ...
```

**Latency envelope (first-order estimates; hardware model, to be confirmed by the pilot):**

- Single-stream decode latency ≈ `Σ stage_compute + (hops × RTT)`. At 10 ms RTT per hop,
  Qwen2.5-7B fp16 (0.54 GB/layer) on 8× RTX-3060-class nodes costs ≈ 112 ms/token (~9 tok/s);
  a single RTX 4090 would serve the same model at ~66 tok/s — *but a 12 GB card cannot host
  7B fp16 at all (15.2 GB weights)*, and a Qwen3-32B-class model (65.6 GB fp16) fits on **no**
  single consumer GPU. Distributed sharding is therefore not an optimization but the
  **enabler** for pools of 8–12 GB consumer cards to serve ≥7B models.
- Pipelined aggregate throughput ≈ `batch / max(stage_compute)` — hop RTT overlaps across
  in-flight requests; batch-8 ceilings are ~1.7k tok/s (8×3060-class, 7B) and ~900 tok/s
  (8×3090-class, 32B-class model).
- Decode network cost is negligible (~0.14 MB/s per hop at 20 tok/s); **prefill** dominates:
  a 4k-token prompt moves ~30 MB per hop (bf16) ≈ 2.3 s at 100 Mbps (open question §20.11).

Consequence for the MVP: optimize **aggregate pipelined throughput and model capacity**,
not single-stream latency; single-stream interactivity is a known, accepted trade-off.
Prefer same-region nodes; when the pool spans regions, order stages so adjacent stages sit
on low-RTT pairs (TBD optimization, §20.12).

## 10. Communication Protocol

| Link | Transport (MVP) | Format |
|---|---|---|
| Client ↔ Coordinator | HTTPS REST + SSE (or WS, TBD) | JSON |
| Node ↔ Coordinator | WSS (JSON messages) or HTTPS/gRPC — TBD | JSON/protobuf TBD |
| Node ↔ Node (data plane) | MVP: relay via coordinator WSS; v0.2: direct gRPC over TLS | binary activations (safetensors-like frames TBD) |

NAT friendliness: nodes connect **outbound only** (one persistent WSS to the coordinator);
nodes never accept inbound connections in the MVP, so home/cloud nodes behind NAT work
unmodified. The direct node↔node data plane (§19) must then handle NAT traversal
(hole punching / TURN — TBD).

Message families (node ↔ coordinator): `REGISTER`, `REGISTER_ACK`, `HEARTBEAT`,
`METRICS_REPORT`, `JOB_ASSIGN`, `JOB_STATUS`, `ACTIVATION_RELAY`, `TOKEN_BATCH`,
`LEDGER_EVENT`, `SHARD_MANIFEST`. Schemas are pydantic models in `dain_common`
(S1); `TOKEN_BATCH` carries decoded token strings from the sampling stage to the
client relay; `ACTIVATION_RELAY` headers distinguish `hidden` (forward) and
`sampled_token` (last stage → entry) roles.

Constraints to respect in schema design: heartbeats ≤ 1 KiB; activation frames chunked and
streamable; all messages versioned; idempotent job operations (safe retry).

## 11. Node Registration

```
node ─► coordinator: REGISTER {
  node_id (stable, self-generated keypair/key — TBD),
  capability_manifest (§6),
  auth_token / mTLS cert
}
coordinator ─► node: REGISTER_ACK {
  heartbeat_interval, node_token (per-node credential for WS auth; issued once,
                      persisted across re-registrations),
  scoring_snapshot_refs,
  assigned_shards[] (may be empty → standby), model_store_url
}
```

- Coordinator validates manifest (plausibility checks + on-demand micro-benchmark challenge
  to detect inflated claims — run a small timed kernel and compare vs claimed TFLOPS; MVP:
  optional, TBD).
- Node fetches shards by content hash, verifies digest, reports `SHARD_READY`.
- Re-registration after OFFLINE reuses node_id; history (uptime, ledger) persists.

## 12. Scheduling

```
function schedule_placement(model, pool):
    feasible = [n for n in pool if n.state == ONLINE and n.VRAM_free >= shard_min(n)]
    ranked   = sort_by_score_desc(feasible)
    K        = clamp(ceil(model_layers / layers_per_node_target), 8, 16)
    selected = take(ranked, K)
    if len(selected) < min_nodes(model): return WAIT   # or degraded mode (§13)
    assign partition sizes ∝ measured throughput per node (§8.1)
    designate 1–2 highest-score non-selected nodes as warm backups
      (standby: shard manifests known, shards cached if capacity allows)
    emit JOB_ASSIGN / partition-table update
```

Placement is recomputed on: node join/leave, DEGRADED transition, or periodic rebalance
(default: every 10 min, only if imbalance > 20%, TBD). Client requests are admitted while
`QUEUED ≤ queue_limit`; otherwise HTTP 429 with retry-after.

## 13. Fault Tolerance

- **Detection**: heartbeat timeout (3 missed intervals → OFFLINE);
  stage-level watchdog (per-stage deadline = 4 × p99 stage latency → stage failed).
- **Tracking**: coordinator records per-job stage progress via `JOB_STATUS` acks at each hop;
  activations of in-flight stages are buffered at the sending node until the receiver acks.
- **Retry**: a failed *stage* is retried on (a) the same node if the failure was transient
  (timeout on a busy node, once), else (b) a backup node that loads the same shard.
  Only the failed stage onward is recomputed; completed prefix activations are replayed from
  the buffer — the job is **not** restarted from the prompt unless the KV-cache-holding
  node for already-generated context is lost.
- **Reassignment**: on OFFLINE, affected partitions migrate to backups; queued jobs re-route;
  in-flight jobs retry on the new placement.
- **Degraded operation**: if the pool drops below K but ≥ `min_nodes` (e.g., 4, model-
  dependent — with re-partitioning to fewer, larger shards; possible only if VRAM fits),
  the coordinator re-partitions across remaining nodes and keeps serving at lower
  throughput; below `min_nodes`, requests are queued/rejected with a clear status.
- **Idempotency**: all job messages carry `job_id` + `stage_idx` + `attempt`, so retries
  never double-count ledger entries (§15 dedupes on this key).

## 14. Monitoring

- Coordinator exposes Prometheus metrics (`/metrics`) — TBD names:
  `dain_nodes_online`, `dain_node_score`, `dain_jobs_active`, `dain_stage_latency_p50/p99`,
  `dain_stage_retries_total`, `dain_tokens_total`, `dain_job_failures_total`,
  `dain_ledger_credits_total`.
- Heartbeats carry: GPU util %, VRAM used, net BW, temp, power (if available).
- A minimal dashboard (grafana or simple HTML) lists nodes, scores, placements, active jobs.
- Structured logs with `job_id`/`node_id` correlation IDs.

## 15. Compute Accounting

Payment-agnostic ledger of **measured useful work** per node.

Ledger entry (append-only, keyed by `(job_id, stage_idx, attempt)` — idempotent):

```
LedgerEvent {
  node_id, job_id, model_id,
  partition_id,                 # layer range or expert set
  tokens_in, tokens_out,
  flops_est,                    # from per-partition FLOP model (see below)
  compute_seconds,              # wall/GPU busy time on the node
  gpu_util_avg, cpu_util_avg,
  energy_kwh_est (optional),    # util power model per GPU class, TBD
  outcome: SUCCESS | RETRIED_AWAY | FAILED,
  ts
}
```

FLOP estimation uses a static per-partition model:
`flops ≈ 2 · params_active · tokens` (dense) or
`2 · (shared_params + Σ_active experts params) · tokens` (MoE), computed by the coordinator
from the partition table — nodes cannot self-report FLOPs (untrusted input, §16).

Credit function (illustrative default; weights configurable):

```
credit(event) = w_tok · (tokens_in + tokens_out)
              + w_flop · flops_est
              + w_time · compute_seconds · util_factor
  × (1.0 if SUCCESS else 0.2 if RETRIED_AWAY else 0)
```

API: `GET /ledger/node/{id}`, `GET /ledger/summary?since=`,
`POST /ledger/export` (period aggregate). Settlement (fiat, credits, on-chain) is
explicitly out of scope for the MVP; the export is the integration point.

Verification of claimed work (lightweight, MVP): cross-check `compute_seconds` and token
counts against coordinator-observed stage timings and client-visible output length;
statistical outliers flag the node for score penalties. Stronger proof-of-inference
(re-execution sampling, commitments): future work (§19).

## 16. Security

- Node↔coordinator: TLS; node identity via registration token (MVP) → mTLS + per-node
  client certs (v0.2). Coordinator CA pins.
- Client API: API keys (MVP), per-key rate limits.
- Model confidentiality: weights are licensed property — nodes are semi-trusted partners
  (MVP: legal/ToS agreement); shard delivery over authenticated TLS, hash-verified.
  Note: any node holding shards can in principle exfiltrate them; MVP accepts this risk for
  a vetted pool; TEEs are future work (§19).
- Result integrity: nodes are untrusted for accounting claims; coordinator-derived FLOPs and
  output-length cross-checks (§15). Byzantine resistance (wrong activations) is a known open
  problem — MVP mitigates via reputation scoring and optional random re-execution of a
  sampled stage on another node (cost: 2× for sampled jobs), TBD.
- Activation relay through coordinator in MVP gives it visibility for audit (and makes it a
  bandwidth bottleneck — accepted trade-off for v0.1).
- Input safety: prompt content is treated as untrusted data, never executed; size limits;
  per-node resource caps (max batch, max activation frame size).

## 17. Environmental Measurement

**Objective (to be investigated, not assumed):** determine whether orchestrating inference
across existing distributed hardware reduces the need for additional centralized AI
infrastructure, compared with serving the same workload from new dedicated datacenter
capacity.

Hypotheses to test experimentally:

- H1: better utilization of existing idle hardware (measured: average GPU utilization of
  participating nodes with/without DAIN workload vs their baseline idle time).
- H2: avoided dedicated capacity (measured: throughput served by DAIN vs equivalent
  datacenter GPU-hours displaced — estimated, with stated assumptions).
- H3: embodied-carbon effect (measured: avoided new-hardware demand, using published
  embodied-carbon-per-GPU factors; estimate with uncertainty bounds, TBD methodology).
- H4: cooling/facility overhead (measured: PUE proxy — nodes in ambient-cooled environments
  vs datacenter PUE baseline).
- H5: electricity per useful token (measured: `kWh / 1k tokens` on DAIN nodes via power
  telemetry or per-GPU power models, vs published baselines for equivalent centralized
  serving; must include networking and coordinator overhead).

Method: instrument the ledger (§15) with energy estimates; run controlled A/B measurement
campaigns; report with uncertainty. The spec makes **no claim** of environmental benefit
prior to these measurements. Counter-hypotheses (network transfer energy, lower utilization
efficiency of heterogeneous hardware, hardware acceleration of aging due to added duty
cycles) are explicitly part of the measurement plan.

## 18. MVP Scope

In scope:
- 1 coordinator VPS, 8–16 nodes, 1 model.
- Dev/CI model `dain-tiny-16L`: a seeded 16-layer Llama-family transformer with a
  byte-level tokenizer, exported to per-layer shards — hermetic (no network, no
  gated weights) and numerically parity-checkable against the full model.
- Reference models: **TinyLlama-1.1B** (dev/CI, CPU-simulable), **Qwen2.5-7B-Instruct fp16**
  (pilot; 15.2 GB weights — exceeds 12 GB cards, so sharding is *required* on consumer
  pools), **Qwen3-32B fp16** (capacity showcase; 65.6 GB — fits no single consumer GPU),
  **OLMoE-1B-7B** (MoE stretch).
- Layer-range partitioning, semi-static placement, throughput-proportional stage sizing (§8.1).
- Node scoring (§7) with configurable weights.
- Prompt routing, streaming inference (SSE relay), pipelining across stages.
- Heartbeat monitoring, node state machine, stage retry, backup reassignment, degraded mode.
- Ledger with per-node usage records + export API.
- Client REST API: `POST /v1/completions` (streaming), `GET /v1/models`,
  `GET /v1/nodes`, `GET /v1/jobs/{id}`.

Explicitly deferred: dynamic expert rebalancing, tensor parallelism, direct P2P data plane,
KV-cache migration, blockchain settlement, strong result verification, coordinator HA,
multi-model routing, tenant isolation, energy telemetry beyond coarse estimates.

## 19. Future Extensions

- Direct node↔node data plane (gRPC/WebRTC), topology-aware placement (latency matrix).
- Dynamic MoE expert replication and hot-expert migration.
- Coordinator replication (Raft) / multi-coordinator federation.
- On-chain or fiat settlement plugged into the ledger export; staking/slashing for
  providers.
- TEE-based node attestation (SEV-SNP/TDX), result-integrity proofs (re-execution
  commitments, zkML — speculative).
- Speculative decoding across pipeline stages; disaggregated prefill/decode tiers.
- Elastic multi-model pool; batch schedulers with SLO classes.

## 20. Open Technical Questions

1. Activation relay through coordinator vs direct P2P in v0.1 — bandwidth budget?
2. Quantization strategy per node (uniform vs mixed-precision partitions)? Accuracy impact
   of heterogeneous precision across a pipeline?
3. Optimal `m_ref` baselines and EWMA half-lives for scoring stability vs responsiveness?
4. MoE expert routing on lossy links: fallback when the expert-hosting node is degraded —
   replicate, remap to closest expert, or run on CPU?
5. Stage-deadline tuning vs false-positive retries under load spikes.
6. How to verify node capability claims cheaply (benchmark challenge design)?
7. Energy measurement fidelity: onboard power telemetry vs power models — error bounds?
8. Ledger double-entry design if multiple coordinators are later introduced.
9. Context-length growth: which node stores KV, and what happens on its failure?
10. Minimum viable re-execution sampling rate for result-integrity guarantees?
11. Activation compression for prefill transfer: a 4k-token prefill moves ~30 MB per hop
    (Qwen2.5-7B, bf16) ≈ 2.3 s at 100 Mbps — does pipelined prefill hide this, or is
    fp8/quantized activation transfer required?
12. Stage ordering across regions: place adjacent stages on the lowest-RTT node pairs
    (greedy path optimization) vs pure score ranking — at what RTT does topology
    dominate the score?

## 21. Reference Technology Stack (MVP)

Locked for the first implementation; changing a choice requires a written rationale here.

| Concern | Choice | Rationale |
|---|---|---|
| Client | React 18 + TypeScript + Vite, static SPA | SSE-streaming chat UI + node dashboard; static build deploys anywhere free; strong tooling |
| Client hosting | Netlify (static) | Free tier, trivial CI deploys, SSE passthrough works; Render is the fallback |
| Coordinator | Python 3.11+, FastAPI + Uvicorn, pydantic v2, `websockets` | Orchestration is I/O-bound (no primary compute); async-native SSE/WS; fastest iteration |
| Coordinator state | SQLite (WAL) → Postgres post-MVP | Zero-ops on a VPS; schema kept portable for the HA path (§5) |
| Node agent | Python 3.11+, asyncio, PyTorch + transformers, psutil | GPU math runs in CUDA — Python only orchestrates and serializes; one language shares protocol code with the coordinator |
| Shared code | `dain-common` package: pydantic message schemas (§10), FLOP tables, scoring config | Single source of truth imported by coordinator, node, sim, and tests |
| Dev simulation | Multi-process local cluster: coordinator + N node agents, TinyLlama-1.1B on CPU, deterministic FakeExecutor for CI | Full pipeline testable without any GPU |
| Tooling | uv — one venv + lockfile **per project** (Common/Coordinator/Node/Sim; `dain-common` consumed via editable path dependency), ruff, pytest, mypy (non-gating) | Fast, deterministic, agent-friendly loop; every component standalone-syncable and deployable |

Deliberately **not** chosen: Rust/Go/C++ for coordinator or node — they pay off only in the
activation-relay hot path and deployment footprint, not in MVP iteration speed; the
`dain-common` schemas keep a later partial rewrite of just the relay (e.g., in Go or Rust)
feasible without touching control logic. Node.js for the coordinator — weaker story for
shared ML-adjacent code with the node agent.

Escape hatch: if the S10 pilot shows the Python relay CPU-bound below target throughput,
split **only** the relay into a compiled sidecar service; do not rewrite the control plane.

---

*End of skeletal specification. Expand per-section before implementation planning.*
