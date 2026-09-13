# DAIN Instant-Serving Architecture (S22 plan)

Status: approved design, not yet implemented. Companion to `Docs/spec.md`
(amendments listed in §8) and `Docs/stages.md` (stages S22a–S22f below).
Grounded in measurements on the reference 2-core Ice Lake host (2026-09-13,
`llama-3.2-1b`, 16 layers, fp16-runtime); every latency claim below cites those
numbers. Scratch benchmarks live in `TEMP/` (gitignored).

## 1. Executive Summary

Replace request-driven shard fetching with a **provisioned serving pool**:

- Each node maintains a coordinator-assigned **model portfolio** in the
  background — download → verify → derived-fp16 cache → warm stage → **warm
  probe** — and is only schedulable for a model once it reports READY with
  *measured* throughput and RSS.
- The scheduler becomes **replica-first**: a request for model M goes to one
  node holding the whole model whenever an unsaturated one exists (fewest
  nodes); pipeline mode (k>1) is the fallback for models no single free node
  can host. Chat is **stateless**: the transcript rides in the prompt, no
  session affinity.
- Nodes gain a **SessionManager**: one warm model serves N concurrent streams
  via continuous batching; "busy" becomes a load vector, not a hard state.
- Scaling path is explicit: O(1) request-path indexing now, Postgres + cells
  (per-cell schedulers behind a directory) for horizontal scale, direct
  node↔node relay + content-addressed mirrors for WAN bytes.

Product goal this serves: a user prompt feels like ChatGPT (~2 s first token
warm, measured 1.8 s), while the backend stays distributed, fewest-nodes, and
adaptive to node count, type, saturation, and traffic.

## 2. Measured ground truth (reference host, 1B fp16-runtime)

| Phase (real node code paths) | Time | Notes |
|---|---|---|
| Cold: packed int4 → dequant → build | ~50 s | "first ever run on a machine" |
| Warm files: derived fp16 cache → build | ~29 s | dequant cache = 2× faster rebuild |
| Lukewarm: mmap-assign weights (no RAM copy) | ~18 s build; decode disk-bound when RAM-short | RSS 1.6 GB, evictable; **greedy parity confirmed** |
| Warm: stage resident | **1.8 s** | ~3.4 GB RSS for 1B |
| Prefill 6 tokens (warm) | ~1.8 s | grows with transcript; client "send history" toggle governs |
| Decode, RAM-sufficient vs paging (3.4 GB RSS on 2.6 GB free) | 0.26 vs ~0.5 s/tok | overcommit works but ~2× slower |

Consequences baked into the design: warm retention is the only route to
"instant"; the derived-fp16 cache is a persistent provisioner artifact; the
mmap tier is a policy option (free RAM ⇄ speed); capacity must use measured
RSS/tok-s, never theory.

## 3. Architecture decisions

### 3.1 Node-side Provisioner with explicit per-model state

**Choice:** each node runs a Provisioner reconciling a coordinator-pushed
desired model set through the state machine
`assigned → downloading → files_ready → warm → serving` (+ `evicting`,
`error`).

Rationale: makes "first run sets up model files before accepting jobs" a
first-class lifecycle instead of an accident of request timing; the
coordinator sees pool-wide readiness and can route around slow nodes.

Trade-off: node disk/RAM is planned by the coordinator (assignment policy),
not discovered lazily — needs the eviction policy (3.6) to stay safe.

Non-negotiables (user requirement 1): a node restart **never re-downloads**
what it already verified — startup reconciliation is: hash-verify local cache
(sidecars) → download only missing shards → **reuse the persistent derived
fp16 cache** (`<model_cache>/<model_id>/derived/`, keyed to manifest
`source_config_hash` + quantization spec; a re-export invalidates it) → warm
build if the warm budget allows → **warm probe** (user requirement 2): a
1–2-token generation that exercises all kernels, proves the build works, and
reports measured `toks_s` + RSS in the heartbeat. A node that fails its probe
never serves that model and reports `error` with the reason.

### 3.2 Replica-first scheduling (fewest nodes)

**Choice:** for model M the coordinator keeps a live index
`M → ready replicas ranked by (affinity-free load, RAM/VRAM headroom, measured
toks_s)`. A request goes to **one whole-model node** whenever one is
unsaturated; pipeline placement (k>1) is the fallback only when no single free
node fits the model (or all replicas + queue are saturated and scale-out
cannot keep up).

Rationale: k=1 removes relay hops, per-request placement math, and most
failure surface; small models land on small machines. The k=1 shape is
literally today's protocol with one stage, so JobAssign / ledger / fault
tolerance work unchanged.

Trade-off: replica serving needs the model's full weight set on one node;
pipeline mode remains in the codebase and the test matrix (parity tests stay
green) for models larger than any node — including overcommit.

### 3.3 Stateless chat (no session affinity)

**Choice:** every request carries its full transcript in the prompt (the
client already does this via its "send conversation history" setting). No
sticky routing, no KV affinity, no prefix cache in v1.

Rationale (user requirement 3): nodes turn on and off and users return days
later — a session bound to a node is a liability; stateless requests make
every replica interchangeable, which is what makes the scheduler simple and
the system scale horizontally. Follow-ups cost prefill over the transcript
(measured ~sublinear; the client's history toggle caps it).

Trade-off: long transcripts re-prefill each turn. A KV-prefix cache is a
future, optional layer (spec §19) — it must never be required for correctness.

### 3.4 SessionManager: continuous batching on one warm model

**Choice:** the node's one-generation-per-node exclusion (`node_lock`, shared
`StageModel.cache`, batch-dim-1 forwards — the three verified blockers) is
replaced by a SessionManager: per-session KV caches passed through
`_run_layers`, batch-dim-B forwards, token streams multiplexed by
`request_id`. Node "BUSY" becomes a load vector in metrics
(`active_sessions`, `queue_depth`, `toks_s_headroom`); the state machine keeps
only liveness semantics.

Rationale: without intra-node batching, a warm replica serves one user at a
time and the fewest-nodes goal collapses under the first concurrent user.
Batching is what makes replica-first viable under traffic and what makes
saturation-adaptive routing real.

Trade-off: batching changes numerics only via padding discipline (pad-to-multiple,
per-row logits) — parity tests must compare batched vs batch-1 exactly.

### 3.5 Adaptive pool policies (the controller loop)

**Choice:** a controller task (tick ~5 s) maintains desired state:

- `desired_replicas(M) = clamp(ceil(observed arrivals × target_wait /
  per-replica capacity), warm_floor(M), pool capacity)`; scale out after
  saturation persists 2 ticks, in after 10 min idle (hysteresis).
- Assignments balance by remaining disk + warm budget; **pipeline-only
  models** (no node can host whole, even overcommit) are flagged in the
  catalog and always planned as pipelines.
- Eviction under contention: LRU by (last_served, demand), warm stages before
  files, never below `warm_floor`; revocations are graceful (drain sessions).
- Admission: unsaturated replica → schedule; else queue with honest position +
  ETA (SSE status frames); queue wait estimate > pipeline TTFT budget →
  scale-out or pipeline fallback.

Rationale: this is the "adaptive to everything" requirement in one place, with
hysteresis preventing flapping.

Trade-off: controller tuning becomes an operational knob set (all
config-driven with safe defaults).

### 3.6 Warming UX and status frames

**Choice:** new optional SSE frames `{"type":"status","stage":"queued"|"provisioning"|"warming", "progress", "eta_s"}` (unknown to old clients = ignored, verified
client-side) plus a `queued` job state. The 30 s client idle watchdog is
already covered by ≤10 s heartbeats once the generator runs; pre-dispatch
phases emit their own keepalives.

### 3.7 Scalability path

**Choice:** ship the single coordinator with O(1) request-path indexing and
persisted desired state (new `assignments` table; today's plans are
memory-only and die on restart). Release-grade horizontal scale = **cells**:
a directory (model catalog + node↔cell by latency) + per-cell schedulers (the
current coordinator) + direct node↔node activation relay for pipeline mode
(the relay header is self-describing; JobAssign already carries the full stage
graph; coordinator relay stays as NAT fallback) + `mirror_urls` in manifests
so shards are CDN/object-storage addressable (sha256 already guards bytes).

Trade-off: cells and direct relay are deferred behind interfaces so nothing
blocks S22a–S22d.

## 4. Use-case flows (simple terms)

- **Fresh machine, first ever run:** register → assignment push → background
  download (chunked, P2P) → derived-fp16 cache → warm build → warm probe →
  READY. Client-visible only as a new node appearing in the dashboard; it
  never gates a user's request because other warm replicas serve meanwhile
  (on a one-node pool, the first request shows honest "warming" progress).
- **Node restart (cache present):** verify sidecars (no download, no
  dequant-cache rebuild) → warm build from derived cache → probe → serving in
  ~half the first-ever time (measured 29 s vs 50 s for 1B; warm-build tier is
  the target for "restart → ready").
- **Prompt on a warm pool:** request → replica index → best unsaturated
  replica → SESSION_START → batched decode → tokens stream. **~1.8 s first
  token measured.** No pipeline, no relay, no placement math.
- **Concurrent prompts:** second/third… session joins the warm model's batch.
  Saturation visible in metrics; queueing only when every replica is full;
  controller scales out before users notice.
- **Follow-up in chat:** full transcript in prompt, any replica — no affinity
  state, no stuck sessions, days-later continuity for free.
- **Model too big for any one node:** catalog flags pipeline-only; placement
  splits across fewest warm (or file-ready) nodes; activations relay
  node↔node (coordinator fallback); parity tests unchanged.
- **Node dies:** replicas rebalance; queued/warming work moves to other ready
  nodes; in-flight stream re-prefills from transcript on the next replica.
- **Model deleted / new model exported:** assignment revocation / push,
  graceful drain, disk LRU evicts files then warm stages.
- **Coordinator restart:** nodes keep provisioning/serving from persisted
  desired state; ledger/nodes/tokens already survive (verified).

## 5. Phased implementation plan (per stage: green gates, no behavior change until proven)

- **S22a — Provisioner + assignments (files tier).** Coordinator: assignments
  computation + persistence, `RegisterAck.assigned_shards` filled, new
  `ASSIGNMENT` envelope + `MODEL_STATUS` report; node: Provisioner task in
  `agent.py` (background, bandwidth-capped via existing download knobs),
  derived-fp16 cache in `ModelStoreClient`/`storage_int4.py`. Scheduling
  unchanged (still request-path fetch, now a cache hit). Files tier = the
  29 s rebuild path.
- **S22b — Warm tier + ready-gating.** Warm budget settings
  (`DAIN_WARM_BUDGET_GB`, `DAIN_WARM_MODELS`), StageModel retention with LRU,
  warm probe (measured toks_s + RSS into heartbeats), scheduler reads
  readiness; warming requests admitted with status frames. mmap tier as a
  per-node policy (`DAIN_WARM_MODE=mmap|resident|files`).
- **S22c — Replica-first placement + stateless queueing.** Placement policy
  rewrite (k=1 preferred, pipeline fallback), queue with ETA + status frames,
  pipeline-only model flag; spec §12 amendment recorded.
- **S22d — SessionManager + continuous batching.** Node-side sessions, batched
  forwards, load-vector metrics, saturation routing; batch parity tests.
- **S22e — WAN bytes.** `mirror_urls` + direct node↔node relay (fallback to
  coordinator relay); CDN-friendly immutable shard URLs.
- **S22f — Cells + Postgres.** Directory + per-cell schedulers; only when
  scale demands it.

## 6. Implementation order

| Step | Stage | Key files | Complexity |
|---|---|---|---|
| 1 | S22a | `partition.py` (assignments), `store.py` (assignments table), `schemas.py` (ASSIGNMENT/MODEL_STATUS), `Node/agent.py` + new `provisioner.py`, `llm.py` (derived cache) | Medium |
| 2 | S22b | `Node/jobs.py` (stage retention), `provisioner.py` (probe), `nodes.py`/`schemas.py` (status+RSS), `partition.py` (readiness gate) | Medium |
| 3 | S22c | `partition.py` (replica-first), `api.py` (queue/status frames), `jobs.py` (queued state), `Client` status UI | Medium-High |
| 4 | S22d | `Node/llm.py` (batch forwards, per-session cache), new `Node/sessions.py`, metrics | High |
| 5 | S22e | `schemas.py` (mirror_urls), `peer_server.py`/`jobs.py` (direct relay), `model_export.py` (publish) | Medium |
| 6 | S22f | `app.py` split, Postgres, directory service | High |

## 7. Key risks & mitigations

1. **Batching changes numerics** → pad-to-multiple + exact batch-vs-batch-1
   parity tests gated on S22d; single-session path stays bit-identical.
2. **Warm budget OOM on small hosts** → budget is enforced pre-build (probe
   RSS), derived cache + mmap tiers degrade gracefully; overcommit flag
   unchanged for volunteer nodes.
3. **Assignment churn / thundering herd on join** → hysteresis + per-node
   download caps (existing parallel-chunk knobs) + staggered provisioning.
4. **Coordinator restart loses in-flight plans** → desired state persisted
   (S22a), nodes are authoritative over local readiness.
5. **Spec drift** → every stage records its §-amendment in the commit + this
   doc's §8; parity and chaos suites must stay green at every gate.
6. **Memory-allocator retention** (measured: RSS not fully returned after
   teardown) → capacity accounting uses *current* RSS from heartbeats, not
   deltas; node restart reclaims if needed.

## 8. Spec amendments this plan makes (to be noted in stage commits)

- §12: K = clamp(ceil(L/target), 8, 16) replaced by **replica-first** policy
  (k=1 preferred; pipeline fallback; deployment min/max stages still honored).
- §3 non-goals: "no autoscaling" is amended — the controller (3.5) is in scope.
- §10/§16: coordinator relay becomes the *fallback* data path; direct
  node↔node relay ships when S22e lands (spec Q1 resolved).

## 9. Non-goals

Speculative prefetch of models with no assignment; gossip/WAN P2P for shard
bytes (CDN mirrors instead); KV-prefix caching (future, optional); cross-model
memory sharing; TEEs; training.
