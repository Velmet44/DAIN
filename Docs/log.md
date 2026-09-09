# DAIN — Development Log

Append-only work log, newest entry last. One entry per work session / stage.
Format: what was done, decisions made, deviations from Docs/stages.md, gate results.

---

## 2026-09-08 — Specification & planning (pre-S0)

- Wrote `Docs/spec.md` v0.1 (skeletal, 20 sections) per project brief.
- Ran feasibility simulations to pressure-test the spec instead of asserting it:
  - Scoring/ranking over a synthetic 20-node heterogeneous pool → clean separation
    (0.14–0.36), top-K admits all high-end nodes. Formula kept.
  - Stage-sizing comparison: score-proportional layer split gave **0%** throughput gain
    over equal split; throughput-proportional split gave **~43% over score-proportional**
    and **~2.8× over equal**. §8.1 rewritten to size stages by measured throughput.
- v0.2 additions from a second feasibility pass (real model dims, memory bandwidth):
  - §9 latency envelope: single-stream decode ≈ Σ stage_compute + hops×RTT (7B on 8×3060-class
    ≈ 112 ms/tok); 7B fp16 (15.2 GB) cannot fit a 12 GB card; 32B-class fits no consumer GPU
    → sharding is an *enabler*, not just an optimization. Batch-8 pipelined ceilings ~1.7k tok/s.
  - §10 NAT model: nodes connect outbound-only (no inbound ports).
  - §18 reference models: TinyLlama-1.1B / Qwen2.5-7B / Qwen3-32B / OLMoE-1B-7B.
  - §21 technology stack locked; §20 open questions 11–12 added.

## 2026-09-08 — S0: Bootstrap & toolchain ✅ (commit 2cd11ad)

- uv workspace with members `Common`/`Coordinator`/`Node`/`Sim` → packages
  `dain-common`/`dain-coordinator`/`dain-node`/`dain-sim`; CPython 3.12 pinned via
  `.python-version`; root `package = false`, `link-mode = "copy"` (cross-filesystem cache).
- Root `dev` dependency-group installs all four members + pytest/ruff/httpx, so a plain
  `uv sync` yields a complete development environment (first sync did not — fixed).
- FastAPI coordinator skeleton with `GET /healthz`; `python -m dain_coordinator` entry point.
- Live-socket smoke: server spawned from the venv, `{"status":"ok"}` received, clean shutdown.
- Gates: `uv run ruff check .` clean · `uv run pytest` 2 passed.
- Decisions: Python 3.12 (system has 3.14; 3.12 has the broadest PyTorch wheel support);
  ruff rules E/F/W/I/UP/B; third-party starlette TestClient deprecation warnings noted,
  not actionable by us.
- Deviations: curl checkpoint satisfied via scripted smoke (same assertion, venv python).

## 2026-09-08 — S1: Protocol & shared schemas ✅ (commit 2983e9a)

- `schemas.py`: `Envelope{v, type, ts, payload}` — protocol-version-validated,
  `extra="forbid"`, frozen; all 9 message families with payload models
  (`REGISTER…SHARD_MANIFEST`); `NodeState`/`JobState`/`TaskOutcome` as `StrEnum`;
  `CapabilityManifest` plausibility validators (free ≤ total, TFLOPS bounds);
  `JobAssign` graph validators (contiguous stages, prompt on entry stage only,
  ascending layers); `LedgerEvent.idempotency_key = (job_id, stage_idx, attempt)`;
  `parse_payload()` dispatch table `PAYLOAD_TYPES`.
- `config.py`: `ProtocolConfig` (5 s heartbeat × 3 missed = 15 s offline),
  `ScoringWeights` (spec §7 defaults; network 0.15 split into net_bw 0.10 + net_lat 0.05,
  reliability 0.15 into uptime 0.10 + failure_rate 0.05; sum-to-1 validated),
  `ScoringRefs`, `ScoringConfig`.
- `scoring.py`: `norm`/`norm_inv` saturating normalizations, `ewma`,
  `is_feasible` (hard VRAM gate; CPU-only nodes feasible only for 0-VRAM partitions),
  `score_node` — **weight redistribution for structurally missing components**
  (no power telemetry → no energy score; CPU-only → no mem_bw score), soft penalty
  (×0.5) when uptime < 0.8. Returns per-component breakdown for observability.
- `accounting.py`: `MODELS` registry for the 4 reference models;
  `flops_per_token` = 2·params_active; `flops_for_stages` = uniform per-layer
  approximation; `credit()` with outcome factors SUCCESS 1.0 / RETRIED_AWAY 0.2 / FAILED 0.0,
  utilization falls back GPU → CPU → 1.0.
- Tests: 42 new (schemas 20, scoring 13, accounting 9). Golden scores pinned from a
  one-off reference run: n4090 0.6990619 > n3060 0.5010521 > ncpu 0.2884220 > nflakey
  0.2210550. Suite total **44 passed**, ruff clean.
- Bring-up lessons (fixture drift caught by the goldens, exactly their purpose):
  default GPU fixture omitted `mem_bw_gbs` (→ redistribution changed the score), and the
  CPU-only fixture used different net values than the golden run. Rule going forward:
  golden fixtures must be *literally identical* to the golden computation.
- Decisions/deviations: `StrEnum` per ruff UP042 (py312); `zip(strict=False)` for the
  adjacent-pairs layer-ordering check (B905 satisfied explicitly, semantics intended);
  weight sub-split documented in `ScoringWeights` docstring; `.gitattributes` added for
  LF normalization after CRLF warning noise on Windows.
- Next: **S2 — Coordinator core** (registry, auth, heartbeats, state machine) per
  `Docs/stages.md`.

## 2026-09-08 — Repository restructure: per-project isolation (user request)

- Dropped the root uv workspace: root `pyproject.toml`, `uv.lock`, `.venv`,
  `.python-version`, and root caches removed. The root now holds only shared docs/config:
  `README.md`, `.gitignore`, `.gitattributes`, `Docs/`, `Deploy/`, `Builds/` (gitignored),
  and the four project directories (+ `Client/` from S9).
- Each Python project is now **standalone**: own `pyproject.toml`, `uv.lock`, `.venv`,
  `.python-version` (3.12), ruff + pytest config, and its own `tests/` directory.
  Tests relocated: schema/scoring/accounting/smoke suites → `Common/tests/`, healthz →
  `Coordinator/tests/`, placeholders in `Node/tests/` and `Sim/tests/`; root `Tests/`
  dissolved.
- `dain-common` is consumed via editable path dependency
  (`{ path = "../Common", editable = true }`) in Coordinator/Node/Sim — single protocol
  source of truth, no root workspace.
- Conventions locked and documented: gates run **inside the project directory**;
  cross-component integration tests live in `Sim/tests/`; Sim entry points are modules
  (`python -m dain_sim.cluster|chaos|benchmarks`).
- `Docs/stages.md` updated everywhere (§2 DoD, §3 architecture, S0–S11 command paths);
  `README.md` rewritten; spec §21 tooling row amended.
- Rationale: user asked for a clean root and per-project isolation (one venv each for
  node and coordinator). This also matches the deployment reality — coordinator ships to
  a VPS, nodes ship to provider machines, each syncing independently.
- Accepted trade-off: dependency resolution duplicated across four lockfiles (all on
  3.12; uv's shared wheel cache makes this cheap), and a protocol change requires
  re-running `uv sync` in consumer projects.
- Gates re-run in all four projects: Common **43 passed**, Coordinator **1 passed**,
  Node **1 passed**, Sim **1 passed**; ruff clean everywhere.
- Next: **S2 — Coordinator core**.

## 2026-09-08 — S2: Coordinator core ✅

- `dain_common` (S1 pkg): `RegisterAck` gained `node_token` — the per-node credential
  issued at first registration and used for WS auth (spec §11 amended to match).
- `settings.py`: `CoordinatorSettings` (env-driven; heartbeat 5 s × 3 missed = 15 s
  offline; join token; min_score 0.05; overload/thermal thresholds; `uptime_alpha`).
- `store.py`: `SQLiteRegistry` — single connection + lock, WAL, portable SQL schema
  (`nodes` + `state_history`); JSON columns for manifest/metrics/score components;
  only the WAL pragma is SQLite-specific. NodeRow ↔ row mapping is the serialization
  boundary.
- `nodes.py`: `NodeService` — registration (join token → per-node token via
  `secrets.token_hex`, comparison via `secrets.compare_digest`), heartbeat handling
  (server-clock timestamps, seq-gap warnings, uptime EWMA), score recomputation on
  every report, and the **spec §6 state machine enforced by an explicit transition
  table** (`ALLOWED_TRANSITIONS`, incl. from-None registration edge). DEGRADED on:
  score < min_score / thermal > 90 °C / 3 consecutive overload strikes (>97 % util);
  automatic recovery on a clean report; OFFLINE on heartbeat timeout; re-registration
  and WS reconnect paths both restore ONLINE with history preserved. `mark_busy` /
  `mark_online` exposed for the S6 scheduler. WS message dispatch (malformed → 4400,
  identity mismatch → 4400, unauthenticated → 4401); later-stage message families are
  logged and deferred (JOB_STATUS S4+, SHARD_MANIFEST S5, LEDGER_EVENT S8).
- `monitor.py`: `HeartbeatMonitor` background task (tick 0.5 s, exception-proof);
  `api.py`: REST `POST /node/register`, `POST /node/deregister`, WS `/node/ws`,
  admin `GET /admin/nodes[?state=]` and `GET /admin/nodes/{id}` with full history.
  `logging_setup.py`: JSON formatter with node_id/job_id correlation (text default).
- Tests (Coordinator, compressed timings ≈ 50×): 20 passed — registration/auth
  lifecycle incl. the gate sequence register→ONLINE→silence→OFFLINE→re-register→ONLINE,
  thermal/overload degrade + recovery, deregister auth (403/404), state-machine unit
  tests against the spec table, restart persistence (records + tokens + history +
  monitor working across processes), wire-validity of the REST body.
- Sim: `tests/helpers.py` (in-process uvicorn cluster + fake WS nodes) and the
  **20-node × 60 s real-socket stability checkpoint — passed (64.9 s)**: 20/20 ONLINE,
  exactly one history entry per node (zero spurious flips), heartbeat counts healthy.
  Sim dev deps += dain-coordinator (editable path), httpx, websockets.
- Bring-up lessons: starlette WS close codes surface on `WebSocketDisconnect.code`
  (assert the attribute, not str()); close-before-accept raises at connect, close
  after accept must be *received* to be observed; dataclasses use `dataclasses.replace`
  (no `model_copy`); a helper named `test_settings` gets collected by pytest — renamed
  to `make_settings`.
- Decisions: sync sqlite3 + lock accepted at MVP scale (≤ 100 nodes, sub-ms ops) with
  async/Postgres deferred to the HA path; admin endpoints unauthenticated until S9
  hardening; DEGRADED evaluation is immediate for score/thermal, strike-counted (3)
  for overload.
- Next: **S3 — Node agent skeleton + local cluster sim** per `Docs/stages.md`.

## 2026-09-08 — S3: Node agent skeleton + local cluster sim ✅

- `logging_setup` moved to `dain_common` (both agents share the JSON-correlation
  formatter); Coordinator imports updated.
- `dain_node` (deps += httpx, psutil, websockets):
  - `settings.py`: `NodeSettings` (DAIN_COORD_URL is a **base** URL — agent derives
    `/node/ws` and REST endpoints; state path; reconnect bounds; stub net figures).
  - `identity.py`: persistent node identity (`node_id` + issued `node_token`,
    atomic tmp+replace writes, corrupt-file → fresh join). Spec §11: re-registration
    reuses node_id; history survives because it is keyed by node_id.
  - `capabilities.py`: psutil CPU/RAM always; torch/CUDA GPU (vram via `mem_get_info`,
    TFLOPS/mem-BW from a name table — *claimed*, unverified per spec §7) else CPU-only.
  - `executor.py`: `Executor` base (no-op lifecycle hooks) + deterministic
    `FakeExecutor` (sha1-of-prompt token stream); S4 wires the real HF executor.
  - `agent.py`: `NodeAgent` — REST registration each session (join token first,
    per-node token after; rejected token → automatic re-join under same node_id),
    heartbeat loop (ack'ed interval, seq, psutil/CUDA metrics), receive loop raced
    against the stop event (`asyncio.wait FIRST_COMPLETED`) so a silent server can
    never block shutdown, capped exponential backoff (0.5→8 s), `StopGuard` for
    SIGINT/SIGTERM/SIGBREAK → deregister → exit 0. Incoming JOB_ASSIGN is logged and
    deferred to S4.
- `dain_sim.server`: in-process uvicorn coordinator helper (moved out of test helpers).
- `dain_sim.cluster`: one-command cluster — spawns coordinator + N real
  `python -m dain_node` subprocesses (per-node state dirs, varied reported bandwidth),
  polls the registry for `--duration`, graceful-stops all agents, prints a per-node
  health summary (final state / transitions / heartbeats / score / shutdown reason),
  exit 0 iff: all heartbeating, zero spurious OFFLINE flips, all deregistered.
- Checkpoints:
  - `uv run python -m dain_sim.cluster --nodes 12 --duration 60` → **HEALTHY, exit 0**:
    12/12 online throughout, 12 heartbeats each, exactly 2 transitions (registered +
    deregistered), differentiated scores (0.320–0.344), all `deregistered`.
  - `tests/test_reconnect.py` → passed: hard-kill (no deregister) → OFFLINE in ~3 s
    (well within the 15 s budget) → restart with persisted identity → same node_id
    re-registered ONLINE → graceful stop → `deregistered`.
- Gates: Common 43, Coordinator 20, Node 11, Sim 3 — all passed, ruff clean.
- Bring-up lessons: the agent's receive loop blocked `async for` on a silent server,
  which postponed deregister until `heartbeat_timeout` won the race — fixed by racing
  receive vs stop-event and cancelling pending tasks; `_summarize()` call-signature
  bug only surfaced via the real CLI run (exit-code assertions in bash pipelines are
  polluted by grep — capture with `$?` directly after `;`, not inside a pipe);
  `Executor` kept a plain base class (no ABC) because its hooks have defaults.
- Decisions: seq resets to 0 across agent restarts (server logs a gap warning;
  persistent counters deferred until needed); net BW/latency are configured stubs
  until S10; admin REST stays unauthenticated until S9.
- Next: **S4 — Single-node inference path** (real model, streaming).

## 2026-09-08 — S4: Single-node inference path ✅

- Common: `TOKEN_BATCH` message family; `ModelManifest` (served by the model store);
  `ActivationRelayHeader.role` (`hidden` | `sampled_token`); `StageAssignment.expert_ids`
  (S11 skeleton). Spec §10/§18 amended.
- Node: torch 2.14.0+cpu + transformers pinned **exactly 4.46.3** (the stage runner uses
  Llama decoder-layer internals — parity tests are the correctness gate). Modules:
  `shard_export.py` (seeded dev model `dain-tiny-16L` → per-layer safetensors shards +
  manifest), `byte_tokenizer.py` (vocab 256, byte 0 = EOS), `model_store` IO in
  `dain_common.model_store`, `llm.py` (`ModelStoreClient` with sha256-verified shard
  download/caching; `StageModel` — explicit weight mapping, only assigned layers
  instantiated; own 4D causal-mask builder — no HF internals; greedy/temperature
  sampling), `jobs.py` (`JobHandler`: JOB_ASSIGN → generation → TOKEN_BATCH).
  **Protocol invariant: only the sampling stage emits TOKEN_BATCH**; entry stage owns
  the loop and reports lifecycle via JOB_STATUS.
- Coordinator: `connections.py` (per-node WS registry, send locks, disconnect hook),
  `jobs.py` (`JobTracker` state machine QUEUED→DISPATCHED→RUNNING→STREAMING→
  COMPLETED/FAILED with idempotent terminal writes + per-job SSE queues;
  `ActivationRelay` routes hidden/sampled_token chunks between stage nodes),
  `/v1/completions` (SSE + non-streaming JSON), `/v1/models`, `/v1/jobs/{id}`,
  `/model/manifest|shard` (node-token authenticated, path-traversal guarded),
  API-key auth on /v1.
- Tests: Node 15 (shard parity vs HF reference = **0.000000 max diff**; greedy
  determinism; 4-stage in-process pipeline parity < 1e-3 + identical greedy tokens;
  tokenizer/identity/executor), Coordinator 20, Sim e2e checkpoint: coordinator +
  1 real agent subprocess → ≥20 streamed SSE tokens, valid framing, job COMPLETED,
  auth 401, non-streaming parity. All gates green.
- Bring-up lessons (the big one): `sample_token` indexed `logits[0, -1]` assuming
  [B,T,V], but the runner emits [B,V] — argmax over a scalar silently returned
  0, which the eos check swallowed as an immediate EOS. Diagnosed by comparing
  node logits tails against the HF reference (identical!) → the bug was in shape
  handling, not the model. Also: registration now carries `model_store_url`
  (derived from request base); node setup failures are reported to the client as
  error TOKEN_BATCHes instead of dying silently; orphaned agent processes from
  crashed runs can impersonate a pool (kill strays / use unique join tokens in
  shared environments).
- Decisions: dev model is hermetic (seeded tiny Llama) — real TinyLlama/Qwen use the
  same code path when weights are present; transformers pinned exactly; single-node
  = a stage covering all layers (one code path for S4/S5).
- Next: **S5 — Distributed pipeline**.

## 2026-09-09 — S5: Distributed pipeline ✅

- `partition.py`: `build_placement` — K = clamp(ceil(L/target), 1, max_k, |pool|);
  feasible = ONLINE + live WS connection + per-stage capacity share; ranked by score;
  **stage sizes ∝ throughput proxy** (claimed TFLOPS, CPU cores as relative proxy)
  via largest-remainder integer split with min 1 layer (spec §8.1 — the §12 formal
  scheduler lands in S6 on top of this). Largest-remainder handles the min-1 clamp
  overshoot explicitly.
- Completions: multi-stage dispatch — one JOB_ASSIGN per stage (`my_stage_idx` per
  node, prompt only on entry); BUSY marking for exclusive per-node execution
  (MVP has no intra-node batching; deadlock-free because BUSY nodes are excluded
  from new placements); `_release` returns nodes BUSY→ONLINE in a finally (both
  streaming and non-streaming paths). Stage latencies exposed in `/v1/jobs/{id}`.
- Node: per-node `node_lock` serializing generation/steps (defense in depth),
  grad disabled globally (inference-only — fixes `numpy()` on requires_grad),
  **early-activation buffer** for activations arriving before the stage runtime
  registers (race: entry prefills immediately), ack-per-step via JOB_STATUS.
- Checkpoint: `test_pipeline_parity.py` — 1-agent reference completion ≡ 4-agent
  distributed completion (16 layers → 4 stages × 4 layers) ≡ in-process HF greedy
  reference; job view shows 4 stages with layer ranges [0,4,8,12] and latencies.
  Full Sim suite green 2× consecutively (plus early flakes diagnosed below).
- Bring-up lessons: activations for not-yet-built stage runtimes must be BUFFERED,
  never dropped (entry prefills immediately); a node is not dispatchable between
  REST registration and WS attach — placement now requires live connections and
  /admin/nodes exposes `connected`; heredoc-based python patches in Git Bash are
  fragile (silent non-application) — verify patches by asserting the new content;
  the machine has 4 cores — full-suite CPU contention produces timing flakes, so
  tests wait on *state*, with generous bounds, never on sleep().
- Next: **S6 — Scoring-driven scheduling**.

## 2026-09-09 — S6: Scoring-driven scheduling, top-K, backups, queueing ✅

- `partition.py` (S6 formalization of the S5 sizing): `plan_placement` is a **pure
  module** over registry snapshots (no I/O — unit-testable) implementing spec §12 in
  order: feasibility (ONLINE + live WS connection) → score-ranked with deterministic
  tie-breaks (score desc, throughput desc, node_id) → **top-K** with
  `K = clamp(ceil(L / layers_per_node_target), min_k, max_k)` → capacity filter
  (per-stage VRAM/RAM share) → throughput-proportional sizing via largest-remainder
  integer split (min 1 layer) → **warm backups** (next-ranked nodes up to
  `backup_count`). Returns None on infeasible / below `min_nodes` (caller → HTTP 429
  / degraded mode); fewer-than-desired-K plans are flagged `degraded`.
- `PlacementRecorder` + `PlacementEvent`: ring buffer of recompute events
  (request/join/leave/degraded/recovered) surfaced via the placement log (spec §12
  observability).
- Admission/backpressure (`api.py`): `plan_placement` is called per request; an
  infeasible pool or a full queue returns HTTP **429 with `Retry-After: 5`**;
  bounded active jobs (`queue_limit`) and per-key concurrency (`max_concurrent_per_key`)
  enforce backpressure. Failing dispatch (busy/unreachable stage node) releases nodes
  and returns 503.
- Placement recompute on pool changes (`app.py`): `recompute_pool_events` recomputes
  the plan for every model in the store on node `leave` (WS disconnect) and on
  `degraded` entry / `recovered` transitions (via `service.on_pool_change`); a store
  path failure never breaks the triggering path.
- Common: schema message-family count test updated 10 → 11 (the S5 `SHARD_MANIFEST`
  addition had drifted an old hard-coded count — the `set(PAYLOAD_TYPES) ==
  set(MessageType)` assertion was already the correctness gate).
- Checkpoints:
  - `Coordinator/tests/test_scheduler.py` (10) — K clamped by target+pool; selection
    ranks by score; sizing ∝ throughput (fast node > slow, min 1 layer); low-score &
    capacity-excluded & DEGRADED nodes excluded; None on no-online pool; backups
    disjoint from stage nodes; small model → single stage / fine-grained spread.
  - `Sim/tests/test_admission.py` (1) — 50 concurrent requests vs 2-node pool with
    `queue_limit=3`, `max_concurrent_per_key=2`: every request is 200 or 429, some of
    each, no crash/deadlock, pool recovers to all-ONLINE after.
  - `Sim/tests/test_placement_recompute.py` (1) — job1 on 4/5 nodes (backup = the
    5th); kill the sampling-stage node → it goes OFFLINE; job2's placement excludes
    the dead node and still covers every layer contiguously.
- Gates: Common 43, Coordinator 30, Node 15, Sim 7 — all green, ruff clean everywhere.
- Bring-up lesson (documented 4-core contention): `test_placement_recompute`'s first
  **non-streaming** job ran 4 stages of real CPU inference on a 4-core host, and under
  full-suite contention exceeded the default 60 s `job_timeout_s`, surfacing as an
  httpx `ReadTimeout`. Fixed by aligning with the S5 parity test's `job_timeout_s=90.0`
  (S5 rule: tests wait on state with generous bounds, never on sleep). The scheduler
  logic itself was correct — purely a tight inference-bound timeout.
- Decisions: `min_stages`/`max_stages` are deployment configuration (production
  reference `DAIN_MIN_STAGES=8` in `Deploy/.env.example`; dev/sim pools use a lower
  floor); BUSY nodes are excluded from new placements so the no-intra-node-batching
  MVP cannot deadlock (established in S5); backups are designated at placement time
  and consumed for reassignment in S7.
- Next: **S7 — Fault tolerance & degraded mode** (stage watchdog, retry with attempt
  counter, backup reassignment, degraded re-partition, `dain_sim.chaos`).
