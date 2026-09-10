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

## 2026-09-09 — S7: Fault tolerance & degraded mode

- **`Coordinator/dain_coordinator/faults.py`** — new `FaultManager` (spec §13):
  - `handle_node_lost(node_id)`: fires on WS disconnect and on heartbeat-timeout OFFLINE
    eviction (wired via `connections.on_disconnect` + new `service.on_node_lost` hook in
    `nodes.py`). Re-assigns **every in-flight stage** hosted on the lost node.
  - `tick()`: stage watchdog — a stage with no progress for `4 × p99(stage latency)`
    (clamped to `[stage_deadline_min_s, stage_deadline_max_s]`) is reassigned; a whole-job
    stall (no token at all since dispatch) restarts from the prompt, bounded by
    `max_job_restarts`.
  - `_reassign`: picks a warm backup (ONLINE + live WS + not already on this job), re-sends
    `JOB_ASSIGN` for the reassigned stage, marks `RETRYING`, increments the per-stage
    `attempt` (max `max_stage_attempts`), and fires `STAGE_RETRY` at the upstream so the
    completed prefix is replayed instead of restarting from the prompt.
  - Idempotency contract: `(job_id, stage_idx, attempt)` stays unique across retries —
    the S8 ledger uniqueness key.
- Fixes uncovered by S7 bring-up:
  - **`dataclasses.replace` vs `StageAssignment`**: `StageAssignment` is a pydantic model,
    so replacement now uses `model_copy(update={"node_id": replacement})`.
  - **`recompute_pool` latent bug**: `recompute_pool_events` in `app.py` was calling
    `recompute_pool` without the required `min_nodes=` keyword (S6 latent) — now explicit.
  - **Non-entry replacement stall (root cause of the S7 saga)**: on reassignment,
    `on_job_assign` queued buffered `_early` activations into `rt.inbound`, but only the
    ENTRY stage consumes `rt.inbound`; a non-entry (e.g. sampling-stage) replacement never
    processed them → hung until the 90 s `job_timeout_s`. Fix: `_bootstrap_stage(rt,
    buffered)` builds the stage then feeds each buffered activation through `_run_step`.
  - **Replay-buffer clobbering**: `_send_activation` now buffers only `role=="hidden"`
    activations; upstream `sampled_token` relays were overwriting the replay buffer with
    the token result, so `on_stage_retry` had nothing to replay.
  - `relay.route` defers with a warning (`relay_target_unreachable`) instead of failing the
    job when a stage node has just died — the watchdog/reassign path owns recovery.
- **`Sim/dain_sim/chaos.py`** — rewritten as a harness: `run_chaos(...)` returns a report;
  `_scenario_complete` streams a job then kills the sampling-stage node at the first token
  and asserts completion + unique `_attempt` keys; `_scenario_degraded` serves with 3 of 8
  nodes; `_scenario_reject` expects clean 429/503 below `min_nodes`. `main()` CLI:
  `--nodes N --kill-at T:nodeID --expect complete|degraded|reject --tokens`. Node
  subprocesses log to per-node `<node_id>.log` with `PYTHONUNBUFFERED=1` (nodes' own log
  lines are otherwise invisible in captured test output).
- **`Sim/tests/test_chaos_matrix.py`** (4 new checkpoint tests): kill 1 of 6 mid-job →
  completes via backup (single stage-3 retry, verified: `retries={'3': 1}`); kill 3 of 8 →
  degraded serving continues; kill below `min_nodes` → clean rejection; no duplicate ledger
  `attempt` keys across the run.
- Gates: Common 43, Coordinator 30, Node 15, Sim 11 (7 + 4 new) — all green, ruff clean.
  First full-suite Sim run flaked `test_pipeline_parity`/`test_single_node_e2e` on
  4-core contention (node WS attach racing the first POST → no connected node → 429);
  both pass standalone and on a clean full-suite rerun. No code change needed — the
  per-test `wait_connected` + the first-request path are already state-based.
- Decisions: reassignment is single-stage (only the failed stage onward is recomputed;
  the completed prefix is replayed from the upstream's buffer — never a full restart unless
  the KV-holding/entry node is lost); no coordinator HA / cross-node KV migration / consensus
  (spec §13 explicitly out of scope for S7).
- Next: **S8 — Accounting ledger** (`(job_id, stage_idx, attempt)` keys, token/score
  accounting, settle on completion).

## 2026-09-09 — S8: Accounting ledger (spec §15)

- Implemented `Coordinator/dain_coordinator/ledger.py`: coordinator-side `Ledger` emits one
  append-only event per `(job_id, stage_idx, attempt)` when a job reaches a terminal state
  (wired via `JobTracker.on_terminal`, guarded once per job). Store adds a `ledger` table with
  `UNIQUE(job_id, stage_idx, attempt)` + `INSERT OR IGNORE` → retry/replay storms never
  double-count.
- Trust model (§15/§16): `flops_est` computed from the Common FLOP tables
  (`flops_for_stages`, no node-claimed numbers); `tokens_in` from `estimate_prompt_tokens`
  (≈4 chars/token), `tokens_out` from the coordinator's own stream; `energy_kwh_est` left null
  until node power telemetry lands. Outcomes: SUCCESS (final attempt of a finished stage, ×1.0),
  RETRIED_AWAY (abandoned attempts, ×0.2, attributed to the node that actually ran them via
  `attempt_nodes`), FAILED (unfinished stages, ×0.0). Credit weights configurable via
  `settings.accounting_weights` (defaults match Common goldens).
- Verification (§15): each stage's wall time cross-checked vs the p99 latency window; stages
  >15 s and >10× p99 are flagged (`verified=false`, note) and fire a score-penalty hook
  (`apply_verification_penalty` pins uptime below `min_uptime_soft × 0.75`). Thresholds loose
  because 4-core contention legitimately stretches stages.
- API: `GET /ledger/node/{id}`, `GET /ledger/summary` (per-node credit buckets + flagged
  counts), `POST /ledger/export` (CSV/JSON), all behind the existing API key.
- Tests: `Coordinator/tests/test_ledger.py` (6 — emission, dedupe, FAILED×0.0, verification
  flags, summary/export/CSV, auth) and `Sim/tests/test_ledger_reconcile.py` (reconciles the
  ledger over HTTP after a real retry storm in the S7 chaos scenario). Added `dain-tiny-16L`
  ModelSpec + `estimate_prompt_tokens` + goldens in `Common/tests/test_accounting.py`.
- Gates: Common 44, Coordinator 36 (30 + 6), Node 15, Sim 12 (11 + 1) — all green, ruff clean
  across all four packages. One contended full-suite Sim flake of
  `test_single_node_streaming_e2e` (node WS attach racing the first POST → 429) re-ran green;
  parity test also verified solo.
- Next: **S9 — Web client + deployment** (needs human-provided hosting credentials §4).

### 2026-09-09 — S9 hardening + CI (session 3)

**Goal:** Complete S9 hardening (admin auth, rate limiting, CORS, Netlify→GitHub Pages
migration) and add Python CI (step 2).

**S9 hardening:**
- Added `require_admin` dependency (X-Admin-Key header or Bearer token, timing-safe
  comparison via `secrets.compare_digest`) to `admin_router` in `api.py:90`.
- Added `admin_api_key` field to `CoordinatorSettings` (env: `DAIN_ADMIN_API_KEY`, default
  `dain-dev-admin-key`). Updated `Deploy/.env.example` with the new variable.
- Changed `rate_limit_per_min` default from `0` (disabled) to `60` (enabled).
- Updated all test fixtures (`Coordinator/tests/conftest.py`, `Sim/tests/helpers.py`,
  `Sim/dain_sim/cluster.py`, `Sim/dain_sim/chaos.py`) to set `admin_api_key=ADMIN_KEY`
  in `CoordinatorSettings` and pass `ADMIN_HEADERS = {"X-Admin-Key": ADMIN_KEY}` to
  all admin API calls.
- Updated `Sim/tests/test_20_nodes.py`, `test_reconnect.py`, `test_single_node_e2e.py`,
  `test_pipeline_parity.py`, `test_admission.py`, `test_placement_recompute.py` with
  admin auth headers on all `/admin/*` calls.

**GitHub Pages consolidation (replacing Netlify):**
- Deprecated `netlify.toml` (file retained with deprecation notice, safe to delete).
- Updated `Client/vite.config.ts`, `Client/.env.example` to remove Netlify references.
- Updated `.github/workflows/deploy.yml` env section (placeholder URL, correct comments).
- Updated `Docs/stages.md` S9 section: Netlify → GitHub Pages throughout (task list,
  checkpoints, status table).
- Updated `Docs/spec.md` §21 reference stack: Netlify → GitHub Pages.
- Updated `Deploy/.env.example` with CORS, rate limit, and admin key documentation.

**CI for Python tests (step 2):**
- Created `.github/workflows/ci.yml`: 4 parallel jobs (Common, Coordinator, Node, Sim)
  each running `uv sync`, `uv run pytest -q`, `uv run ruff check .` on push/PR to main.
- Uses `astral-sh/setup-uv@v5` with caching for fast runs.

**Gates:** Common 46, Coordinator 36 — all green, ruff clean across all four packages.
All 19 lint issues from the admin-auth additions (line length, import ordering) resolved.
Sim/Node integration tests not run in this session (require model export + torch); they
are gated by the CI workflow on push.

### 2026-09-09 — S9 hardening + same-PC dev launch (session 4)

**Local-run bug fixes (root cause of "node stuck" / "chat shows no model"):**
- Coordinator bound port found busy (previous orphaned run) → coordinator now
  auto-picks a free port near 8000 (`_find_open_port` in `__main__.py`).
- Node got `ws://localhost:8000 ` (trailing space from the old .bat env lines) →
  `NodeSettings.ws_base_url` now strips whitespace; the launcher no longer leaks
  trailing spaces (`set "VAR=val"` quoting).
- Node `warmup_manifest_failed ... err=model store base URL not set yet` → warmup
  ran before registration (store base URL only arrives in `RegisterAck`). Warmup
  moved into the reconnect loop, once after first successful register
  (`NodeAgent.run` / `run_agent` in `agent.py`).

**Script layout — one self-contained launcher (project folder is relocatable):**
- `bootstrap.ps1`/`bootstrap.sh` moved to `Scripts/` (repo-root anchored via
  `Split-Path`/`..`).
- Removed `start.bat`/`stop.bat`; single new **`Scripts/start-samepc.ps1`**
  launches coordinator + N nodes + web client. No absolute paths in the repo
  (`rg E:\DAIN` clean) — the folder can be copied anywhere or to another PC.
- Tabs: spawns each component as a tab of the current Windows Terminal window
  when `$env:WT_SESSION` is set; falls back to separate windows otherwise.
- Stopping = Ctrl+C in each window; no stop script.
- README quick-start updated to point at `Scripts/start-samepc.ps1`.

**GitHub Pages:** deploy workflow re-ran clean; site live at
https://velmet44.github.io/DAIN/.

### 2026-09-09 — same-PC launcher: fix + verified run (session 5)

**Bug (reported: launcher hung at "Waiting for coordinator..." with no windows):**
- `Start-Component` used `$MyInvocation.MyCommand.Path`, which is empty inside a
  function → children launched with `-File ""` → coordinator never started.
  Replaced with `$script:ScriptPath = $PSCommandPath` captured at script top level.
- Non-ASCII (em-dash `—`) smugged through an edit → file became unparseable in
  PowerShell 5.1 (Smart quote bytes break string parsing). Rewrote file as pure
  ASCII and re-saved with a UTF-8 BOM; `Parser.ParseFile` clean afterwards.

**Verified end-to-end (non-interactive smoke run):**
- Launcher detected coordinator on port 8000 (auto-port not needed this time),
  spawned coordinator + 1 node + Vite client.
- `GET /healthz` → `{"status":"ok"}`; `GET /admin/nodes` (X-Admin-Key) → node
  `node-DESKTOP-2RFHTDL-5235` online & connected, score 0.30. Node and client
  processes present. All test processes cleaned up afterwards.

**Gates:** launcher parses clean, full cluster brings itself up on one command.

### 2026-09-10 — web client: stale-config fix + verbose console logging (session 6)

**Bug (reported: launcher opens everything, but browser chat gets `401 Unauthorized` on
`127.0.0.1:8000/v1/models` and can't send).**

- Root cause: not the launcher env — verified shell `VITE_API_URL`/`VITE_API_KEY` reach
  the Vite dev server (probed the transformed `src/api.ts`). The failure came from the
  client's `useLocalStorage`: any stored value (e.g. `dain:base_url=127.0.0.1:8000`,
  `dain:api_key=""` from earlier dev sessions) **permanently** beat the launcher's
  per-run env defaults → empty API key → 401, on the hardcoded fallback URL.
- Fix in `Client/src/components.tsx`: `useLocalStorage` now honors a value only when the
  user explicitly edited it (stored as JSON `["value", true]`); legacy raw-string or
  non-edited entries are ignored and the env-provided default wins on every load.
  `Client/src/App.tsx`, `chat.tsx`, `api.ts`, `dashboard.tsx` unchanged in behavior.
- After the fix the browser picks up the coordinator URL + API key from the launcher env
  automatically; no stale storage can pin old values anymore.

**Verbose client console logging (requested):**
- Added `Client/src/logs.ts`: tagged helpers `logInfo/logOk/logWarn/logError` (always
  on) and `logDebug` (dev-server only).
- Wired into: env config on load, base-url/API-key edits, model fetch + picker,
  completion POST, SSE lifecycle (job dispatched, per-token frames via logDebug, final
  finish/token counts, `[DONE]`), abort/failure paths, dashboard poll results, and
  localStorage edit/ignore decisions. API keys are never logged (only `set`/`EMPTY`).
- `npm run build` (tsc --noEmit + vite) clean — fixed three tsc errors introduced during
  the edit chain (dropped return, missing `SseFrame.status`, nullable abort signal).

**Gates:** TS strict build green; client bundles successfully.

**Follow-up (same run):** after the localStorage fix the 401 was gone (models load,
key set) but Send/Enter still did nothing. Cause: `VITE_MODEL_ID` is empty in the
launcher flow and the old legacy `dain:model` value was now (correctly) ignored, so
`modelId` stayed `""` and `send()` early-returned at `!modelId` with no feedback.
- `chat.tsx`: auto-selects the first model from `/v1/models` when none is chosen
  (logged as `auto-selecting first model`). Send + Enter now work out of the box.
- Build re-verified clean.

**Follow-up (same run):** first full end-to-end send over the web client. Diagnosis of
"no streaming / weird logs / node silent":

- **Node gave no console output** — `Node/dain_node/__main__.py` never called
  `configure_logging`, so info logs were dropped by the root logger's default WARNING
  level (the coordinator configures it in `app.py`). Now calls
  `configure_logging(json_mode=settings.log_json, level=INFO)`; node window shows
  lifecycle + `generation_start` / new `generation_done` job=.. tokens=.. reason=..
  per job (added in both `_run_single_stage` and the distributed sampler in
  `Node/dain_node/jobs.py`).
- **Coordinator INFO spam** — every token batch logged
  `message_ignored type=token_batch` and every lifecycle `job_status_deferred (S4+)`
  even though those messages are (correctly) processed by the WS loop in `api.py`.
  Demoted those four deferred/ignored branches in `Coordinator/dain_coordinator/nodes.py`
  to DEBUG. Token relay is no longer per-token chatty at INFO.
- **"No streaming" is not a bug** — the dev model `dain-tiny-16L` (16 layers, hidden=64,
  random weights) generates 58 tokens in ~8 ms on this CPU; each token IS its own
  TOKEN_BATCH → SSE frame (visible per-token in the browser console `[dain:debug]`),
  but all frames land in one read burst so React paints once. Visible streaming
  appears with a real model (S10). "Random text" is expected: the dev model is
  untrained seed-initialized proto-text (per chat hint).

**Gates:** ruff clean + node tests (16) and coordinator registration/state tests (17) green.

**Follow-up (same run, 2 nodes):** with `numNodes=2` only one node appeared in the client
dashboard and the coordinator log showed both registrations with the **same**
`node_id=node-DESKTOP-2RFHTDL-5235`, followed by `connection_replaced` /
`heartbeat_seq_gap` ping-pong.
- Root cause: node identity persists to `node_state.json` and loading it **overrides**
  the env `DAIN_NODE_ID` (`identity.py`). The launcher started every node window in the
  same `Node/` cwd with the same default state path, so node N+1 loaded node 1's
  identity and both claimed one coordinator node.
- Fix in `Scripts/start-samepc.ps1` (node branch): each node now gets its own
  `DAIN_NODE_STATE_PATH=node_state-<index>.json` and `DAIN_MODEL_CACHE=shard_cache-<index>`
  so same-PC nodes keep distinct persisted identities (distinct caches also avoid
  concurrent shard-write races). Script re-verified with `Parser.ParseFile` (clean, ASCII).

**Follow-up (same run, CI red):** GitHub Actions "Python tests & lint / Sim (integration)"
failed — every spawned node crashed on POSIX, so nodes never came ONLINE and all Sim e2e
tests (admission, chaos matrix, ledger reconcile, pipeline parity, reconnect, single-node)
cascade-failed.
- Root cause: `Node/dain_node/agent.py` `StopGuard.install()` referenced
  `signal.SIGBREAK` unconditionally; `SIGBREAK` is Windows-only and does not exist on
  Linux/CI (Ubuntu), so the import-time attribute lookup raised `AttributeError` and the
  agent died before registering. `test_reconnect.py` already guarded its own SIGBREAK use
  via `os.name == "nt"`; the agent's signal list did not.
- Fix: `StopGuard.install()` now uses `getattr(signal, "SIGBREAK", None)` so the
  Windows-only signal is skipped on POSIX (existing `if sig is None: continue` guard then
  handles it).
- Regression test added: `Node/tests/test_agent.py` — `test_stopguard_install_is_portable`
  installs `StopGuard` under a running loop and asserts no exception (would have failed on
  any CI node run).

**Gates:** ruff clean + node tests (17) green.

**Verification (same run):** live 2-node run on the same PC confirmed healthy — distinct
`node-1`/`node-2` identities register and stay connected (no `connection_replaced` /
`heartbeat_seq_gap`), a 16-layer job fans out 4 shards / 2 stages / 2 nodes, both nodes
return `busy→online`, and ledger credits are recorded per stage. Stage assignment
rotated between jobs (job1 stage0=node-1, job2 stage0=node-2), i.e. placement balances.
- Note: node `generation_done tokens` (sampled bytes) can exceed coordinator
  `token_final tokens` (emitted UTF-8 chars) because `ByteStreamer` skips continuation
  bytes (multi-byte chars stream as one frame per decoded char). Cosmetic counter
  difference, not a data-loss bug.
- A stale `node-DESKTOP-2RFHTDL-5235` (pre-fix shared-identity era) is age-out cleanup
  at startup; coordinator marks it offline and it no longer re-registers.

### 2026-09-10 — public client on GitHub Pages: endpoint verified (session 6 follow-up)

- Deployment mode decision (user): **GitHub Pages only** — the web client is public at
  **https://velmet44.github.io/DAIN/**, but there is **no public coordinator** yet.
  The client is a static SPA; the coordinator URL is entered in the Chat tab
  (`dain:base_url` localStorage, launcher-provided default). Pages is a green field; the
  repo has **no** `deploy.yml` workflow (deploy is manual/pages-configured, not CI).
- Verified: `https://velmet44.github.io/DAIN/` returns the built SPA
  (`/DAIN/assets/index-*.js` + `*.css`, React 18.3.1 production bundle) with HTTP 200;
  assets resolve. Chat/dashboard will only show data once a coordinator URL that is
  reachable from the browser is entered.
- README updated with the public endpoint under "Quick start".

### 2026-09-10 — pre-S10 gate sweep + hygiene (session 6 follow-up)

- **Local gate sweep:** Common **44 passed** ✓ (ruff clean), Coordinator **36 passed** ✓
  (ruff clean), Node **17 passed** ✓ (ruff clean); Sim integration suite **left to a
  manual run** (pytest too slow inside the sandbox; CI is green with the SIGBREAK fix).
- **Coordinator format drift:** `ruff format` normalized 6 files (incl. a stray UTF-8 BOM
  at the top of `tests/test_ledger.py`); tests re-passed after the reformat.
- **Hygiene:** deleted the stale shared-era `Node/node_state.json`; gitignored
  `node_state-*.json` and `shard_cache*/` so per-machine identities and caches can never
  be committed.
- **Decisions (user):** CI confirmed green; **coordinator runs on the user's own PC for
  now** (no public coordinator — GitHub Pages hosts the client only). Multi-PC testing is
  viable immediately: node agents connect **outbound** (NAT-friendly, spec §10), so any
  machine running the packaged node (`Scripts` portable `DAIN.zip`, commit bce1ecd) can
  register against the home-PC coordinator over LAN; shards are pulled over HTTP from the
  coordinator's model store (one-time per-node cache). This is the S10 onboarding path
  minus GPUs.

### 2026-09-10 — dev tooling: cache cleanup + portable packaging scripts (session 7)

- Added **`Scripts/clean_cache.bat`** — removes all regenerable cache/build artifacts,
  recursive from the repo root: `__pycache__`, `.pytest_cache`, `.ruff_cache`,
  `.mypy_cache`, `Node/build`, `Node/dist`, `Client/dist`, `Client/node_modules/.vite`,
  `Node/node_state-*.json`, `Node/shard_cache*`. **Keeps** all `.venv` environments and
  the SQLite databases (explicit user requirement). Prints a per-item removal list + a
  summary count, then pauses.
- Added **`Scripts/package.bat`** + **`Scripts/package.ps1`** — packages the whole
  relocatable project into a zip at the repo root, auto-bumping the name if present
  (`DAIN.zip` → `DAIN1.zip` → `DAIN2.zip` …). Staging strategy: recursive copy to a
  temp dir that **prunes excluded names at each directory level** (never descends into
  excluded trees), then zips via .NET `ZipFile.CreateFromDirectory` (includes hidden
  files like `.gitignore`).
- **Excluded** (recreatable/ignoreable): `__pycache__`, `.pytest_cache`, `.ruff_cache`,
  `.mypy_cache`, `.venv`, `node_modules`, `.git`, `build/`, `dist/`, `shard_cache*`,
  `node_state*.json`, `*.sqlite3*`, `*.pyc`, `*.spec`, `.DS_Store`, `Thumbs.db`,
  previous `DAIN*.zip`.
- **Included even though gitignored**: `model_store/dain-tiny-16L` (the dev model is
  essential for a fresh machine to infer). `Deploy/caddy.exe` ships too (needed for the
  prod edge, no download step).
- Verified end-to-end in a test run: `Scripts/package.bat` produced a 19.38 MB zip with
  124 entries; zip contents audited — all six projects (`Common/Coordinator/Node/Sim/
  Client/Deploy`), `Docs/`, `.github/workflows`, `model_store`, `Scripts/` present;
  marker scan over entry names showed **zero** `venv|__pycache__|node_modules|.git|
  sqlite|shard_cache|node_state|build/|dist` hits. Test zip removed after verification.
- Portability contract confirmed: the zip copies cleanly to another PC / folder; the
  recipient unzips and runs `Scripts/bootstrap.ps1` (recreates all four `.venv` via
  `uv sync` and `Client/node_modules` via `npm ci`). Agreed with the earlier session-4
  decision that the repo contains no absolute paths (`rg E:\DAIN` clean) so the folder
  is relocatable.

### 2026-09-10 — coordinator admin web UI (session 8)

- Added a **self-contained admin SPA** served by the coordinator at `GET /admin` and
  `GET /admin/` (`Coordinator/dain_coordinator/admin_ui.html`, zero build step, vanilla
  JS + CSS, dark theme, ASCII). Read once at import via `_admin_ui()` in `app.py`;
  excluded from the OpenAPI schema (`include_in_schema=False`).
- **Security model:** the HTML is an unauthenticated shell — every data call it makes
  is individually authed, so nothing leaks. The page takes two keys (X-Admin-Key for
  `/admin/*`, X-API-Key for `/v1/*` + `/ledger/*`), persisted in `localStorage`
  (`dain_admin` / `dain_api`); 401s surface as a hint banner.
- **What it shows** (polled every 5 s, pausable, manual refresh): metric strip (models,
  online/other nodes, active jobs, total credits), nodes table (state pill, score, GPU/
  VRAM, agent version, last-heartbeat age), click-to-expand node detail (score-component
  bars + full state-history trail), recent placement events, ledger summary per unit
  (success/retried/failed/flagged/credits + totals), model list. Data feeds:
  `GET /admin/nodes`, `/admin/nodes/{id}`, `/v1/nodes`, `/ledger/summary`, `/v1/models`.
- Tests: `Coordinator/tests/test_admin_ui.py` (4 — page served with the right content
  type, slashless `/admin` alias, no-auth shell, and the data APIs staying locked to
  their own keys incl. wrong-key 401).
- Gates: Coordinator **40 passed** (+4), ruff clean.
- Answer to the earlier admin question: the coordinator now has a browser page —
  open `http://<coord-host>:8000/admin/` and enter the admin + API keys.
### 2026-09-10 - real-model path: export_hf_model + HF tokenizer (session 8)

- Implemented the real-model path end to end so actual Hugging Face checkpoints (Llama family) can
  run through the distributed pipeline (S18; first step of S10 real-model bring-up).
- **Common:** `ModelManifest` gained optional `tokenizer_file` + `tokenizer_hash` (must be set
  together; None = dev byte-level tokenizer); fixed `dain_common/__init__.py` to actually export
  `ModelManifest` (it was in `__all__` but missing from the import list).
- **Node export:** `shard_export.py` refactored byte/embed/layer shard writing into a shared
  `_write_shards`; added `export_hf_model(out_dir, model_id, model=, tokenizer=, dtype=)` for
  Llama-family checkpoints (constructing a model/tokenizer locally makes tests hermetic;
  `hf_name` pulls from the Hub). Stores the fast `tokenizer.json` beside the shards and records
  its sha256 in the manifest. Rejects non-Llama architectures (Qwen2 = S10 future stage).
- **Node tokenizer:** new `dain_node/hf_tokenizer.py` (`HFTokenizer`: encode/decode/feed via
  `tokenizers.Tokenizer.from_file`); `StageModel` now takes the dtype from `manifest.dtype`
  (fp16 supported; rope verified to cast to input dtype) and builds an HFTokenizer when a
  tokenizer path is given, else the ByteTokenizer.
- **Node distribution:** `ModelStoreClient.ensure_tokenizer` downloads + verifies the
  tokenizer over the store HTTP API; `jobs.py` streams real tokens via a `TokenStreamer`
  (swapped in when the manifest carries a tokenizer) and the activation relay now uses the
  stage dtype (was hardcoded fp32) - both send sites and the receiver in `_run_step` are
  dtype-aware (`_ACTIVATION_DTYPES`).
- **Coordinator:** `GET /model/tokenizer/{model_id}` serves the model's tokenizer.json
  (404 for byte-level models), with the same safe-id guard as shards.
- **Tests:** Common 47 (+3 manifest tokenizer-field tests), Coordinator 36, Node 20 (+4 new
  in `tests/test_hf_export.py`: fp16 shard export, tokenizer round-trip/feed, fp16 StageModel
  load + forward, non-Llama rejection - the BPE tokenizer is trained on a tiny local corpus
  so no hub access is needed). New Sim e2e `tests/test_hf_model_e2e.py`: 2 agents split a
  12-layer fp16 Llama (2 stages x 6 layers), streamed completion equals a greedy HF
  reference decoded with the same tokenizer. Full Sim job-path regressions (pipeline parity,
  single-node e2e, reconnect/replay) still pass; the remaining heavy Sim suite is a manual
  user run.
- **Fixes found while testing:** e2e 429 was a registration-to-WS race (nodes listed ONLINE
  a moment before the WebSocket was counted as connected) - hardened with a
  `wait_connected` gate like the parity test. `httpx.ResponseNotRead` was a test bug
  (reading `.text` on a streamed response).
- **Portability (user requirement, refreshed):** no absolute paths anywhere in code
  (`rg E:\DAIN` hits only this log), bootstrap scripts anchor on their own location, and all
  runtime defaults are cwd-relative (`model_store`, `node_state.json`, `shard_cache`), so
  the folder still works when moved to another location or machine after
  `Scripts/bootstrap.ps1`.

### 2026-09-10 — LAN coordinator auto-discovery (session 9, commit 2e7becf)

**Problem:** a node's `config.json` defaulted to `ws://localhost:8000`, so every LAN node
needed a hand-typed coordinator IP (`coord_url`) before it could join.

**Solution — UDP broadcast probe (stdlib-only, no new deps):**
- **Common (`dain_common/node_discovery.py`):** v1 protocol — node broadcasts
  `{"v":1,"op":"discover","k":"<join_token>"}` (≤512 B JSON datagrams); coordinator replies
  unicast `{"v":1,"op":"hello","port":8000}`. The node builds
  `ws://<reply-source-ip>:<port>` where the source address is *authoritative* — it is the
  interface that can actually reach that node, so a multi-homed / DHCP coordinator always
  advertises the right address. Token check is timing-safe (`secrets.compare_digest`).
  WAN/public: disable discovery and set `coord_url` explicitly (subnet-scoped by design).
- **Coordinator (`dain_coordinator/discovery.py`):** `DiscoveryResponder` — non-blocking UDP
  socket, answers only authenticated probes, drops malformed/wrong-token silently, logs
  `discovery_request source=`. Wired into `create_app`'s lifespan (start/run/close); an
  occupied discovery port logs `discovery_disabled bind_failed` and the coordinator keeps
  running. New settings: `DAIN_DISCOVERY_PORT=8456`, `DAIN_DISCOVERY_ENABLED=true`.
- **Node (`dain_node/discovery.py`):** `discover_coordinator()` broadcasts the probe, collects
  replies until timeout, returns the best `ws://…`. Handles the Windows ICMP quirk where a
  silent host's "port unreachable" surfaces as `ConnectionResetError` on `recvfrom`
  (swallowed, keep listening).
- **`__main__.py`:** on startup in standalone mode, an empty `coord_url` triggers one probe
  (≤2 s); on success the URL is written into `config.json` atomically (`write_config`, tmp +
  replace) so later runs skip the probe and the user can read exactly what the node joined.
  ConfigWatcher baseline is taken *after* the write, so the auto-write never looks like a
  user edit. Empty `coord_url` in `NodeSettings.from_config()` falls back to localhost.
- **`Scripts/build-node.ps1`**: new default config ships `coord_url=""` (auto-discover);
  publish message updated ("extract the zip and run DainNode.exe — it auto-discovers the
  coordinator on the LAN; edit config.json only for remote/WAN use"). `Deploy/.env.example`
  documents the two new discovery vars.
- **Also committed (was seated in the working tree):** the `/msg` chat UI
  (`chat_ui.html`, `app.py` route, `Coordinator/tests/test_chat_ui.py`) per user decision.
- **Gates:** Common **54 passed** ✓, Coordinator **48 passed** ✓, Node **25 passed** ✓,
  ruff clean across all three (Sim suite left to the usual manual/CI run).
- WIP context: user's next focus is WAN (TLS/wss via reverse proxy, per-node credentials,
  outbound-only nodes); LAN improvements after this are coordinator IP stability/DNS and
  flaky-WiFi heartbeat tolerance.

### 2026-09-10 — coordinator config.json: all options, auto-created (session 10, commit 180c241)

**Problem:** the coordinator only accepted env vars (`DAIN_*`), so every run needed a long
env block (it was "annoying to put env vars every time"). The node already had config.json;
the coordinator ended up with the same pattern.

- **`settings.py`:** added `from_config(data, base_dir)` covering **every** field (verified by
  `test_env_map_covers_every_default_key` — no setting can exist only in env or only in the
  file), plus `DEFAULT_CONFIG` (the full 36-option file) and `COORDINATOR_ENV` (field → env
  var map). Also plugged five env gaps that previously had *no* env mapping at all:
  `DAIN_MONITOR_TICK_S`, `DAIN_UPTIME_ALPHA`, `DAIN_OVERLOAD_UTIL_PCT`, `DAIN_OVERLOAD_STRIKES`,
  `DAIN_TEMP_DEGRADE_C`, `DAIN_MAX_COMPLETION_TOKENS`. `_truthy()` accepts JSON booleans and
  "1"/"true"/"yes"/"on". Relative `db_path`/`model_store_dir` resolve against the config's
  folder; unknown keys are ignored (extra doc fields never break loading); `cors_origins`
  accepts a list *or* a comma string.
- **`config.py` (new):** `find_base_dir()` (dev → `Coordinator/`) and `resolve_settings()`:
  if `Coordinator/config.json` is missing it **writes a full default config** on first run
  (`config_created` logged, message "edit it and restart"), then overlays environment
  variables per-field so launchers (`Scripts/start-samepc.ps1`, Sim harness env) keep their
  precedence — config provides defaults, env wins when set.
- **`__main__.py`:** `python -m dain_coordinator` now just works — no env needed; logs
  `coordinator_start host=… port=… config=… config_created=…`.
- **`.gitignore`:** `Coordinator/config.json` (machine-specific, auto-generated).
- **Not file-configurable (kept as code defaults):** `scoring`/`accounting_weights` are
  nested config objects — reference values already live in `Deploy/.env.example`/Common.
- **Tests:** `Coordinator/tests/test_coordinator_config.py` (13: round-trip of every default,
  full override, comma-vs-list CORS, relative/absolute path resolution, default-file
  creation, existing-file use, env-over-config, env-over-missing-file, env-map coverage,
  base-dir sanity). Real smoke: first `resolve_settings()` in `Coordinator/` created the
  36-key `config.json` exactly as intended.
- **Gates:** Coordinator **61 passed** ✓ (48 + 13), ruff clean. Common/Node untouched.
- Behavior notes for the user: changing `Coordinator/config.json` takes effect on the next
  start (read once, no watchdog — a mid-flight reload could strand running jobs); env vars
  still override individual fields; `scoring`/`accounting_weights` stay code-level.
- Key defaults are now random 8-char strings (part of the same session-10 work) —
  `DEFAULT_JOIN_TOKEN=Jj3L7ewD`, `DEFAULT_API_KEY=DzOjEXqs`,
  `DEFAULT_ADMIN_API_KEY=2UPZQJln` — applied to coordinator + node code defaults,
  `Coordinator/config.json`, `Scripts/build-node.ps1`, `Scripts/start-samepc.ps1` prompts,
  and `Deploy/.env.example`. Sim/Coordinator test fixtures keep their own explicit word keys
  (`conftest`/`helpers` pass them in `make_settings`), so suites are untouched by default
  changes.

### 2026-09-10 — admin page: rotate/reset keys at runtime (session 11, commit 962512c)

**Problem:** changing the join token or API keys meant editing `config.json` and restarting —
the admin page could see nodes but not manage the cluster's own credentials.

- **Admin API:**
  - `GET /admin/settings/keys` → current `join_token`/`api_key`/`admin_api_key` plus their
    shipped defaults, which fields are env-overridden at startup, and whether a config file is
    writable.
  - `PUT /admin/settings/keys` → set any of the three (min 8 chars via pydantic, unknown keys
    422) or restore any of them to default via `reset_<field>: true`. Changes apply
    **immediately** by swapping the live settings object (`app.state.settings` +
    `NodeService.settings`) — auth checks read it per request; the discovery responder's token
    rotates through a new `set_join_token()`. Existing node sessions are untouched (they use
    per-node tokens).
  - Persistence: written atomically into the coordinator's `config.json` (merge + tmpfile+
    replace in `config.persist_config`). A first-ever edit bootstraps the file *from the live
    settings*, so unrelated keys don't silently snap back to defaults. Without a config file
    (`writable: false`) changes are in-memory only. If the field came from an env var at
    startup, the response notes "env still wins at restart".
- **Admin UI:** new "Keys & tokens" card — three fields prefilled with the current values, a
  per-key **Reset** button (restores the shipped default), and **Apply changes** (PUTs the
  three values; the page then re-keys its own session with the new client/admin key so it keeps
  working mid-session). Auto-visible whenever an admin key is provided; a note line explains
  persistence + env-override behavior.
- **Settings:** `DEFAULT_JOIN_TOKEN`/`DEFAULT_API_KEY`/`DEFAULT_ADMIN_API_KEY` now exist as
  named constants so "reset to default" and code defaults can never drift.
- **Gates:** Coordinator **70 passed** (61 + 9 new key tests covering rotate API/admin/
  join-token, short-key 422, unknown-key 422, reset, restart persistence, in-memory-only,
  no-op unchanged, first-edit bootstrap), ruff clean. Node/Common untouched.

### 2026-09-10 — admin page upgrade: node/job/model/log controls (session 12, commit 0785808)

**Goal:** the admin page is now a control surface, not just a read-only dashboard. New
Coordinator endpoints (all under `/admin`, admin-key gated) + UI:

- **Nodes:** `POST /admin/nodes/{id}/offline` (evict — transitions to offline, fails the
  node's active jobs), `POST /admin/nodes/{id}/online` (recover). 409 on illegal transitions.
  Detail panel now has **Recover (ONLINE)** / **Evict (OFFLINE)** buttons.
- **Jobs:** `GET /admin/jobs` (recent + active, newest first), `POST /admin/jobs/{id}/cancel`
  (releases the stage nodes + fails the job). New Jobs card lists them with a **Cancel**
  button on active ones. (Ordinary jobs *can* be cancelled; jobs in-progress on one
  coordinator can't be resumed on another.)
- **Models:** `GET /admin/models` (with on-disk sizes), `POST /admin/models/{id}/delete`
  (path-traversal-safe: safe-char check + `resolve()` containment + rmtree), `POST
  /admin/models/rescan` (recompute placements immediately via the existing
  `recompute_pool`). Models card shows size + **Delete** (confirm); a **Rescan placements**
  action the user asked for.
- **Logs:** new in-process ring (`dain_coordinator.logs.RingLogHandler`, 2000 lines) attached
  to the root logger at app build; `GET /admin/logs?lines=N` tails it. Log viewer card with
  line-count + auto-refresh.
- **Ledger:** **Export JSON** / **Export CSV** buttons hit the existing authenticated
  `/ledger/export`.
- Fixed pre-existing CSS bug: NodeState/JobState are lowercase StrEnum values, but the pill
  classes in `admin_ui.html` were `st-ONLINE`-style => states never actually showing colors.
  Pill classes now `st-online`/`st-busy`/`st-degraded`/`st-offline`.
- **Gates:** Coordinator **79 passed** (70 + 9 new: evict/recover lifecycle, 409s, unknown
  node 409, model list size/delete/rescan, path-traversal, jobs list/cancel/409s/404, logs
  tail), ruff clean. Node/Common untouched.

### 2026-09-10 — hardening sweep: 7 confirmed bug fixes (session 13)

Post-release audit found seven real bugs; all fixed in one pass.

- **Node agent version drift** (`Node/dain_node/agent.py`): `Register.agent_version` was
  hardcoded `"0.1.0"` while the package is 1.0.0 — the coordinator recorded a wrong version
  for every node. Now imports `__version__` from `dain_node`.
- **`_restart_from_entry` never re-dispatched** (`Coordinator/dain_coordinator/faults.py`):
  the whole-job-stall restart only reset bookkeeping and set the job QUEUED, so it sat until
  the watchdog failed it. It now re-fires `JOB_ASSIGN` for every stage of the current
  placement (stage 0 carries the prompt; nodes already replace their runtime for a repeated
  assign, so this is safe), resets `last_token_at`/`first_token_at`/`dispatched_at`, and
  transitions to DISPATCHED. Dead nodes' sends fail and the normal watchdog→`_reassign` path
  picks backups. Also fixed the **double restart increment** (tick + restart both bumped
  `job.restarts`, so `max_job_restarts=1` gave zero useful restarts): `restarts` is now
  incremented only inside `_restart_from_entry`, and tick uses `>=`.
- **Port fallback not advertised** (`Coordinator/dain_coordinator/__main__.py`): when the
  preferred port was busy, the free port was passed to uvicorn but never written back into
  settings — the UDP discovery responder told nodes `ws://host:8000` while the server listened
  on 8001. Settings are now rebuilt with the bound port (`dataclasses.replace`) before
  `create_app`.
- **Client disconnect leaked the job** (`Coordinator/dain_coordinator/api.py`): the SSE (and
  non-streaming) `finally` released the busy nodes but never reached a terminal state, so
  orphaned jobs burned watchdog restarts and never emitted a ledger event. The `finally` now
  calls `jobs.fail_job(..., "client disconnected")` when the record is still non-terminal
  (no-op on the normal completion path).
- **Shard download tmp-file race** (`Node/dain_node/llm.py`): shard and tokenizer downloads
  used a fixed `path.tmp`, so two concurrent jobs fetching the same missing shard could
  interleave writes and cache a corrupt file. Temp names are now unique (`secrets.token_hex`),
  matching the identity-state pattern.
- **XSS in chat rendering** (`Client/src/components.tsx`): `marked` output was injected via
  `dangerouslySetInnerHTML` with no sanitizer. Now passed through DOMPurify (new dependency,
  bundled types).
- **Non-ASCII secret headers 500'd** (`Coordinator/dain_coordinator/api.py`):
  `secrets.compare_digest` raises `TypeError` on non-ASCII `str`, so a header with non-ASCII
  bytes returned 500 instead of 401. Both `require_api_key` and `require_admin` now compare
  UTF-8-encoded bytes.
- **Gates:** Coordinator **78 passed**, Common **54 passed**, Node **25 passed**, ruff clean
  on all touched Python files; Client `tsc --noEmit` + vite build clean. Sim (subprocess
  cluster e2e) not re-run this session.

### 2026-09-10 — GGUF import: converter + script + admin trigger (session 14)

Drop a Llama-family `.gguf` into `model_store/`, run one command (or click one
button on the admin page), get DAIN sharded safetensors + manifest. Idempotent.

- **Converter** (`Node/dain_node.import_gguf`, new): reads GGUF metadata
  (rejects non-llama architectures), dequantizes to fp32, then reuses
  `shard_export._write_shards` unchanged (fp16 default, `--layers-per-shard`,
  `--model-id`, `--tokenizer <hf-repo>` fallback for SPM vocabs, `--force`).
  Loading deliberately bypasses `from_pretrained(gguf_file=...)`: that path's
  accelerate meta-device init **segfaults** on the pinned torch-CPU/Windows
  stack (verified against transformers 4.46.3) — instead
  `load_gguf_checkpoint` + `load_state_dict(assign=True)`, which round-trips
  weights exactly (incl. the llama.cpp q/k rope half-split permute, GQA-aware).
  Tied embeddings synthesize `lm_head` like `export_hf_model`. New deps: `gguf`
  (no accelerate — dropped after the segfault finding).
- **Idempotency**: `model_store/.gguf-imports.json` maps file name → sha256 +
  model_id. Skip while the model dir has a manifest AND the sha matches;
  deleting the dir or changing the file re-imports. `--force` overrides.
- **Script** (`Scripts/import-gguf.ps1`, new): `-GgufFile/-ModelStore/-ModelId/
  -Dtype/-LayersPerShard/-Tokenizer/-Force`, anchors on its own location.
- **Admin trigger** (coordinator, no torch added): `GET /admin/models/imports`
  (lists `*.gguf` with pending/importing/imported status from the converter
  marker, `node_project` capability, last error) + `POST /admin/models/import`
  (filename path-traversal-guarded to a store-local basename; 409 while a run
  is active; 503 with clear reason when no Node checkout/uv). Runs
  `uv run --project <Node> python -m dain_node.import_gguf` as a subprocess,
  streaming output into the ring log (visible on the admin Logs card), then
  auto-recomputes placements. `node_project_dir` setting (config.json key +
  `DAIN_NODE_PROJECT_DIR` env) overrides the default sibling `Node/`.
  Test seam: `app.state.gguf_runner` (async callable replacing the subprocess).
- **Admin UI**: Models card gains a GGUF section — per-file status pills
  (pending/importing…/imported), per-file Import button, "Import pending
  GGUFs" action, last error line; hidden when there are no .gguf files.
- **Portability hardening**: `bootstrap.ps1` now self-heals a moved/broken
  `.venv` — on sync or smoke-test failure it deletes venvs and re-syncs from
  the lockfile once before giving up (verified: fresh-clone and moved-venv
  scenarios both pass; uv venvs point outside the project dir).
- **Gates:** Node **29 passed** (4 new GGUF tests: exact round-trip incl. GQA
  permute, idempotency/lifecycle, non-llama rejection, CLI), Coordinator
  **83 passed** (5 new endpoint tests: listing status classification,
  filename validation 400/404, subprocess argv + 409-busy + auto-rescan,
  failure reporting, 503 without Node project), Common 54, ruff clean
  everywhere. **Live smoke**: real `.gguf` in `model_store/` →
  `Scripts/import-gguf.ps1` → 2 shards + tokenizer + manifest, second run
  skips, `list_models` sees the import, artifacts cleaned up.

### 2026-09-10 — keyless localhost admin (session 15)

The admin page (and the coordinator's chat page) now work with **no keys at
all** when opened on the same machine — remote hosts still need keys.

- **Server** (`api.py`): `_local_trusted(request)` grants keyless access to
  `/admin/*` and `/v1/*` only when ALL of these hold:
  1. the socket peer is a loopback address (direct local connection);
  2. the `Host` header is `127.0.0.1` / `localhost` / `[::1]` — defeats DNS
     rebinding, where a visited site resolves its own domain to 127.0.0.1 and
     the browser sends *that* hostname as Host;
  3. if an `Origin` header is present it is localhost — a cross-site drive-by
     POST from any visited web page always carries its own Origin, so it is
     rejected; curl / same-origin local requests pass.
  An empty key header value counts as "not provided" (so the local UI works
  with blank key fields). An explicit *wrong* key is still 401 even from
  localhost. Node auth (`/model/*`, WS) is untouched — nodes are remote.
- **Admin UI** (`admin_ui.html`): no more "enter key first" gating — every
  card loads keylessly; the API key header is only sent when the user typed
  one. On 401 (opened remotely, or revoked) the page degrades gracefully:
  models/nodes fall back to the public `/v1` views, and the keys card shows
  "unauthorized — enter the admin key above".
- **Tests** (`tests/test_admin_localhost.py`, 5 new): keyless admin + client
  API from loopback; empty-header ≙ absent; DNS-rebinding Host blocked;
  cross-site Origin blocked while localhost Origins pass; remote peer 401
  without key / 200 with key; wrong key still 401 locally.
- **Gates:** Coordinator **88 passed** (83 + 5), ruff clean. Node/Common/
  Client untouched.
