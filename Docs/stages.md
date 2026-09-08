# DAIN — Development Stages (Agent Execution Plan)

Companion to [spec.md](spec.md). This document defines the **only** execution order.
It is written for an AI agent driving the repository: every stage has a goal, concrete
deliverables, pass/fail checkpoints, and hard constraints. When this file and the spec
disagree, the spec wins and this file must be amended explicitly.

---

## 1. Agent working agreement (read before every session)

1. **One stage at a time, strictly in order.** Never start stage N+1 until every checkpoint
   of stage N passes and the STATUS table (§5) is updated. Never skip or reorder.
2. **Checkpoint-gated.** Checkpoints are commands with expected outcomes. A stage is done
   only when all checkpoints pass, deliverables exist, and the work is committed
   (`stage <N>: <summary>`, at least one commit per stage).
3. **Keep it green.** Tests from all previous stages must pass at every commit. If an old
   test breaks, fix forward — never delete or weaken a test to pass a checkpoint.
4. **Constraints are hard limits.** The "Do NOT" list per stage prevents scope creep. If a
   constraint genuinely blocks the stage goal, stop and escalate to the human (§4).
5. **Spec is law.** If code contradicts `Docs/spec.md`, either the code is wrong or the spec
   needs an amendment; note the amendment in the stage commit.
6. **Small diffs.** Prefer many small commits within a stage over one large drop.
7. **No drive-by refactors** of earlier stages except to fix a failing test, and justify it
   in the commit message.
8. **Timebox experiments.** Any spike/prototype lives in a branch or `Sim/scratch/` and is
   either merged deliberately or discarded — never left half-merged.

## 2. Global definition of done (applies to every stage)

- [ ] All stage checkpoints pass (commands + expected output below)
- [ ] `uv run pytest -q` green and `uv run ruff check .` clean
- [ ] New protocol/parsing/accounting code has full test coverage; other code proportionate
- [ ] README/docstrings updated where behavior changed
- [ ] STATUS table (§5) ticked and committed

## 3. Repository architecture

The existing skeleton (`Client/`, `Coordinator/`, `Node/`, `Docs/`, `Builds/`) is adopted
and extended — directories are **not** renamed.

```
DAIN/
├── Docs/              spec.md · stages.md · pilot reports
├── Common/            dain_common/    shared: pydantic schemas, scoring, FLOP tables (no I/O)
├── Coordinator/       dain_coordinator/  FastAPI app:
│                                         api/ (client REST+SSE, node WS, admin)
│                                         registry/ · scheduler/ · monitor/ · relay/ · ledger/
├── Node/              dain_node/      agent: ws client, heartbeat, executors/{fake,hf},
│                                      shard_store, capability probe (psutil/torch)
├── Client/            React 18 + Vite + TypeScript SPA (chat UI + node dashboard)
├── Sim/               cluster.py (spawn coordinator + N nodes), chaos.py, benchmarks
├── Tests/             unit/ · integration/  (pytest; imports dain_common / app factories)
├── Deploy/            systemd units, env templates, Netlify/VPS/Render notes
├── Builds/            build artifacts only (gitignored)
├── pyproject.toml     uv workspace: Common, Coordinator, Node, Sim
└── README.md
```

Runtime model (dev = everything on one machine):

- **Coordinator** listens on `:8000` — client REST+SSE (`/v1/*`), node WebSocket
  (`/node/ws`), metrics (`/metrics`). It is the only listener.
- **Nodes connect outbound** to the coordinator (persistent WSS) — no inbound ports, so
  nodes behind NAT work (spec §10). The same code path serves CPU-sim and GPU nodes.
- Key env vars: `DAIN_COORD_URL`, `DAIN_NODE_TOKEN`, `DAIN_MODEL`, `DAIN_DB_PATH`,
  `DAIN_HEARTBEAT_S`. One `.env.example` in `Deploy/` is the reference.
- Dev models: **TinyLlama-1.1B on CPU** for sim/CI; **Qwen2.5-7B fp16** for the GPU pilot
  (spec §18). Model download is a `Sim/` utility, never committed to the repo.

## 4. Decision points (escalate to the human; everything else is the agent's call)

- **S0**: repo layout & toolchain approval (this document).
- **S9**: hosting credentials — Netlify account, coordinator host (user VPS vs Render),
  domain, API-key policy.
- **S10**: GPU pilot participants, acceptance of Qwen/Qwen3 model licenses, power telemetry
  availability.

## 5. Status table (update + commit at each stage gate)

| Stage | Name | Status |
|---|---|---|
| S0 | Bootstrap & toolchain | ✅ 2026-09-08 (2cd11ad) |
| S1 | Protocol & shared schemas (`Common`) | ✅ 2026-09-08 (2983e9a) |
| S2 | Coordinator core: registry, auth, heartbeats, state machine | ☐ |
| S3 | Node agent skeleton + local cluster sim | ☐ |
| S4 | Single-node inference path (real model, streaming) | ☐ |
| S5 | Distributed pipeline: partitioning + activation relay | ☐ |
| S6 | Scoring-driven scheduling, top-K, backups, queueing | ☐ |
| S7 | Fault tolerance & degraded mode | ☐ |
| S8 | Accounting ledger | ☐ |
| S9 | Web client + deployment (Netlify + public coordinator) | ☐ |
| S10 | GPU pilot & measurement campaign | ☐ |
| S11 | (Stretch) MoE expert placement — OLMoE-1B-7B | ☐ |

---

## 6. Stages

### S0 — Bootstrap & toolchain

**Goal:** runnable, linted, tested skeleton; conventions locked.
**Preconditions:** repo exists (git initialized), `Docs/` populated, Python 3.11+ installed.

Tasks:
1. `pyproject.toml` as a **uv workspace** with members `Common`, `Coordinator`, `Node`,
   `Sim`; packages `dain-common`, `dain-coordinator`, `dain-node`, `dain-sim`.
2. Tooling: ruff (lint+format), pytest config, `.gitignore` (`Builds/`, `__pycache__`,
   `.env`, model caches), `.env.example` in `Deploy/`.
3. Skeleton packages with `python -m dain_coordinator --version` style entry points;
   hello-world FastAPI app at `GET /healthz`.
4. `README.md` (what DAIN is, 5-line quickstart, link to `Docs/`).
5. First commit on a branch `main` if the repo has no commits yet.

Checkpoints:
```bash
uv sync                          # resolves workspace, no errors
uv run pytest -q                 # 1 smoke test green
uv run ruff check .              # clean
uv run python -m dain_coordinator &  curl -s localhost:8000/healthz   # {"status":"ok"}
```

Do NOT: implement any DAIN logic, add websockets/pydantic models beyond the health app, or
touch `Client/`.

---

### S1 — Protocol & shared schemas (`Common`)

**Goal:** the wire protocol (spec §10/§11), scoring model (§7), and FLOP/credit math (§15)
as a dependency-free, fully tested package.
**Preconditions:** S0 green.

Tasks:
1. Pydantic v2 models + envelope: `REGISTER`, `REGISTER_ACK`, `HEARTBEAT`, `METRICS_REPORT`,
   `JOB_ASSIGN`, `JOB_STATUS`, `ACTIVATION_RELAY` header, `LEDGER_EVENT`, `SHARD_MANIFEST`;
   `NodeState` enum (`ONLINE|BUSY|DEGRADED|OFFLINE`); job states
   (`QUEUED→DISPATCHED→RUNNING→STREAMING→COMPLETED|FAILED|RETRYING`).
2. Capability manifest model (spec §6.2) with plausibility validators.
3. Scoring: `norm/norm_inv`, penalty application, weight map (defaults from spec §7),
   EWMA helpers. Pure functions, no I/O.
4. FLOP tables (`2·params·tokens`, MoE variant) + credit function (spec §15) with
   `(job_id, stage_idx, attempt)` idempotency key type.

Checkpoints:
```bash
uv run pytest Tests/unit/test_schemas.py -q      # envelope round-trips, versioning, malformed rejects
uv run pytest Tests/unit/test_scoring.py -q      # golden tests: synthetic 20-node pool → expected
                                                 # ranking & scores (seeded, from spec §7 weights)
uv run pytest Tests/unit/test_accounting.py -q   # FLOP table golden values for Qwen2.5-7B (2·7.6e9·tok)
```

Do NOT: any network I/O, any coordinator/node package code, client work.

---

### S2 — Coordinator core: registry, auth, heartbeats, state machine

**Goal:** nodes can register, authenticate, heartbeat, and transition per spec §6 state
machine; state persists in SQLite and survives restart.
**Preconditions:** S1 green.

Tasks:
1. FastAPI app: `POST /node/register` (issue node token), `WS /node/ws` (heartbeats +
   metrics), `POST /node/deregister`. Admin: `GET /admin/nodes`.
2. Registry on SQLite (WAL): nodes, capabilities, scores, state history. Schema portable to
   Postgres (no SQLite-specific SQL outside the store module).
3. Heartbeat monitor: missed-interval detection (3 × 5 s → OFFLINE), state machine exactly
   per spec §6 table, score recomputation on each report (S1 functions).
4. Structured logging with `node_id` correlation.

Checkpoints:
```bash
uv run pytest Tests/integration/test_registration.py -q   # fake WS node: register→ONLINE→
                                                          # stop heartbeats→OFFLINE→re-register→ONLINE
uv run pytest Tests/integration/test_coordinator_restart.py -q  # state survives process restart
uv run pytest Tests/integration/test_20_nodes.py -q       # 20 concurrent fake nodes, 60 s, 0 spurious
                                                          # state flips, all transitions logged
```

Do NOT: scheduling, partition tables, ledger, relay, any model execution.

---

### S3 — Node agent skeleton + local cluster sim

**Goal:** a real node agent process and a one-command local cluster; the coordinator sees
N stable nodes.
**Preconditions:** S2 green.

Tasks:
1. `dain_node`: outbound WSS client (auto-reconnect with backoff), capability probe
   (psutil; torch/CUDA if present, else CPU-only), heartbeat loop with metrics payload,
   `Executor` interface with a **deterministic `FakeExecutor`** (canned activations/tokens).
2. `Sim/cluster.py`: spawns coordinator + N node subprocesses from one config; health
   summary; `--duration`, `--nodes` flags; exit code reflects cluster health.
3. Graceful shutdown (SIGINT → deregister → exit) on both sides.

Checkpoints:
```bash
uv run python Sim/cluster.py --nodes 12 --duration 60 ; echo $?   # exit 0: 12/12 ONLINE,
                                                                 # 0 missed heartbeats, 0 reconnect storms
uv run pytest Tests/integration/test_reconnect.py -q      # kill a node process → OFFLINE within 15 s;
                                                          # restart → ONLINE, same node_id, history kept
```

Do NOT: GPU/model code (FakeExecutor only), partition assignment, job dispatch.

---

### S4 — Single-node inference path (real model, streaming)

**Goal:** the full client→coordinator→node→client loop with a real model on **one** node —
establishes routing, job tracking, and SSE relay before distribution.
**Preconditions:** S3 green; TinyLlama-1.1B downloaded via `Sim/download_model.py`.

Tasks:
1. `HFExecutor` (transformers, CPU fp32/int8): loads full model, generate() streaming.
2. Client API: `POST /v1/completions` (SSE streaming), `GET /v1/models`,
   `GET /v1/jobs/{id}`; job tracker state machine per spec §5; token relay node→coordinator→
   client over the node WSS.
3. Model store stub: shard manifest for a single full-model shard, hash-verified download
   to a node-local cache.
4. Simple API-key auth for client routes (spec §16).

Checkpoints:
```bash
uv run pytest Tests/integration/test_single_node_e2e.py -q   # coordinator+1 node: prompt →
   # ≥20 streamed SSE tokens, valid SSE framing, job COMPLETED, tokens>0 in job record
uv run python Sim/cluster.py --nodes 2 --chat     # manual smoke: interactive prompt answers
```

Do NOT: partitioning/multi-hop, scheduling beyond "pick the only node", failover logic.

---

### S5 — Distributed pipeline: partitioning + activation relay

**Goal:** one model **actually split across N nodes**; activations flow node→node via the
coordinator relay; numerical parity with single-node execution.
**Preconditions:** S4 green.

Tasks:
1. Shard export: per-layer safetensors files + `SHARD_MANIFEST` (content hashes);
   node-local cache with digest verification.
2. Partition table + throughput-proportional sizing (spec §8.1): stage sizes ∝ measured
   node throughput, min 1 layer; coordinator-measured throughput (not node-claimed).
3. Multi-hop relay: `ACTIVATION_RELAY` chunked frames, per-hop acks (sender buffers until
   ack — spec §13), per-stage progress in job tracker.
4. Entry-node dispatch of job descriptors with the full stage graph (spec §9.1–9.7).

Checkpoints:
```bash
uv run pytest Tests/integration/test_pipeline_parity.py -q   # TinyLlama on 4 sim nodes vs
   # single-node reference: max|Δlogits| < 1e-3 (fp32 CPU), identical token stream (greedy)
uv run pytest Tests/integration/test_stage_progress.py -q    # job record shows per-stage
   # latencies & token counts; stage timing matches spec §9 envelope expectations (±50%)
uv run python Sim/cluster.py --nodes 8 --chat    # manual: generation flows through 8 stages
```

Do NOT: dynamic re-placement, failover (fixed node set — kills crash the job, that is S7),
MoE, batching beyond trivial.

---

### S6 — Scoring-driven scheduling, top-K, backups, queueing

**Goal:** full spec §12: feasibility filter → score ranking → top-K → throughput-weighted
sizing → warm backups → admission control.
**Preconditions:** S5 green.

Tasks:
1. Scheduler as pure module over registry snapshots (unit-testable without I/O).
2. `K = clamp(ceil(L / layers_per_node_target), 8, 16)`; infeasible/insufficient pool →
   `WAIT` (HTTP 429 + retry-after); warm backup designation (1–2 next-ranked nodes).
3. Placement recompute on join/leave/DEGRADED (spec §12); events surfaced via `/v1/nodes`.
4. Backpressure: bounded queue, per-key rate limits.

Checkpoints:
```bash
uv run pytest Tests/unit/test_scheduler.py -q        # mixed synthetic pool → expected K,
                                                     # membership, sizing; min-VRAM exclusion works
uv run pytest Tests/integration/test_admission.py -q # 50 concurrent requests vs small pool:
                                                     # some 429 with retry-after, no crash, no deadlock
uv run pytest Tests/integration/test_placement_recompute.py -q  # node leaves → placement
                                                     # updated, new jobs use it, old jobs finish
```

Do NOT: periodic rebalancer (defer), MoE, multi-model.

---

### S7 — Fault tolerance & degraded mode

**Goal:** spec §13 end-to-end: detection, stage retry, backup reassignment, degraded
operation — proven by chaos tests.
**Preconditions:** S6 green.

Tasks:
1. Stage watchdog (deadline = 4 × p99 stage latency), retry with `attempt` counter
   (max 3), recompute from buffered prefix activations only (spec §13).
2. Reassignment: OFFLINE node's partitions → warm backups; in-flight jobs retry on new
   placement; queued jobs re-route.
3. Degraded mode: re-partition across remaining nodes when pool < K but ≥ `min_nodes`;
   clean rejection below it.
4. `Sim/chaos.py`: deterministic kill schedule (node index, time) for repeatable tests.

Checkpoints:
```bash
uv run python Sim/chaos.py --nodes 6 --kill-at 5s:node3 --expect-complete ; echo $?   # 0
uv run pytest Tests/integration/test_chaos_matrix.py -q   # scripted: kill 1 mid-job → completes
   # via retry/backup; kill 3 of 8 → degraded serving continues; kill below min_nodes → clean
   # 429/503 with status, no hang; zero duplicate ledger keys emitted (keys recorded from S5)
```

Do NOT: coordinator HA, cross-node KV migration, consensus.

---

### S8 — Accounting ledger

**Goal:** spec §15 ledger live: events persisted on completion, idempotent, credit function
applied, export API.
**Preconditions:** S7 green.

Tasks:
1. `LEDGER_EVENT` ingestion → append-only table keyed `(job_id, stage_idx, attempt)`;
   dedupe on replay/retry.
2. `flops_est` from coordinator FLOP tables (never node-claimed); cross-checks vs observed
   stage timings (spec §15 verification) with flagging to score penalties.
3. API: `GET /ledger/node/{id}`, `GET /ledger/summary`, `POST /ledger/export` (CSV/JSON).

Checkpoints:
```bash
uv run pytest Tests/integration/test_ledger_reconcile.py -q  # replay S7 chaos scenario:
   # Σ credits per node == expected from job records; retry storms produce exactly one
   # SUCCESS event per (job, stage); FAILED/RETRIED_AWAY outcomes weighted per spec §15
uv run pytest Tests/unit/test_credit.py -q                   # credit golden tests
```

Do NOT: payments, settlement, crypto, price configuration.

---

### S9 — Web client + deployment

**Goal:** public system: chat UI + dashboard on Netlify, coordinator reachable on the
public internet, one sim node attached.
**Preconditions:** S8 green; human provided hosting credentials (§4).

Tasks:
1. `Client/`: React 18 + Vite + TS — chat view (SSE streaming, markdown rendering),
   dashboard view (nodes, scores, placements, active jobs from `/v1/nodes`), API-key entry.
2. `Deploy/`: Netlify config (SPA fallback, env `VITE_API_URL`); coordinator deploy recipe
   (systemd unit for a VPS **or** Render service, env template, HTTPS assumed at platform).
3. CORS locked to the Netlify domain; rate limits on public API.

Checkpoints:
```bash
cd Client && npm ci && npm run build          # builds clean, type-check passes
npm run e2e                                    # local: browser chat streams from local cluster
# deployed: https://<site>.netlify.app chat streams from https://<coord-host>/v1/completions
#           dashboard shows ≥1 ONLINE node with live score
```

Do NOT: user accounts, TLS termination by hand, multi-coordinator setup.

---

### S10 — GPU pilot & measurement campaign

**Goal:** validate the spec's latency envelope (§9) on real hardware and produce the first
§17 environmental data.
**Preconditions:** S9 deployed; 2–4 real GPU nodes volunteered (scale to 8–16 after); model
licenses accepted (§4).

Tasks:
1. Onboard real nodes (agent install script in `Deploy/`); Qwen2.5-7B-Instruct fp16 sharded
   across the pool.
2. `Sim/benchmarks.py`: single-stream tok/s, batch-1/8 pipelined throughput, per-stage
   latency, per-hop RTT; compare measured vs §9 predictions.
3. Energy capture where available (GPU power telemetry → ledger `energy_kwh_est`);
   `kWh/1k tokens` per H5.
4. Write `Docs/pilot-report.md`: measured vs predicted table, deviations, spec amendments.

Checkpoints:
```bash
uv run python Sim/benchmarks.py --nodes <pilot> --model qwen2.5-7b --out Docs/pilot-data.json
# exit 0; report shows measured tok/s within 2× of §9 envelope prediction; every stage of the
# pipeline observed healthy ≥30 min under load
```

Do NOT: MoE, 8–16 node scale-out (only after report), pricing.

---

### S11 — (Stretch) MoE expert placement — OLMoE-1B-7B

**Goal:** expert-set placement (spec §8.2) with router-aware activation routing and parity.
**Preconditions:** S10 report reviewed; human approved the stretch.

Tasks: expert-set partition table entries; router-logits → expert-node routing with reduce
at the next dense stage; hot-expert statistics (replication deferred); parity + chaos
checkpoints as S5/S7 but for OLMoE.

Checkpoints:
```bash
uv run pytest Tests/integration/test_moe_parity.py -q   # OLMoE distributed vs single-node
                                                        # reference logits parity < 1e-3
uv run python Sim/chaos.py --model olmoe --kill-at 5s:node2 --expect-complete ; echo $?  # 0
```

---

## 7. Stage sizing & pacing guide

| Stage | Expected agent sessions | Risk focus |
|---|---|---|
| S0–S1 | 1–2 | toolchain correctness, protocol completeness |
| S2–S3 | 2–3 | async correctness, state machine fidelity |
| S4 | 1–2 | streaming plumbing end-to-end |
| S5 | 2–4 | the hardest stage: parity + relay backpressure |
| S6–S8 | 3–4 | scheduler/chaos/ledger invariants |
| S9 | 1–2 | deployment glue |
| S10 | 2+ | real-world measurement, spec validation |

If a stage exceeds 2× its budget, stop, write a short blockers note in `Docs/`, and
escalate rather than hacking through a constraint.
