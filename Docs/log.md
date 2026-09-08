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
