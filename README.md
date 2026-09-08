# DAIN — Decentralized AI Inference Network

A coordinator-orchestrated inference network that partitions one large model
( dense or MoE ) across 8–16 independent, heterogeneous compute nodes.

- Specification: [Docs/spec.md](Docs/spec.md)
- Development stages: [Docs/stages.md](Docs/stages.md)
- Work log: [Docs/log.md](Docs/log.md)

## Repository layout

The **root holds no buildable code** — only shared docs, config, and the project
directories. Each Python project is fully self-contained: its own `pyproject.toml`,
`uv.lock`, `.venv` (local, gitignored), `.python-version` (3.12), lint/test config,
and `tests/`.

| Path | What it is |
|---|---|
| `Docs/` | `spec.md` · `stages.md` · `log.md` · pilot reports |
| `Common/` | `dain-common` — wire-protocol schemas, node scoring, FLOP/credit math (no I/O) |
| `Coordinator/` | `dain-coordinator` — control-plane API (FastAPI): registry, scheduling, relay, ledger |
| `Node/` | `dain-node` — compute-node agent: registration, heartbeats, executors |
| `Sim/` | `dain-sim` — local multi-process cluster, chaos harness, benchmarks |
| `Client/` | React + Vite + TypeScript web client (arrives in stage S9) |
| `Deploy/` | env templates (`.env.example`), systemd units, hosting notes |
| `Builds/` | local build artifacts (gitignored) |

`Coordinator`, `Node`, and `Sim` depend on `dain-common` through an **editable path
dependency** (`../Common`) — one source of truth for the protocol, no root workspace.
Cross-component integration tests live in `Sim/tests/`; unit tests live with their owner.

## Working with the projects

Requires [uv](https://docs.astral.sh/uv/). All commands run **inside the project
directory**:

```bash
cd Coordinator
uv sync                            # create/update .venv from uv.lock
uv run pytest                      # this project's tests
uv run ruff check .                # lint
uv run python -m dain_coordinator  # serve the API on :8000
curl -s localhost:8000/healthz     # {"status":"ok"}
```

The same pattern works in `Common`, `Node`, and `Sim`. To drive a full local cluster
(coordinator + N node agents on one machine, CPU model):

```bash
cd Sim && uv run python -m dain_sim.cluster --nodes 4 --duration 60
```

Nodes connect **outbound** to the coordinator (spec §10) — nodes need no inbound
ports, so home/NAT machines work unmodified.
