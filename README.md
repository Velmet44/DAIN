# DAIN — Decentralized AI Inference Network

A coordinator-orchestrated inference network that partitions one large model
( dense or MoE ) across 8–16 independent, heterogeneous compute nodes.

- Specification: [Docs/spec.md](Docs/spec.md)
- Development stages: [Docs/stages.md](Docs/stages.md)
- Work log: [Docs/log.md](Docs/log.md)

## Repository layout

| Directory | Package | Role |
|---|---|---|
| `Common/` | `dain-common` | Wire protocol schemas, node scoring, FLOP/credit math (no I/O) |
| `Coordinator/` | `dain-coordinator` | Control-plane VPS: API, registry, scheduling, relay, ledger |
| `Node/` | `dain-node` | Compute-node agent: registration, heartbeats, executors |
| `Client/` | — | React + Vite + TypeScript web client (later stage) |
| `Sim/` | `dain-sim` | Local multi-process cluster simulation, chaos & benchmarks |
| `Tests/` | — | pytest unit & integration suites |
| `Deploy/` | — | systemd units, env templates, hosting notes |
| `Docs/` | — | Spec, stages, logs, reports |

## Quickstart (development)

Requires [uv](https://docs.astral.sh/uv/). All Python runs inside the uv-managed
venv (`.venv`, Python 3.12).

```bash
uv sync                          # create venv, install workspace + dev tools
uv run pytest                    # run the test suite
uv run ruff check .              # lint
uv run python -m dain_coordinator  # start coordinator on :8000
curl -s localhost:8000/healthz   # {"status":"ok"}
```

Nodes connect **outbound** to the coordinator (spec §10) — no inbound ports on
nodes, so NAT/home machines work unmodified.
