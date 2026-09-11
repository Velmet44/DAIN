# DAIN — Decentralized AI Inference Network

A coordinator-orchestrated inference network that partitions one large model
( dense or MoE ) across 8–16 independent, heterogeneous compute nodes.

- Specification: [Docs/spec.md](Docs/spec.md)
- Development stages: [Docs/stages.md](Docs/stages.md)
- Work log: [Docs/log.md](Docs/log.md)

## Quick start (same PC, web interface)

Requires [uv](https://docs.astral.sh/uv/) and [Node.js 18+](https://nodejs.org).

### Option A: One script

Run **`Scripts\start-samepc.ps1`**:

```powershell
powershell -ExecutionPolicy Bypass -File Scripts\start-samepc.ps1
```

It opens the coordinator, N node agents, and the web client (as tabs in the
same Windows Terminal window if you launch it from there, or in separate windows
otherwise) and prompts for API keys when asked (Enter accepts the dev defaults).
Stop everything with **Ctrl+C** in each window.

### Option B: Manual (3 terminals)

```bash
# 1. Bootstrap everything (one-time)
powershell -ExecutionPolicy Bypass -File Scripts\bootstrap.ps1

# 2. Export the dev model (one-time)
cd Node && uv run python -c "from dain_node.shard_export import export_tiny_llama; export_tiny_llama('../model_store')"

# 3. Start coordinator (terminal 1)
cd Coordinator
$env:DAIN_MODEL_STORE_DIR="../model_store"
uv run python -m dain_coordinator

# 4. Start node agent (terminal 2)
cd Node
$env:DAIN_MODEL_STORE_DIR="../model_store"
uv run python -m dain_node

# 5. Start web client (terminal 3)
cd Client
npm run dev
```

Open **http://localhost:5173/DAIN/** — type a message in the Chat tab.

### Public client (GitHub Pages)

**https://velmet44.github.io/DAIN/** — static build of the same SPA. Enter a
coordinator URL in the Chat tab to connect (default `http://127.0.0.1:8000`; with no
public coordinator, point it at a reachable home/VPS coordinator).

The `deploy` workflow bakes the coordinator origin into the bundle and **fails
the build** when it isn't configured — it never deploys a site pointing at
`127.0.0.1`. Before delivery, set (repo Settings → Secrets and variables → Actions):

- **Variable `DAIN_API_URL`** — public coordinator origin, must be `https://` for Pages;
- **Secret `DAIN_API_KEY`** — the `DAIN_API_KEY` that coordinator runs with
  (a static site exposes it; prototype posture);
- optionally **`DAIN_MODEL_ID`** (default `dain-tiny-16L`) and **`DAIN_DEFAULT_MAX_TOKENS`**.

Or for a one-command cluster (no browser, chat REPL):

```bash
cd Sim && uv run python -m dain_sim.cluster --nodes 4 --chat
```

### Importing GGUF models

Drop any **Llama-architecture** `.gguf` file into `model_store/` and convert it to
DAIN's sharded safetensors + manifest format:

```powershell
powershell -ExecutionPolicy Bypass -File Scripts\import-gguf.ps1
```

That imports every pending `.gguf` in the store (pass `-GgufFile <file>`,
`-ModelId`, `-Dtype fp16|fp32`, `-Tokenizer <hf-repo>` to customize). It is
idempotent — already-imported files are skipped (`-Force` re-imports). The same
converter is available from the coordinator's admin page (**Models → GGUF files**),
which shells out to the Node converter without pulling torch into the
coordinator. Note that DAIN executes fp16/fp32, not GGUF quants: a Q4 7B
(~4 GB) becomes ~14 GB of fp16 shards.

### Exporting INT4 models (TorchAO)

Point DAIN at a **local HuggingFace Llama checkpoint** to export TorchAO INT4 shards.
Quantized shards are `.pt` files with an `int4_cpu` packing layout; the scheduler
only places them on nodes whose software advertises torchao + that layout, so a
node without torchao never reaches a packed model's shards:

```powershell
powershell -ExecutionPolicy Bypass -File Scripts\export-model.ps1 `
  -SourceDir C:\models\Llama-3.1-8B -ModelId llama-3.1-8b-int4
```

Pass `-ActivationDtype bf16`, `-GroupSize`, `-LayersPerShard` to customize
(`-DryRun` validates without writing). The same exporter is available from the
admin page (**Model Export** card) — `source_dir` must resolve under an
approved `export_roots` (or `DAIN_EXPORT_ROOTS`, semicolon-separated). The
exporter writes `manifest.json` + `tokenizer.json` into the store model dir and
runs as a subprocess so the coordinator never imports torch/torchao.

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
| `Client/` | React + Vite + TypeScript web client (S9) |
| `Deploy/` | env templates (`.env.example`), Caddyfile, caddy binary |
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
