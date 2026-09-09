#!/usr/bin/env bash
# DAIN bootstrap — rebuild the full dev environment from a fresh copy (macOS/Linux).
#
#     ./bootstrap.sh            # python envs + client deps + smoke test
#     ./bootstrap.sh -m <dir>   # also export tiny-llama shards into <dir>
#
# Prerequisites (not installed): uv  <https://astral.sh/uv>, Node.js 18+.
# Idempotent; resolves only from committed lockfiles (no absolute paths baked in).

set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_store=""
while getopts "m:" opt; do
  case "$opt" in
    m) model_store="$OPTARG" ;;
    *) exit 1 ;;
  esac
done

command -v uv >/dev/null || { echo "uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
command -v npm >/dev/null || { echo "npm not found — install Node.js 18+ (https://nodejs.org)" >&2; exit 1; }

for pkg in Common Node Coordinator Sim; do
  echo "uv sync: $pkg"
  (cd "$root/$pkg" && uv sync --quiet)
done

echo "npm ci: Client"
(cd "$root/Client" && npm ci && npm run build)

echo "smoke test: imports"
uv run --project "$root/Sim" python -c "import dain_sim.cluster, dain_coordinator.app, dain_node.agent, dain_common; print('imports OK')"

if [ -n "$model_store" ]; then
  echo "export tiny llama -> $model_store"
  uv run --project "$root/Node" python -c "from dain_node.shard_export import export_tiny_llama; export_tiny_llama(r'$model_store')"
fi

echo "== bootstrap OK — environment is self-contained =="