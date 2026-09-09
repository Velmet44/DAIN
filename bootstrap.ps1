# DAIN bootstrap — rebuild the full dev environment from a fresh copy.
#
# Run from anywhere inside the repo (it anchors on this file's location):
#     powershell -ExecutionPolicy Bypass -File DAIN\bootstrap.ps1
# Use -ModelStore <dir> to also export the tiny-llama shards into a folder
# that a coordinator's DAIN_MODEL_STORE_DIR can point at.
#
# Prerequisites on the machine (not installed by this script):
#   - Python-free: uv  (https://astral.sh/uv)  — drives all Python envs
#   - Node.js 18+    (https://nodejs.org)      — drives the web client
#   - git            (optional, only for updating)
#
# Idempotent: safe to re-run; everything resolves from committed lockfiles so
# no machine-specific absolute paths are ever baked in.

param(
    [string]$ModelStore = ""
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "== DAIN bootstrap ==" -ForegroundColor Cyan

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Install it first:" -ForegroundColor Red
    Write-Host "  powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`""
    exit 1
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "npm not found. Install Node.js 18+ first: https://nodejs.org" -ForegroundColor Red
    exit 1
}

# --- Python envs (uv sync reads .python-version + uv.lock: exact, portable) ---
foreach ($pkg in "Common", "Node", "Coordinator", "Sim") {
    Write-Host "uv sync: $pkg" -ForegroundColor Yellow
    & uv sync --project (Join-Path $root $pkg) --quiet
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

# --- Web client (npm ci reads package-lock.json: exact, portable) ---
Write-Host "npm ci: Client" -ForegroundColor Yellow
Push-Location (Join-Path $root "Client")
try {
    & npm ci
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & npm run build
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
finally {
    Pop-Location
}

# --- Smoke test: every package importable from the Sim environment (which
#     depends on all four as editable path packages) + build a tiny model ---
Write-Host "smoke test: imports" -ForegroundColor Yellow
& uv run --project (Join-Path $root "Sim") python -c "import dain_sim.cluster, dain_coordinator.app, dain_node.agent, dain_common; print('imports OK')"
if ($LASTEXITCODE -ne 0) { exit 1 }

if ($ModelStore) {
    Write-Host "export tiny llama -> $ModelStore" -ForegroundColor Yellow
    $quote = $ModelStore.Replace("'", "''")
    & uv run --project (Join-Path $root "Node") python -c "from dain_node.shard_export import export_tiny_llama; export_tiny_llama(r'$quote')"
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

Write-Host "== bootstrap OK — environment is self-contained ==" -ForegroundColor Green