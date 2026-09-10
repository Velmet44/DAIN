# Import GGUF models into the DAIN model store (session 14).
#
# Converts every Llama-family .gguf sitting in the model store that has not
# been imported yet (or a single file via -GgufFile) into DAIN's sharded
# safetensors + manifest format. Weights are dequantized to fp16/fp32 — DAIN
# does not execute GGUF quants. Already-imported files are skipped unless
# -Force. The same converter is reachable from the admin page's Models card.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File Scripts\import-gguf.ps1
#   powershell ... -GgufFile model_store\Llama-3.2-1B.Q4_K_M.gguf -Dtype fp16
#   powershell ... -GgufFile model.3.2-1B.gguf -ModelId llama-3.2-1b -Tokenizer meta-llama/Llama-3.2-1B
[CmdletBinding()]
param(
    [string]$GgufFile = "",
    [string]$ModelStore = "",
    [string]$ModelId = "",
    [ValidateSet("fp16", "fp32")]
    [string]$Dtype = "fp16",
    [int]$LayersPerShard = 4,
    [string]$Tokenizer = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$script:Root = Split-Path -Parent $PSScriptRoot

if (-not $ModelStore) { $ModelStore = Join-Path $script:Root "model_store" }
$nodeDir = Join-Path $script:Root "Node"
if (-not (Test-Path -LiteralPath (Join-Path $nodeDir "pyproject.toml"))) {
    throw "Node project not found at $nodeDir - run this from a full DAIN checkout."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found on PATH - install it from https://docs.astral.sh/uv/."
}
if (-not (Test-Path -LiteralPath $ModelStore)) {
    New-Item -ItemType Directory -Path $ModelStore | Out-Null
    Write-Host "created model store: $ModelStore"
}

$cliArgs = @("run", "--project", $nodeDir, "python", "-m", "dain_node.import_gguf", $ModelStore)
if ($GgufFile) {
    $resolved = Resolve-Path -LiteralPath $GgufFile -ErrorAction Stop
    $cliArgs += $resolved.Path
}
if ($ModelId)    { $cliArgs += @("--model-id", $ModelId) }
if ($Tokenizer)  { $cliArgs += @("--tokenizer", $Tokenizer) }
if ($Force)      { $cliArgs += "--force" }
$cliArgs += @("--dtype", $Dtype, "--layers-per-shard", $LayersPerShard)

Write-Host "DAIN GGUF import: store=$ModelStore dtype=$Dtype"
& uv @cliArgs
if ($LASTEXITCODE -ne 0) {
    throw "import failed (uv exit code $LASTEXITCODE)"
}
