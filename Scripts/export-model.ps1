# Export a local HuggingFace model to DAIN TorchAO-INT4 shards (session N).
#
# Quantizes the source directory into INT4-weight-only DAIN shards and writes
# them into the model store alongside a manifest.json and tokenizer.json. The
# node runs as a subprocess so the Coordinator never imports torch/torchao.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File Scripts\export-model.ps1 `
#       -SourceDir C:\models\Llama-3.1-8B -ModelId llama-3.1-8b-int4
#   powershell ... -SourceDir .\src -ModelId tiny -ActivationDtype bf16 -Force
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$SourceDir,
    [Parameter(Mandatory=$true)]
    [string]$ModelId,
    [string]$ModelStore = "",
    [ValidateSet("int4")]
    [string]$Quantization = "int4",
    [int]$GroupSize = 128,
    [ValidateSet("fp16", "bf16")]
    [string]$ActivationDtype = "fp16",
    [int]$LayersPerShard = 4,
    [switch]$Force,
    [switch]$DryRun
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

$resolved = Resolve-Path -LiteralPath $SourceDir -ErrorAction Stop
if (-not (Test-Path -LiteralPath $resolved.Path)) {
    throw "Source directory not found: $resolved"
}

$cliArgs = @(
    "run", "--project", $nodeDir, "python", "-m", "dain_node.model_export",
    "--source-dir", $resolved.Path,
    "--model-id", $ModelId,
    "--output-store", $ModelStore,
    "--quantization", $Quantization,
    "--group-size", $GroupSize,
    "--activation-dtype", $ActivationDtype,
    "--layers-per-shard", $LayersPerShard,
    "--json-progress"
)
if ($Force)      { $cliArgs += "--force" }
if ($DryRun)     { $cliArgs += "--dry-run" }

Write-Host "DAIN model export: source=$resolved model_id=$ModelId quant=$Quantization dtype=$ActivationDtype"
& uv @cliArgs
if ($LASTEXITCODE -ne 0) {
    throw "export failed (uv exit code $LASTEXITCODE)"
}
