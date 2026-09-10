# build-node.ps1 — Build the DAIN Node standalone distribution.
#
#   powershell -ExecutionPolicy Bypass -File Scripts\build-node.ps1
#
# Produces a self-contained folder and zip that users can extract and run:
#
#   Builds/DainNode-v0.1.0/
#     DainNode.exe
#     _internal/
#     config.json        (default settings — user edits this)
#     shard_cache/       (empty, created by the node on first job)
#
#   Builds/DainNode-v0.1.0.zip

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot   # repo root

# ── 1. Read version ──────────────────────────────────────────────────────
$initPy = Join-Path $root "Node\dain_node\__init__.py"
$match = Select-String -Path $initPy -Pattern '__version__\s*=\s*"([^"]+)"'
if (-not $match) {
    Write-Host "Cannot read version from $initPy" -ForegroundColor Red
    exit 1
}
$version = $match.Matches[0].Groups[1].Value
Write-Host "== building DainNode v$version ==" -ForegroundColor Cyan

# ── 2. Build the exe via PyInstaller ─────────────────────────────────────
$buildScript = Join-Path $root "Node\tools\build_exe.ps1"
& powershell -ExecutionPolicy Bypass -File $buildScript
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller build failed" -ForegroundColor Red
    exit 1
}

# ── 3. Assemble distribution folder ──────────────────────────────────────
$distSrc  = Join-Path $root "Node\dist\dain-node"
$buildsDir = Join-Path $root "Builds"
if (-not (Test-Path -LiteralPath $buildsDir)) {
    New-Item -ItemType Directory -Path $buildsDir -Force | Out-Null
}
$distName = "DainNode-v$version"
$distDir  = Join-Path $buildsDir $distName

if (Test-Path -LiteralPath $distDir) {
    Remove-Item -LiteralPath $distDir -Recurse -Force
}
Write-Host "  Assembling $distDir ..."
Copy-Item -LiteralPath $distSrc -Destination $distDir -Recurse

# ── 4. Rename exe ────────────────────────────────────────────────────────
$oldExe = Join-Path $distDir "dain-node.exe"
$newExe = Join-Path $distDir "DainNode.exe"
if (Test-Path -LiteralPath $oldExe) {
    Rename-Item -LiteralPath $oldExe -NewName "DainNode.exe"
}

# ── 5. Write default config.json ─────────────────────────────────────────
$configObj = [ordered]@{
    coord_url    = "ws://localhost:8000"
    join_token   = "dain-dev-join-token"
    node_id      = ""
    heartbeat_s  = 5
    model        = ""
    cache_dir    = "shard_cache"
    state_path   = "node_state.json"
    net_bw_mbps  = 100
    net_lat_ms   = 50
    log_json     = $false
}
$configPath = Join-Path $distDir "config.json"
$configObj | ConvertTo-Json -Depth 4 | Set-Content -Path $configPath -Encoding UTF8
Write-Host "  Created config.json"

# ── 6. Create empty shard_cache ──────────────────────────────────────────
$cacheDir = Join-Path $distDir "shard_cache"
New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null

# ── 7. Zip ───────────────────────────────────────────────────────────────
$zipPath = Join-Path $buildsDir "$distName.zip"
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($distDir, $zipPath)
Remove-Item -LiteralPath $distDir -Recurse -Force

$sizeMB = [math]::Round((Get-Item -LiteralPath $zipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "  Built DainNode v$version" -ForegroundColor Green
Write-Host "  Zip: $zipPath ($sizeMB MB)"
Write-Host ""
Write-Host "  Users: extract the zip, edit config.json, run DainNode.exe"
Write-Host ""
