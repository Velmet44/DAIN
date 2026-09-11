# build-node.ps1 — Build the DAIN Node standalone distribution.
#
#   powershell -ExecutionPolicy Bypass -File Scripts\build-node.ps1
#
# Produces a self-contained folder and zip that users can extract and run:
#
#   Builds/DainNode-v0.1.0/
#     DainNode.exe          (single-file PyInstaller onefile build)
#     config.json           (default settings — user edits this)
#     shard_cache/          (empty, created by the node on first job)
#
#   Builds/DainNode-v0.1.0.zip
#
# The onefile exe extracts to %TEMP% at launch, so it is safe in OneDrive or
# network folders (no directory tree to dehydrate).

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
# onefile build → a single exe in dist/ (no folder, no _internal tree).
$distSrc  = Join-Path $root "Node\dist\dain-node.exe"
$buildsDir = Join-Path $root "Builds"
if (-not (Test-Path -LiteralPath $buildsDir)) {
    New-Item -ItemType Directory -Path $buildsDir -Force | Out-Null
}
$distName = "DainNode-v$version"
$distDir  = Join-Path $buildsDir $distName

if (Test-Path -LiteralPath $distDir) {
    Remove-Item -LiteralPath $distDir -Recurse -Force
}
New-Item -ItemType Directory -Path $distDir -Force | Out-Null
Write-Host "  Assembling $distDir ..."
Copy-Item -LiteralPath $distSrc -Destination (Join-Path $distDir "DainNode.exe")

# ── 5. Write default config.json ─────────────────────────────────────────
# coord_url empty = auto-discover the coordinator on the LAN (first run probes
# a UDP broadcast; the node fills in ws://<coordinator-ip>:<port> itself).
$configObj = [ordered]@{
    coord_url    = ""
    join_token   = "Jj3L7ewD"
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
# BOM-free UTF-8 (PS 5.1 -Encoding UTF8 would write a BOM, which breaks every
# json loader in the app — config.py reads with utf-8-sig, but stay clean anyway).
$json = $configObj | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText($configPath, $json, (New-Object System.Text.UTF8Encoding($false)))
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
Write-Host "  Users: extract the zip and run DainNode.exe (single file + config.json)"
Write-Host "         it auto-discovers the coordinator on the LAN (join_token must match);"
Write-Host "         edit config.json only for remote/WAN use."
Write-Host ""
