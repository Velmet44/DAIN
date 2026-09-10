param(
    [Parameter(Mandatory=$true)]
    [string]$Root
)

$ErrorActionPreference = 'Stop'

$excludedDir = @(
    '__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache',
    '.venv', 'node_modules', '.git', 'build', 'dist',
    '.package_stage', '.egg-info'
)
$excludedDirLike = @('shard_cache*')
$excludedFileLike = @(
    'node_state*.json',
    'DAIN*.zip',
    '*.sqlite3', '*.sqlite3-shm', '*.sqlite3-wal',
    '*.pyc', '*.pyo', '*.spec', '*.log',
    '.DS_Store', 'Thumbs.db', 'desktop.ini'
)

Write-Host ""
Write-Host "============================================"
Write-Host "  DAIN Package"
Write-Host "  Root: $Root"
Write-Host "============================================"
Write-Host ""

$zipPath = Join-Path $Root "DAIN.zip"
$index = 1
while (Test-Path -LiteralPath $zipPath) {
    $zipPath = Join-Path $Root ("DAIN{0}.zip" -f $index)
    $index++
}
Write-Host "  Target: $zipPath"
Write-Host ""

$stage = Join-Path $env:TEMP "dain_package_stage"
if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

$script:fileCount = 0
$script:dirCount = 0

function Copy-Tree {
    param([string]$Src, [string]$Dst)
    Get-ChildItem -LiteralPath $Src -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.PSIsContainer) {
            $skip = $excludedDir -contains $_.Name
            if (-not $skip) {
                foreach ($p in $excludedDirLike) {
                    if ($_.Name -like $p) { $skip = $true; break }
                }
            }
            if ($skip) { return }
            $newDst = Join-Path $Dst $_.Name
            New-Item -ItemType Directory -Path $newDst -Force | Out-Null
            $script:dirCount++
            Copy-Tree $_.FullName $newDst
        } else {
            foreach ($p in $excludedFileLike) {
                if ($_.Name -like $p) { return }
            }
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Dst $_.Name) -Force
            $script:fileCount++
        }
    }
}

Write-Host "  Staging files..."
Copy-Tree $Root $stage
Write-Host "    staged $script:fileCount files, $script:dirCount directories"
Write-Host ""
Write-Host "  Creating archive..."

Add-Type -AssemblyName System.IO.Compression.FileSystem
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
[System.IO.Compression.ZipFile]::CreateFromDirectory($stage, $zipPath)

Remove-Item -LiteralPath $stage -Recurse -Force

$sizeMB = [math]::Round((Get-Item -LiteralPath $zipPath).Length / 1MB, 2)
Write-Host "  Created: $zipPath ($sizeMB MB, $script:fileCount files)"
Write-Host ""
Write-Host "  Excluded: caches, .venv, node_modules, build/, dist/,
              shard_cache*, node_state*.json, sqlite databases,
              .git, previous DAIN*.zip"
Write-Host "  Note: on the target PC run 'Scripts\bootstrap.ps1' to
              recreate .venv and node_modules."
Write-Host ""