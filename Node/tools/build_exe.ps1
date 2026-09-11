# Builds the node agent into a standalone Windows bundle with PyInstaller.
#
#   powershell -ExecutionPolicy Bypass -File Node\tools\build_exe.ps1
#   # optional: -Mode onedir  (folder of exe + libs, faster startup)
#
# Output: Node\dist\dain-node.exe             (onefile: single .exe)
#         Node\dist\dain-node\dain-node.exe   (onedir: folder of exe + libs)
#
# The onefile bundle is fully self-contained (torch, transformers, node agent,
# shared dain_common package); the target PC needs nothing but the .exe. It is
# safe to copy into OneDrive-synced or network folders — content is extracted
# to %TEMP% at launch, so no directory tree gets dehydrated. The single .exe is
# slower to start than onedir (extracts torch + transformers each launch). No
# absolute paths are embedded, so the result runs from any location.

param(
    [ValidateSet("onedir", "onefile")]
    [string]$Mode = "onefile"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot   # Node/
$here = $PSScriptRoot                      # Node/tools/

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found - install: https://astral.sh/uv" -ForegroundColor Red
    exit 1
}

Write-Host "== building dain-node ($Mode) ==" -ForegroundColor Cyan

# Ensure the interpreter + dev deps (pyinstaller) are ready.
& uv sync --project $root --quiet
if ($LASTEXITCODE -ne 0) { exit 1 }

$flag = if ($Mode -eq "onefile") { "--onefile" } else { "--onedir" }

# Anchors the build: PyInstaller writes build/, dist/ and the .spec next to the
# *caller's* working directory by default, so running from Scripts/ would scatter
# output there. Pin every output path to the Node/ project root instead.
Push-Location $root
try {
# --paths $root keeps dain_common importable from physical source regardless of
# how the editable dependency is laid out in site-packages.
    &  uv run --project $root python -m PyInstaller `
        --noconfirm --clean `
        $flag `
        --name dain-node `
        --paths $root `
        --collect-all transformers `
        --distpath "$root\dist" `
        --workpath "$root\build" `
        --specpath "$root" `
        "$here\node_entry.py"
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
finally {
    Pop-Location
}

if ($Mode -eq "onedir") {
    Write-Host "OK: $root\dist\dain-node\dain-node.exe" -ForegroundColor Green
} else {
    Write-Host "OK: $root\dist\dain-node.exe" -ForegroundColor Green
}