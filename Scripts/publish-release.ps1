# publish-release.ps1 — Build the node distribution and publish it to GitHub Releases.
#
#   powershell -ExecutionPolicy Bypass -File Scripts\publish-release.ps1 -Version 0.2.0
#   powershell -ExecutionPolicy Bypass -File Scripts\publish-release.ps1 -Version 0.2.0 -Notes "Bug fixes"
#   powershell -ExecutionPolicy Bypass -File Scripts\publish-release.ps1 -Version 0.2.0 -Draft
#
# Prerequisites:
#   - gh CLI authenticated (run: gh auth login)
#   - Node/dain_node/__init__.py __version__ matches -Version (the script updates it)

param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [string]$Title = "",

    [string]$Notes = "",

    [switch]$Draft,

    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# ── 0. Preflight ─────────────────────────────────────────────────────────
foreach ($cmd in @("gh", "git")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "$cmd not found. Install it first." -ForegroundColor Red
        exit 1
    }
}

$tag = "v$Version"
Write-Host "== publish DainNode $tag ==" -ForegroundColor Cyan

# ── 1. Update __version__ in __init__.py ──────────────────────────────────
$initPy = Join-Path $root "Node\dain_node\__init__.py"
$pattern = '__version__\s*=\s*"([^"]+)"'
$match = Select-String -Path $initPy -Pattern $pattern
if (-not $match) {
    Write-Host "Cannot read version from $initPy" -ForegroundColor Red
    exit 1
}
$currentVer = $match.Matches[0].Groups[1].Value
if ($currentVer -ne $Version) {
    Write-Host "  Updating __init__.py: $currentVer -> $Version"
    $content = Get-Content -LiteralPath $initPy -Raw
    $content = $content -replace $pattern, "__version__ = `"$Version`""
    Set-Content -LiteralPath $initPy -Value $content -NoNewline -Encoding UTF8
    # Also update pyproject.toml version field
    $pyproject = Join-Path $root "Node\pyproject.toml"
    if (Test-Path -LiteralPath $pyproject) {
        $ptContent = Get-Content -LiteralPath $pyproject -Raw
        $ptContent = $ptContent -replace 'version\s*=\s*"[^"]*"', "version = `"$Version`""
        Set-Content -LiteralPath $pyproject -Value $ptContent -NoNewline -Encoding UTF8
    }
} else {
    Write-Host "  __init__.py already at $Version"
}

# ── 2. Build the zip (unless skipped) ────────────────────────────────────
if (-not $SkipBuild) {
    Write-Host "`n  Building distribution..."
    $buildScript = Join-Path $PSScriptRoot "build-node.ps1"
    & powershell -ExecutionPolicy Bypass -File $buildScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed" -ForegroundColor Red
        exit 1
    }
}

# Locate the zip
$buildsDir = Join-Path $root "Builds"
$zipPath = Join-Path $buildsDir "DainNode-v$Version.zip"
if (-not (Test-Path -LiteralPath $zipPath)) {
    Write-Host "Expected zip not found: $zipPath" -ForegroundColor Red
    Write-Host "Run without -SkipBuild, or run build-node.ps1 first." -ForegroundColor Yellow
    exit 1
}
$sizeMB = [math]::Round((Get-Item -LiteralPath $zipPath).Length / 1MB, 1)
Write-Host "  Zip: $zipPath ($sizeMB MB)"

# ── 3. Git tag + push ────────────────────────────────────────────────────
Write-Host "`n  Creating git tag $tag ..."
$existingTag = git tag -l $tag 2>$null
if ($existingTag -eq $tag) {
    Write-Host "  Tag $tag already exists, deleting remote tag..."
    git tag -d $tag
    git push origin :refs/tags/$tag
}
git tag -a $tag -m "DainNode $Version"
git push origin $tag

# ── 4. Build release notes ───────────────────────────────────────────────
if (-not $Title) { $Title = "DainNode $Version" }
if (-not $Notes) {
    $Notes = @"
## DainNode v$Version

### Distribution

Download **DainNode-v$Version.zip**, extract, edit \`config.json\`, then run \`DainNode.exe\`.

| Field | Description |
|---|---|
| \`coord_url\` | WebSocket URL of the coordinator (e.g. \`ws://your-coordinator:8000\`) |
| \`join_token\` | Token to register with the coordinator |
| \`model\` | Model to load (e.g. \`dain-tiny-16L\`) |
| \`cache_dir\` | Where to store model shards |
| \`heartbeat_s\` | Heartbeat interval in seconds |

Editing \`config.json\` while the node is running triggers an automatic restart with the new settings.

### Changes

$Notes
"@
}

# ── 5. Create GitHub release ─────────────────────────────────────────────
Write-Host "`n  Creating GitHub release $tag ..."
$releaseArgs = @(
    "release", "create", $tag,
    $zipPath,
    "--title", $Title,
    "--notes", $Notes,
    "--repo", (git remote get-url origin)
)
if ($Draft) {
    $releaseArgs += "--draft"
}
$releaseArgs += "--generate-notes"

& gh @releaseArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create GitHub release" -ForegroundColor Red
    exit 1
}

$repoUrl = git remote get-url origin
Write-Host ""
Write-Host "  Published!" -ForegroundColor Green
Write-Host "  Release: $repoUrl/releases/tag/$tag"
Write-Host ""
