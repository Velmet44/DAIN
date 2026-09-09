param(
    [ValidateSet("launcher", "coordinator", "node", "client", IgnoreCase = $true)]
    [string]$Mode = "launcher",
    [int]$NodeIndex = 1
)

# ============================================================================
#  DAIN single-file launcher for local (same-PC) testing.
#
#  Fully self-contained: every path derives from this file's own location, so
#  the whole repo folder can be copied anywhere (or to another Windows PC)
#  and still work — there are no absolute paths baked in.
#
#  Run it three ways:
#    powershell -ExecutionPolicy Bypass -File Scripts\start.ps1               # launcher (default)
#    Double-click the file in Explorer (edit association as needed)            # launcher
#    powershell -File Scripts\start.ps1 -Mode coordinator                      # one component
#
#  The launcher starts the coordinator, N node agents, and the web client —
#  each in its own window/tab. Stop everything with Ctrl+C in each window
#  (or taskkill /IM python.exe /F as a last resort).
# ============================================================================

$ErrorActionPreference = "Stop"

# --- Self-contained root: parent of the Scripts folder this file lives in ---
$script:ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:Root = Split-Path -Parent $script:ScriptsDir
$script:ModelStore = Join-Path $script:Root "model_store"

function Exec-Component {
    param(
        [string]$Mode,
        [int]$NodeIndex
    )

    switch ($Mode.ToLower()) {
        "coordinator" {
            if (-not $env:DAIN_MODEL_STORE_DIR) { $env:DAIN_MODEL_STORE_DIR = $script:ModelStore }
            Write-Host "=== DAIN Coordinator ===" -ForegroundColor Cyan
            Push-Location (Join-Path $script:Root "Coordinator")
            try { uv run python -m dain_coordinator }
            finally { Pop-Location }
        }
        "node" {
            if (-not $env:DAIN_MODEL_STORE_DIR) { $env:DAIN_MODEL_STORE_DIR = $script:ModelStore }
            if (-not $env:DAIN_NODE_ID) { $env:DAIN_NODE_ID = "node-$NodeIndex" }
            if (-not $env:DAIN_MODEL) { $env:DAIN_MODEL = "dain-tiny-16L" }
            Write-Host "=== DAIN Node $($env:DAIN_NODE_ID) ===" -ForegroundColor Green
            Push-Location (Join-Path $script:Root "Node")
            try { uv run python -m dain_node }
            finally { Pop-Location }
        }
        "client" {
            if (-not $env:VITE_API_URL) { $env:VITE_API_URL = "http://localhost:8000" }
            if (-not $env:VITE_API_KEY) { $env:VITE_API_KEY = "dain-dev-key" }
            Write-Host "=== DAIN Web Client ===" -ForegroundColor Magenta
            Push-Location (Join-Path $script:Root "Client")
            try {
                if (-not (Test-Path (Join-Path $PWD "node_modules"))) { npm ci }
                npm run dev
            }
            finally { Pop-Location }
        }
    }
    if ($Mode.ToLower() -in @("coordinator", "node")) {
        Write-Host "`nPress Enter to close this window..." -ForegroundColor DarkGray
        Read-Host
    }
}

# --- Direct component mode: run this process as one component ---
if ($Mode.ToLower() -ne "launcher") {
    Exec-Component -Mode $Mode -NodeIndex $NodeIndex
    exit 0
}

# --- Launcher mode -----------------------------------------------------------

Write-Host ""
Write-Host " === DAIN Local Dev Launcher ===" -ForegroundColor Blue
Write-Host ""

# Prerequisites
foreach ($tool in @("uv", "node")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Host "[ERROR] $tool not found." -ForegroundColor Red
        Write-Host "  Run Scripts\bootstrap.ps1 first (installs deps), then retry."
        Read-Host "Press Enter to exit..."
        exit 1
    }
}

# Optional keys (Enter = defaults)
$adminKey = Read-Host "  Admin API key     [dain-dev-admin-key]"
if (-not $adminKey) { $adminKey = "dain-dev-admin-key" }

$clientKey = Read-Host "  Client API key    [dain-dev-key]"
if (-not $clientKey) { $clientKey = "dain-dev-key" }

$joinToken = Read-Host "  Join token        [dain-dev-join-token]"
if (-not $joinToken) { $joinToken = "dain-dev-join-token" }

$numNodes = Read-Host "  Number of nodes   [1]"
if (-not $numNodes) { $numNodes = 1 }

Write-Host ""
Write-Host "  Starting cluster with:" -ForegroundColor Gray
Write-Host "    Admin key:      $adminKey"
Write-Host "    Client API key: $clientKey"
Write-Host "    Join token:     $joinToken"
Write-Host "    Nodes:          $numNodes"
Start-Sleep -Seconds 2

# Shared env (inherited by every child process we spawn)
$env:DAIN_ADMIN_API_KEY = $adminKey
$env:DAIN_API_KEY = $clientKey
$env:DAIN_JOIN_TOKEN = $joinToken
$env:DAIN_MODEL_STORE_DIR = $script:ModelStore

# Spawn a component in its own window (or a tab when running inside Windows Terminal)
function Start-Component {
    param(
        [string]$Title,
        [string]$Mode,
        [int]$NodeIndex = 1
    )
    $childArgs = @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$($MyInvocation.MyCommand.Path)`"",
        "-Mode", $Mode,
        "-NodeIndex", "$NodeIndex"
    )
    if ($env:WT_SESSION) {
        # Inside Windows Terminal? Add a tab to THIS window.
        wt -w 0 nt --title $Title -- powershell $childArgs
    } else {
        Start-Process powershell -ArgumentList $childArgs
    }
}

# 1) Coordinator
Start-Component -Title "DAIN Coordinator" -Mode "coordinator"

# 2) Wait for it, discover its port (coordinator auto-picks a free one)
Write-Host "`nWaiting for coordinator and detecting its port..." -ForegroundColor Gray
$coordPort = $null
$ports = @(8000, 8001, 8002, 8080, 8888, 9000)
for ($try = 0; $try -lt 25 -and -not $coordPort; $try++) {
    foreach ($p in $ports) {
        try {
            $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 "http://localhost:$p/healthz"
            $coordPort = $p
            break
        } catch { }
    }
    if (-not $coordPort) { Start-Sleep -Seconds 1 }
}
if (-not $coordPort) {
    Write-Host "[WARN] Coordinator not detected after 25s, assuming port 8000." -ForegroundColor Yellow
    $coordPort = 8000
}
Write-Host "Coordinator lives on port $coordPort." -ForegroundColor Green
$env:DAIN_COORD_URL = "ws://localhost:$coordPort"
$env:VITE_API_URL = "http://localhost:$coordPort"
$env:VITE_API_KEY = $clientKey

# 3) Node agents (share the detected port)
for ($i = 1; $i -le [int]$numNodes; $i++) {
    $env:DAIN_NODE_ID = "node-$i"
    Start-Component -Title "DAIN Node $i" -Mode "node" -NodeIndex $i
    Start-Sleep -Milliseconds 800
}

# 4) Web client
Start-Component -Title "DAIN Web Client" -Mode "client"

# 5) Summary
Write-Host ""
Write-Host " === All terminals launched! ===" -ForegroundColor Green
Write-Host "  Coordinator:  http://localhost:$coordPort"
Write-Host "  Web client:   http://localhost:5173/DAIN/"
Write-Host "  Admin API:    curl -H `"X-Admin-Key: $adminKey`" http://localhost:$coordPort/admin/nodes"
Write-Host ""
Write-Host "  Ctrl+C in each window stops it. This launcher window stays open."
Read-Host "`nPress Enter to open the web client in your browser..."
Start-Process "http://localhost:5173/DAIN/"