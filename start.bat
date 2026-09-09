@echo off
title DAIN - Starting local cluster...
echo.
echo  ===================================
echo    DAIN Local Dev Launcher
echo  ===================================
echo.

REM --- Check prerequisites ---
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] uv not found. Install: powershell -ExecutionPolicy Bypass -File bootstrap.ps1
    pause
    exit /b 1
)
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    pause
    exit /b 1
)

REM --- Ask for optional keys (Enter = use defaults) ---
echo Press Enter to accept defaults shown in brackets.
echo.
set /p DAIN_ADMIN_KEY="  Admin API key     [dain-dev-admin-key]: "
if "%DAIN_ADMIN_KEY%"=="" set DAIN_ADMIN_KEY=dain-dev-admin-key

set /p DAIN_CLIENT_KEY="  Client API key    [dain-dev-key]: "
if "%DAIN_CLIENT_KEY%"=="" set DAIN_CLIENT_KEY=dain-dev-key

set /p DAIN_JOIN_TOK="  Join token        [dain-dev-join-token]: "
if "%DAIN_JOIN_TOK%"=="" set DAIN_JOIN_TOK=dain-dev-join-token

set /p NUM_NODES="  Number of nodes   [1]: "
if "%NUM_NODES%"=="" set NUM_NODES=1

echo.
echo  Starting cluster with:
echo    Admin key:      %DAIN_ADMIN_KEY%
echo    Client API key: %DAIN_CLIENT_KEY%
echo    Join token:     %DAIN_JOIN_TOK%
echo    Nodes:          %NUM_NODES%
echo.
timeout /t 2 >nul

REM --- Terminal 1: Coordinator ---
start "DAIN Coordinator" cmd /k "title DAIN Coordinator && cd /d %~dp0Coordinator && set DAIN_MODEL_STORE_DIR=%~dp0model_store&& set DAIN_ADMIN_API_KEY=%DAIN_ADMIN_KEY% && set DAIN_API_KEY=%DAIN_CLIENT_KEY% && set DAIN_JOIN_TOKEN=%DAIN_JOIN_TOK% && echo Starting coordinator... && uv run python -m dain_coordinator"

REM --- Wait for coordinator to start ---
echo Waiting for coordinator to start (3s)...
timeout /t 4 >nul

REM --- Terminal 2..N: Node agents ---
setlocal enabledelayedexpansion
for /l %%i in (1,1,%NUM_NODES%) do (
    start "DAIN Node %%i" cmd /k "title DAIN Node %%i && cd /d %~dp0Node && set DAIN_MODEL_STORE_DIR=%~dp0model_store&& set DAIN_COORD_URL=ws://localhost:8000 && set DAIN_JOIN_TOKEN=%DAIN_JOIN_TOK% && set DAIN_NODE_ID=node-%%i && set DAIN_MODEL=dain-tiny-16L && echo Starting node %%i... && uv run python -m dain_node"
    timeout /t 1 >nul
)
endlocal

REM --- Terminal 3: Web client ---
if exist "%~dp0Client\node_modules" (
    start "DAIN Web Client" cmd /k "title DAIN Web Client && cd /d %~dp0Client && set VITE_API_URL=http://localhost:8000 && set VITE_API_KEY=%DAIN_CLIENT_KEY% && echo Starting dev server... && npm run dev"
) else (
    start "DAIN Web Client" cmd /k "title DAIN Web Client && cd /d %~dp0Client && echo Installing dependencies... && npm ci && set VITE_API_URL=http://localhost:8000 && set VITE_API_KEY=%DAIN_CLIENT_KEY% && echo Starting dev server... && npm run dev"
)

echo.
echo  ===================================
echo    All terminals launched!
echo  ===================================
echo.
echo  Coordinator:  http://localhost:8000
echo  Web client:   http://localhost:5173/DAIN/
echo  Admin API:    curl -H "X-Admin-Key: %DAIN_ADMIN_KEY%" http://localhost:8000/admin/nodes
echo.
echo  Press any key to open the web client in your browser...
pause >nul
start http://localhost:5173/DAIN/
