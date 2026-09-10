@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>nul
set "ROOT=%~dp0.."
cd /d "%ROOT%"

echo ============================================
echo   DAIN Cache Cleanup
echo   Root: %ROOT%
echo ============================================
echo.

set "DELETED_COUNT=0"

rem ---------- Python caches ----------
echo [Python caches]
for /r "%ROOT%" /d %%D in (__pycache__) do (
    if exist "%%D" (
        rd /s /q "%%D" 2>nul
        set /a DELETED_COUNT+=1
        echo   Removed: %%D
    )
)

rem ---------- Tool caches ----------
echo [Tool caches]
for %%C in (.pytest_cache .ruff_cache .mypy_cache) do (
    for /r "%ROOT%" /d %%D in (%%C) do (
        if exist "%%D" (
            rd /s /q "%%D" 2>nul
            set /a DELETED_COUNT+=1
            echo   Removed: %%D
        )
    )
)

rem ---------- Build artifacts ----------
echo [Build artifacts]
for %%D in (
    "%ROOT%\Node\build"
    "%ROOT%\Node\dist"
    "%ROOT%\Client\dist"
) do (
    if exist "%%~D" (
        rd /s /q "%%~D" 2>nul
        set /a DELETED_COUNT+=1
        echo   Removed: %%~D
    )
)

rem ---------- Vite dependency cache ----------
echo [Vite cache]
if exist "%ROOT%\Client\node_modules\.vite" (
    rd /s /q "%ROOT%\Client\node_modules\.vite" 2>nul
    set /a DELETED_COUNT+=1
    echo   Removed: %ROOT%\Client\node_modules\.vite
)

rem ---------- Node runtime state ----------
echo [Node runtime state]
for %%F in ("%ROOT%\Node\node_state-*.json") do (
    if exist "%%F" (
        del /q "%%F" 2>nul
        set /a DELETED_COUNT+=1
        echo   Removed: %%F
    )
)

rem ---------- Shard caches ----------
echo [Shard caches]
for /d %%D in ("%ROOT%\Node\shard_cache*") do (
    if exist "%%D" (
        rd /s /q "%%D" 2>nul
        set /a DELETED_COUNT+=1
        echo   Removed: %%D
    )
)

echo.
echo ============================================
echo   Summary
echo ============================================
echo   Cache items removed: %DELETED_COUNT%
echo   .venv environments and sqlite databases were kept.
echo.
pause