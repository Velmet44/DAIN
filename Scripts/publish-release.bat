@echo off
setlocal
set "ROOT=%~dp0.."
cd /d "%ROOT%"

echo ============================================
echo   DAIN Publish Release
echo ============================================
echo.

set /p VERSION="  Version (e.g. 0.2.0): "
if "%VERSION%"=="" (
    echo   Version is required.
    pause
    exit /b 1
)

set /p NOTES="  Release notes (optional, press Enter to skip): "

if "%NOTES%"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0publish-release.ps1" -Version "%VERSION%"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0publish-release.ps1" -Version "%VERSION%" -Notes "%NOTES%"
)

echo.
pause
