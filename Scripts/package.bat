@echo off
setlocal
set "ROOT=%~dp0.."
cd /d "%ROOT%"

echo ============================================
echo   DAIN Package
echo   Root: %ROOT%
echo ============================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0package.ps1" -Root "%ROOT%"
set "RESULT=%errorlevel%"

echo.
if "%RESULT%"=="0" (
    echo   Package created successfully.
) else (
    echo   Packaging FAILED with code %RESULT%.
)
echo.
pause