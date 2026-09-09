@echo off
title DAIN - Stopping...
echo Stopping all DAIN processes...

REM Kill coordinator, node agents, and vite dev server
taskkill /FI "WINDOWTITLE eq DAIN Coordinator*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq DAIN Node*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq DAIN Web Client*" /F >nul 2>&1

REM Also kill any stray python processes running dain
for /f "tokens=2" %%p in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr "PID:"') do (
    wmic process where "ProcessId=%%p" get CommandLine 2>nul | findstr "dain" >nul && taskkill /PID %%p /F >nul 2>&1
)

echo Done.
timeout /t 2 >nul
