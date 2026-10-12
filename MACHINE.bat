@echo off
title THE MACHINE - Condor Desk
cd /d "%~dp0"
echo ================================================================
echo    THE MACHINE - unified condor desk
echo    engine + monitors + live dashboard
echo ================================================================
echo.
echo    DEFAULT = PAPER (places nothing). Dashboard auto-opens.
echo.
echo    To go LIVE: python machine.py --live --i-understand
echo    (double-check the ticket before you arm it)
echo ================================================================
echo.
start "" "%~dp0dashboard.html"
.venv\Scripts\python.exe machine.py
echo.
pause
