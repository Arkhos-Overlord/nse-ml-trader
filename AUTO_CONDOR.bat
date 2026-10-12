@echo off
title Auto-Condor Engine - VRP Iron Condor
cd /d "%~dp0"
echo ================================================
echo   AUTO-CONDOR ENGINE
echo ================================================
echo.
echo   DEFAULT = PAPER MODE (places nothing).
echo.
echo   To go LIVE (real orders on your Fyers account):
echo     open a terminal here and run:
echo       python auto_condor.py --live --i-understand
echo.
echo   This launcher runs PAPER only. Press a key.
echo ================================================
pause
echo.
.venv\Scripts\python.exe auto_condor.py --once
echo.
pause
