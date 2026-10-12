@echo off
title Condor Desk - VRP Iron Condor Live Signal
cd /d "%~dp0"
echo Starting Condor Desk...
echo.
.venv\Scripts\python.exe condor_desk.py
echo.
echo ================================================
echo   Press any key to close.
echo ================================================
pause >nul
