@echo off
title Condor Desk - FYERS LIVE
cd /d "%~dp0"
echo ================================================
echo   CONDOR DESK - FYERS LIVE
echo ================================================
echo.
echo First, make sure you filled in fyers_env.txt
echo (client_id + secret from https://myapi.fyers.in)
echo.
pause
echo.
.venv\Scripts\python.exe fyers_live.py
echo.
echo ================================================
echo   Press any key to close.
echo ================================================
pause >nul
