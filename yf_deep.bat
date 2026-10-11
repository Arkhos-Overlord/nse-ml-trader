@echo off
cd /d C:\Users\sydne\_assess\Kimi-Auto-trader
set PY=C:\Users\sydne\_assess\Kimi-Auto-trader\.venv\Scripts\python.exe
echo === yfinance deep pull + ORB re-test %date% %time% === >> yf_deep.log
%PY% pull_intraday_deep.py >> yf_deep.log 2>&1
%PY% pull_history_yf.py >> yf_deep.log 2>&1
%PY% pull_intraday_stocks.py >> yf_deep.log 2>&1
%PY% pull_intraday_stocks_wide.py >> yf_deep.log 2>&1
%PY% pull_earnings.py >> yf_deep.log 2>&1
%PY% orb_2d_gates.py >> yf_deep.log 2>&1
%PY% stock_gap_fade.py >> yf_deep.log 2>&1
%PY% pead_final.py >> yf_deep.log 2>&1
%PY% vrp_strangle_final.py >> yf_deep.log 2>&1
echo === done %date% %time% === >> yf_deep.log
