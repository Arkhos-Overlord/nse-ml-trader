"""Instrument the backtest loop to see WHY entries never happen on 15m.

Runs a minimal version of the walk-forward with prints at each decision point:
regime tradeable? setup found? units>0? cost gate pass? One bar at a time until
we see the first (non-)entry.
"""
import warnings

import pandas as pd
import yfinance as yf

from volforge.backtest import _prep
from volforge.regime import HMMRegimeEngine
from volforge.risk import RiskManager
from volforge.strategies import Signal, best_setup

warnings.filterwarnings("ignore")


def load(sym):
    d = yf.download(sym, period="60d", interval="15m", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return _prep(d.dropna())


df = load("^NSEI")
engine = HMMRegimeEngine()
risk = RiskManager()
min_train, refit_every, min_rr = 150, 30, 1.5
fitted_at = -10**9

print("bar | regime | tradeable | mult | setup_sig | rr | units | cost_gate | ENTER")
entered = 0
for i in range(min_train, len(df) - 1):
    if i - fitted_at >= refit_every or engine.model is None:
        engine.fit(df.iloc[:i])
        fitted_at = i
    engine.predict(df.iloc[:i])
    mult = engine.position_size_multiplier()
    tradeable = engine.is_tradeable()
    setup = best_setup(df.iloc[:i + 1], min_rr=min_rr)
    has_sig = setup.signal != Signal.NONE
    units = risk.position_size(setup.entry, setup.stop_loss, mult) if has_sig else 0
    cost_ok = False
    if has_sig and units > 0:
        move_pct = setup.reward / (setup.entry + 1e-9) * 100
        cost_ok = move_pct > risk.cfg.round_trip_cost_bps / 100
    would_enter = tradeable and has_sig and units > 0 and cost_ok
    if would_enter:
        entered += 1
    # print only interesting bars: has signal but didn't enter, or entered
    if has_sig or would_enter:
        if i < min_train + 40 or would_enter:
            print(f"{i} | {engine.current_regime.value:8} | {str(tradeable):5} | "
                  f"{mult:.2f} | {setup.signal.value:4} | {setup.rr_ratio:.1f} | "
                  f"{units:4} | {str(cost_ok):5} | {'YES' if would_enter else ''}")
    if entered >= 5:
        break

print(f"\nTotal entries in traced run: {entered}")
print("If 0 entries but has_sig=True: the blocker is regime/mult/units/cost.")
print("Look for which column is False/0 on signal bars.")
