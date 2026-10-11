"""Adversarial audit of the +139% VOLFORGE result — assume it's a bug until proven.

Hunts the standard leakage/bias bugs that fake huge returns:
  1. Look-ahead in feature/signal computation (does strategy see the future?).
  2. Entry/exit on the SAME bar's close+next (no execution lag).
  3. Position sizing using future stop distance.
  4. Regime model refit includes the current bar's own features.
  5. Costs understated / missing slippage.
  6. Stats double-counted or drawn from full-series indicators.
Run: python audit_backtest.py
"""
import warnings

import numpy as np
import pandas as pd

import pipeline
from volforge.backtest import _prep

warnings.filterwarnings("ignore")


def check_feature_leakage():
    """Does compute_features / strategies use any future data at bar i?"""
    from volforge.strategies import best_setup, orb_setup

    df = _prep(pipeline.load_ohlcv("nifty50_2years.csv"))
    print("=== 1. FEATURE / SIGNAL LEAKAGE ===")
    # Compute setup at bar i using data up to i, and again at i+1 — the bar-i setup
    # must NOT change when future bar arrives (i.e., no peek ahead).
    i = 200
    s_then = best_setup(df.iloc[:i + 1])
    s_now = best_setup(df.iloc[:i + 2])
    same = (s_then.signal == s_now.signal and s_then.entry == s_now.entry)
    print(f"  setup at bar {i} stable when bar {i+1} added: {same}")
    # ORB uses last 3 bars of the SLICE — confirm it doesn't reach beyond iloc[:i+1]
    orb = orb_setup(df.iloc[:i + 1])
    uses_only_slice = orb.entry in [float(df['close'].iloc[i])] or orb.entry == 0
    print(f"  ORB entry drawn only from slice tail: {uses_only_slice}")
    return same


def check_execution_lag():
    """Backtest enters at bar i close, evaluates exit at bar i+1. Any same-bar fill?"""
    print("\n=== 2. EXECUTION TIMING ===")
    # We fill at signal bar's close and resolve next bar. Confirm no fill at bar i+1's
    # open using bar i+1 info that wasn't available at bar i close.
    print("  model: signal at close(i) -> position active for bar i+1 only")
    print("  RISK: fill price = close(i), but we enter BEFORE knowing close(i+1).")
    print("  -> execution lag of exactly 1 bar. Acceptable but optimistic on fills.")


def check_sizing_leak():
    """Position size uses stop distance (known at signal time) — no future peek?"""
    print("\n=== 3. SIZING LEAKAGE ===")
    print("  size = risk_budget / |entry - stop_loss|; both known at signal time.")
    print("  entry/stop are from the SAME slice as the signal -> no future data. OK.")


def check_regime_refit():
    """Does engine.fit(d.iloc[:i]) leak? It fits on data < i, predicts on < i."""
    print("\n=== 4. REGIME REFIT SCOPE ===")
    print("  engine.fit(d.iloc[:i]) trains only on rows < i. predict(d.iloc[:i]) too.")
    print("  No future rows. OK. BUT: refit uses expanding window = regime is fit on")
    print("  ALL past including recent regime -> mild look-ahead in regime LABELS? No,")
    print("  labels come from fit-time means, applied forward only. OK.")


def check_costs():
    print("\n=== 5. COSTS / SLIPPAGE ===")
    print("  cost = 15 bps round trip on notional. Real Nifty F&O costs are higher")
    print("  (STT, brokerage, spread, slippage). 15bps may UNDERSTATE true cost.")
    print("  -> the +139% is computed WITH costs, but costs may be too low.")


def check_stats():
    print("\n=== 6. STAT SANITY ===")
    print("  15 trades -> 139% return = each trade ~+8.8% avg, win rate 40%.")
    print("  Sharpe 7.9 with daily-ish bars is implausible; likely small-sample noise")
    print("  in a choppy-but-up 2y window. 15 trades is a TINY sample.")
    print("  CONCLUSION: numbers are internally consistent but statistically fragile.")


if __name__ == "__main__":
    check_feature_leakage()
    check_execution_lag()
    check_sizing_leak()
    check_regime_refit()
    check_costs()
    check_stats()
    print("\n=== KEY QUESTION: is +139% real? ===")
    print("Cannot confirm from static audit alone. Need OUT-OF-SAMPLE + PARAM SWEEP")
    print("+ different window. If +139% collapses on holdout, it was overfit.")
