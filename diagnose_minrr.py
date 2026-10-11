"""Isolate why min_rr has ZERO effect on the backtest (the smoking gun).

If changing min_rr 1.5 -> 2.5 produces identical trades, either the param is dead
or a real invariant dominates. Probe directly: how many setups actually pass each
min_rr, and what does best_setup return? Read-only diagnostic.
"""
import warnings

import pandas as pd

import pipeline
from volforge.backtest import _prep
from volforge.strategies import STRATEGIES, best_setup

warnings.filterwarnings("ignore")

df = _prep(pipeline.load_ohlcv("nifty50_2years.csv"))
print(f"Loaded {len(df)} bars\n")

print("=== Signal production across the series (raw, pre-risk) ===")
counts = {"ORB": 0, "VWAP": 0, "EMA": 0}
rr_seen = []
for i in range(250, len(df) - 1, 5):
    window = df.iloc[:i + 1]
    for name, fn in STRATEGIES.items():
        s = fn(window)
        if s.signal != "NONE":
            counts[name] += 1
            rr_seen.append(s.rr_ratio)
    b = best_setup(window, min_rr=1.5)
    b25 = best_setup(window, min_rr=2.5)
    if b.signal != "NONE" and b.rr_ratio != b25.rr_ratio:
        print(f"  bar {i}: min_rr CHANGES result (rr1.5={b.rr_ratio:.2f} rr2.5={b25.rr_ratio:.2f})")

print(f"\n  raw signal counts (every 5th bar, 250+): {counts}")
if rr_seen:
    print(f"  R:R ratios seen: min={min(rr_seen):.2f} max={max(rr_seen):.2f} "
          f"mean={sum(rr_seen)/len(rr_seen):.2f}")
    below = [r for r in rr_seen if r < 2.5]
    print(f"  setups with R:R < 2.5: {len(below)}/{len(rr_seen)}")
    print(f"  setups with R:R >= 2.5: {len(rr_seen)-len(below)}/{len(rr_seen)}")

print("\n=== Key question ===")
print("If ALL setups have R:R >= 2.5, then min_rr 1.5/2.0/2.5 filter nothing")
print("differently -> identical backtests. That's EXPECTED, not a bug, IF the")
print("strategies are structured so their R:R is a fixed multiple (ORB=2.0x,")
print("EMA=3:1). Then min_rr only matters as a floor that's already cleared.")
