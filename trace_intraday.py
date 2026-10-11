"""Why 0 trades on intraday? Trace the pipeline bar-by-bar.

For each bar: what does each strategy say? does the regime gate open? does the
cost gate block? This isolates WHICH stage is killing every trade.
Read-only diagnostic.
"""
import warnings

import pandas as pd
import yfinance as yf

from volforge.backtest import _prep
from volforge.regime import HMMRegimeEngine
from volforge.strategies import STRATEGIES, Signal

warnings.filterwarnings("ignore")


def load(sym):
    d = yf.download(sym, period="60d", interval="15m", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return _prep(d.dropna())


df = load("^NSEI")
print(f"Loaded {len(df)} 15m bars\n")

# fit regime once on a big window
engine = HMMRegimeEngine()
engine.fit(df.iloc[:400])
regime, conf = engine.predict(df)
print(f"Current regime: {regime.value} conf={conf:.2f} tradeable={engine.is_tradeable()}")
print(f"position multiplier now: {engine.position_size_multiplier():.2f}\n")

print("=== per-bar: strategy signals over last 300 bars ===")
sig_counts = {n: 0 for n in STRATEGIES}
valid_rr = 0
cost_blocked = 0
gate_blocked = 0
total = 0
for i in range(400, len(df) - 1, 3):
    w = df.iloc[:i + 1]
    total += 1
    any_sig = False
    for name, fn in STRATEGIES.items():
        s = fn(w)
        if s.signal != Signal.NONE:
            sig_counts[name] += 1
            any_sig = True
            # does it pass R:R 1.5?
            if s.is_valid(1.5):
                valid_rr += 1
                # cost gate: expected move > 15bps?
                move_pct = abs(s.target - s.entry) / s.entry * 100
                if move_pct <= 0.15:
                    cost_blocked += 1
    # regime gate
    r, c = engine.predict(w)
    if any_sig and not engine.is_tradeable():
        gate_blocked += 1

print(f"  sampled {total} bars")
print(f"  raw signals: {sig_counts}")
print(f"  passed R:R>=1.5: {valid_rr}")
print(f"  blocked by cost gate (move<=15bps): {cost_blocked}")
print(f"  bars with a signal but regime NOT tradeable: {gate_blocked}")
print("\n=== diagnosis ===")
if sum(sig_counts.values()) == 0:
    print("  strategies produce ZERO signals -> strategy logic never fires intraday")
elif valid_rr == 0:
    print("  signals fire but none pass R:R>=1.5 -> setups too shallow")
elif cost_blocked > valid_rr * 0.5:
    print("  most valid setups blocked by COST gate -> moves too small to pay costs")
elif gate_blocked > total * 0.5:
    print("  regime gate blocks most bars -> engine is too conservative")
