"""What trades did VOLFORGE actually take? Print every entry/exit with R:R.

If R:R hits 168x, a stop distance is ~0 (degenerate). See the real trade log and
whether sizing/exits are sane. Read-only.
"""
import warnings

import pandas as pd

import pipeline
from volforge.backtest import _prep
from volforge.strategies import Signal, best_setup

warnings.filterwarnings("ignore")

df = _prep(pipeline.load_ohlcv("nifty50_2years.csv"))
print("=== every best_setup across the series (R:R + entry + SL) ===")
rows = []
for i in range(250, len(df) - 1, 3):
    s = best_setup(df.iloc[:i + 1], min_rr=1.5)
    if s.signal != Signal.NONE:
        rows.append({"bar": i, "strat": s.strategy, "sig": s.signal.value,
                     "entry": round(s.entry, 1), "sl": round(s.stop_loss, 1),
                     "tgt": round(s.target, 1), "risk": round(s.risk, 2),
                     "rr": round(s.rr_ratio, 2)})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
print(f"\n=== sanity ===")
print(f"  setups: {len(t)}")
print(f"  risk==0 (degenerate stop): {(t['risk'] <= 0.01).sum()}")
print(f"  rr > 10 (absurd): {(t['rr'] > 10).sum()}")
print(f"  rr distribution: {sorted(t['rr'].round(1).tolist())[:5]} ... {sorted(t['rr'].round(1).tolist())[-5:]}")
print("\n  VERDICT: if many rr>10 or risk~0, strategies emit garbage setups that")
print("  survive the R:R gate and inflate returns. The 'edge' is the bug.")
