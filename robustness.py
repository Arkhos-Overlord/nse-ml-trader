"""Out-of-sample + parameter-robustness test for VOLFORGE.

The +139% came from ONE window. This proves whether it's real or overfit:
  1. Holdout split: train window vs untouched future window.
  2. Parameter sweep: vary min_rr, hold_bars, refit_every, risk% — a real edge is
     ROBUST to small param changes; an overfit edge collapses.
  3. Walk-forward blocks: run on contiguous sub-periods; stable = real.
Run: python robustness.py
"""
import warnings

import numpy as np
import pandas as pd

import pipeline
from volforge.backtest import _prep, walk_forward
from volforge.risk import RiskConfig

warnings.filterwarnings("ignore")


def run(df, **kw):
    try:
        return walk_forward(df, **kw)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "total_return_pct": np.nan}


def main():
    df = _prep(pipeline.load_ohlcv("nifty50_2years.csv"))
    n = len(df)

    print("=== 1. HOLD-OF-SAMPLE (first 60% = train, last 40% = untouched) ===")
    split = int(n * 0.6)
    for label, d in [("train(first60%)", df.iloc[:split]),
                     ("HOLDOUT(last40%)", df.iloc[split:])]:
        r = run(d)
        print(f"  {label}: ret={r.get('total_return_pct')}% "
              f"trades={r.get('trades')} B&H={r.get('buy_hold_return_pct')}%")

    print("\n=== 2. PARAMETER SWEEP (robust = stays profitable across all) ===")
    grid = []
    for min_rr in [1.5, 2.0, 2.5]:
        for hold in [6, 8, 12]:
            for refit in [30, 50, 80]:
                r = run(df, min_rr=min_rr, hold_bars=hold, refit_every=refit)
                grid.append((min_rr, hold, refit, r.get("total_return_pct", np.nan),
                             r.get("trades", 0)))
    g = pd.DataFrame(grid, columns=["min_rr", "hold", "refit", "ret%", "trades"])
    print(g.to_string(index=False))
    prof = g["ret%"] > 0
    print(f"\n  profitable in {prof.sum()}/{len(g)} param combos "
          f"({prof.mean()*100:.0f}%)  median ret={g['ret%'].median():.1f}%")
    print("  -> ROBUST if >70% combos profitable AND median beats cash.")
    print("  -> OVERFIT if only the original params worked.")

    print("\n=== 3. CONTIGUOUS BLOCKS (stable across time = real) ===")
    blocks = 4
    for b in range(blocks):
        lo, hi = int(n * b / blocks), int(n * (b + 1) / blocks)
        r = run(df.iloc[lo:hi])
        print(f"  block{b+1} [{lo}:{hi}]: ret={r.get('total_return_pct')}% "
              f"trades={r.get('trades')} B&H={r.get('buy_hold_return_pct')}%")


if __name__ == "__main__":
    main()
