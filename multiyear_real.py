"""Multi-year real-options robustness: does the VRP edge hold across regimes?

One low-vol year (2024) isn't proof. Test 2022 (high vol), 2023, 2024, 2025
on REAL option prices. A real edge survives ALL regimes; an overfit/regime-
dependent one collapses in a high-vol year.

Run: python multiyear_real.py
"""
import warnings

import numpy as np
import pandas as pd

from real_options_backtest import backtest_real, build_vix_signal

warnings.filterwarnings("ignore")


def main():
    print("Building VIX VRP signal (3y)...")
    sig = build_vix_signal()

    years = [2022, 2023, 2024, 2025]
    all_results = []
    all_trades = []
    for yr in years:
        path = f"NIFTY_{yr}.parquet"
        try:
            opt = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001
            print(f"  {yr}: not downloaded ({e})")
            continue
        trades = backtest_real(opt, sig)
        if trades.empty:
            print(f"  {yr}: 0 trades")
            continue
        all_trades.append(trades)
        all_results.append({
            "year": yr, "trades": len(trades),
            "total_pnl": trades.pnl.sum(),
            "avg_pnl": trades.pnl.mean(),
            "win%": (trades.pnl > 0).mean() * 100,
            "worst": trades.pnl.min(),
            "avg_vix": trades.vix.mean(),
        })

    if not all_results:
        print("no data")
        return
    r = pd.DataFrame(all_results)
    print("\n=== MULTI-YEAR REAL OPTIONS BACKTEST ===")
    print(r.to_string(index=False))

    combined = pd.concat(all_trades)
    print(f"\n=== COMBINED ({len(combined)} trades, {r.year.min()}-{r.year.max()}) ===")
    print(f"  total PnL: Rs{combined.pnl.sum():,.0f}")
    print(f"  avg/trade: Rs{combined.pnl.mean():,.0f}")
    print(f"  win rate: {(combined.pnl > 0).mean()*100:.0f}%")
    print(f"  worst trade: Rs{combined.pnl.min():,.0f}")
    print(f"  profitable years: {(r.total_pnl > 0).sum()}/{len(r)}")

    # the honest verdict
    print("\n=== VERDICT ===")
    if (r.total_pnl > 0).all():
        print("  EDGE ROBUST: profitable in EVERY year incl. high-vol.")
    elif (r.total_pnl > 0).mean() >= 0.75:
        print("  EDGE MOSTLY HOLDS: profitable most years, weak in some regime.")
    else:
        print("  EDGE REGIME-DEPENDENT: only works in low-vol. NOT robust.")

    # per-lot annual
    print(f"\n  avg annual PnL (all yrs): Rs{r.total_pnl.mean():,.0f}/lot")
    print(f"  avg trades/yr: {r.trades.mean():.0f}")
    print(f"  HONEST: real prices, real tail. This is the true multi-year test.")


if __name__ == "__main__":
    main()
