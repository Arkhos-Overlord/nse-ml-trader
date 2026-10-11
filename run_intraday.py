"""Fill GAP 2: run VOLFORGE on real INTRADAY 15m data (where vol-clustering lives).

Fetches 60d of 15m bars for Nifty + BankNifty, runs the same fixed machine
(regime + strategies + risk, walk-forward), and reports honestly. This is the
timeframe the Phase-1 diagnosis said the signal actually lives in.
Run: python run_intraday.py
"""
import warnings

import pandas as pd
import yfinance as yf

from volforge.backtest import _prep, walk_forward

warnings.filterwarnings("ignore")


def load_intraday(sym, label, interval="15m", period="60d"):
    d = yf.download(sym, period=period, interval=interval, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna()
    print(f"  {label} {interval}: {len(d)} bars "
          f"{d.index[0].date()} -> {d.index[-1].date()}")
    return d


if __name__ == "__main__":
    for sym, label in [("^NSEI", "NIFTY"), ("^NSEBANK", "BANKNIFTY")]:
        print(f"\n=== VOLFORGE on {label} 15m (real intraday data) ===")
        try:
            df = _prep(load_intraday(sym, label))
            if len(df) < 100:
                print(f"  too few bars ({len(df)}), skipping")
                continue
            r = walk_forward(df, min_train=150, refit_every=30, hold_bars=8, min_rr=1.5)
            print("  --- RESULT ---")
            for k in ["total_return_pct", "trades", "win_rate_pct",
                      "max_drawdown_pct", "buy_hold_return_pct",
                      "beats_cash", "beats_buy_hold"]:
                print(f"    {k}: {r.get(k)}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}")
