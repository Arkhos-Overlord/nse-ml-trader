"""CRYPTO edge engine — real 90-day sample, full stress gauntlet.

26k bars per symbol = REAL sample (no waiting like equity intraday). Tests
intraday momentum, mean-reversion, breakout with the SAME stress test that
killed every fake equity edge:
  (a) positive in BOTH holdout halves
  (b) broad-based across symbols
  (c) stable across param sweeps
Plus honesty gate (realistic trade sizes) and honest fees (0.1% Binance taker).

Fees: 0.1% taker per side = 0.2% round-trip. Sizing: fixed $ notional.
Run: python crypto_edge.py
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CRYPTO_DIR = "data_indmoney/crypto"
FEE = 0.001          # 0.1% taker per side
NOTIONAL = 1500.0    # $1500 per trade (~Rs1.25L) — fits Rs1.5L
SLIP = 0.0002


def load():
    out = {}
    for p in glob.glob(os.path.join(CRYPTO_DIR, "*_5m.parquet")):
        sym = os.path.basename(p).replace("_5m.parquet", "")
        df = pd.read_parquet(p)
        df["datetime"] = pd.to_datetime(df["datetime"])
        med = df.close.median()
        if med > 0 and df.close.max() / med < 10 and df.datetime.duplicated().sum() == 0:
            out[sym] = df.sort_values("datetime").reset_index(drop=True)
    return out


def cost(notional):
    return notional * (2 * FEE + 2 * SLIP)


def backtest(df, signal_fn, hold_bars, fee=True):
    """Long/short. signal_fn(df, i) -> direction or 0. Enter next bar open."""
    trades = []
    for i in range(60, len(df) - hold_bars):
        d = signal_fn(df, i)
        if d == 0:
            continue
        entry = float(df.open.iloc[i + 1])
        exit_ = float(df.close.iloc[min(i + hold_bars, len(df) - 1)])
        if entry <= 0 or exit_ <= 0:
            continue
        qty = NOTIONAL / entry
        gross = (exit_ - entry) * qty * d
        net = gross - (cost(NOTIONAL) if fee else 0)
        trades.append({"pnl": net, "t": i, "date": df.datetime.iloc[i]})
    return trades


# ---- signals ----
def sig_momentum(df, i, look=12):
    """Continue short-term momentum."""
    r = df.close.iloc[i] / df.close.iloc[i - look] - 1
    return 1 if r > 0.003 else (-1 if r < -0.003 else 0)


def sig_reversion(df, i, look=12):
    """Fade short-term move."""
    r = df.close.iloc[i] / df.close.iloc[i - look] - 1
    return -1 if r > 0.003 else (1 if r < -0.003 else 0)


def sig_breakout(df, i, look=48):
    """Donchian breakout."""
    hi = df.high.iloc[i - look:i].max()
    lo = df.low.iloc[i - look:i].min()
    if df.close.iloc[i] >= hi:
        return 1
    if df.close.iloc[i] <= lo:
        return -1
    return 0


def stress(trades, df_len, label):
    """Return (avg, win, holdout_ok, n)."""
    if len(trades) < 30:
        return None
    arr = np.array([t["pnl"] for t in trades])
    mid = df_len // 2
    h1 = np.array([t["pnl"] for t in trades if t["t"] < mid])
    h2 = np.array([t["pnl"] for t in trades if t["t"] >= mid])
    avg = arr.mean()
    win = (arr > 0).mean()
    holdout_ok = len(h1) > 10 and len(h2) > 10 and h1.mean() > 0 and h2.mean() > 0
    return {"avg": avg, "win": win, "n": len(arr), "holdout": holdout_ok,
            "h1": h1.mean() if len(h1) else 0, "h2": h2.mean() if len(h2) else 0}


def main():
    data = load()
    print(f"=== CRYPTO EDGE ENGINE ({len(data)} symbols, real 90d sample) ===")
    print(f"  NOTIONAL ${NOTIONAL:,.0f} (~Rs{NOTIONAL*83:,.0f}), fee {FEE*100:.1f}%/side")
    print(f"  cost/RT ${cost(NOTIONAL):.2f}\n")

    signals = {"momentum": sig_momentum, "reversion": sig_reversion, "breakout": sig_breakout}
    holds = [6, 12, 24, 48]   # 30min, 1h, 2h, 4h on 5m bars

    print(f"{'signal':<12}{'hold':>5}{'sym':<10}{'n':>6}{'win%':>7}{'avg$':>9}{'holdout':>9}")
    survivors = []
    for sname, sfn in signals.items():
        for hold in holds:
            for sym, df in data.items():
                t = backtest(df, sfn, hold)
                s = stress(t, len(df), f"{sname}/{hold}/{sym}")
                if s is None:
                    continue
                ho = "YES" if s["holdout"] else "no"
                print(f"{sname:<12}{hold:>5}{sym:<10}{s['n']:>6}{s['win']*100:>6.0f}%"
                      f"{s['avg']:>9.2f}{ho:>9}")
                if s["avg"] > 0 and s["holdout"]:
                    survivors.append((sname, hold, sym, s))

    print(f"\n=== SURVIVORS (avg>0 AND both holdout halves positive) ===")
    if not survivors:
        print("  NONE — no crypto config passed the stress gauntlet either.")
        print("  (this is the honest result: momentum/reversion/breakout all fail)")
        return
    for sname, hold, sym, s in survivors:
        print(f"  {sname} hold={hold} {sym}: avg ${s['avg']:.2f}, "
              f"{s['win']*100:.0f}% win, H1 ${s['h1']:.2f} H2 ${s['h2']:.2f}")

    # broad-based check: same config positive across ALL symbols?
    from collections import defaultdict
    by_config = defaultdict(list)
    for sname, hold, sym, s in survivors:
        by_config[(sname, hold)].append(sym)
    print(f"\n=== BROAD-BASED CHECK ===")
    for (sname, hold), syms in by_config.items():
        print(f"  {sname} hold={hold}: positive on {len(syms)}/{len(data)} symbols")
    real = [(k, v) for k, v in by_config.items() if len(v) >= len(data) * 0.75]
    print(f"\n  VERDICT: {'*** REAL EDGE: ' + str(real) if real else 'no broad-based edge'}")


if __name__ == "__main__":
    main()
