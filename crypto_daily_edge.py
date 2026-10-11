"""Crypto DAILY momentum — documented edge, biggest real sample (1000 days).

Crypto 5-min = fee-dominated (proven). But crypto DAILY momentum is one of the
most documented anomalies in finance (unlike equity). 1000 days x 4 symbols =
real sample. Test time-series momentum at daily resolution with full stress:
holdout halves + broad-based + param stability + honest fees.

Fees: 0.1%/side. NOTIONAL $1500. Long/short both tested.
Run: python crypto_daily_edge.py
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DIR = "data_indmoney/crypto_daily"
FEE = 0.001
NOTIONAL = 1500.0


def load():
    out = {}
    for p in glob.glob(os.path.join(DIR, "*_1d.parquet")):
        sym = os.path.basename(p).replace("_1d.parquet", "")
        df = pd.read_parquet(p)
        df["datetime"] = pd.to_datetime(df["datetime"])
        out[sym] = df.sort_values("datetime").reset_index(drop=True)
    return out


def cost(n):
    return n * 2 * FEE


def backtest(df, signal_fn, hold):
    trades = []
    for i in range(60, len(df) - hold):
        d = signal_fn(df, i)
        if d == 0:
            continue
        entry = float(df.open.iloc[i + 1])
        exit_ = float(df.close.iloc[min(i + hold, len(df) - 1)])
        if entry <= 0 or exit_ <= 0:
            continue
        qty = NOTIONAL / entry
        gross = (exit_ - entry) * qty * d
        trades.append({"pnl": gross - cost(NOTIONAL), "t": i})
    return trades


def sig_tsmom(df, i, look):
    """Time-series momentum: sign of past `look`-day return."""
    r = df.close.iloc[i] / df.close.iloc[i - look] - 1
    return 1 if r > 0 else -1


def sig_tsmom_thresh(df, i, look, th=0.05):
    r = df.close.iloc[i] / df.close.iloc[i - look] - 1
    if r > th:
        return 1
    if r < -th:
        return -1
    return 0


def sig_ma_cross(df, i, fast=10, slow=50):
    if df.close.iloc[i] > df.close.iloc[i - fast:i].mean() > df.close.iloc[i - slow:i].mean():
        return 1
    if df.close.iloc[i] < df.close.iloc[i - fast:i].mean() < df.close.iloc[i - slow:i].mean():
        return -1
    return 0


def stress(trades, n):
    if len(trades) < 30:
        return None
    arr = np.array([t["pnl"] for t in trades])
    mid = n // 2
    h1 = np.array([t["pnl"] for t in trades if t["t"] < mid])
    h2 = np.array([t["pnl"] for t in trades if t["t"] >= mid])
    return {"avg": arr.mean(), "win": (arr > 0).mean(), "n": len(arr),
            "holdout": len(h1) > 10 and len(h2) > 10 and h1.mean() > 0 and h2.mean() > 0,
            "h1": h1.mean() if len(h1) else 0, "h2": h2.mean() if len(h2) else 0}


def main():
    data = load()
    print(f"=== CRYPTO DAILY EDGE ({len(data)} symbols, "
          f"{min(len(d) for d in data.values())} days, real sample) ===")
    print(f"  NOTIONAL ${NOTIONAL:,.0f}, fee {FEE*100:.1f}%/side\n")

    configs = []
    for look in [7, 14, 30, 60, 90]:
        configs.append((f"tsmom_{look}d", lambda df, i, lk=look: sig_tsmom(df, i, lk)))
    for look in [30, 60]:
        configs.append((f"tsmom_th_{look}d", lambda df, i, lk=look: sig_tsmom_thresh(df, i, lk)))
    configs.append(("ma_cross_10_50", sig_ma_cross))
    holds = [1, 3, 7, 14]

    print(f"{'strategy':<16}{'hold':>5}{'sym':<10}{'n':>6}{'win%':>7}{'avg$':>9}{'holdout':>9}")
    survivors = []
    for cname, cfn in configs:
        for hold in holds:
            for sym, df in data.items():
                t = backtest(df, cfn, hold)
                s = stress(t, len(df))
                if s is None:
                    continue
                ho = "YES" if s["holdout"] else "no"
                print(f"{cname:<16}{hold:>5}{sym:<10}{s['n']:>6}{s['win']*100:>6.0f}%"
                      f"{s['avg']:>9.2f}{ho:>9}")
                if s["avg"] > 0 and s["holdout"]:
                    survivors.append((cname, hold, sym, s))

    print(f"\n=== SURVIVORS (avg>0 AND both holdout halves positive) ===")
    if not survivors:
        print("  NONE — crypto daily momentum also fails the stress gauntlet.")
        print("  Honest result on 1000 real days.")
        return
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for cname, hold, sym, s in survivors:
        by_cfg[(cname, hold)].append(sym)
        print(f"  {cname} hold={hold} {sym}: avg ${s['avg']:.2f} "
              f"{s['win']*100:.0f}% H1 ${s['h1']:.2f} H2 ${s['h2']:.2f}")
    print(f"\n=== BROAD-BASED ===")
    for (cname, hold), syms in by_cfg.items():
        print(f"  {cname} hold={hold}: {len(syms)}/{len(data)} symbols")
    real = [(k, v) for k, v in by_cfg.items() if len(v) >= len(data) * 0.75]
    print(f"\n  VERDICT: {'*** REAL EDGE: ' + str(real) if real else 'no broad-based edge'}")


if __name__ == "__main__":
    main()
