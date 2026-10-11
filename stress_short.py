"""Stress-test the 20d_high SHORT edge — is it real or luck?

One positive result (net +Rs34/trade, 53% win) is NOT an edge until it survives:
  1. HOLDOUT: split the year in half; must be positive in BOTH halves
  2. HOLDS: sweep hold periods 1-20 days
  3. LOOKBACK: sweep the "high" lookback (10/20/40/60 day)
  4. PER-STOCK: is it broad-based or one lucky name?
  5. BEST/WORST YEAR QUARTER: consistency across time

An edge that only works in one sub-period or one stock is overfitting/luck.
Run: python stress_short.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/daily_1y.parquet"
NOTIONAL = 50000.0
BROKERAGE = 20
STT = 0.001
EXCH = 0.0000297
GST = 0.18
SLIP = 0.0005


def load():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    ok = []
    for lb, g in df.groupby("label"):
        med = g.close.median()
        if med > 0 and g.close.max() / med < 10 and g.datetime_ist.duplicated().sum() == 0:
            ok.append(g)
    return pd.concat(ok)


def cost(notional):
    return 2 * BROKERAGE + notional * 2 * (STT / 2 + EXCH * (1 + GST) + SLIP)


def short_trades(g, lookback, hold):
    """Short when close >= lookback-day high. Returns list of net pnls."""
    g = g.sort_values("datetime_ist").reset_index(drop=True)
    out = []
    for i in range(lookback, len(g) - hold):
        if g.close.iloc[i] >= g.high.iloc[i - lookback:i].max():
            entry = float(g.open.iloc[i + 1])
            ex = float(g.close.iloc[min(i + hold, len(g) - 1)])
            if entry <= 0 or ex <= 0:
                continue
            qty = int(NOTIONAL / entry)
            gross = (entry - ex) * qty          # short: profit when price falls
            out.append(gross - cost(entry * qty))
    return out


def stats(trades):
    if len(trades) < 15:
        return None
    a = np.array(trades)
    return {"n": len(a), "avg": a.mean(), "win": (a > 0).mean(),
            "total": a.sum(), "worst": a.min()}


def main():
    df = load()
    stocks = df[df.label != "NIFTY50"]
    print("=== STRESS TEST: short 20d-high breakout ===")
    print(f"  universe {stocks.label.nunique()} stocks, "
          f"{stocks.datetime_ist.dt.date.nunique()} days\n")

    # 1. hold sweep
    print("--- hold period sweep (lookback=20) ---")
    print(f"{'hold':>5}{'n':>6}{'avg':>10}{'win%':>7}{'total':>12}")
    best_hold = None
    for hold in [1, 2, 3, 5, 7, 10, 15, 20]:
        allt = []
        for _, g in stocks.groupby("label"):
            allt.extend(short_trades(g, 20, hold))
        s = stats(allt)
        if s:
            print(f"{hold:>5}{s['n']:>6}{s['avg']:>10,.0f}{s['win']*100:>6.0f}%{s['total']:>12,.0f}")
            if best_hold is None or s["avg"] > best_hold[1]["avg"]:
                best_hold = (hold, s)

    # 2. lookback sweep at best hold
    h = best_hold[0] if best_hold else 3
    print(f"\n--- lookback sweep (hold={h}) ---")
    print(f"{'lb':>5}{'n':>6}{'avg':>10}{'win%':>7}{'total':>12}")
    for lb in [10, 20, 40, 60]:
        allt = []
        for _, g in stocks.groupby("label"):
            allt.extend(short_trades(g, lb, h))
        s = stats(allt)
        if s:
            print(f"{lb:>5}{s['n']:>6}{s['avg']:>10,.0f}{s['win']*100:>6.0f}%{s['total']:>12,.0f}")

    # 3. HOLDOUT: first half vs second half
    print(f"\n--- HOLDOUT: first half vs second half (lb=20, hold={h}) ---")
    dates = sorted(stocks.datetime_ist.dt.date.unique())
    mid = dates[len(dates) // 2]
    for name, mask in [("H1", stocks.datetime_ist.dt.date < mid),
                       ("H2", stocks.datetime_ist.dt.date >= mid)]:
        allt = []
        for _, g in stocks[mask].groupby("label"):
            allt.extend(short_trades(g, 20, h))
        s = stats(allt)
        if s:
            print(f"  {name}: {s['n']} trades, avg Rs{s['avg']:,.0f}, "
                  f"{s['win']*100:.0f}% win, total Rs{s['total']:,.0f}")

    # 4. per-stock breakdown
    print(f"\n--- per-stock (lb=20, hold={h}) ---")
    per = []
    for lb, g in stocks.groupby("label"):
        s = stats(short_trades(g, 20, h))
        if s:
            per.append((lb, s["n"], s["avg"], s["win"]))
    per.sort(key=lambda x: -x[2])
    pos = sum(1 for p in per if p[2] > 0)
    print(f"  {pos}/{len(per)} stocks positive")
    for lb, n, avg, win in per:
        flag = "+" if avg > 0 else "-"
        print(f"    {flag} {lb:<12} n={n:>3} avg Rs{avg:>8,.0f} {win*100:>3.0f}%")

    print(f"\n=== VERDICT ===")
    if best_hold and best_hold[1]["avg"] > 0:
        print(f"  best: hold={best_hold[0]}, avg Rs{best_hold[1]['avg']:,.0f}/trade")
        print(f"  SURVIVES stress only if: positive in BOTH halves AND broad-based")
    else:
        print("  no positive configuration found")


if __name__ == "__main__":
    main()
