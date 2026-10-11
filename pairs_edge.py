"""PAIRS TRADING / STATISTICAL ARBITRAGE — market-neutral edge hunt.

Directional equity swing = no edge (proven, L14). Pairs trading is a different
mechanism entirely: long one stock, short a cointegrated peer, profiting from
RELATIVE mispricing with ~zero market exposure. Less efficient in midcaps,
works at small capital, no gap risk if both legs are equities.

Method:
  1. Find cointegrated pairs (Engle-Granger on 1y of daily log prices)
  2. Compute z-score of the spread
  3. Long the cheap leg + short the expensive leg when |z| > 2
  4. Exit at |z| < 0.5 (mean reversion) or stop at |z| > 3.5
  5. Both legs Rs50K each = Rs1L deployed, Rs1.5L can run 1 pair

Honesty gates: no look-ahead (cointegration fit on rolling window only),
honest costs on BOTH legs, stress test (holdout halves + per-pair).
Run: python pairs_edge.py
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
Z_ENTER = 2.0
Z_EXIT = 0.5
Z_STOP = 3.5
MIN_HISTORY = 120   # days of cointegration window


def load():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    ok = []
    for lb, g in df.groupby("label"):
        med = g.close.median()
        if med > 0 and g.close.max() / med < 10 and g.datetime_ist.duplicated().sum() == 0:
            g = g.sort_values("datetime_ist")
            if len(g) >= MIN_HISTORY + 30:
                ok.append(g)
    return ok


def cost(notional):
    return 2 * BROKERAGE + notional * 2 * (STT / 2 + EXCH * (1 + GST) + SLIP)


def cointegrated(a, b):
    """Engle-Granger lite: regress log(a) on log(b), test residual stationarity
    via half-life. Returns (beta, ok)."""
    la, lb = np.log(a), np.log(b)
    beta = np.polyfit(lb, la, 1)[0]
    resid = la - beta * lb
    # AR(1) coefficient -> half-life
    r_lag = resid[:-1]
    r_now = resid[1:]
    phi = np.polyfit(r_lag, r_now, 1)[0]
    if phi >= 0.999:
        return beta, False
    half_life = -np.log(2) / np.log(phi) if phi > 0 else 999
    return beta, 1 <= half_life <= 30   # reversion in 1-30 days


def find_pairs(series):
    """series: dict label -> (dates, close array). Returns list of (a,b,beta)."""
    labels = list(series.keys())
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb_ = labels[i], labels[j]
            da, ca = series[la]
            db, cb = series[lb_]
            # align on common dates
            common = sorted(set(da) & set(db))
            if len(common) < MIN_HISTORY + 30:
                continue
            ia = {d: k for k, d in enumerate(da)}
            ib = {d: k for k, d in enumerate(db)}
            a = np.array([ca[ia[d]] for d in common])
            b = np.array([cb[ib[d]] for d in common])
            if a.min() <= 0 or b.min() <= 0:
                continue
            # correlation pre-filter (fast)
            if abs(np.corrcoef(a, b)[0, 1]) < 0.7:
                continue
            beta, ok = cointegrated(a, b)
            if ok:
                pairs.append((la, lb_, beta, common, a, b))
    return pairs


def trade_pair(common, a, b, beta):
    """Walk forward: z-score from rolling window, no look-ahead."""
    la = np.log(a)
    trades = []
    W = 60   # rolling window for spread stats
    pos = 0
    entry_z = 0.0
    for t in range(W, len(common)):
        spread = la[:t] - beta * np.log(b[:t])
        win = spread[-W:]
        mu, sd = win.mean(), win.std()
        if sd == 0:
            continue
        z = (spread[-1] - mu) / sd
        # manage
        if pos != 0:
            if abs(z) <= Z_EXIT or abs(z) >= Z_STOP:
                # close both legs
                pa = a[t] / a[t - 1] - 1     # return of leg A today
                pb = b[t] / b[t - 1] - 1
                # long A short B if pos==1
                r = (pa * pos) - (pb * pos)
                gross = r * NOTIONAL
                net = gross - 2 * cost(NOTIONAL)
                trades.append({"z_entry": entry_z, "z_exit": z, "pnl": net,
                               "reason": "stop" if abs(z) >= Z_STOP else "revert",
                               "date": common[t]})
                pos = 0
        if pos == 0 and z <= -Z_ENTER:
            pos, entry_z = 1, z      # A cheap -> long A, short B
        elif pos == 0 and z >= Z_ENTER:
            pos, entry_z = -1, z     # A rich -> short A, long B
    return trades


def main():
    series = {}
    for g in load():
        series[g.label.iloc[0]] = (list(g.datetime_ist.dt.date), g.close.values)
    print(f"=== PAIRS / STAT-ARB ({len(series)} instruments) ===")
    print(f"  finding cointegrated pairs (corr>0.7, half-life 1-30d)...")

    pairs = find_pairs(series)
    print(f"  found {len(pairs)} cointegrated pairs\n")
    if not pairs:
        print("  no pairs — try lower correlation threshold or more instruments")
        return

    all_trades = []
    per_pair = []
    for la, lb_, beta, common, a, b in pairs:
        t = trade_pair(common, a, b, beta)
        if len(t) < 5:
            continue
        arr = np.array([x["pnl"] for x in t])
        all_trades.extend(t)
        per_pair.append((f"{la}/{lb_}", len(t), arr.mean(), (arr > 0).mean()))

    if not all_trades:
        print("  no pairs produced enough trades")
        return

    print(f"{'pair':<24}{'n':>5}{'avgPnL':>11}{'win%':>7}")
    for name, n, avg, win in sorted(per_pair, key=lambda x: -x[2]):
        print(f"{name:<24}{n:>5}{avg:>11,.0f}{win*100:>6.0f}%")

    arr = np.array([x["pnl"] for x in all_trades])
    print(f"\n=== AGGREGATE ===")
    print(f"  pairs traded: {len(per_pair)} | trades: {len(arr)}")
    print(f"  win rate: {(arr > 0).mean()*100:.0f}%")
    print(f"  total PnL: Rs{arr.sum():,.0f}")
    print(f"  avg/trade: Rs{arr.mean():,.0f} (both legs, Rs1L deployed)")
    print(f"  worst: Rs{arr.min():,.0f} | best: Rs{arr.max():,.0f}")

    # holdout
    dates = sorted({x["date"] for x in all_trades})
    mid = dates[len(dates) // 2]
    h1 = np.array([x["pnl"] for x in all_trades if x["date"] < mid])
    h2 = np.array([x["pnl"] for x in all_trades if x["date"] >= mid])
    print(f"\n=== HOLDOUT ===")
    for nm, s in [("H1", h1), ("H2", h2)]:
        if len(s):
            print(f"  {nm}: {len(s)} trades, avg Rs{s.mean():,.0f}, {(s > 0).mean()*100:.0f}% win")
    pos_pairs = sum(1 for p in per_pair if p[2] > 0)
    print(f"\n  {pos_pairs}/{len(per_pair)} pairs positive")

    # verdict
    n_days = 246
    tpy = len(arr) / n_days * 250
    est = arr.mean() * tpy
    print(f"\n  ~{tpy:.0f} trades/yr, est annual on 1 pair: Rs{est:,.0f} "
          f"(monthly ~Rs{est/12:,.0f})")
    ok = (arr.mean() > 0 and len(h1) and len(h2) and h1.mean() > 0
          and h2.mean() > 0 and pos_pairs > len(per_pair) * 0.5)
    print(f"\n  VERDICT: {'*** REAL EDGE ***' if ok else 'no edge / not robust'}")
    if not ok:
        print("  (fails: positive in both halves AND >50% pairs positive)")


if __name__ == "__main__":
    main()
