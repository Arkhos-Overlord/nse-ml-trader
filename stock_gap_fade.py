"""GAP-FADE on single stocks — where the literature says it works.

Lit (TickerGuard pre-registered, 4068 stocks; Zarattini 2025): gap-fade is the
ONLY intraday edge that survives, and it lives on SINGLE STOCKS with news
catalysts, short-biased, low hit rate, fat right tail. Index gap-fade failed
(no catalyst). Now test it properly on 12 liquid Indian stocks.

Equity intraday (MIS) economics — this is the Rs1.5L-fit path futures wasn't:
  - MIS leverage ~5x -> Rs1.5L controls ~Rs7.5L of stock intraday
  - brokerage flat Rs20/order, intraday STT 0.025% sell, no delivery STT
  - square off same day (no gap risk)
A Rs500 stock at 5x leverage = Rs100 cash per share -> Rs1.5L holds ~1500
shares = Rs7.5L exposure. Cost ~0.1% RT. This is affordable AND has a cushion.

Gap-fade: stock opens >thr% above prior close (news overreaction) -> SHORT,
expect intraday correction; gap down -> LONG. Square off at close.
Filter for 'stock in play': only fade gaps on ABOVE-AVERAGE volume days
(Zarattini: the edge needs the catalyst = volume).

Honesty gates: hit rate (lit says LOW is correct), median (negative is
correct), tail (fat right tail is the edge), holdout, full-capital, per-stock.

Run: python stock_gap_fade.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/intraday_stocks.parquet"
CAPITAL = 150000.0
LEVERAGE = 5.0                 # MIS intraday leverage
BROKERAGE = 20.0
STT_SELL = 0.00025             # intraday STT on sell
EXCH_GST = 0.00004
SLIP = 0.001                   # 0.1% slippage on liquid large-caps


def load():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df["date"] = df.datetime_ist.dt.date
    df["hm"] = df.datetime_ist.dt.strftime("%H:%M")
    return df.sort_values(["label", "datetime_ist"]).reset_index(drop=True)


def equity_cost(exposure):
    return 2 * BROKERAGE + exposure * STT_SELL + exposure * EXCH_GST + exposure * SLIP


def gap_fade_stocks(df, gap_thr=0.02, vol_mult=1.5, require_vol=True):
    """Fade opening gaps on single stocks. 'Stock in play' filter: only when
    first-hour volume > vol_mult x average first-hour volume (news catalyst)."""
    trades = []
    for lb, g in df.groupby("label"):
        daily = g.groupby("date").agg(o=("open", "first"), c=("close", "last"))
        daily = daily.sort_index()
        dates = list(daily.index)
        for i in range(1, len(dates)):
            prev_close = daily.c.iloc[i - 1]
            o = daily.o.iloc[i]
            gap = o / prev_close - 1
            if abs(gap) < gap_thr:
                continue
            # volume catalyst: first-hour volume vs stock's average
            gd = g[g.date == dates[i]].sort_values("datetime_ist")
            first_hr = gd[gd.hm <= "10:15"]
            if require_vol and len(first_hr) > 0:
                avg_fh = g[g.hm <= "10:15"].groupby("date").volume.sum().mean()
                if first_hr.volume.sum() < vol_mult * avg_fh:
                    continue  # no catalyst, skip
            direction = -1 if gap > 0 else 1   # fade
            entry = o * (1 + SLIP * direction)
            exit_ = daily.c.iloc[i]
            trades.append({"label": lb, "date": dates[i], "dir": direction,
                           "entry": entry, "exit": exit_, "gap": gap})
    return trades


def simulate(trades):
    """MIS equity: 5x leverage, 1 position at a time, compound, net."""
    capital, peak, maxdd = CAPITAL, CAPITAL, 0
    last = None
    for t in trades:
        if last is not None and t["date"] <= last:
            continue
        exposure = capital * LEVERAGE
        qty = int(exposure / t["entry"])
        if qty < 1:
            continue
        gross = (t["exit"] - t["entry"]) * qty * t["dir"]
        cost = equity_cost(qty * t["entry"])
        capital += gross - cost
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
        last = t["date"]
    return capital, maxdd


def report(name, trades):
    if not trades:
        print(f"  {name:<30} 0 trades")
        return None
    n = len(trades)
    # per-trade net on a standard exposure
    exp0 = CAPITAL * LEVERAGE
    r = []
    for t in trades:
        qty = exp0 / t["entry"]
        g = (t["exit"] - t["entry"]) * qty * t["dir"]
        c = equity_cost(exp0)
        r.append(g - c)
    net = np.array(r)
    cap, maxdd = simulate(trades)
    ts = net.mean() / (net.std() / np.sqrt(n)) if net.std() > 0 else 0
    print(f"  {name:<30} n={n:<4} hit={100*(net>0).mean():3.0f}% "
          f"net/tr=Rs{net.mean():+8.0f} med=Rs{np.median(net):+8.0f} "
          f"t={ts:+.2f} best=Rs{net.max():+7.0f} worst=Rs{net.min():+8.0f} "
          f"end=Rs{cap:,.0f} DD={maxdd*100:.0f}%")
    return net


def main():
    df = load()
    print("=== SINGLE-STOCK GAP-FADE (where the lit says it works) ===")
    print(f"  {df.label.nunique()} stocks x {df.date.nunique()} days, MIS 5x leverage")
    print(f"  lit signature: LOW hit + negative median + fat right tail = REAL\n")
    for thr in [0.015, 0.02, 0.03]:
        for vm in [1.5, 2.0]:
            tr = gap_fade_stocks(df, thr, vm)
            report(f"fade gap>{thr*100:.0f}% vol>{vm}x", tr)
    print()
    # no volume filter (baseline)
    for thr in [0.02, 0.03]:
        tr = gap_fade_stocks(df, thr, require_vol=False)
        report(f"fade gap>{thr*100:.0f}% (no vol filt)", tr)

    # direction (lit: short-biased)
    print(f"\n--- DIRECTION (lit says short-biased) ---")
    tr = gap_fade_stocks(df, 0.02, 1.5)
    for dn, up in [("fade UP (short)", True), ("fade DOWN (long)", False)]:
        grp = [t for t in tr if (t["gap"] > 0) == up]
        report(f"  {dn}", grp)

    # per-stock (is it broad or one stock carrying it?)
    print(f"\n--- PER-STOCK (broad or one lucky stock?) ---")
    for lb in sorted(set(t["label"] for t in tr)):
        grp = [t for t in tr if t["label"] == lb]
        if grp:
            exp0 = CAPITAL * LEVERAGE
            net = np.array([( (t["exit"]-t["entry"])*(exp0/t["entry"])*t["dir"] - equity_cost(exp0)) for t in grp])
            print(f"  {lb:<10} n={len(grp):<3} hit={100*(net>0).mean():3.0f}% "
                  f"net/tr=Rs{net.mean():+7.0f}")

    # holdout
    print(f"\n--- HOLDOUT ---")
    dates = sorted(set(t["date"] for t in tr))
    mid = dates[len(dates) // 2]
    for nm, grp in [("first 30d", [t for t in tr if t["date"] < mid]),
                    ("last 30d", [t for t in tr if t["date"] >= mid])]:
        report(f"  {nm}", grp)


if __name__ == "__main__":
    main()
