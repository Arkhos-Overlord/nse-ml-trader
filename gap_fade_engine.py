"""OPENING-GAP FADE — the one intraday edge that survives pre-registration.

Literature (decoded this session):
- TickerGuard pre-registered study (4,068 US stocks, 6.8M stock-days, 2010-2026):
  5 of 58 intraday variants passed — ALL were opening-gap trades, 4 SHORT.
  Winner: FADE large upward gaps, +0.36%/trade net, but 39% hit rate, negative
  median (few big winners carry it).
- Zarattini & Aziz (2025, Swiss Finance Inst): ORB works ONLY on 'Stocks in
  Play' (news catalysts) — top 20 = +1600% net, Sharpe 2.81. On normal stocks
  ORB fails. The edge is the CATALYST, not the mechanic.

The convergent finding: the only robust intraday edge is FADING large opening
gaps (overreaction corrects). This matches my own 'Nifty downside-break' lead
(short breaks continued, but that was continuation — this is the FADE, which
the literature actually supports).

On an INDEX (Nifty), a 'gap' = the open vs prior close. Fade = if Nifty gaps
UP big, short it expecting intraday correction; gap DOWN big, go long.
Indices don't have single-stock news, but they DO have overnight global-flow
overreactions that mean-revert intraday.

Test on 60 days deep intraday, futures economics, ALL honesty gates. The
literature warns: low hit rate, negative median, fat right tail — so measure
the TAIL, not just the mean.

Run: python gap_fade_engine.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/intraday_deep.parquet"
LOT, BROKERAGE = 65, 20.0
STT_SELL, EXCH_GST, SLIP = 0.0001, 0.00004, 0.0002
CAPITAL = 150000.0
MARGIN = 150000.0


def load(label="NIFTY50"):
    df = pd.read_parquet(STORE)
    df = df[(df.interval == "5m") & (df.label == label)].copy()
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df.sort_values("datetime_ist").reset_index(drop=True)
    df["date"] = df.datetime_ist.dt.date
    df["hm"] = df.datetime_ist.dt.strftime("%H:%M")
    return df


def fut_cost(px):
    return 2 * BROKERAGE + LOT * px * (STT_SELL + EXCH_GST + SLIP)


def gap_fade(df, gap_thr=0.003, exit_mode="close"):
    """Fade the opening gap. Gap = today's open vs prior close. If gap up >
    thr -> SHORT expecting correction; gap down -> LONG. No look-ahead (prior
    close + today's open only). Exit at close or first favorable touch."""
    daily = df.groupby("date").agg(o=("open", "first"), c=("close", "last"))
    daily = daily.sort_index()
    trades = []
    dates = list(daily.index)
    for i in range(1, len(dates)):
        prev_close = daily.c.iloc[i - 1]
        o = daily.o.iloc[i]
        gap = o / prev_close - 1
        if abs(gap) < gap_thr:
            continue
        direction = -1 if gap > 0 else 1   # FADE the gap
        entry = o * (1 + SLIP * direction)
        exit_ = daily.c.iloc[i]
        trades.append({"date": dates[i], "dir": direction, "entry": entry,
                       "exit": exit_, "gap": gap, "px": o})
    return trades


def simulate(trades):
    capital, peak, maxdd = CAPITAL, CAPITAL, 0
    pnls = []
    for t in trades:
        lots = max(1, int(capital / MARGIN))
        gross = (t["exit"] - t["entry"]) * lots * LOT * t["dir"]
        cost = lots * fut_cost(t["px"])
        pnl = gross - cost
        capital += pnl
        pnls.append(pnl)
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
    return capital, maxdd, np.array(pnls)


def report(name, trades):
    if not trades:
        print(f"  {name:<26} 0 trades")
        return None
    n = len(trades)
    px = np.mean([t["px"] for t in trades])
    cp = fut_cost(px)
    r = np.array([(t["exit"] - t["entry"]) * LOT * t["dir"] for t in trades])
    net = r - cp
    cap, maxdd, pnls = simulate(trades)
    ts = net.mean() / (net.std() / np.sqrt(n)) if net.std() > 0 else 0
    print(f"  {name:<26} n={n:<4} hit={100*(net>0).mean():3.0f}% "
          f"net/tr=Rs{net.mean():+7.0f} med=Rs{np.median(net):+7.0f} "
          f"t={ts:+.2f} best=Rs{net.max():+6.0f} worst=Rs{net.min():+7.0f} "
          f"end=Rs{cap:,.0f} DD={maxdd*100:.0f}%")
    return net, cap, maxdd


def main():
    print("=== OPENING-GAP FADE (the one edge that survives pre-registration) ===")
    print(f"  Lit: TickerGuard 5/58 passed = all gap-fade, 4 short. Zarattini:")
    print(f"  ORB works only with news catalysts. Fade large opening gaps.\n")
    for label in ["NIFTY50", "BANKNIFTY"]:
        df = load(label)
        ndays = df.date.nunique()
        print(f"--- {label}: {ndays} days ---")
        for thr in [0.002, 0.003, 0.005]:
            tr = gap_fade(df, thr)
            report(f"gap-fade thr{thr*100:.1f}%", tr)
        # direction split (lit says short-biased)
        tr = gap_fade(df, 0.003)
        for dn, dd in [("fade UP gaps (short)", -1), ("fade DOWN gaps (long)", 1)]:
            grp = [t for t in tr if (t["gap"] > 0) == (dd == -1)]
            report(f"  {dn}", grp)
        print()

    print("--- HONESTY GATES (Nifty gap-fade thr0.3%) ---")
    df = load("NIFTY50")
    tr = gap_fade(df, 0.003)
    dates = sorted(set(t["date"] for t in tr))
    mid = dates[len(dates) // 2]
    for nm, grp in [("first half", [t for t in tr if t["date"] < mid]),
                    ("second half", [t for t in tr if t["date"] >= mid])]:
        if grp:
            px = np.mean([t["px"] for t in grp])
            r = np.array([(t["exit"] - t["entry"]) * LOT * t["dir"] for t in grp])
            net = r - fut_cost(px)
            print(f"  {nm:<11} n={len(grp):<3} net/tr=Rs{net.mean():+7.0f} "
                  f"hit={100*(net>0).mean():.0f}%")
    print(f"\n  KEY: lit says LOW hit rate + negative median + fat right tail is")
    print(f"  the SIGNATURE of a real gap-fade edge. Check hit rate + median.")


if __name__ == "__main__":
    main()
