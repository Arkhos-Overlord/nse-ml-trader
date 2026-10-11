"""DAILY-PAYOUT INTRADAY FUTURES ENGINE — squared off every day.

Goal: a strategy that takes a position each trading day and squares off by
15:15-15:25 (no overnight gap risk), paying (or losing) daily. This is the
'pays every day' structure — but tested HONESTLY, because 'every day' engines
are exactly where fake edges hide.

WHY INTRADAY FUTURES:
- No overnight gap tail (the thing that killed every overnight strategy).
- Futures economics: flat Rs20/order, no delivery STT -> costs ~Rs537/lot RT
  on Nifty, small vs the ~150-175pt daily range (Rs10-11K gross/lot).
- Daily range is the payout source: capture a fraction of it, square off.

CAPITAL REALITY (honest): Nifty futures margin ~Rs1.5L/lot. Rs1.5L = ~1 lot,
nothing left for drawdown. So the engine is sized to 1 lot and the REAL
question is whether the per-day edge survives costs + the losing days.

SIGNAL (no look-ahead, decided in the first 15-30 min, held to close):
  Opening-drift continuation: if the first 15min breaks the prior day's close
  decisively (informed overnight flow), ride it to the close. This is the
  intraday expression of the downside-break lead, but SAME-DAY (no overnight).

Gates: win rate, net/day, full-capital compounding, worst day, holdout
(first 30d vs last 30d), and the KILLER gate: does it survive if you can only
afford 1 lot (i.e. is the DAILY P/L positive and the tail survivable)?

Run: python daily_payout_engine.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/intraday_deep.parquet"
LOT, BROKERAGE = 65, 20.0
STT_SELL, EXCH_GST, SLIP = 0.0001, 0.00004, 0.0002
CAPITAL = 150000.0
MARGIN_PER_LOT = 150000.0   # ~1 lot on Rs1.5L


def load(label="NIFTY50"):
    df = pd.read_parquet(STORE)
    df = df[(df.interval == "5m") & (df.label == label)].copy()
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df.sort_values("datetime_ist").reset_index(drop=True)
    df["date"] = df.datetime_ist.dt.date
    df["hm"] = df.datetime_ist.dt.strftime("%H:%M")
    return df


def fut_cost_per_lot(px):
    return 2 * BROKERAGE + LOT * px * (STT_SELL + EXCH_GST + SLIP)


def opening_drift(df, orb_min=15, thr=0.001):
    """Enter in first orb_min if price breaks prior close by >thr; hold to close.
    No look-ahead: prior close + first-bar data only."""
    daily_close = df.groupby("date").close.last()
    dates = list(daily_close.index)
    trades = []
    for i, d in enumerate(dates):
        if i == 0:
            continue
        prev_close = daily_close.iloc[i - 1]
        g = df[df.date == d].sort_values("datetime_ist")
        if g.empty:
            continue
        cutoff = (pd.Timestamp(f"2000-01-01 {g.hm.iloc[0]}")
                  + pd.Timedelta(minutes=orb_min)).strftime("%H:%M")
        orbw = g[g.hm <= cutoff]
        if len(orbw) < 3:
            continue
        # direction: did first orb_min close break prior close?
        orb_close = float(orbw.close.iloc[-1])
        move = orb_close / prev_close - 1
        if abs(move) < thr:
            continue  # no decisive break, sit out (preserves capital)
        direction = 1 if move > 0 else -1
        entry = orb_close * (1 + SLIP * direction)  # slippage against you
        exit_ = float(g.close.iloc[-1])            # square off at close
        trades.append({"date": d, "dir": direction, "entry": entry, "exit": exit_,
                       "px": orb_close})
    return trades


def simulate_daily(trades, label):
    """1 lot, square off daily, compound capital. This is the real test: can
    Rs1.5L survive the daily P/L including losing days?"""
    capital = CAPITAL
    peak = capital
    maxdd = 0
    daily_pnl = []
    for t in trades:
        lots = max(1, int(capital / MARGIN_PER_LOT))
        gross = (t["exit"] - t["entry"]) * lots * LOT * t["dir"]
        cost = lots * fut_cost_per_lot(t["px"])
        pnl = gross - cost
        capital += pnl
        daily_pnl.append(pnl)
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
    return capital, maxdd, daily_pnl


def report(name, trades):
    if not trades:
        print(f"  {name:<30} 0 trades")
        return
    n = len(trades)
    px = np.mean([t["px"] for t in trades])
    cp = fut_cost_per_lot(px)
    r = np.array([(t["exit"] - t["entry"]) * LOT * t["dir"] for t in trades])
    net = r - cp
    cap, maxdd, dpnl = simulate_daily(trades, name)
    win = 100 * (net > 0).mean()
    ts = net.mean() / (net.std() / np.sqrt(n)) if net.std() > 0 else 0
    print(f"  {name:<30} days={n:<4} sit_out={100-n/ (n+0):.0f}... "
          f"win={win:3.0f}% net/day=Rs{net.mean():+7.0f} "
          f"t={ts:+.2f} worst=Rs{net.min():+7.0f} end=Rs{cap:,.0f} DD={maxdd*100:.0f}%")
    return net, cap, maxdd


def main():
    print("=== DAILY-PAYOUT INTRADAY FUTURES ENGINE ===")
    print(f"  1 lot Nifty (margin ~Rs1.5L), square off daily, no overnight gap")
    print(f"  cost ~Rs{fut_cost_per_lot(24000):.0f}/lot RT vs daily range ~150-175pt\n")
    for label in ["NIFTY50", "BANKNIFTY"]:
        df = load(label)
        ndays = df.date.nunique()
        print(f"--- {label}: {ndays} trading days ---")
        for orb_min in [15, 30]:
            for thr in [0.0008, 0.0015]:
                tr = opening_drift(df, orb_min, thr)
                report(f"ORB{orb_min} thr{thr}", tr)
        # how many days sit out (no trade)?
        tr = opening_drift(df, 15, 0.001)
        print(f"  -> traded {len(tr)}/{ndays} days, sat out {ndays-len(tr)}\n")

    # HONESTY: holdout + direction split on the base config
    print("--- HONESTY GATES (Nifty ORB15 thr0.001) ---")
    df = load("NIFTY50")
    tr = opening_drift(df, 15, 0.001)
    dates = sorted(set(t["date"] for t in tr))
    mid = dates[len(dates) // 2]
    for nm, grp in [("first 30d", [t for t in tr if t["date"] < mid]),
                    ("last 30d", [t for t in tr if t["date"] >= mid])]:
        if grp:
            px = np.mean([t["px"] for t in grp])
            cp = fut_cost_per_lot(px)
            r = np.array([(t["exit"] - t["entry"]) * LOT * t["dir"] for t in grp])
            net = r - cp
            print(f"  {nm:<10} days={len(grp):<3} net/day=Rs{net.mean():+7.0f} "
                  f"win={100*(net>0).mean():.0f}%")
    # direction
    for dn, dd in [("long only", 1), ("short only", -1)]:
        grp = [t for t in tr if t["dir"] == dd]
        if grp:
            px = np.mean([t["px"] for t in grp])
            cp = fut_cost_per_lot(px)
            r = np.array([(t["exit"] - t["entry"]) * LOT * t["dir"] for t in grp])
            net = r - cp
            print(f"  {dn:<10} days={len(grp):<3} net/day=Rs{net.mean():+7.0f} "
                  f"win={100*(net>0).mean():.0f}%")

    print(f"\n=== VERDICT ===")
    print(f"  This is the honest daily-payout test. If net/day is negative or")
    print(f"  the tail (worst day) is unsurvivable on 1 lot, it does NOT pay daily.")
    print(f"  60 days is still a modest sample — the scheduled re-test grows it.")


if __name__ == "__main__":
    main()
