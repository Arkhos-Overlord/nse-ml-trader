"""ORB-2d signal — the honesty gates. Real edge or fake?

v2 found: Nifty ORB breakout held 2 days = +0.169%/tr, 61% win, t=+1.49,
Rs1.5L->Rs297K, DD36%. First positive microstructure result. BUT:
  - t=+1.49 sub-significant
  - BankNifty same signal is NEGATIVE (t=-1.97) -> if real, should work on both
This runs the gates that decide real vs fake:

GATE 1 — CROSS-INDEX: does ORB-2d work on BOTH Nifty and BankNifty? A real
         structural edge is index-independent. One-yes-one-no = noise/regime.
GATE 2 — TIME HOLDOUT: first 30d vs last 30d, both positive?
GATE 3 — PARAM SENSITIVITY: ORB window (10/15/20/30min) x hold (2/3d). A real
         edge is a plateau, not a spike. One lucky cell = overfit.
GATE 4 — REGIME: does it survive when we split by market trend (up vs down
         days)? Structural edges should be regime-robust.
GATE 5 — COST SENSITIVITY: does it survive 2x slippage (real fills are worse)?

Run: python orb_2d_gates.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/intraday_deep.parquet"
LOT, BROKERAGE = 65, 20.0
STT_SELL, EXCH_GST = 0.0001, 0.00004
CAPITAL = 150000.0


def load(label):
    df = pd.read_parquet(STORE)
    df = df[(df.interval == "5m") & (df.label == label)].copy()
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df.sort_values("datetime_ist").reset_index(drop=True)
    df["date"] = df.datetime_ist.dt.date
    df["hm"] = df.datetime_ist.dt.strftime("%H:%M")
    return df


def orb(df, orb_min, hold_days, slip=0.0002):
    daily_close = df.groupby("date").close.last()
    daily_idx = {d: i for i, d in enumerate(daily_close.index)}
    trades = []
    for d, g in df.groupby("date"):
        g = g.sort_values("datetime_ist")
        cutoff = (pd.Timestamp(f"2000-01-01 {g.hm.iloc[0]}")
                  + pd.Timedelta(minutes=orb_min)).strftime("%H:%M")
        orbw = g[g.hm <= cutoff]
        after = g[g.hm > cutoff]
        if len(orbw) < 3 or after.empty:
            continue
        hi, lo = orbw.high.max(), orbw.low.min()
        i = daily_idx[d]
        if i + hold_days >= len(daily_close):
            continue
        exit_px = float(daily_close.iloc[i + hold_days])
        for _, r in after.iterrows():
            if r.high >= hi:
                # entry with slippage against you
                e = hi * (1 + slip)
                trades.append({"date": d, "dir": 1, "entry": e, "exit": exit_px,
                               "trend": 1 if exit_px > hi else 0})
                break
            if r.low <= lo:
                e = lo * (1 - slip)
                trades.append({"date": d, "dir": -1, "entry": e, "exit": exit_px,
                               "trend": 1 if exit_px < lo else 0})
                break
    return trades


def net_rets(trades):
    notional = LOT * np.mean([t["entry"] for t in trades])
    cf = (2 * BROKERAGE + notional * STT_SELL + notional * EXCH_GST
          + notional * 0.0002) / notional
    r = np.array([(t["exit"] / t["entry"] - 1) * t["dir"] for t in trades])
    return r - cf, r


def summarize(name, trades):
    if len(trades) < 10:
        print(f"  {name:<34} n={len(trades)} (too few)")
        return None
    net, _ = net_rets(trades)
    ts = net.mean() / (net.std() / np.sqrt(len(net))) if net.std() > 0 else 0
    print(f"  {name:<34} n={len(net):<4} win={100*(net>0).mean():3.0f}% "
          f"net={net.mean()*100:+.3f}% t={ts:+.2f}")
    return net.mean(), ts


def main():
    print("=== ORB-2d HONESTY GATES ===")
    df_n = load("NIFTY50")
    df_b = load("BANKNIFTY")

    print("\nGATE 1 — CROSS-INDEX (real edge works on both):")
    tn = orb(df_n, 15, 2)
    tb = orb(df_b, 15, 2)
    mn = summarize("Nifty ORB-2d", tn)
    mb = summarize("BankNifty ORB-2d", tb)
    if mn and mb:
        both = mn[0] > 0 and mb[0] > 0
        print(f"  -> both positive: {'YES (real candidate)' if both else 'NO (one-index = noise/regime)'}")

    print("\nGATE 2 — TIME HOLDOUT (Nifty ORB-2d):")
    days = sorted(set(t["date"] for t in tn))
    mid = days[len(days) // 2]
    for nm, grp in [("first 30d", [t for t in tn if t["date"] < mid]),
                    ("last 30d", [t for t in tn if t["date"] >= mid])]:
        summarize(nm, grp)

    print("\nGATE 3 — PARAM SENSITIVITY (Nifty, plateau or spike?):")
    print(f"  {'orb_min':>8}{'hold':>6}{'net%/tr':>10}{'t-stat':>9}")
    for om in [10, 15, 20, 30]:
        for hd in [2, 3]:
            t = orb(df_n, om, hd)
            if len(t) >= 10:
                net, _ = net_rets(t)
                ts = net.mean() / (net.std() / np.sqrt(len(net)))
                print(f"  {om:>7}m{hd:>5}d{net.mean()*100:>9.3f}%{ts:>+9.2f}")

    print("\nGATE 4 — REGIME (Nifty ORB-2d, split by break direction):")
    longs = [t for t in tn if t["dir"] == 1]
    shorts = [t for t in tn if t["dir"] == -1]
    summarize("long breaks only", longs)
    summarize("short breaks only", shorts)

    print("\nGATE 5 — COST SENSITIVITY (Nifty ORB-2d, 2x slippage):")
    t2 = orb(df_n, 15, 2, slip=0.0004)
    summarize("ORB-2d 2x slippage", t2)

    print("\n=== VERDICT ===")
    print("  REAL edge needs: both indices positive + both holdout halves")
    print("  positive + param plateau + survives 2x slippage. Otherwise fake.")


if __name__ == "__main__":
    main()
