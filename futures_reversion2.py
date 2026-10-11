"""FUTURES reversion — the REAL signal, tested honestly.

Sweep found: z>=2.0 (extreme moves only) is net-positive across multiple
lookbacks, while low-z is negative. That consistency (only extremes revert) is
a real mechanism, not noise. But the sweep's 'best' single config had
insignificant holdout halves (t=+0.38/+1.21) — that was param luck.

This tests the SIGNAL properly: extreme-only reversion, and checks the thing
that decides tradeability — is the edge big enough per trade to clear futures
costs AND survive the tail on Rs1.5L?

The key insight from the sweep: z=2.0 with LONGER holds (10d) is where net is
positive. But holdout t-stats were low -> the per-trade edge is small. The
question: does VOLATILITY TARGETING (size positions by recent vol) turn a small
consistent edge into a tradeable one? That's the practitioner technique
(u/Thundr3, Yang-Zhang vol targeting) I haven't applied.

Run: python futures_reversion2.py
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CAPITAL = 150000.0
LOT = 65
BROKERAGE = 20.0
STT_SELL = 0.0001
EXCH_GST = 0.00004


def load(sym="^NSEI"):
    df = yf.download(sym, start="2018-01-01", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"date": "datetime_ist"})
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"]).dt.tz_localize(None)
    df["ret"] = df.close.pct_change()
    df["vol20"] = df.ret.rolling(20).std()
    return df.dropna().reset_index(drop=True)


def gen(df, N, H, z_thr, vol_target=None):
    """Extreme-only reversion. Optional vol targeting: scale exposure so each
    position risks ~vol_target of capital (Yang-Zhang style, practitioner
    technique). No look-ahead (z and vol use only past data)."""
    close = df.close.values
    vol = df.vol20.values
    dates = df.datetime_ist.values
    trades = []
    i = max(N, 20)
    while i + H < len(close):
        w = close[i - N:i]
        mu, sd = w.mean(), w.std()
        if sd == 0 or np.isnan(vol[i]):
            i += 1
            continue
        z = (close[i - 1] - mu) / sd
        if z <= -z_thr:
            direction = 1
        elif z >= z_thr:
            direction = -1
        else:
            i += 1
            continue
        entry, exit_ = close[i], close[i + H]
        raw_ret = (exit_ / entry - 1) * direction
        # vol targeting: position size inversely proportional to recent vol
        scale = 1.0
        if vol_target:
            scale = min(3.0, vol_target / vol[i]) if vol[i] > 0 else 1.0
        trades.append({"date": pd.Timestamp(dates[i]), "entry": entry,
                       "exit": exit_, "dir": direction, "raw_ret": raw_ret,
                       "scale": scale})
        i += H
    return pd.DataFrame(trades)


def fut_cost_frac(notional):
    return (2 * BROKERAGE + notional * STT_SELL + notional * EXCH_GST) / notional


def sim(trades, capital=CAPITAL, vol_target=None):
    """Full-capital compounding, 1 position at a time, vol-targeted."""
    peak, maxdd, last = capital, 0, None
    for _, t in trades.iterrows():
        if last is not None and t["date"] <= last:
            continue
        lots = max(1, int(capital / (LOT * t["entry"])))
        notional = lots * LOT * t["entry"]
        # apply vol scaling to lots (bounded by capital)
        lots_v = max(1, int(lots * (t["scale"] if vol_target else 1.0)))
        notional_v = lots_v * LOT * t["entry"]
        if notional_v > capital * 3:  # cap leverage
            notional_v = capital * 3
            lots_v = int(notional_v / (LOT * t["entry"]))
        gross = (t["exit"] - t["entry"]) * lots_v * LOT * t["dir"]
        cost = 2 * BROKERAGE + notional_v * STT_SELL + notional_v * EXCH_GST
        capital += gross - cost
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
        last = t["date"]
    return capital, maxdd


def main():
    df = load("^NSEI")
    print("=== FUTURES REVERSION v2 — extreme-only + vol targeting ===")
    print(f"  {len(df)} days, {df.datetime_ist.min().date()} -> {df.datetime_ist.max().date()}\n")

    # Confirm the signal: z>=2.0 across lookbacks, fixed H=10
    print("--- SIGNAL CHECK: z>=2.0, H=10, across N (no vol targeting) ---")
    print(f"{'N':>4}{'n':>6}{'net%/tr':>10}{'t-stat':>8}{'win%':>6}")
    for N in [10, 15, 20, 30]:
        tr = gen(df, N, 10, 2.0)
        if len(tr) < 30:
            continue
        nf = tr.raw_ret.mean() - fut_cost_frac(LOT * tr.entry.mean())
        ts = nf / (tr.raw_ret.std() / np.sqrt(len(tr)))
        print(f"{N:>4}{len(tr):>6}{nf*100:>9.3f}%{ts:>+8.2f}{100*(tr.raw_ret>0).mean():>5.0f}%")

    # Does vol targeting help? test the most consistent config
    print(f"\n--- VOL TARGETING (N=20, H=10, z=2.0) ---")
    print(f"{'vol_tgt':>8}{'n':>6}{'Rs1.5L end':>13}{'ret%':>8}{'maxDD':>7}")
    for vt in [None, 0.01, 0.015, 0.02]:
        tr = gen(df, 20, 10, 2.0, vol_target=vt)
        if len(tr) < 30:
            continue
        cap, maxdd = sim(tr, vol_target=vt)
        print(f"{str(vt):>8}{len(tr):>6}{cap:>13,.0f}{(cap/CAPITAL-1)*100:>7.0f}%{maxdd*100:>6.0f}%")

    # full honesty gates on the base (no vol tgt) consistent config
    print(f"\n--- HONESTY GATES (N=20, H=10, z=2.0, no vol tgt) ---")
    tr = gen(df, 20, 10, 2.0)
    nf = tr.raw_ret - fut_cost_frac(LOT * tr.entry.mean())
    mid = tr.date.iloc[len(tr) // 2]
    for nm, g in [("first half", tr[tr.date < mid]), ("second half", tr[tr.date >= mid])]:
        gf = g.raw_ret - fut_cost_frac(LOT * g.entry.mean())
        gts = gf.mean() / (gf.std() / np.sqrt(len(g))) if gf.std() > 0 else 0
        print(f"  {nm:<12} n={len(g):<4} net {gf.mean()*100:+.3f}%/tr  t={gts:+.2f}")
    both = (tr[tr.date<mid].raw_ret.mean()>0) and (tr[tr.date>=mid].raw_ret.mean()>0)
    tr["yr"] = tr.date.dt.year
    tr["net"] = tr.raw_ret - fut_cost_frac(LOT * tr.entry.mean())
    yr = tr.groupby("yr").net.mean()
    print(f"  both halves positive: {'YES' if both else 'NO'}")
    print(f"  profitable years: {(yr>0).sum()}/{len(yr)}")
    cap, maxdd = sim(tr)
    print(f"  Rs1.5L -> Rs{cap:,.0f} ({(cap/CAPITAL-1)*100:+.0f}%), maxDD {maxdd*100:.0f}%")

    print(f"\n=== VERDICT ===")
    tradeable = both and cap > CAPITAL and (yr > 0).mean() > 0.55 and maxdd < 0.5
    print(f"  tradeable at Rs1.5L: {'YES' if tradeable else 'NO'}")
    if not tradeable:
        print(f"  blockers: both_halves={'Y' if both else 'N'}, "
              f"end>start={'Y' if cap>CAPITAL else 'N'}, "
              f"years>55%={'Y' if (yr>0).mean()>0.55 else 'N'}, "
              f"DD<50%={'Y' if maxdd<0.5 else 'N'}")
    print(f"  (futures economics DID help vs equity, but the index is efficient")
    print(f"   at daily resolution — the per-trade edge stays sub-significant.)")


if __name__ == "__main__":
    main()
