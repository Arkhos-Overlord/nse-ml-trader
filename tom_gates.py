"""TOM edge — the real gates. Calendar effects decay fastest; prove it's not
a spike before believing it.

GATE 1: YEAR-BY-YEAR consistency (calendar effects are known -> arbitraged;
   a real one persists across the 8.8y, not just recent).
GATE 2: TOM WIDTH sweep (last/first 2,3,4,5 days) — plateau or spike?
GATE 3: NET of a tradable structure. TOM is a long-only daily edge. Two ways
   to trade it: (a) Nifty futures long during TOM (margin ~1 lot), (b) long
   Nifty ETF during TOM (no margin, small capital, tiny spread). Test BOTH net
   of costs on Rs1.5L.
GATE 4: does it beat just buying-and-holding? (edge must be incremental)
GATE 5: regime — is it concentrated in one market regime (bull) or broad?

Run: python tom_gates.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HIST = "data_indmoney/daily_history.parquet"
LOT, BROKERAGE = 65, 20.0
STT_SELL, EXCH_GST, SLIP = 0.0001, 0.00004, 0.0002
CAPITAL = 150000.0


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def load_nifty():
    df = pd.read_parquet(HIST)
    df = df[df.label == "NIFTY50"].copy()
    df["date"] = pd.to_datetime(df["datetime_ist"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    df["ret"] = df.close.pct_change()
    df["dom"] = pd.to_datetime(df.date).dt.day
    df["yr"] = pd.to_datetime(df.date).dt.year
    return df.dropna(subset=["ret"])


def mark_tom(df, w):
    """last w cal days of month OR first w days of next."""
    dd = pd.to_datetime(df.date)
    dim = dd.dt.days_in_month
    days_from_eom = dim - dd.dt.day
    df["in_tom"] = (days_from_eom < w) | (df.dom <= w)
    return df


def main():
    df = load_nifty()
    print("=== TOM EDGE — REAL GATES (calendar effects decay fastest) ===")
    print(f"  Nifty {df.date.min()} -> {df.date.max()}, {len(df)} days, "
          f"{df.yr.nunique()} years\n")

    print("GATE 2 — WIDTH SWEEP (plateau or spike?):")
    for w in [2, 3, 4, 5]:
        d = mark_tom(df.copy(), w)
        tom = d[d.in_tom]["ret"].values
        print(f"  w={w} (±{w}d): TOM n={len(tom):<4} mean={tom.mean()*100:+.4f}%/d "
              f"t={tstat(tom):+.2f}  rest mean={d[~d.in_tom]['ret'].mean()*100:+.4f}%/d")

    print("\nGATE 1 — YEAR-BY-YEAR (w=3, persistence across 8.8y?):")
    d = mark_tom(df.copy(), 3)
    pos_years = 0
    years = sorted(d.yr.unique())
    for yr in years:
        tom = d[(d.yr == yr) & (d.in_tom)]["ret"].values
        rest = d[(d.yr == yr) & (~d.in_tom)]["ret"].values
        if len(tom) > 3 and len(rest) > 20:
            spread = tom.mean() - rest.mean()
            flag = "+" if spread > 0 else "-"
            if spread > 0:
                pos_years += 1
            print(f"  {yr}: TOM {tom.mean()*100:+.3f}%/d (n={len(tom)}) vs rest "
                  f"{rest.mean()*100:+.3f}%/d -> spread {spread*100:+.3f}% {flag}")
    print(f"  -> positive spread in {pos_years}/{len(years)} years")

    print("\nGATE 4 — NET TRADABLE STRUCTURE on Rs1.5L:")
    d = mark_tom(df.copy(), 3)
    # (a) futures long during TOM, flat otherwise
    for w in [3]:
        d = mark_tom(df.copy(), w)
        cap, peak, maxdd, trades = CAPITAL, CAPITAL, 0, 0
        for _, r in d.iterrows():
            if not r.in_tom:
                continue
            lots = max(1, int(cap / (0.12 * r.close)))  # ~12% margin/lot
            px = r.close
            gross = (d.close.shift(-1).loc[r.name] if False else 0)
            # use next-day close as exit proxy won't work cleanly; use same-day
            # TOM = hold through TOM window; approximate: 1-day hold each TOM day
            trades += 1
        # simpler + honest: model TOM as "buy at TOM-day open, hold to TOM end"
    # (b) ETF long during TOM: cost = 0.1% round trip, no margin
    print("  (a) Nifty FUTURES long in TOM (1 lot, ~Rs1.5L margin):")
    d = mark_tom(df.copy(), 3)
    cap, peak, maxdd = CAPITAL, CAPITAL, 0
    for i in range(len(d) - 1):
        if not d.in_tom.iloc[i]:
            continue
        lots = max(1, int(cap / (0.12 * d.close.iloc[i])))
        nxt = d.close.iloc[i + 1]
        px = d.close.iloc[i]
        gross = (nxt - px) * lots * LOT
        cost = lots * (2 * BROKERAGE + LOT * px * (STT_SELL + EXCH_GST + SLIP))
        cap += gross - cost
        peak = max(peak, cap)
        maxdd = max(maxdd, (peak - cap) / peak)
    bh = (d.close.iloc[-1] / d.close.iloc[0]) * CAPITAL
    print(f"      end Rs{cap:,.0f} ({(cap/CAPITAL-1)*100:+.1f}%) DD={maxdd*100:.0f}% "
          f"vs buy-hold Rs{bh:,.0f} ({(bh/CAPITAL-1)*100:+.0f}%)")

    print("  (b) Nifty ETF long in TOM (0.1% RT cost, no margin, full Rs1.5L):")
    d = mark_tom(df.copy(), 3)
    cap, peak, maxdd = CAPITAL, CAPITAL, 0
    for i in range(len(d) - 1):
        px = d.close.iloc[i]
        nxt = d.close.iloc[i + 1]
        expo = cap * (1 if d.in_tom.iloc[i] else 0.0)
        gross = (nxt - px) / px * expo
        cost = expo * 0.001 if d.in_tom.iloc[i] else 0.0
        cap += gross - cost
        peak = max(peak, cap)
        maxdd = max(maxdd, (peak - cap) / peak)
    print(f"      end Rs{cap:,.0f} ({(cap/CAPITAL-1)*100:+.1f}%) DD={maxdd*100:.0f}% "
          f"vs buy-hold Rs{bh:,.0f} ({(bh/CAPITAL-1)*100:+.0f}%)")

    print("\nGATE 5 — REGIME (is TOM just bull-market drift?):")
    d = mark_tom(df.copy(), 3)
    # split by market trend (200d MA)
    d["ma200"] = d.close.rolling(200).mean()
    d["up"] = d.close > d.ma200
    for reg in [True, False]:
        sub = d[d.up == reg]
        tom = sub[sub.in_tom]["ret"].values
        rest = sub[~sub.in_tom]["ret"].values
        if len(tom) > 5:
            print(f"  {'BULL' if reg else 'BEAR'}: TOM {tom.mean()*100:+.3f}%/d "
                  f"(t={tstat(tom):+.2f}) vs rest {rest.mean()*100:+.3f}%/d "
                  f"-> spread {(tom.mean()-rest.mean())*100:+.3f}%")

    print("\n=== VERDICT ===")
    print("  REAL TOM edge = positive spread in MOST years + broad across regimes")
    print("  + net tradable structure beats buy-hold. Spike-in-one-year = decayed.")


if __name__ == "__main__":
    main()
