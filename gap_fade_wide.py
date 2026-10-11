"""GAP-FADE on the WIDE universe (57 liquid F&O stocks) — resolve the lead.

Standing lead: short-side single-stock gap-fade (fade large UP gaps), was
t=+2.34 on n=6 / 12 stocks. Too small. Now: 57 stocks x 60 days. Run the
honest gates at full sample:

- full universe short-side + long-side
- year... no, 60d -> holdout halves
- per-stock breadth (real edge is broad, not 1-2 lucky names)
- 'stock in play' volume-catalyst filter (Zarattini: the edge NEEDS the catalyst)
- net of MIS equity costs, full Rs1.5L compounding
- the LIT SIGNATURE: short-biased, high hit, small tail (few big winners)

Run: python gap_fade_wide.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/intraday_stocks_wide.parquet"
CAPITAL = 150000.0
LEVERAGE = 5.0
BROKERAGE = 20.0
STT_SELL = 0.00025
EXCH_GST = 0.00004
SLIP = 0.001


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def load():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df["date"] = df.datetime_ist.dt.date
    df["hm"] = df.datetime_ist.dt.strftime("%H:%M")
    return df.sort_values(["label", "datetime_ist"])


def equity_cost_frac():
    return (2 * BROKERAGE + STT_SELL + EXCH_GST + SLIP)  # per unit exposure


def gap_fade(df, gap_thr=0.02, vol_mult=None, require_vol=False):
    trades = []
    for lb, g in df.groupby("label"):
        daily = g.groupby("date").agg(o=("open", "first"), c=("close", "last"))
        daily = daily.sort_index()
        dates = list(daily.index)
        # average first-hour volume for catalyst filter
        fh = g[g.hm <= "10:15"].groupby("date").volume.sum()
        avg_fh = fh.mean() if len(fh) > 5 else 0
        for i in range(1, len(dates)):
            prev_close = daily.c.iloc[i - 1]
            o = daily.o.iloc[i]
            gap = o / prev_close - 1
            if abs(gap) < gap_thr:
                continue
            if require_vol and avg_fh > 0:
                if fh.get(dates[i], 0) < vol_mult * avg_fh:
                    continue
            direction = -1 if gap > 0 else 1
            entry = o * (1 + SLIP * direction)
            exit_ = daily.c.iloc[i]
            trades.append({"label": lb, "date": dates[i], "dir": direction,
                           "entry": entry, "exit": exit_, "gap": gap})
    return trades


def net_per_trade(t):
    """Net Rs on a fixed MIS exposure (Rs7.5L = 1.5L x 5)."""
    expo = CAPITAL * LEVERAGE
    qty = expo / t["entry"]
    gross = (t["exit"] - t["entry"]) * qty * t["dir"]
    cost = 2 * BROKERAGE + expo * STT_SELL + expo * EXCH_GST + expo * SLIP
    return gross - cost


def simulate(trades):
    """1 position at a time (one per day), compound, net."""
    capital, peak, maxdd = CAPITAL, CAPITAL, 0
    last = None
    for t in sorted(trades, key=lambda x: x["date"]):
        if last is not None and t["date"] <= last:
            continue
        capital += net_per_trade(t)
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
        last = t["date"]
    return capital, maxdd


def report(name, trades):
    if not trades:
        print(f"  {name:<30} 0 trades")
        return None
    net = np.array([net_per_trade(t) for t in trades])
    cap, maxdd = simulate(trades)
    n = len(trades)
    ts = tstat(net)
    print(f"  {name:<30} n={n:<4} hit={100*(net>0).mean():3.0f}% "
          f"net/tr=Rs{net.mean():+7.0f} med=Rs{np.median(net):+7.0f} "
          f"t={ts:+.2f} best=Rs{net.max():+7.0f} worst=Rs{net.min():+7.0f} "
          f"end=Rs{cap:,.0f} DD={maxdd*100:.0f}%")
    return net


def main():
    df = load()
    print("=== GAP-FADE WIDE UNIVERSE (57 liquid F&O stocks) ===")
    print(f"  lit signature: short-biased, high hit, small tail\n")
    for thr in [0.02, 0.03]:
        print(f"  --- gap > {thr*100:.0f}% (no vol filter) ---")
        tr = gap_fade(df, thr)
        report("all (long+short)", tr)
        report("  SHORT only (fade up)", [t for t in tr if t["dir"] == -1])
        report("  LONG only (fade down)", [t for t in tr if t["dir"] == 1])
    print()
    for thr in [0.02, 0.03]:
        print(f"  --- gap > {thr*100:.0f}% + volume catalyst (1.5x) ---")
        tr = gap_fade(df, thr, require_vol=True, vol_mult=1.5)
        report("all", tr)
        report("  SHORT only", [t for t in tr if t["dir"] == -1])
        report("  LONG only", [t for t in tr if t["dir"] == 1])

    # the key gate: per-stock breadth on the short side
    print("\n--- PER-STOCK BREADTH (short-side, gap>2%, no vol filter) ---")
    tr = [t for t in gap_fade(df, 0.02) if t["dir"] == -1]
    pos, tot = 0, 0
    for lb in sorted(set(t["label"] for t in tr)):
        grp = [t for t in tr if t["label"] == lb]
        net = np.array([net_per_trade(t) for t in grp])
        tot += 1
        if net.mean() > 0:
            pos += 1
        print(f"  {lb:<12} n={len(grp)} hit={100*(net>0).mean():3.0f}% "
              f"net/tr=Rs{net.mean():+7.0f}")
    print(f"  -> positive in {pos}/{tot} stocks (broad={pos/tot*100:.0f}%)")

    # holdout
    print("\n--- HOLDUT (short-side, gap>2%) ---")
    dates = sorted(set(t["date"] for t in tr))
    mid = dates[len(dates) // 2]
    for nm, grp in [("first 30d", [t for t in tr if t["date"] < mid]),
                    ("last 30d", [t for t in tr if t["date"] >= mid])]:
        if grp:
            net = np.array([net_per_trade(t) for t in grp])
            print(f"  {nm:<10} n={len(grp):<3} hit={100*(net>0).mean():3.0f}% "
                  f"net/tr=Rs{net.mean():+7.0f} t={tstat(net):+.2f}")

    print("\n=== VERDICT ===")
    print("  REAL = short-side positive, broad across MOST stocks, both holdout")
    print("  halves positive, |t|>2. Concentrated or one-half = fake.")


if __name__ == "__main__":
    main()
