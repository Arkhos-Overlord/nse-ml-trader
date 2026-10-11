"""ETH edge refinement — push quarterly consistency above 60%.

Current ETH edge is 2-sided (real) but 6/11 quarters (55%, regime-dependent).
Goal: find a config that is (a) 2-sided, (b) >=60% positive quarters,
(c) positive both holdout halves. Sweep threshold/lookback/hold and score
each on QUARTERLY consistency, not just total PnL (total hides regime luck).

Run: python eth_refine.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FEE = 0.001
NOTIONAL = 1500.0


def load():
    df = pd.read_parquet("data_indmoney/crypto_daily/ETHUSDT_1d.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def cost(n):
    return n * 2 * FEE


def evaluate(df, look, hold, thresh):
    t = []
    for i in range(max(look, 60), len(df) - hold):
        r = df.close.iloc[i] / df.close.iloc[i - look] - 1
        d = 1 if r > thresh else (-1 if r < -thresh else 0)
        if d == 0:
            continue
        entry = float(df.open.iloc[i + 1])
        exit_ = float(df.close.iloc[min(i + hold, len(df) - 1)])
        if entry <= 0 or exit_ <= 0:
            continue
        qty = NOTIONAL / entry
        t.append({"date": df.datetime.iloc[i], "dir": d,
                  "pnl": (exit_ - entry) * qty * d - cost(NOTIONAL)})
    if len(t) < 50:
        return None
    t = pd.DataFrame(t)
    t["q"] = t.date.dt.to_period("Q")
    q = t.groupby("q").pnl.sum()
    pos_q = (q > 0).sum() / len(q)
    avg = t.pnl.mean()
    win = (t.pnl > 0).mean()
    lng = t[t.dir == 1].pnl.sum()
    sht = t[t.dir == -1].pnl.sum()
    mid = len(df) // 2
    h1 = t[t.date < df.datetime.iloc[mid]].pnl.mean()
    h2 = t[t.date >= df.datetime.iloc[mid]].pnl.mean()
    return {"look": look, "hold": hold, "thresh": thresh, "n": len(t),
            "pos_q": pos_q, "n_q": len(q), "avg": avg, "win": win,
            "two_sided": lng > 0 and sht > 0, "h1": h1, "h2": h2,
            "total": t.pnl.sum()}


def main():
    df = load()
    print("=== ETH REFINEMENT: sweep for >=60% quarterly consistency ===")
    print(f"  {len(df)} days. Scoring on QUARTERLY consistency (hides regime luck)\n")

    results = []
    for look in [14, 30, 60, 90]:
        for hold in [3, 7, 14, 21]:
            for thresh in [0.0, 0.03, 0.05, 0.08]:
                r = evaluate(df, look, hold, thresh)
                if r:
                    results.append(r)

    # filter: 2-sided + both holdout halves positive + avg>fee
    good = [r for r in results if r["two_sided"] and r["h1"] > 0 and r["h2"] > 0
            and r["avg"] > cost(NOTIONAL)]
    good.sort(key=lambda x: (-x["pos_q"], -x["avg"]))

    print(f"{'look':>5}{'hold':>5}{'thr':>6}{'n':>5}{'posQ':>6}{'avg$':>8}{'win%':>6}{'2sided':>8}")
    for r in good[:20]:
        print(f"{r['look']:>5}{r['hold']:>5}{r['thresh']:>6.2f}{r['n']:>5}"
              f"{r['pos_q']*100:>5.0f}%{r['avg']:>8.2f}{r['win']*100:>5.0f}%"
              f"{'YES' if r['two_sided'] else 'no':>8}")

    robust = [r for r in good if r["pos_q"] >= 0.6]
    print(f"\n=== CONFIGS passing >=60% quarterly + 2-sided + both halves ===")
    if robust:
        for r in robust:
            print(f"  look={r['look']} hold={r['hold']} thresh={r['thresh']}: "
                  f"{r['pos_q']*100:.0f}% quarters, avg ${r['avg']:.2f}, "
                  f"{r['n']} trades")
        best = robust[0]
        print(f"\n  *** ROBUST ETH EDGE FOUND ***")
        print(f"  best: look={best['look']} hold={best['hold']} thresh={best['thresh']}")
        print(f"  {best['pos_q']*100:.0f}% positive quarters, avg ${best['avg']:.2f}/trade")
    else:
        print("  NONE reach 60% quarterly consistency.")
        print("  ETH edge is real but regime-dependent — not robust enough to trade.")


if __name__ == "__main__":
    main()
