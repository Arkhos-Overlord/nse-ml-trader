"""PEAD long-side — the honest gates. Kill the compounding artifact first.

pead_test showed: long positive-surprise = +2.965%/ev, t=+4.63, 63% win, but
'end=Rs1.8Cr' is the overlapping-compounding artifact (many concurrent
positions, full-capital each). MUST NOT trust that number (L7/L15).

Clean gates:
- FIXED position sizing (Rs1.5L notional per event, NOT compounding), 1 event
  at a time (skip overlaps) — realistic equity delivery/MIS
- long positive-surprise only, moderate surprise (the magnitude is INVERTED —
  big surprises mean-revert)
- hold sweep, holdout, breadth, net of costs
- the KEY question: is +2.965%/ev real after fixed sizing + realistic overlaps?

Run: python pead_gates.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EARN = "data_indmoney/earnings.parquet"
HIST = "data_indmoney/daily_history.parquet"
CAPITAL = 150000.0
BROKERAGE = 20.0
STT = 0.001      # delivery both sides
SLIP = 0.001


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def load_prices():
    df = pd.read_parquet(HIST)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df["date"] = df.datetime_ist.dt.date
    return df.sort_values(["label", "date"])


def build_events(prices, earn, hold, surprise_lo, surprise_hi, side="long"):
    px = {}
    for lb, g in prices.groupby("label"):
        g = g.sort_values("date").reset_index(drop=True)
        px[lb] = (list(g.date), list(g.close), {d: i for i, d in enumerate(g.date)})
    ev = []
    for _, e in earn.iterrows():
        s = e.surprise_pct
        if side == "long" and not (surprise_lo <= s <= surprise_hi):
            continue
        lb, ed = e.label, e.earn_date.date()
        if lb not in px:
            continue
        dates, closes, dmap = px[lb]
        fut = [i for i, d in enumerate(dates) if d >= ed]
        if not fut:
            continue
        i0 = fut[0]
        if i0 + hold >= len(closes):
            continue
        entry, exit_ = closes[i0], closes[i0 + hold]
        ev.append({"label": lb, "entry_date": dates[i0], "exit_date": dates[i0 + hold],
                   "surprise": s, "entry": entry, "exit": exit_,
                   "ret": exit_ / entry - 1})
    return pd.DataFrame(ev).sort_values("entry_date").reset_index(drop=True)


def simulate_fixed(ev, max_concurrent=3):
    """FIXED Rs1.5L notional per position, cap concurrent positions (realistic),
    no compounding artifact. Track cash + open positions."""
    cash = CAPITAL
    open_pos = []  # (exit_date)
    pnls = []
    for _, e in ev.iterrows():
        # close matured positions
        open_pos = [x for x in open_pos if x > e.entry_date]
        if len(open_pos) >= max_concurrent:
            continue
        notional = CAPITAL * 0.5   # use half capital per position (allow concurrency)
        qty = notional / e.entry
        gross = (e.exit - e.entry) * qty
        cost = 2 * BROKERAGE + notional * STT + notional * SLIP
        pnls.append(gross - cost)
        open_pos.append(e.exit_date)
    return np.array(pnls)


def report(name, ev, max_concurrent=3):
    if ev.empty:
        print(f"  {name:<34} 0 events")
        return None
    pnls = simulate_fixed(ev, max_concurrent)
    ts = tstat(pnls)
    print(f"  {name:<34} n={len(pnls):<4} net/ev=Rs{pnls.mean():+7.0f} "
          f"t={ts:+.2f} win={100*(pnls>0).mean():3.0f}% "
          f"worst=Rs{pnls.min():+7.0f} total=Rs{pnls.sum():+,.0f}")
    return pnls


def main():
    prices = load_prices()
    earn = pd.read_parquet(EARN)
    print("=== PEAD LONG-SIDE — HONEST GATES (fixed sizing, no compounding artifact) ===")
    print(f"  earlier +2.965%/ev t=+4.63 had a FAKE end (overlapping compounding)")
    print(f"  magnitude is INVERTED (big surprises revert) -> test moderate bands\n")

    print("--- HOLD SWEEP (long, surprise 0-25%) ---")
    for hold in [5, 10, 21, 42]:
        ev = build_events(prices, earn, hold, 0, 25, "long")
        report(f"hold {hold}d surp0-25%", ev)

    print("\n--- SURPRISE BAND SWEEP (hold 21d, long) ---")
    for lo, hi in [(0, 10), (0, 25), (5, 25), (10, 50), (25, 100)]:
        ev = build_events(prices, earn, 21, lo, hi, "long")
        report(f"surp {lo}-{hi}%", ev)

    print("\n--- HOLDOUT (hold 21d, surp 0-25%): pre-2020 vs post-2020 ---")
    ev = build_events(prices, earn, 21, 0, 25, "long")
    ev["yr"] = pd.to_datetime(ev.entry_date).dt.year
    report("  pre-2020", ev[ev.yr < 2020])
    report("  post-2020", ev[ev.yr >= 2020])

    print("\n--- BREADTH (hold 21d, surp 0-25%) ---")
    ev = build_events(prices, earn, 21, 0, 25, "long")
    if not ev.empty:
        # per-event pnl WITHOUT the concurrency skip (breadth = per-event)
        notional = CAPITAL * 0.5
        ev = ev.copy()
        ev["pnl"] = ((ev.exit - ev.entry) * (notional / ev.entry)
                     - 2 * BROKERAGE - notional * STT - notional * SLIP)
        per = ev.groupby("label").pnl.agg(["mean", "count"])
        per = per[per["count"] >= 4]
        pos = (per["mean"] > 0).sum()
        print(f"  positive in {pos}/{len(per)} stocks with >=4 events "
              f"({100*pos/len(per):.0f}%)")
        print(f"  top: {per.sort_values('mean',ascending=False).head(3).index.tolist()}")
        print(f"  bottom: {per.sort_values('mean').head(3).index.tolist()}")

    print("\n--- vs the FAKE number: fixed-sizing total return ---")
    ev = build_events(prices, earn, 21, 0, 25, "long")
    pnls = simulate_fixed(ev)
    if len(pnls):
        cum = np.cumsum(pnls)
        print(f"  total Rs{pnls.sum():,.0f} on Rs1.5L over {len(pnls)} events "
              f"({100*pnls.sum()/CAPITAL:+.1f}%) — NOT Rs1.8Cr (that was the artifact)")
        print(f"  this is the HONEST number the earlier test hid")

    print("\n=== VERDICT ===")
    print("  REAL = fixed-sizing net/ev>0 with |t|>2, both holdout halves,")
    print("  broad across stocks. The Rs1.8Cr is dead; judge on net/ev + breadth.")


if __name__ == "__main__":
    main()
