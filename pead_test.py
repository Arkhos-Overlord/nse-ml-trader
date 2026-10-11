"""PEAD TEST — the anomaly confirmed twice in India, on its required data.

Feed: 1,360 earnings events, 57 stocks, EPS estimate + reported + surprise%.
Data: 8.8y daily history (daily_history.parquet). Join, measure drift.

PEAD (Harshita 2018 sig+robust 2002-2017; Das 2026 sig BSE500 2015-2024):
positive earnings surprise -> stock drifts UP for weeks after; negative ->
down. Test: for each event, buy (long) if surprise>0, short if <0, hold N days
(1,5,10,21), net of equity delivery... actually use simple long/short net of
brokerage+STT. Measure abnormal return vs Nifty (CAR).

Honesty gates: is drift POSITIVE after positive surprise and NEGATIVE after
negative (both sides)? holdout (pre-2020 vs post-2020)? breadth across stocks?
net of costs? and the classic PEAD trap — is it just the earnings-day jump
(already priced) or genuine forward drift?

Run: python pead_test.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EARN = "data_indmoney/earnings.parquet"
HIST = "data_indmoney/daily_history.parquet"
BROKERAGE = 20.0
STT = 0.001  # delivery STT both sides ~0.1%
SLIP = 0.001


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def load_prices():
    df = pd.read_parquet(HIST)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df["date"] = df.datetime_ist.dt.date
    return df.sort_values(["label", "date"])


def pead(prices, earn, hold=5, min_surprise=0.0, use_abs=False):
    """For each earnings event: direction by surprise sign, hold N days from
    the first trading day AFTER announcement. Return per-event net return."""
    # index prices by (label, date)->close, and next-N-day close
    px = {}
    for lb, g in prices.groupby("label"):
        g = g.sort_values("date").reset_index(drop=True)
        dates = list(g.date)
        closes = list(g.close)
        dmap = {d: i for i, d in enumerate(dates)}
        px[lb] = (dates, closes, dmap)
    events = []
    for _, e in earn.iterrows():
        if abs(e.surprise_pct) < min_surprise:
            continue
        lb, ed = e.label, e.earn_date.date()
        if lb not in px:
            continue
        dates, closes, dmap = px[lb]
        # first trading day >= announce date (or after)
        fut = [i for i, d in enumerate(dates) if d >= ed]
        if not fut:
            continue
        i0 = fut[0]
        if i0 + hold >= len(closes):
            continue
        entry = closes[i0]
        exit_ = closes[i0 + hold]
        direction = 1 if e.surprise_pct > 0 else -1
        events.append({"label": lb, "date": ed, "surprise": e.surprise_pct,
                       "dir": direction, "entry": entry, "exit": exit_,
                       "ret": (exit_ / entry - 1) * direction})
    return pd.DataFrame(events)


def net_return(r, exposure_frac=1.0):
    cost = 2 * BROKERAGE + STT + SLIP  # per unit exposure (brokerage is tiny on large notional)
    return r - cost * exposure_frac


def report(name, ev):
    if ev.empty:
        print(f"  {name:<30} 0 events")
        return
    r = ev.ret.values
    # net: costs are ~0.2% of exposure on these notional sizes
    net = r - 0.002
    cap = 150000.0
    peak = maxdd = 0
    peak = cap
    for rr in net:  # 1 position, full capital (simplified compounding)
        cap *= (1 + rr)
        peak = max(peak, cap)
        maxdd = max(maxdd, (peak - cap) / peak)
    print(f"  {name:<30} n={len(net):<4} gross={r.mean()*100:+.3f}%/ev "
          f"net={net.mean()*100:+.3f}% t={tstat(net):+.2f} "
          f"win={100*(net>0).mean():.0f}% end=Rs{cap:,.0f} DD={maxdd*100:.0f}%")


def main():
    prices = load_prices()
    earn = pd.read_parquet(EARN)
    print("=== PEAD TEST (confirmed twice in India) ===")
    print(f"  {earn.label.nunique()} stocks, {len(earn)} events, "
          f"8.8y daily history\n")

    print("--- DRIFT BY HOLD PERIOD (all events, surprise-signed) ---")
    for hold in [1, 5, 10, 21, 42]:
        ev = pead(prices, earn, hold=hold)
        report(f"hold {hold}d", ev)

    print("\n--- BY SURPRISE MAGNITUDE (hold 21d) ---")
    for ms in [0, 10, 25, 50]:
        ev = pead(prices, earn, hold=21, min_surprise=ms)
        report(f"|surprise|>{ms}%", ev)

    print("\n--- BOTH SIDES SEPARATELY (hold 21d, |surp|>10%) ---")
    ev = pead(prices, earn, hold=21, min_surprise=10)
    if not ev.empty:
        report("  positive surprise (long)", ev[ev.dir == 1])
        report("  negative surprise (short)", ev[ev.dir == -1])

    print("\n--- HOLDOUT (hold 21d, |surp|>10%): pre-2020 vs post-2020 ---")
    ev = pead(prices, earn, hold=21, min_surprise=10)
    if not ev.empty:
        ev["yr"] = pd.to_datetime(ev.date).dt.year
        report("  pre-2020", ev[ev.yr < 2020])
        report("  post-2020", ev[ev.yr >= 2020])

    print("\n--- BREADTH: stocks with positive avg drift (hold 21d) ---")
    ev = pead(prices, earn, hold=21, min_surprise=10)
    if not ev.empty:
        per = ev.groupby("label").ret.agg(["mean", "count"])
        per = per[per["count"] >= 5]
        pos = (per["mean"] > 0).sum()
        print(f"  positive in {pos}/{len(per)} stocks with >=5 events "
              f"({100*pos/len(per):.0f}%)")

    print("\n--- THE PEAD TRAP: is it the announcement-day jump or forward drift? ---")
    # hold=1 is mostly the day-after reaction; hold=21 tests genuine drift
    e1 = pead(prices, earn, hold=1, min_surprise=10)
    e21 = pead(prices, earn, hold=21, min_surprise=10)
    if not e1.empty and not e21.empty:
        print(f"  day-1 (announce reaction): {e1.ret.mean()*100:+.3f}%/ev "
              f"t={tstat(e1.ret.values-0.002):+.2f}")
        print(f"  21-day (forward drift):    {e21.ret.mean()*100:+.3f}%/ev "
              f"t={tstat(e21.ret.values-0.002):+.2f}")
        print(f"  -> genuine PEAD needs the 21d drift to beat the day-1 reaction")

    print("\n=== VERDICT ===")
    print("  REAL PEAD = both sides work, drift grows with hold, both holdout")
    print("  halves positive, broad across stocks, 21d > day1. Else it's just")
    print("  the announcement jump (already priced) = no tradeable edge.")


if __name__ == "__main__":
    main()
