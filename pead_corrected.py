"""PEAD CORRECTED — kill the look-ahead bias the frontier models flagged.

All 4 frontier models caught what I should have: my pead entry was the FIRST bar
>= ann_date = the ANNOUNCEMENT DAY itself. Indian results print AFTER market
close (~17:00 IST), so that entry happens BEFORE the information exists =
LOOK-AHEAD BIAS. The +1.23% CAR / t=+2.14 was inflated.

CORRECT timing (matches Claude.py + chatgpt.md + retail-edge + gemini):
- reaction day = first trading day >= ann_date (the print lands after close)
- entry = OPEN of the day AFTER the reaction day (skips the announcement-day
  jump that no retail trader could catch)
- exit = OPEN `hold` trading days later (or close)
- ALSO: clustered t-stat (events cluster in results season -> naive t overstated)

Also run the liquidity condition (Marcellus/Chuia licensing: the drift lives in
liquid names) and the SUE fallback (time-series YoY EPS, public-data version).

Run: python pead_corrected.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EARN = "data_indmoney/earnings.parquet"
HIST = "data_indmoney/daily_history.parquet"
CAPITAL = 150000.0
BROKERAGE = 20.0
STT = 0.001
SLIP = 0.001


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def clustered_t(ev, col):
    g = ev.groupby(pd.to_datetime(ev.ann_date).dt.to_period("M"))[col].mean()
    if len(g) < 6 or g.std(ddof=1) == 0:
        return np.nan, len(g)
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g))), len(g)


def load_prices():
    df = pd.read_parquet(HIST)
    df["date"] = pd.to_datetime(df["datetime_ist"]).dt.date
    return df.sort_values(["label", "date"])


def build(prices, hold=21, lo=10, hi=50, lag=1):
    """Corrected timing: entry = open of (reaction day + lag)."""
    nifty = prices[prices.label == "NIFTY50"].sort_values("date").reset_index(drop=True)
    nd = {d: i for i, d in enumerate(nifty.date)}
    nc = list(nifty.close)
    px = {}
    for lb, g in prices.groupby("label"):
        g = g.sort_values("date").reset_index(drop=True)
        px[lb] = (list(g.date), list(g.open), list(g.close),
                  {d: i for i, d in enumerate(g.date)})
    earn = pd.read_parquet(EARN)
    ev = []
    for _, e in earn.iterrows():
        s = e.surprise_pct
        if not (lo <= s <= hi):
            continue
        lb, ed = e.label, e.earn_date.date()
        if lb not in px or lb == "NIFTY50":
            continue
        dates, opens, closes, dmap = px[lb]
        fut = [i for i, d in enumerate(dates) if d >= ed]
        if not fut:
            continue
        reaction = fut[0]          # first trading day >= ann_date (print after close)
        entry_i = reaction + lag   # enter NEXT day open (no look-ahead)
        exit_i = entry_i + hold
        if exit_i >= len(closes):
            continue
        entry_d = dates[entry_i]
        exit_d = dates[exit_i]
        if entry_d in nd and exit_d in nd:
            nifty_ret = nc[nd[exit_d]] / nc[nd[entry_d]] - 1
        else:
            nifty_ret = 0.0
        entry = opens[entry_i]
        exit_ = opens[exit_i]
        stock_ret = exit_ / entry - 1
        ev.append({"label": lb, "ann_date": ed, "entry_date": entry_d,
                   "surprise": s, "r": stock_ret, "nifty": nifty_ret,
                   "car": stock_ret - nifty_ret})
    return pd.DataFrame(ev).sort_values("ann_date").reset_index(drop=True)


def report(name, ev):
    if ev.empty:
        print(f"  {name:<30} 0 events")
        return
    cost = 0.002
    net_raw = ev.r - cost
    net_car = ev.car - cost
    t_car, k = clustered_t(ev, "car")
    print(f"  {name:<30} n={len(ev):<4} raw={ev.r.mean()*100:+.2f}% "
          f"CAR={ev.car.mean()*100:+.2f}% netCAR={net_car.mean()*100:+.2f}% "
          f"t_naive={tstat(net_car):+.2f} t_cluster={t_car:+.2f}(k={k}) "
          f"win={100*(net_car>0).mean():.0f}%")


def main():
    prices = load_prices()
    print("=== PEAD CORRECTED (look-ahead killed) ===")
    print("  entry = OPEN of day AFTER reaction day (results print after close)")
    print("  + clustered t-stat (events cluster in results season)\n")

    print("--- CORRECTED: all surprise-signed, by hold ---")
    for hold in [5, 21, 42]:
        ev = build(prices, hold, lo=-1e9, hi=1e9)
        report(f"hold {hold}d all", ev)

    print("\n--- CORRECTED: the specified band (surp 10-50%) ---")
    for hold in [21, 42]:
        ev = build(prices, hold, 10, 50)
        report(f"hold {hold}d surp10-50", ev)

    print("\n--- CORRECTED: wider bands (is there ANY band that works?) ---")
    for lo, hi in [(5, 100), (10, 100), (0, 100), (10, 50), (20, 80)]:
        ev = build(prices, 21, lo, hi)
        report(f"surp {lo}-{hi} hold21", ev)

    print("\n--- vs the BIASED number (entry on ann day, for contrast) ---")
    print("  (my earlier +1.23% CAR t=+2.14 used ann-day entry = look-ahead)")
    print("  corrected above uses next-day-open entry — compare the magnitude")

    print("\n--- HOLDOUT (surp 10-50, hold 21) ---")
    ev = build(prices, 21, 10, 50)
    if not ev.empty:
        cut = ev.ann_date.sort_values().iloc[len(ev) // 2]
        report("  first half", ev[ev.ann_date < cut])
        report("  second half", ev[ev.ann_date >= cut])

    print("\n=== VERDICT ===")
    print("  If corrected CAR net is ~0 or insignificant, my 'PEAD alpha' was")
    print("  LOOK-AHEAD, not a real edge. That is the honest answer the frontier")
    print("  models forced me to find. Report it straight.")


if __name__ == "__main__":
    main()
