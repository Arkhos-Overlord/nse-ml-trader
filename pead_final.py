"""PEAD FINAL GATE — is it alpha or beta? + year persistence.

Long-side PEAD surp 10-50% passed: +Rs2,143/ev, t=+3.49, 64% win, 65% breadth,
worst -Rs7,969, +80% over 113 events (fixed sizing). Before I call it REAL:

GATE A — BETA CHECK: strip the market. Is the post-earnings drift ABNORMAL
   (vs Nifty over the same window) or just 'stocks rose because market rose'?
   Compute CAR = stock return - Nifty return over the hold. If CAR still
   positive with |t|>2, it's alpha.

GATE B — YEAR PERSISTENCE: calendar/fundamental anomalies survive only if they
   persist across years. Positive in how many of the ~8 years?

GATE C — RECENCY/OVERLAP: are the +Rs2,143 concentrated in a few recent events
   or steady? (survivorship in the earnings feed — recent data is cleaner).

Run: python pead_final.py
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
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def load_prices():
    df = pd.read_parquet(HIST)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df["date"] = df.datetime_ist.dt.date
    return df.sort_values(["label", "date"])


def build(prices, hold=21, lo=10, hi=50):
    # nifty series for benchmark
    nifty = prices[prices.label == "NIFTY50"].sort_values("date").reset_index(drop=True)
    nd = {d: i for i, d in enumerate(nifty.date)}
    nc = list(nifty.close)
    px = {}
    for lb, g in prices.groupby("label"):
        g = g.sort_values("date").reset_index(drop=True)
        px[lb] = (list(g.date), list(g.close), {d: i for i, d in enumerate(g.date)})
    earn = pd.read_parquet(EARN)
    ev = []
    for _, e in earn.iterrows():
        s = e.surprise_pct
        if not (lo <= s <= hi):
            continue
        lb, ed = e.label, e.earn_date.date()
        if lb not in px or lb == "NIFTY50":
            continue
        dates, closes, dmap = px[lb]
        fut = [i for i, d in enumerate(dates) if d >= ed]
        if not fut:
            continue
        i0 = fut[0]
        if i0 + hold >= len(closes):
            continue
        # nifty over same window
        entry_d = dates[i0]
        exit_d = dates[i0 + hold]
        if entry_d in nd and exit_d in nd:
            j0, j1 = nd[entry_d], nd[exit_d]
            nifty_ret = nc[j1] / nc[j0] - 1
        else:
            nifty_ret = 0.0
        stock_ret = closes[i0 + hold] / closes[i0] - 1
        ev.append({"label": lb, "entry_date": entry_d, "surprise": s,
                   "stock_ret": stock_ret, "nifty_ret": nifty_ret,
                   "car": stock_ret - nifty_ret})
    return pd.DataFrame(ev).sort_values("entry_date").reset_index(drop=True)


def main():
    prices = load_prices()
    print("=== PEAD FINAL GATE — alpha or beta? ===")
    ev = build(prices, 21, 10, 50)
    print(f"  {len(ev)} events (surp 10-50%, hold 21d)\n")

    print("GATE A — BETA STRIP (CAR = stock - Nifty over same window):")
    car = ev.car.values - 0.002  # net of costs
    print(f"  raw stock ret:  {ev.stock_ret.mean()*100:+.3f}%/ev t={tstat(ev.stock_ret):+.2f}")
    print(f"  Nifty (beta):   {ev.nifty_ret.mean()*100:+.3f}%/ev t={tstat(ev.nifty_ret):+.2f}")
    print(f"  ABNORMAL (CAR): {ev.car.mean()*100:+.3f}%/ev t={tstat(ev.car):+.2f}")
    print(f"  CAR net costs:  {car.mean()*100:+.3f}%/ev t={tstat(car):+.2f} "
          f"win={100*(car>0).mean():.0f}%")
    # beta = covariance / variance
    beta = np.cov(ev.stock_ret, ev.nifty_ret)[0, 1] / np.var(ev.nifty_ret) \
        if np.var(ev.nifty_ret) > 0 else 0
    print(f"  beta vs Nifty:  {beta:.2f}")
    print(f"  -> if CAR net still |t|>2, it's ALPHA not beta")

    print("\nGATE B — YEAR PERSISTENCE:")
    ev["yr"] = pd.to_datetime(ev.entry_date).dt.year
    pos_y, tot_y = 0, 0
    for yr, g in ev.groupby("yr"):
        if len(g) < 3:
            continue
        tot_y += 1
        car_y = g.car.mean() - 0.002
        flag = "+" if car_y > 0 else "-"
        if car_y > 0:
            pos_y += 1
        print(f"  {yr}: n={len(g):<3} CAR={g.car.mean()*100:+.3f}%/ev "
              f"net={car_y*100:+.3f}% t={tstat(g.car-0.002):+.2f} {flag}")
    print(f"  -> positive CAR in {pos_y}/{tot_y} years")

    print("\nGATE C — RECENCY (first half vs second half of events):")
    h = len(ev) // 2
    print(f"  1st half: n={h} CAR={ev.car[:h].mean()*100:+.3f}% t={tstat(ev.car[:h]-0.002):+.2f}")
    print(f"  2nd half: n={len(ev)-h} CAR={ev.car[h:].mean()*100:+.3f}% "
          f"t={tstat(ev.car[h:]-0.002):+.2f}")

    print("\n=== FINAL VERDICT ===")
    print("  REAL PEAD edge = CAR net of costs |t|>2 + positive in MOST years")
    print("  + both halves positive. That is alpha, tradeable at Rs1.5L.")


if __name__ == "__main__":
    main()
