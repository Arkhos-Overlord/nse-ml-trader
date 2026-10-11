"""ADVERSARIAL AUDIT of the one candidate that passed the standard gate.

PEAD band (surp 10-50%, 42d) passed G1-G5: net +9.82%/yr vs Nifty +9.17%,
5/5 walk-forward folds positive, DD -19.8%, Rs1.5L -> Rs1.86L over 8.6y.

This is the FIRST candidate to clear an externally-authored standard. Per the
lesson I just encoded: confirmation bias peaks at first success. So audit it
HARD before believing it. Specific threats:

  A. SURVIVORSHIP — universe is CURRENT tickers yfinance still serves. Names
     that got big by doing well are overrepresented. Fix: drop the top
     performers and re-run; if the edge dies, it was survivorship.
  B. POST-HOC BAND — surp 10-50% was chosen AFTER seeing results (LMArena
     flagged this too). Fix: test NEIGHBORING bands. A real edge is a plateau,
     not a spike at one band.
  C. CLUSTERING — earnings cluster in results season; event-level t overstates.
     Fix: month-clustered t + calendar-year persistence.
  D. CORPORATE ACTIONS — unadjusted splits/bonuses fabricate returns. Fix: drop
     the events whose window contains a >20% jump while Nifty moved <5%.
  E. LOOK-AHEAD — already fixed (entry day AFTER ann), but re-verify.
  F. IS IT JUST BETA — the strategy is long-only in a rising market. Fix:
     beta-strip; if abnormal return is ~0, it's beta.

Run: python pead_adversarial.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EARN = "data_indmoney/earnings.parquet"
HIST = "data_indmoney/daily_history.parquet"
CAPITAL = 150000.0
SLOTS = 5
SLOT_CAP = CAPITAL / SLOTS

STT = 0.001
EXCHANGE = 0.0000297
SEBI = 1e-6
GST = 0.18
STAMP = 0.00015
SLIP = 0.001
BROKERAGE = 20.0


def cost_pct(notional):
    fees = (EXCHANGE * 2 + SEBI * 2) * notional + 2 * BROKERAGE
    gst = GST * (EXCHANGE + SEBI) * notional
    return (STT * 2 * notional + STAMP * notional + fees + gst
            + SLIP * 2 * notional) / notional


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def clustered_t(ev):
    g = ev.groupby(pd.to_datetime(ev.ann_date).dt.to_period("M"))["ret"].mean()
    if len(g) < 6 or g.std(ddof=1) == 0:
        return np.nan, len(g)
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g))), len(g)


def load_prices():
    df = pd.read_parquet(HIST)
    df["date"] = pd.to_datetime(df["datetime_ist"]).dt.date
    return df.sort_values(["label", "date"])


def build_events(prices, lo, hi, hold, exclude_tickers=None):
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
        if exclude_tickers and lb in exclude_tickers:
            continue
        dates, opens, closes, dmap = px[lb]
        fut = [i for i, d in enumerate(dates) if d > ed]
        if not fut:
            continue
        ei = fut[0]
        xi = ei + hold
        if xi >= len(opens):
            continue
        # D. corporate action scan: any |daily ret| > 20% inside window while
        # Nifty moved < 5% -> suspect unadjusted split/bonus
        bad = False
        for k in range(ei, min(xi + 1, len(closes) - 1)):
            sr = closes[k + 1] / closes[k] - 1
            d0, d1 = dates[k], dates[k + 1]
            if d0 in nd and d1 in nd:
                nr = nc[nd[d1]] / nc[nd[d0]] - 1
            else:
                nr = 0.0
            if abs(sr) > 0.20 and abs(nr) < 0.05:
                bad = True
                break
        if bad:
            continue
        nifty_ret = (nc[nd[dates[xi]]] / nc[nd[dates[ei]]] - 1
                     if dates[ei] in nd and dates[xi] in nd else 0.0)
        ev.append({"label": lb, "ann_date": ed, "d_in": dates[ei],
                   "d_out": dates[xi], "entry": opens[ei], "exit": opens[xi],
                   "ret": opens[xi] / opens[ei] - 1, "nifty": nifty_ret})
    return pd.DataFrame(ev).sort_values("d_in").reset_index(drop=True)


def run(name, ev, hold):
    if ev.empty:
        print(f"\n  {name}: 0 events")
        return None
    c = cost_pct(SLOT_CAP)
    slots = [[] for _ in range(SLOTS)]
    daily = {}
    pnls = []
    for _, e in ev.iterrows():
        for s in slots:
            if not s or e.d_in > s[-1][0]:
                qty = SLOT_CAP / e.entry
                pnl = (e.exit - e.entry) * qty - c * SLOT_CAP
                s.append((e.d_out, pnl))
                daily[e.d_out] = daily.get(e.d_out, 0.0) + pnl / CAPITAL
                pnls.append(pnl / SLOT_CAP)
                break
    dates = sorted(daily)
    port = np.array([daily[d] for d in dates])
    yrs = max((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25, 1e-9)
    total = np.prod(1 + port) - 1
    scagr = (1 + total) ** (1 / yrs) - 1
    ev = ev.copy()
    ev["net_ret"] = ev.ret - c
    ev["abn"] = ev.ret - ev.nifty - c
    tc, k = clustered_t(ev)
    ev["yr"] = pd.to_datetime(ev.ann_date).dt.year
    pos_years = sum(1 for _, g in ev.groupby("yr") if g.abn.mean() > 0)
    tot_years = ev.yr.nunique()
    print(f"\n  {name}")
    print(f"    events {len(ev)}, taken {len(pnls)}, per-trade net {np.mean(pnls)*100:+.2f}%, "
          f"win {100*(np.array(pnls)>0).mean():.0f}%, worst {min(pnls)*100:+.1f}%")
    print(f"    net CAGR {scagr*100:+.2f}%/yr over {yrs:.1f}y | total Rs{CAPITAL*total:,.0f}")
    print(f"    abnormal (beta-ish, vs Nifty) mean {ev.abn.mean()*100:+.2f}%/event, "
          f"t_event {tstat(ev.abn):+.2f}, t_cluster {tc:+.2f} (k={k})")
    print(f"    calendar-year persistence: positive abnormal in {pos_years}/{tot_years} years")
    return scagr, ev


def main():
    prices = load_prices()
    print("=" * 88)
    print("  ADVERSARIAL AUDIT — PEAD band (surp 10-50%, 42d), the one gate-passer")
    print("=" * 88)

    print("\n[A] BASELINE (with corporate-action scan, entry day AFTER ann)")
    run("surp 10-50, 42d", build_events(prices, 10, 50, 42), 42)

    print("\n[B] POST-HOC BAND TEST — is it a plateau or a spike at 10-50?")
    for lo, hi in [(5, 30), (10, 30), (10, 50), (20, 60), (30, 80), (10, 100), (0, 100)]:
        r = run(f"surp {lo}-{hi}, 42d", build_events(prices, lo, hi, 42), 42)
    print("    -> a REAL edge is a PLATEAU across neighbouring bands; a spike")
    print("       at exactly 10-50 means the band was mined from this sample.")

    print("\n[C] SURVIVORSHIP — drop the 10 biggest winners, re-run")
    ev = build_events(prices, 10, 50, 42)
    if not ev.empty:
        top = ev.groupby("label").ret.mean().nlargest(10).index.tolist()
        print(f"    dropping: {top}")
        run("surp 10-50, 42d, ex-top-10", build_events(prices, 10, 50, 42,
                                                      exclude_tickers=set(top)), 42)

    print("\n" + "=" * 88)
    print("  VERDICT RULE: TRADEABLE only if the band is a PLATEAU, the edge")
    print("  survives dropping top performers, abnormal return (not raw) is")
    print("  significant with clustering accounted for, and it persists across")
    print("  most calendar years. Otherwise NOT YET / DEAD.")


if __name__ == "__main__":
    main()
