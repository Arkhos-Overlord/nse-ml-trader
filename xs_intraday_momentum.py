"""CROSS-SECTIONAL last-half-hour momentum — the RIGHT test.

Earlier time-series pooled intraday momentum was NEGATIVE (-0.019%/day). The
lit (Springer 2024 UK/China/Brazil) explains WHY: intraday momentum is
CLIENTELE-DEPENDENT — strong in institutional markets (US/UK/Brazil), WEAK in
retail-dominated ones (China A-shares), and STRENGTHENS with institutional
ownership. Li 2018: seasonality is significant in ADVANCED, not emerging mkts.

So the naive pooled time-series test was the wrong test. The RIGHT test (matching
the lit + Chuia's liquid-name finding) is CROSS-SECTIONAL on LARGE LIQUID names:
  - Each day, rank the 12 large-caps by their first-half-hour return.
  - LONG the top decile (winners), SHORT the bottom (losers), hold to close.
  - The cross-section cancels market beta; the edge is the dispersion.
Institutional-heavy large-caps are where the lit says intraday momentum lives.

Also: is it better LONG-only top-3 (fits small capital, no shorting cost) vs
long-short? And gate it: year-persistence isn't possible on 60d, so use
holdout + per-stock breadth + net of costs.

PEAD note: confirmed in India twice (Harshita 2018; Das 2026) but needs earnings
dates — no keyless source. Cannot test cleanly. Logged as data-blocked.

Run: python xs_intraday_momentum.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STOCKS = "data_indmoney/intraday_stocks.parquet"
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
    df = pd.read_parquet(STOCKS)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df["date"] = df.datetime_ist.dt.date
    df["hm"] = df.datetime_ist.dt.strftime("%H:%M")
    return df.sort_values(["label", "datetime_ist"])


def build_panel(df):
    """Per stock-day: first-half return (signal) + last-half return (target)."""
    rows = []
    for lb, g in df.groupby("label"):
        g = g.sort_values("datetime_ist")
        for d, day in g.groupby("date"):
            day = day.sort_values("datetime_ist")
            fh = day[day.hm <= "10:00"]
            lh = day[day.hm >= "14:30"]
            if len(fh) < 3 or len(lh) < 3:
                continue
            r_first = fh.close.iloc[-1] / day.open.iloc[0] - 1
            r_last = lh.close.iloc[-1] / lh.open.iloc[0] - 1
            rows.append({"label": lb, "date": d, "r_first": r_first,
                         "r_last": r_last, "px_lh": lh.open.iloc[0]})
    return pd.DataFrame(rows)


def xs_momentum(panel, mode="longshort", topn=3):
    """Cross-sectional: rank by r_first, long winners / short losers, last half."""
    rets = []
    for d, day in panel.groupby("date"):
        day = day.sort_values("r_first")
        n = len(day)
        if n < 6:
            continue
        k = max(1, n // 4)  # quartile
        losers = day.head(k)
        winners = day.tail(k)
        if mode == "longshort":
            r = (winners.r_last.mean() - losers.r_last.mean()) / 2
        elif mode == "longonly":
            r = winners.tail(topn).r_last.mean()
        elif mode == "shortonly":
            r = -losers.head(topn).r_last.mean()
        rets.append(r)
    return np.array(rets)


def net_and_sim(rets, mode):
    # per-trade net of costs on a standard MIS exposure
    expo = CAPITAL * LEVERAGE
    cost_frac = (2 * BROKERAGE + expo * STT_SELL + expo * EXCH_GST
                 + expo * SLIP) / expo
    # number of legs per day
    legs = {"longshort": 2, "longonly": 1, "shortonly": 1}[mode]
    net = rets - legs * cost_frac
    cap, peak, maxdd = CAPITAL, CAPITAL, 0
    for r in net:
        cap *= (1 + r * LEVERAGE / len(rets) * len(rets))  # not meaningful per-day on full cap
    # simpler: compound daily on leveraged exposure
    cap = CAPITAL
    peak = CAPITAL
    maxdd = 0
    for r in net:
        cap *= (1 + r * LEVERAGE)
        peak = max(peak, cap)
        maxdd = max(maxdd, (peak - cap) / peak)
    return net, cap, maxdd, cost_frac


def report(name, rets, mode):
    if len(rets) < 5:
        print(f"  {name:<28} n={len(rets)} (too few)")
        return
    net, cap, maxdd, cf = net_and_sim(rets, mode)
    print(f"  {name:<28} days={len(net):<3} gross={rets.mean()*100:+.3f}%/d "
          f"net={net.mean()*100:+.3f}%/d t={tstat(net):+.2f} "
          f"win={100*(net>0).mean():.0f}% end=Rs{cap:,.0f} DD={maxdd*100:.0f}%")
    return net


def main():
    df = load()
    panel = build_panel(df)
    print("=== CROSS-SECTIONAL LAST-HALF-HOUR MOMENTUM (the right test) ===")
    print(f"  lit: intraday momentum is INSTITUTIONAL-dependent -> test on large-caps")
    print(f"  {panel.label.nunique()} large-caps x {panel.date.nunique()} days\n")
    print("  (time-series pooled was NEGATIVE; cross-section cancels beta)\n")

    for mode in ["longshort", "longonly", "shortonly"]:
        print(f"  --- {mode} ---")
        rets = xs_momentum(panel, mode)
        report(f"quartile {mode}", rets, mode)

    print("\n--- HOLDUT: long-short quartile, 1st vs 2nd half of days ---")
    rets = xs_momentum(panel, "longshort")
    h = len(rets) // 2
    net, _, _, cf = net_and_sim(rets, "longshort")
    print(f"  1st half: n={h} net={net[:h].mean()*100:+.3f}%/d t={tstat(net[:h]):+.2f}")
    print(f"  2nd half: n={len(net)-h} net={net[h:].mean()*100:+.3f}%/d "
          f"t={tstat(net[h:]):+.2f}")

    print("\n--- BREADTH: which stocks contribute (is it broad or one name?) ---")
    # per-stock contribution to long-short
    contrib = {}
    for d, day in panel.groupby("date"):
        day = day.sort_values("r_first")
        n = len(day)
        if n < 6:
            continue
        k = max(1, n // 4)
        for lb in day.tail(k).label:
            contrib[lb] = contrib.get(lb, 0) + 1
        for lb in day.head(k).label:
            contrib[lb] = contrib.get(lb, 0) + 1
    print("  appearances in winner/loser buckets (top 6):")
    for lb, c in sorted(contrib.items(), key=lambda x: -x[1])[:6]:
        print(f"    {lb:<10} {c}")

    print("\n=== VERDICT ===")
    print("  cross-sectional long-short with net>0 and t>2 = institutional-edge")
    print("  confirmed in India. PEAD: data-blocked (no keyless earnings dates).")


if __name__ == "__main__":
    main()
