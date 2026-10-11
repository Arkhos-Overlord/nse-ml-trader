"""PEAD PORTFOLIO — the realistic ₹1.5L simulation, properly gated.

The prior run starved the strategy: 1 position at a time -> 27 trades -> 0
walk-forward folds. A ₹1.5L account runs 5 slots x Rs30K (MIS/delivery mix),
so up to 5 concurrent PEAD positions. That is the realistic structure, and it
produces enough trades to actually validate.

Simulation:
- 5 fixed slots of Rs30K each, no compounding, no overlap within a slot
- entry = open of first day strictly AFTER ann_date (no look-ahead)
- exit = open 42 trading days later
- costs: Indian 2024-25 (STT 0.1% both legs, NSE exch 0.00297%, GST, stamp, slip)
- benchmark: Nifty buy-and-hold over the SAME calendar span
- walk-forward: 3y train / 1y test on the daily portfolio return series

Run: python pead_portfolio.py
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


def sharpe(r):
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    return r.mean() / sd * np.sqrt(252) if sd > 1e-12 else 0.0


def walk_forward(r, train=756, test=252):
    r = np.asarray(r, float)
    folds, i = [], 0
    while i + train + test <= len(r):
        folds.append(sharpe(r[i + train:i + train + test]))
        i += test
    return folds


def build_events(lo, hi, hold):
    prices = pd.read_parquet(HIST)
    prices["date"] = pd.to_datetime(prices["datetime_ist"]).dt.date
    prices = prices.sort_values(["label", "date"])
    px = {}
    for lb, g in prices.groupby("label"):
        g = g.sort_values("date").reset_index(drop=True)
        px[lb] = (list(g.date), list(g.open), {d: i for i, d in enumerate(g.date)})
    earn = pd.read_parquet(EARN)
    ev = []
    for _, e in earn.iterrows():
        s = e.surprise_pct
        if not (lo <= s <= hi):
            continue
        lb, ed = e.label, e.earn_date.date()
        if lb not in px or lb == "NIFTY50":
            continue
        dates, opens, dmap = px[lb]
        fut = [i for i, d in enumerate(dates) if d > ed]
        if not fut:
            continue
        ei = fut[0]
        xi = ei + hold
        if xi >= len(opens):
            continue
        ev.append({"label": lb, "d_in": dates[ei], "d_out": dates[xi],
                   "entry": opens[ei], "exit": opens[xi]})
    return pd.DataFrame(ev).sort_values("d_in").reset_index(drop=True)


def simulate(ev, hold):
    """5 slots, fixed Rs30K each, no compounding. Returns daily portfolio
    return series + trade stats."""
    c = cost_pct(SLOT_CAP)
    slots = [[] for _ in range(SLOTS)]   # each holds list of (d_out, pnl_frac)
    daily = {}
    taken, skipped = 0, 0
    pnls = []
    for _, e in ev.iterrows():
        placed = False
        for s in slots:
            # slot free if no open position or last one exited
            if not s or e.d_in > s[-1][0]:
                qty = SLOT_CAP / e.entry
                gross = (e.exit - e.entry) * qty
                cost = c * SLOT_CAP
                pnl = gross - cost
                s.append((e.d_out, pnl))
                # book pnl on exit date as a return on TOTAL capital
                daily[e.d_out] = daily.get(e.d_out, 0.0) + pnl / CAPITAL
                pnls.append(pnl / SLOT_CAP)
                taken += 1
                placed = True
                break
        if not placed:
            skipped += 1
    return daily, taken, skipped, np.array(pnls)


def gate(name, ev, hold, nifty_dates, nifty_rets):
    daily, taken, skipped, pnls = simulate(ev, hold)
    if not daily:
        print(f"\n  {name}: no trades")
        return
    # align to nifty calendar
    nd = {d: i for i, d in enumerate(nifty_dates)}
    dates = sorted(daily)
    port = np.array([daily[d] for d in dates])
    # benchmark over the same span: Nifty buy-and-hold daily
    start, end = dates[0], dates[-1]
    si = nd.get(start, 0)
    ei = nd.get(end, len(nifty_rets) - 1)
    bench = nifty_rets[si:ei + 1]
    span_yrs = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    n = len(port)
    yrs = max(span_yrs, 1e-9)

    # portfolio CAGR: compound the daily returns, but the strategy is only
    # invested on trade-exit days -> annualize over the calendar span
    total_ret = np.prod(1 + port) - 1
    scagr = (1 + total_ret) ** (1 / yrs) - 1
    btotal = np.prod(1 + bench) - 1
    bcagr = (1 + btotal) ** (1 / yrs) - 1

    # build a full daily series (0 on non-trade days) for walk-forward
    full = pd.Series(0.0, index=pd.to_datetime(sorted(set(dates) | set(nifty_dates[si:ei + 1]))))
    for d in dates:
        full[pd.Timestamp(d)] += daily[d]
    full = full.sort_index()
    oos = walk_forward(full.values)
    neg = sum(1 for s in oos if s < 0)
    g2 = len(oos) > 0 and neg == 0

    cum = np.cumprod(1 + port)
    dd = ((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min()

    print(f"\n{'='*86}")
    print(f"  {name}")
    print(f"{'='*86}")
    print(f"  trades taken: {taken} (skipped for capacity: {skipped}) over {yrs:.1f}y span")
    print(f"  per-trade net: {pnls.mean()*100:+.2f}% of slot, "
          f"win {100*(pnls>0).mean():.0f}%, worst {pnls.min()*100:+.1f}%")
    print(f"  net CAGR (calendar-annualized): {scagr*100:+.2f}%/yr")
    print(f"  Nifty buy-and-hold same span:   {bcagr*100:+.2f}%/yr")
    print(f"  G1 beats bench:  {'PASS' if scagr > bcagr else 'FAIL'} ({(scagr-bcagr)*100:+.2f}pp)")
    if oos:
        print(f"  G2 walk-forward: {len(oos)} folds, {['%+.2f'%s for s in oos]} "
              f"-> {'PASS' if neg == 0 else f'FAIL ({neg} negative)'}")
    else:
        print(f"  G2 walk-forward: FAIL (0 folds)")
    print(f"  G4 survivability: max DD {dd*100:.1f}% on traded capital")
    print(f"  total net return: {total_ret*100:+.1f}% over {yrs:.1f}y "
          f"= Rs{CAPITAL*total_ret:,.0f} on Rs1.5L")
    too_good = scagr > 0.50
    print(f"  G5 too-good:     {'KILL-AND-DEBUG' if too_good else 'ok'}")
    ok = scagr > bcagr and g2 and dd > -0.25 and not too_good
    print(f"  VERDICT: {'TRADEABLE' if ok else 'DEAD' if scagr <= bcagr else 'NOT YET'}")
    return ok


def main():
    prices = pd.read_parquet(HIST)
    prices = prices[prices.label == "NIFTY50"].copy()
    prices["date"] = pd.to_datetime(prices["datetime_ist"]).dt.date
    prices = prices.sort_values("date").reset_index(drop=True)
    nifty_dates = list(prices.date)
    nifty_rets = prices.close.pct_change().fillna(0).values

    print("=" * 86)
    print("  PEAD PORTFOLIO — realistic Rs1.5L sim (5 slots x Rs30K, no compounding)")
    print("  Hurdle: Nifty buy-and-hold. Entry = day AFTER ann (no look-ahead).")
    print("=" * 86)
    for lo, hi, hold, nm in [
        (0.001, 1e9, 42, "PEAD broad drift (surp>0, 42d hold)"),
        (10, 50, 42, "PEAD band (surp 10-50%, 42d hold)"),
        (10, 50, 21, "PEAD band (surp 10-50%, 21d hold)"),
        (0.001, 1e9, 21, "PEAD broad drift (surp>0, 21d hold)"),
    ]:
        ev = build_events(lo, hi, hold)
        gate(nm, ev, hold, nifty_dates, nifty_rets)


if __name__ == "__main__":
    main()
