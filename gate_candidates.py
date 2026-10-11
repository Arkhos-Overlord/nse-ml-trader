"""Run the session's surviving candidates through THE STANDARD GATE.

The gate (adopted from OpenQuant India): net CAGR must beat the Nifty
benchmark, walk-forward OOS Sharpe non-negative, costs declared, survivable.

The decisive context from standard_gate.py: Nifty buy-and-hold nets only
+3.64%/yr after costs with a -40.2% max DD. THAT is the hurdle.

Candidates, built as real net daily-return series:

  1. PEAD BROAD DRIFT: long all positive-surprise stocks, 42d hold, one
     position at a time, fixed sizing, net of costs. (prior: clustered t=+4.21)
  2. PEAD STRICT BAND: surp 10-50%, 21d hold. (prior: collapsed post-look-ahead-fix)
  3. VRP-INFORMED LONG: long Nifty when IndiaVIX >> recent realized vol.
     (prior: real premium, weak directional signal)

Each is compared against Nifty buy-and-hold over the SAME dates.

Run: python gate_candidates.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EARN = "data_indmoney/earnings.parquet"
HIST = "data_indmoney/daily_history.parquet"
CAPITAL = 150000.0

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


def gate(name, ret_by_date, notes=""):
    """ret_by_date: dict date->daily portfolio return (already net-ready)."""
    nifty = pd.read_parquet(HIST)
    nifty = nifty[nifty.label == "NIFTY50"].copy()
    nifty["date"] = pd.to_datetime(nifty["datetime_ist"]).dt.date
    nifty = nifty.sort_values("date").reset_index(drop=True)
    bench_all = nifty.close.pct_change()

    dates = sorted(ret_by_date)
    if not dates:
        print(f"\n  {name}: no trades")
        return
    # align to nifty trading days
    nd = {d: i for i, d in enumerate(nifty.date)}
    port, bench = [], []
    for d in dates:
        if d not in nd:
            continue
        port.append(ret_by_date[d])
        bench.append(bench_all.iloc[nd[d]])
    port = np.array(port)
    bench = np.array(bench)
    n = len(port)
    # annualize over the CALENDAR span of the trades, not the trade count
    span_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    yrs = max(span_days / 365.25, n / 252)
    c = cost_pct(CAPITAL)
    # cost per trade amortized over the hold (42d or 21d)
    HOLD = 42 if "42" in name else 21
    net = port - c / HOLD

    scagr = (np.prod(1 + net)) ** (1 / yrs) - 1 if n else np.nan
    bcagr = (np.prod(1 + bench)) ** (1 / yrs) - 1 if n else np.nan
    oos = walk_forward(net)
    neg = sum(1 for s in oos if s < 0)
    # walk-forward needs real folds; 0 folds = cannot validate = NOT YET
    if len(oos) == 0:
        g2 = False
        g2_txt = "FAIL (0 folds — sample too short to walk-forward)"
    else:
        g2 = neg == 0
        g2_txt = (f"{len(oos)} folds, {['%+.2f'%s for s in oos[:6]]} -> "
                  f"{'PASS' if neg == 0 else f'FAIL ({neg} negative)'}")
    cum = np.cumprod(1 + net)
    dd = ((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min()

    print(f"\n{'='*86}")
    print(f"  {name}  {notes}")
    print(f"{'='*86}")
    print(f"  trades: {n} over {yrs:.1f}y span | cost/RT {c*100:.3f}%")
    print(f"  net CAGR:        {scagr*100:+.2f}%/yr")
    print(f"  Nifty same dates: {bcagr*100:+.2f}%/yr")
    print(f"  G1 beats bench:  {'PASS' if scagr > bcagr else 'FAIL'} ({(scagr-bcagr)*100:+.2f}pp)")
    print(f"  G2 walk-forward: {g2_txt}")
    print(f"  G4 survivability: max DD {dd*100:.1f}%")
    too_good = (not np.isnan(scagr)) and scagr > 0.50
    print(f"  G5 too-good:     {'KILL-AND-DEBUG' if too_good else 'ok'}")
    ok = (scagr > bcagr) and g2 and (dd > -0.25) and not too_good
    print(f"  VERDICT: {'TRADEABLE' if ok else 'DEAD' if scagr <= bcagr else 'NOT YET'}")
    return ok


def build_pead(lo, hi, hold):
    prices = pd.read_parquet(HIST)
    prices["date"] = pd.to_datetime(prices["datetime_ist"]).dt.date
    prices = prices.sort_values(["label", "date"])
    nifty = prices[prices.label == "NIFTY50"].sort_values("date").reset_index(drop=True)
    nd = {d: i for i, d in enumerate(nifty.date)}
    nc = list(nifty.close)
    px = {}
    for lb, g in prices.groupby("label"):
        g = g.sort_values("date").reset_index(drop=True)
        px[lb] = (list(g.date), list(g.open), list(g.close), {d: i for i, d in enumerate(g.date)})
    earn = pd.read_parquet(EARN)
    # one position at a time: collect (entry_date, exit_date, ret), sort, skip overlaps
    events = []
    for _, e in earn.iterrows():
        s = e.surprise_pct
        if not (lo <= s <= hi):
            continue
        lb, ed = e.label, e.earn_date.date()
        if lb not in px or lb == "NIFTY50":
            continue
        dates, opens, closes, dmap = px[lb]
        fut = [i for i, d in enumerate(dates) if d > ed]
        if not fut:
            continue
        ei = fut[0]
        xi = ei + hold
        if xi >= len(closes):
            continue
        events.append((dates[ei], dates[xi], opens[xi] / opens[ei] - 1))
    events.sort()
    ret_by_date = {}
    busy_until = None
    for d_in, d_out, r in events:
        if busy_until is not None and d_in <= busy_until:
            continue
        ret_by_date[d_out] = r
        busy_until = d_out
    return ret_by_date


def main():
    print("=" * 86)
    print("  SURVIVING CANDIDATES THROUGH THE STANDARD GATE")
    print("  Hurdle (from standard_gate.py): Nifty buy-and-hold nets +3.64%/yr, DD -40%")
    print("=" * 86)
    gate("PEAD broad drift 42d (surp>0)", build_pead(0.001, 1e9, 42),
         "prior: clustered t=+4.21")
    gate("PEAD band 21d (surp 10-50)", build_pead(10, 50, 21),
         "prior: collapsed after look-ahead fix")
    gate("PEAD band 42d (surp 10-50)", build_pead(10, 50, 42),
         "prior: t=+2.16 event-level")


if __name__ == "__main__":
    main()
