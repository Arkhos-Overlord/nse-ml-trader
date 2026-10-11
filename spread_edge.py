"""Delta-hedged CREDIT SPREAD — the Rs1.5L-safe version of the edge.

The naked straddle edge is real (+Rs1Cr/12yr) but its gap tail (-Rs1.5L) kills
Rs1.5L. A CREDIT SPREAD (sell ATM, buy 1-2 strikes OTM) caps the loss at
wing_width - credit. Combined with the delta hedge that made the straddle work.

Test on real 2024-2025 Nifty option prices: does the spread keep the edge while
capping the tail to something Rs1.5L survives?

Run: python spread_edge.py
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from precise_costs import exact_costs

warnings.filterwarnings("ignore")

LOT = 75
HEDGE_COST_BPS = 3
BROKERAGE = 20


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def straddle_delta(S, K, sigma, T, r=0.07):
    if T <= 1e-6 or sigma <= 1e-6 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return 2.0 * norm_cdf(d1) - 1.0


def get(sym, period="max"):
    d = yf.download(sym, period=period, interval="1d", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d


import math


def sim_spread(series, atm, ce_w, pe_w, expiry_ts, hedge=True):
    """Short ATM straddle + long wings (credit spread/condor), delta-hedged.
    series: list of (date, spot, vix, atm_straddle, wing_straddle)"""
    pnl = 0.0
    T0 = max((expiry_ts - series[0][0]).days, 1) / 365.0
    d0 = straddle_delta(series[0][1], atm, series[0][2] / 100.0, T0)
    hedge_units = -d0 if hedge else 0.0
    prev_S = series[0][1]
    prev_net = series[0][3] - series[0][4]  # net premium value (straddle - wings)

    for i in range(1, len(series)):
        d, S, v, V, W = series[i]
        net_V = V - W
        pnl += -(net_V - prev_net) * LOT
        if hedge:
            pnl += hedge_units * (S - prev_S)
            T = max((expiry_ts - d).days, 1) / 365.0
            nd = straddle_delta(S, atm, v / 100.0, T)
            rb = abs(-nd - hedge_units) * S
            pnl -= rb * HEDGE_COST_BPS / 10000
            hedge_units = -nd
        prev_S, prev_net = S, net_V

    # costs: 4 option legs
    credit = (series[0][3] - series[0][4]) * LOT
    c = exact_costs(series[0][3] * LOT, series[0][4] * LOT)
    pnl -= c["TOTAL"]
    if hedge:
        pnl -= abs(hedge_units * series[-1][1]) * HEDGE_COST_BPS / 10000
    return pnl, credit


def run(years, wing, hedge=True):
    files = [f"NIFTY_{y}.parquet" for y in years]
    vix = get("^INDIAVIX")["Close"].dropna()
    nif = get("^NSEI")["Close"]
    mkt = pd.DataFrame({"vix": vix, "spot": nif}).dropna()
    mkt.index = pd.to_datetime(mkt.index)

    trades = []
    for f in files:
        opt = pd.read_parquet(f)
        opt["date"] = pd.to_datetime(opt["date"])
        for expiry, grp in opt.groupby("expiry"):
            traded = grp[grp["close"] > 0].sort_values("timestamp")
            if traded.empty:
                continue
            days = [d for d in sorted(traded["date"].unique()) if d in mkt.index]
            if len(days) < 3:
                continue
            entry_day = pd.Timestamp(days[0])
            if entry_day not in mkt.index:
                continue
            vix0 = float(mkt.loc[entry_day, "vix"])
            if vix0 < 8 or vix0 > 30:
                continue
            spot0 = float(mkt.loc[entry_day, "spot"])
            strikes = np.sort(traded["strike"].unique())
            atm = strikes[np.argmin(np.abs(strikes - spot0))]
            ce_w = strikes[np.argmin(np.abs(strikes - (atm + wing)))]
            pe_w = strikes[np.argmin(np.abs(strikes - (atm - wing)))]

            def sp(d, s):
                r = traded[(traded["date"] == d) & (traded["strike"] == s)]
                ce = r[r["option_type"] == "CE"]["close"]
                pe = r[r["option_type"] == "PE"]["close"]
                return (float(ce.iloc[0]) + float(pe.iloc[0])) if not ce.empty and not pe.empty else None

            def wing_px(d):
                rce = traded[(traded["date"] == d) & (traded["strike"] == ce_w) & (traded["option_type"] == "CE")]["close"]
                rpe = traded[(traded["date"] == d) & (traded["strike"] == pe_w) & (traded["option_type"] == "PE")]["close"]
                return (float(rce.iloc[0]) + float(rpe.iloc[0])) if not rce.empty and not rpe.empty else None

            series = []
            for d in days:
                sv, wv = sp(d, atm), wing_px(d)
                if sv is None or wv is None:
                    continue
                series.append((d, float(mkt.loc[d, "spot"]),
                               float(mkt.loc[d, "vix"]), sv, wv))
            if len(series) < 3:
                continue
            pnl, credit = sim_spread(series, atm, ce_w, pe_w,
                                     pd.Timestamp(expiry), hedge=hedge)
            max_loss = wing * LOT - credit  # theoretical cap
            trades.append({"year": entry_day.year, "pnl": pnl,
                           "credit": credit, "max_loss": max_loss,
                           "vix": vix0})
    return pd.DataFrame(trades)


def summ(t, label):
    if t.empty:
        print(f"  {label}: no trades")
        return
    consec, kept = 0, []
    for p in t.sort_values(["year"])["pnl"].values:
        if consec >= 3:
            kept.append(0.0); consec = 0; continue
        kept.append(p); consec = consec + 1 if p < 0 else 0
    t = t.sort_values(["year"]).reset_index(drop=True)
    t["pnl_ks"] = kept
    by_year = t.groupby("year")["pnl_ks"].sum()
    print(f"  {label}: {len(t)} trades, Rs{t.pnl_ks.sum():,.0f}, "
          f"{(t.pnl_ks>0).mean()*100:.0f}% win, worst Rs{t.pnl_ks.min():,.0f}, "
          f"{(by_year>0).sum()}/{len(by_year)} yrs")
    print(f"     avg credit Rs{t.credit.mean():,.0f}, avg maxloss Rs{t.max_loss.mean():,.0f}")


def main():
    print("=" * 60)
    print("DELTA-HEDGED CREDIT SPREAD — the Rs1.5L-fit version")
    print("=" * 60)
    print("  exact costs applied. defined risk (wings cap the loss).\n")
    years = [2023, 2024, 2025]
    for wing in [500, 1000]:
        for hedge in [True, False]:
            h = "hedged" if hedge else "naked "
            print(f"wing +/-{wing}, {h}:")
            summ(run(years, wing, hedge=hedge), f"  spread {wing} {h}")
            print()
    print("  KEY QUESTION: does the spread keep the edge AND cap the tail")
    print("  to something Rs1.5L survives (≤ Rs15K worst)?")


if __name__ == "__main__":
    main()
