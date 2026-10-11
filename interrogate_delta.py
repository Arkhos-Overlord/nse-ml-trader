"""Interrogate the delta-hedged result: is the edge REAL or just hedge execution?

The +Rs80L result has two suspicious facts:
  1. corr(PnL, VRP) = +0.08 (near zero) -> PnL isn't from selling rich vol
  2. worst trade -Rs1.8L -> hedge has gap risk

Tests:
  A. VRP filter vs ALWAYS-ON: does the signal help, or is it noise?
  B. Tighter rebalance (2x/day proxy via finer hedge): does the tail shrink?
  C. Entry only when VRP high (top tercile): does selectivity help?
Honest: if the edge is real, VRP-filtered should beat always-on AND the tail
should be manageable. If always-on wins, the 'edge' is just gamma harvesting.
Run: python interrogate_delta.py
"""
import glob
import math
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

LOT = 75
STT_SELL = 0.000625
BROKERAGE = 20
SLIPPAGE_BPS = 10
SPREAD_BPS = 15
HEDGE_COST_BPS = 3


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


def sim_trade(series, atm, expiry_ts, hedge_frac=1.0):
    """Delta-hedged short straddle over the series. hedge_frac<1 = partial hedge."""
    pnl = 0.0
    T0 = max((expiry_ts - series[0][0]).days, 1) / 365.0
    sig0 = series[0][2] / 100.0
    delta0 = straddle_delta(series[0][1], atm, sig0, T0)
    hedge = -delta0 * hedge_frac
    prev_S, prev_V = series[0][1], series[0][3]
    pnl -= abs(prev_V * LOT) * (SLIPPAGE_BPS + SPREAD_BPS) / 10000 + BROKERAGE * 2
    for i in range(1, len(series)):
        d, S, v, V = series[i]
        pnl += -(V - prev_V) * LOT + hedge * (S - prev_S)
        T = max((expiry_ts - d).days, 1) / 365.0
        nd = straddle_delta(S, atm, v / 100.0, T)
        rb = abs(-nd * hedge_frac - hedge) * S
        pnl -= rb * HEDGE_COST_BPS / 10000
        hedge = -nd * hedge_frac
        prev_S, prev_V = S, V
    pnl -= abs(series[-1][3] * LOT) * (SLIPPAGE_BPS + SPREAD_BPS) / 10000 + BROKERAGE * 2
    pnl -= abs(hedge * series[-1][1]) * HEDGE_COST_BPS / 10000
    pnl -= series[0][3] * LOT * STT_SELL
    return pnl


def build(mkt):
    vix = get("^INDIAVIX")["Close"].dropna()
    nif = get("^NSEI")["Close"]
    mkt = pd.DataFrame({"vix": vix, "spot": nif}).dropna()
    mkt.index = pd.to_datetime(mkt.index)
    ret = mkt["spot"].pct_change()
    mkt["realized_21d"] = ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100
    mkt["vrp"] = mkt["vix"] - mkt["realized_21d"]
    return mkt


def run_all(mkt, mode, hedge_frac=1.0):
    files = sorted(glob.glob("NIFTY_*.parquet"))
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
            entry_day, exit_day = pd.Timestamp(days[0]), pd.Timestamp(days[-1])
            if entry_day not in mkt.index:
                continue
            vix0 = float(mkt.loc[entry_day, "vix"])
            vrp0 = float(mkt.loc[entry_day, "vrp"])
            if vix0 < 8 or vix0 > 30:
                continue
            # MODE filter
            if mode == "vrp_positive" and vrp0 <= 0:
                continue
            if mode == "vrp_top" and vrp0 <= mkt["vrp"].quantile(0.66):
                continue
            spot0 = float(mkt.loc[entry_day, "spot"])
            strikes = np.sort(traded["strike"].unique())
            atm = strikes[np.argmin(np.abs(strikes - spot0))]
            series = []
            for d in days:
                row = traded[(traded["date"] == d) & (traded["strike"] == atm)]
                ce = row[row["option_type"] == "CE"]["close"]
                pe = row[row["option_type"] == "PE"]["close"]
                if ce.empty or pe.empty:
                    continue
                series.append((d, float(mkt.loc[d, "spot"]),
                               float(mkt.loc[d, "vix"]),
                               float(ce.iloc[0]) + float(pe.iloc[0])))
            if len(series) < 3:
                continue
            expiry_ts = pd.Timestamp(expiry)
            pnl = sim_trade(series, atm, expiry_ts, hedge_frac)
            trades.append({"year": entry_day.year, "vrp": vrp0, "pnl": pnl})
    return pd.DataFrame(trades)


def summ(t, label):
    if t.empty:
        print(f"  {label}: no trades")
        return
    t = t.sort_values(["year"]).reset_index(drop=True)
    consec, kept = 0, []
    for p in t["pnl"].values:
        if consec >= 3:
            kept.append(0.0); consec = 0; continue
        kept.append(p); consec = consec + 1 if p < 0 else 0
    t["pnl_ks"] = kept
    by_year = t.groupby("year")["pnl_ks"].sum()
    print(f"  {label}: {len(t)} trades, Rs{t.pnl_ks.sum():,.0f}, "
          f"{(t.pnl_ks>0).mean()*100:.0f}% win, worst Rs{t.pnl_ks.min():,.0f}, "
          f"{(by_year>0).sum()}/{len(by_year)} yrs")


def main():
    mkt = build(None)
    print("=== INTERROGATION: is the edge VRP or just gamma harvesting? ===\n")
    print("A. FILTER COMPARISON (full delta hedge):")
    summ(run_all(mkt, "always"), "always-on     ")
    summ(run_all(mkt, "vrp_positive"), "VRP>0         ")
    summ(run_all(mkt, "vrp_top"), "VRP top-third ")

    print("\nB. HEDGE FRACTION (does partial hedge help tail?):")
    for hf in [1.0, 0.8, 0.6]:
        summ(run_all(mkt, "vrp_positive", hedge_frac=hf), f"hedge {hf:.0%}      ")

    print("\nC. VERDICT:")
    print("  If VRP-filtered beats always-on -> vol signal is real.")
    print("  If always-on wins -> it's gamma harvesting, VRP is decoration.")
    print("  If lower hedge fraction shrinks tail without killing PnL -> safer.")


if __name__ == "__main__":
    main()
