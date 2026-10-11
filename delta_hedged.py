"""DELTA-HEDGED SHORT STRADDLE — the real institutional vol strategy.

The busts showed: short strangle loses when market TRENDS (2017, 2023), regardless
of VIX. The fix professionals use: DELTA-HEDGE. Short the straddle, rebalance the
hedge daily to stay direction-neutral. Then PnL = pure vol harvest:

  PnL ≈ 0.5 * Gamma * S^2 * (sigma_implied^2 - sigma_realized^2) * T

If VRP is real (implied > realized), this makes money in EVERY regime because
direction is hedged away. This is the test that either proves the edge or ends it.

Simulation per expiry week (daily rebalance):
  - short ATM straddle at entry (real option prices)
  - each day: option PnL + hedge PnL, rebalance hedge to -straddle_delta
  - straddle_delta = 2*N(d1) - 1  (BS delta, VIX as implied vol)
  - exit at expiry, close hedge
  - realistic costs on options AND hedge rebalances

12 years, real prices. Run: python delta_hedged.py
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
HEDGE_COST_BPS = 3   # index futures are tight


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def straddle_delta(S, K, sigma, T, r=0.07):
    """Delta of long 1 CE + 1 PE (per unit underlying). ATM ~0, rises with S."""
    if T <= 1e-6 or sigma <= 1e-6 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return 2.0 * norm_cdf(d1) - 1.0


def get(sym, period="max"):
    d = yf.download(sym, period=period, interval="1d", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d


def main():
    # market data
    vix = get("^INDIAVIX")["Close"].dropna()
    nif = get("^NSEI")["Close"]
    mkt = pd.DataFrame({"vix": vix, "spot": nif}).dropna()
    mkt.index = pd.to_datetime(mkt.index)
    ret = mkt["spot"].pct_change()
    mkt["realized_21d"] = ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100
    mkt["vrp"] = mkt["vix"] - mkt["realized_21d"]

    files = sorted(glob.glob("NIFTY_*.parquet"))
    print(f"Delta-hedged short straddle | {len(files)} years | realistic costs\n")

    all_trades = []
    for f in files:
        yr = f.split("_")[1].split(".")[0]
        opt = pd.read_parquet(f)
        opt["date"] = pd.to_datetime(opt["date"])
        for expiry, grp in opt.groupby("expiry"):
            traded = grp[grp["close"] > 0].sort_values("timestamp")
            if traded.empty:
                continue
            days = sorted(traded["date"].unique())
            if len(days) < 3:  # need >=3 days to hedge
                continue
            entry_day = pd.Timestamp(days[0])
            exit_day = pd.Timestamp(days[-1])
            # market data must cover the window
            window = [d for d in days if d in mkt.index]
            if len(window) < 3 or entry_day not in mkt.index:
                continue
            spot0 = float(mkt.loc[entry_day, "spot"])
            vix0 = float(mkt.loc[entry_day, "vix"])
            if vix0 < 8 or vix0 > 30:  # sane vol range
                continue
            strikes = np.sort(traded["strike"].unique())
            atm = strikes[np.argmin(np.abs(strikes - spot0))]

            # per-day straddle value at ATM
            series = []
            for d in window:
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

            # simulate daily delta-hedged short straddle
            expiry_ts = pd.Timestamp(expiry)
            pnl_total = 0.0
            hedge_units = 0.0
            prev_S = series[0][1]
            prev_V = series[0][2 + 1]  # straddle value at entry
            # entry: short straddle, set initial hedge
            T0 = max((expiry_ts - series[0][0]).days, 1) / 365.0
            sig0 = series[0][2] / 100.0
            delta0 = straddle_delta(series[0][1], atm, sig0, T0)
            hedge_units = -delta0  # short delta0 units of underlying to neutralize
            pnl_total -= (abs(prev_V) * LOT) * (SLIPPAGE_BPS + SPREAD_BPS) / 10000  # entry slippage
            pnl_total -= BROKERAGE * 2  # 2 legs

            for i in range(1, len(series)):
                d, S, v, V = series[i]
                # option leg PnL (short): -(V - prev_V)
                opt_pnl = -(V - prev_V) * LOT
                # hedge leg PnL: hedge_units * (S - prev_S)
                hedge_pnl = hedge_units * (S - prev_S)
                pnl_total += opt_pnl + hedge_pnl
                # hedge rebalance cost
                T = max((expiry_ts - d).days, 1) / 365.0
                sig = v / 100.0
                new_delta = straddle_delta(S, atm, sig, T)
                rebalance = abs(-new_delta - hedge_units) * S
                pnl_total -= rebalance * HEDGE_COST_BPS / 10000
                hedge_units = -new_delta
                prev_S, prev_V = S, V

            # close: cover straddle (already at last V), close hedge
            pnl_total -= (abs(series[-1][3]) * LOT) * (SLIPPAGE_BPS + SPREAD_BPS) / 10000
            pnl_total -= BROKERAGE * 2
            pnl_total -= abs(hedge_units * series[-1][1]) * HEDGE_COST_BPS / 10000
            # STT on sold premium
            pnl_total -= (series[0][3] * LOT) * STT_SELL

            all_trades.append({
                "year": entry_day.year, "expiry": expiry, "atm": atm,
                "vix": vix0, "vrp": float(mkt.loc[entry_day, "vrp"]),
                "pnl": pnl_total,
            })

    if not all_trades:
        print("no trades")
        return
    t = pd.DataFrame(all_trades)
    # kill-switch after 3 straight losses (chronological)
    t = t.sort_values(["year", "expiry"]).reset_index(drop=True)
    consec, kept = 0, []
    for p in t["pnl"].values:
        if consec >= 3:
            kept.append(0.0); consec = 0; continue
        kept.append(p); consec = consec + 1 if p < 0 else 0
    t["pnl_ks"] = kept

    print("=== YEAR-BY-YEAR: delta-hedged short straddle (real prices, costs) ===")
    by_year = t.groupby("year").agg(trades=("pnl_ks", "size"),
                                    pnl=("pnl_ks", "sum"),
                                    win=("pnl_ks", lambda s: (s > 0).mean() * 100),
                                    avg_vix=("vix", "mean"),
                                    avg_vrp=("vrp", "mean"))
    print(by_year.to_string())

    print(f"\n=== TOTAL ({len(t)} trades) ===")
    print(f"  total PnL: Rs{t.pnl_ks.sum():,.0f}")
    print(f"  avg/trade: Rs{t.pnl_ks.mean():,.0f}")
    print(f"  win rate: {(t.pnl_ks > 0).mean()*100:.0f}%")
    print(f"  worst trade: Rs{t.pnl_ks.min():,.0f}")
    print(f"  profitable years: {(by_year.pnl > 0).sum()}/{len(by_year)}")

    # does it correlate with VRP? (if yes, the vol edge is real)
    corr = t["pnl"].corr(t["vrp"])
    print(f"\n  corr(PnL, VRP): {corr:+.3f}  <- positive = vol edge is real")
    worst = abs(t.pnl_ks.min())
    print(f"\n  RISK: worst trade Rs{worst:,.0f}")
    print(f"    Rs10K: {'RUINED' if worst > 10000 else 'SURVIVES'}")
    print(f"    Rs50K: {'RUINED' if worst > 50000 else 'SURVIVES'}")
    print(f"\n  HONEST: delta-hedging removes TREND risk. If this is profitable")
    print(f"  across years with a tail that fits capital, the vol edge is REAL.")


if __name__ == "__main__":
    main()
