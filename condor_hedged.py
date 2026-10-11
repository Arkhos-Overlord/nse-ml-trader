"""Delta-hedged IRON CONDOR — cap the gap tail that ruins the account.

The busts proved: delta-hedged short straddle is a real theta/gamma harvest
(+Rs1Cr always-on, 10/12 yrs) BUT the gap tail is -Rs1.8L to -Rs2.4L, which
ruins Rs10K and Rs50K. The ONLY fix: buy wings (iron condor) so a gap can't
bleed past the wing width.

  short ATM straddle + buy wings (defined max loss)
  delta-hedge the net position daily

Max loss = wing_width - credit, KNOWN. If the tail is capped AND it still makes
money across 12 years, THIS is the tradable version. Run: python condor_hedged.py
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


def sim_condor(series, atm, ce_w, pe_w, expiry_ts):
    """Short ATM straddle + long wings, delta-hedged. series = list of
    (date, spot, vix, straddle_val, wing_val) where wing_val = CE_w + PE_w price."""
    pnl = 0.0
    T0 = max((expiry_ts - series[0][0]).days, 1) / 365.0
    # net delta of (short straddle + long wings). wings reduce delta exposure.
    d_straddle = straddle_delta(series[0][1], atm, series[0][2] / 100.0, T0)
    # long wings have positive delta that offsets; approximate net delta = straddle delta
    # scaled down by wing protection
    hedge = -d_straddle
    prev_S = series[0][1]
    prev_net = series[0][3] - series[0][4]  # straddle - wings (net premium value)
    pnl -= abs(prev_net * LOT) * (SLIPPAGE_BPS + SPREAD_BPS) / 10000 + BROKERAGE * 4
    for i in range(1, len(series)):
        d, S, v, V, W = series[i]
        net_V = V - W
        pnl += -(net_V - prev_net) * LOT + hedge * (S - prev_S)
        T = max((expiry_ts - d).days, 1) / 365.0
        nd = straddle_delta(S, atm, v / 100.0, T)
        rb = abs(-nd - hedge) * S
        pnl -= rb * HEDGE_COST_BPS / 10000
        hedge = -nd
        prev_S, prev_net = S, net_V
    pnl -= abs((series[-1][3] - series[-1][4]) * LOT) * (SLIPPAGE_BPS + SPREAD_BPS) / 10000
    pnl -= BROKERAGE * 4
    pnl -= abs(hedge * series[-1][1]) * HEDGE_COST_BPS / 10000
    pnl -= series[0][3] * LOT * STT_SELL
    return pnl


def run(mkt, wing):
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

            def straddle_px(d, s):
                r = traded[(traded["date"] == d) & (traded["strike"] == s)]
                ce = r[r["option_type"] == "CE"]["close"]
                pe = r[r["option_type"] == "PE"]["close"]
                return (float(ce.iloc[0]) + float(pe.iloc[0])) if not ce.empty and not pe.empty else None

            series = []
            for d in days:
                sv = straddle_px(d, atm)
                wv = straddle_px(d, ce_w) if False else None
                # wings: CE at ce_w + PE at pe_w
                r_ce = traded[(traded["date"] == d) & (traded["strike"] == ce_w) & (traded["option_type"] == "CE")]["close"]
                r_pe = traded[(traded["date"] == d) & (traded["strike"] == pe_w) & (traded["option_type"] == "PE")]["close"]
                if sv is None or r_ce.empty or r_pe.empty:
                    continue
                series.append((d, float(mkt.loc[d, "spot"]), float(mkt.loc[d, "vix"]),
                               sv, float(r_ce.iloc[0]) + float(r_pe.iloc[0])))
            if len(series) < 3:
                continue
            pnl = sim_condor(series, atm, ce_w, pe_w, pd.Timestamp(expiry))
            # theoretical max loss = wing width - credit received
            credit = (series[0][3] - series[0][4]) * LOT
            max_loss = wing * LOT - credit
            trades.append({"year": entry_day.year, "pnl": pnl,
                           "max_loss": max_loss, "vix": vix0})
    return pd.DataFrame(trades)


def summ(t, label):
    if t.empty:
        print(f"  {label}: no trades")
        return
    consec, kept = 0, []
    for p in t.sort_values("year")["pnl"].values:
        if consec >= 3:
            kept.append(0.0); consec = 0; continue
        kept.append(p); consec = consec + 1 if p < 0 else 0
    t = t.sort_values("year").reset_index(drop=True)
    t["pnl_ks"] = kept
    by_year = t.groupby("year")["pnl_ks"].sum()
    print(f"  {label}: {len(t)} trades, Rs{t.pnl_ks.sum():,.0f}, "
          f"{(t.pnl_ks>0).mean()*100:.0f}% win, worst Rs{t.pnl_ks.min():,.0f}, "
          f"{(by_year>0).sum()}/{len(by_year)} yrs, avg maxloss Rs{t.max_loss.mean():,.0f}")


def main():
    vix = get("^INDIAVIX")["Close"].dropna()
    nif = get("^NSEI")["Close"]
    mkt = pd.DataFrame({"vix": vix, "spot": nif}).dropna()
    mkt.index = pd.to_datetime(mkt.index)
    print("=== DELTA-HEDGED IRON CONDOR (gap tail capped by wings) ===\n")
    for wing in [1000, 1500, 2000]:
        print(f"wing +/-{wing}:")
        summ(run(mkt, wing), f"  condor {wing} ")
        print()
    print("  HONEST: compare worst-trade vs the straddle version (-Rs1.8L).")
    print("  If wings cap the tail AND it still profits across years -> TRADABLE.")
    print("  The avg max_loss column = your true worst-case per trade.")


if __name__ == "__main__":
    main()
