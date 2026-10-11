"""REGIME-GATED VRP: only sell when VIX is in the safe band, sit out the killers.

The unconditional edge died because 2017/2018/2021/2023 (vol spikes) blew up the
account. Those years share a signature: VIX ELEVATED or SPIKING. The edge hunt
already showed VRP is richest in MID vol and weakest in panic.

Gates (all must pass to sell):
  1. VIX in [vix_lo, vix_hi] band — no selling in panic OR dead-calm
  2. VIX not spiking: recent VIX change below threshold (no vol explosion)
  3. VRP still positive (VIX above realized)
  4. Kill-switch after N consecutive losses

Defined-risk iron condor, realistic costs, 12 years. This is the only version
worth trusting — it REFUSES to trade the regimes that killed the last one.
Run: python regime_gated.py
"""
import glob
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

LOT = 75
STT_SELL = 0.000625
BROKERAGE = 20
SLIPPAGE_BPS = 15
SPREAD_BPS = 20


def costs(v, n_legs=4):
    return v * STT_SELL + BROKERAGE * n_legs * 2 + v * (SLIPPAGE_BPS + SPREAD_BPS) / 10000


def get(sym, period="max"):
    d = yf.download(sym, period=period, interval="1d", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d


def build_signal(vix_lo, vix_hi, spike_max):
    vix = get("^INDIAVIX")["Close"].dropna()
    nif = get("^NSEI")["Close"]
    df = pd.DataFrame({"vix": vix, "spot": nif}).dropna()
    ret = df["spot"].pct_change()
    df["realized_21d"] = ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100
    df["vix_chg"] = df["vix"].pct_change()
    df["vix_5d_chg"] = df["vix"].pct_change(5)
    df["vrp"] = df["vix"] - df["realized_21d"]
    df["sell"] = ((df["vrp"] > 1.0) & (df["vix"] > vix_lo) & (df["vix"] < vix_hi)
                  & (df["vix_chg"] < spike_max) & (df["vix_5d_chg"] < spike_max * 2))
    return df


def backtest(opt, sig, wing, kill=3):
    opt = opt.copy()
    opt["date"] = pd.to_datetime(opt["date"])
    sig = sig.copy()
    sig.index = pd.to_datetime(sig.index)
    trades = []
    for expiry, grp in opt.groupby("expiry"):
        traded = grp[grp["close"] > 0].sort_values("timestamp")
        if traded.empty:
            continue
        days = sorted(traded["date"].unique())
        if len(days) < 2:
            continue
        entry_day, exit_day = pd.Timestamp(days[0]), pd.Timestamp(days[-1])
        if entry_day not in sig.index or not sig.loc[entry_day, "sell"]:
            continue
        spot = sig.loc[entry_day, "spot"]
        strikes = np.sort(traded["strike"].unique())
        atm = strikes[np.argmin(np.abs(strikes - spot))]
        ce_w = strikes[np.argmin(np.abs(strikes - (atm + wing)))]
        pe_w = strikes[np.argmin(np.abs(strikes - (atm - wing)))]

        def px(day, s, o):
            r = traded[(traded["date"] == day) & (traded["strike"] == s)
                       & (traded["option_type"] == o)]
            return float(r["close"].iloc[0]) if not r.empty else None

        vi = [px(entry_day, atm, "CE"), px(entry_day, atm, "PE"),
              px(entry_day, ce_w, "CE"), px(entry_day, pe_w, "PE")]
        vo = [px(exit_day, atm, "CE"), px(exit_day, atm, "PE"),
              px(exit_day, ce_w, "CE"), px(exit_day, pe_w, "PE")]
        if None in vi or None in vo:
            continue
        credit = ((vi[0] + vi[1]) - (vi[2] + vi[3])) * LOT
        debit = ((vo[0] + vo[1]) - (vo[2] + vo[3])) * LOT
        pnl = credit - debit - costs(abs(credit) + abs(debit))
        trades.append({"year": entry_day.year, "pnl": pnl,
                       "vix": sig.loc[entry_day, "vix"],
                       "max_loss": wing * LOT - credit})
    t = pd.DataFrame(trades)
    # apply kill-switch chronologically
    if not t.empty:
        consec, kept = 0, []
        for p in t["pnl"].values:
            if consec >= kill:
                kept.append(0.0); consec = 0; continue
            kept.append(p); consec = consec + 1 if p < 0 else 0
        t["pnl_ks"] = kept
    return t


def main():
    files = sorted(glob.glob("NIFTY_*.parquet"))
    opts = {f.split("_")[1].split(".")[0]: pd.read_parquet(f) for f in files}

    print("=== SWEEP VIX BANDS (which regime is actually safe?) ===\n")
    bands = [(10, 18), (10, 20), (11, 17), (12, 16), (10, 16), (11, 19), (12, 18)]
    results = []
    for lo, hi in bands:
        sig = build_signal(lo, hi, spike_max=0.10)
        all_t = []
        for yr, opt in opts.items():
            t = backtest(opt, sig, wing=1000)
            if not t.empty:
                all_t.append(t)
        if not all_t:
            continue
        c = pd.concat(all_t)
        by_year = c.groupby("year")["pnl_ks"].sum()
        results.append({
            "band": f"{lo}-{hi}", "trades": len(c),
            "total": c.pnl_ks.sum(), "win": (c.pnl_ks > 0).mean() * 100,
            "worst": c.pnl_ks.min(),
            "yrs_profitable": (by_year > 0).sum(), "n_years": len(by_year),
        })
    r = pd.DataFrame(results).sort_values("total", ascending=False)
    print(r.to_string(index=False))

    best = r.iloc[0]
    print(f"\n=== BEST BAND: VIX {best.band} ===")
    print(f"  {best.trades} trades, Rs{best.total:,.0f}, {best.win:.0f}% win, "
          f"worst Rs{best.worst:,.0f}, {best.yrs_profitable}/{best.n_years} yrs")

    # detailed year-by-year for best band
    lo, hi = map(int, best.band.split("-"))
    sig = build_signal(lo, hi, spike_max=0.10)
    all_t = []
    for yr, opt in opts.items():
        t = backtest(opt, sig, wing=1000)
        if not t.empty:
            all_t.append(t)
    c = pd.concat(all_t)
    print(f"\n=== YEAR-BY-YEAR (VIX {lo}-{hi}, condor 1000, costs, kill-switch) ===")
    print(c.groupby("year").agg(trades=("pnl_ks", "size"),
                                pnl=("pnl_ks", "sum"),
                                win=("pnl_ks", lambda s: (s > 0).mean() * 100),
                                avg_vix=("vix", "mean")).to_string())
    worst = abs(c.pnl_ks.min())
    print(f"\n  worst trade: Rs{worst:,.0f}")
    print(f"  Rs10K: {'RUINED' if worst > 10000 else 'SURVIVES'}")
    print(f"  Rs50K: {'RUINED' if worst > 50000 else 'SURVIVES'}")
    print(f"\n  HONEST: this only counts if it's profitable in MOST years with a")
    print(f"  tail that fits the capital. Otherwise still a bust.")


if __name__ == "__main__":
    main()
