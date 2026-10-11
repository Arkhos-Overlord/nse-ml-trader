"""DEFINITIVE TEST: VRP iron condor with REALISTIC costs across ALL years.

Everything prior used close-to-close fills (too optimistic). This models reality:
  - entry/exit at MID +/- slippage (you don't get the close price)
  - bid-ask spread cost on every leg (illiquid wings cost more)
  - STT + brokerage per trade (Nifty options ~ Rs20-40/lot round trip)
  - defined-risk iron condor (buy wings -> capped tail)

If it's STILL profitable across many years with realistic costs, that's proof.
If realistic costs kill it, that's the honest answer. Run after data downloads.
Run: python definitive_test.py
"""
import glob
import warnings

import numpy as np
import pandas as pd

from real_options_backtest import build_vix_signal

warnings.filterwarnings("ignore")

LOT = 75
# realistic Nifty options cost model (per round trip, per leg):
STT_SELL = 0.000625      # 0.0625% on sell premium (STT on options)
BROKERAGE = 20           # Rs20 per order
SLIPPAGE_BPS = 15        # 15bps slippage on entry/exit (worse on wings)
SPREAD_BPS = 20          # bid-ask spread you cross (worse on OTM wings)


def costs(premium_value, n_legs=4):
    """Realistic round-trip cost for an iron condor (4 legs)."""
    stt = premium_value * STT_SELL
    brokerage = BROKERAGE * n_legs * 2  # entry + exit
    slippage = premium_value * (SLIPPAGE_BPS + SPREAD_BPS) / 10000
    return stt + brokerage + slippage


def backtest_condor(opt, sig, wing=1000, use_costs=True):
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
        ce_wing = strikes[np.argmin(np.abs(strikes - (atm + wing)))]
        pe_wing = strikes[np.argmin(np.abs(strikes - (atm - wing)))]

        def px(day, strike, otype):
            r = traded[(traded["date"] == day) & (traded["strike"] == strike)
                       & (traded["option_type"] == otype)]
            return float(r["close"].iloc[0]) if not r.empty else None

        vals_in = [px(entry_day, atm, "CE"), px(entry_day, atm, "PE"),
                   px(entry_day, ce_wing, "CE"), px(entry_day, pe_wing, "PE")]
        vals_out = [px(exit_day, atm, "CE"), px(exit_day, atm, "PE"),
                    px(exit_day, ce_wing, "CE"), px(exit_day, pe_wing, "PE")]
        if None in vals_in or None in vals_out:
            continue
        # net credit: sell ATM (in), buy wings (out)
        credit = ((vals_in[0] + vals_in[1]) - (vals_in[2] + vals_in[3])) * LOT
        debit = ((vals_out[0] + vals_out[1]) - (vals_out[2] + vals_out[3])) * LOT
        pnl = credit - debit
        if use_costs:
            pnl -= costs(abs(credit) + abs(debit))
        max_loss = wing * LOT - credit  # bounded
        trades.append({"expiry": expiry, "year": entry_day.year,
                       "credit": credit, "pnl": pnl, "max_loss": max_loss,
                       "vix": sig.loc[entry_day, "vix"]})
    return pd.DataFrame(trades)


def main():
    print("Building VIX signal (all years)...")
    sig = build_vix_signal()

    files = sorted(glob.glob("NIFTY_*.parquet"))
    print(f"Found {len(files)} year-files: {[f.split('_')[1].split('.')[0] for f in files]}\n")

    for wing in [1000, 1500]:
        print("=" * 55)
        print(f"IRON CONDOR wing +/-{wing}, REALISTIC COSTS, ALL YEARS")
        print("=" * 55)
        all_t = []
        for f in files:
            yr = f.split("_")[1].split(".")[0]
            try:
                opt = pd.read_parquet(f)
            except Exception:  # noqa: BLE001
                continue
            t = backtest_condor(opt, sig, wing=wing, use_costs=True)
            if not t.empty:
                all_t.append(t)
        if not all_t:
            print("  no trades")
            continue
        c = pd.concat(all_t)
        by_year = c.groupby("year").agg(trades=("pnl", "size"),
                                        pnl=("pnl", "sum"),
                                        win=("pnl", lambda s: (s > 0).mean() * 100))
        print(by_year.to_string())
        print(f"\n  TOTAL: {len(c)} trades, Rs{c.pnl.sum():,.0f}")
        print(f"  avg/trade: Rs{c.pnl.mean():,.0f}")
        print(f"  win rate: {(c.pnl > 0).mean()*100:.0f}%")
        print(f"  worst trade: Rs{c.pnl.min():,.0f}")
        print(f"  profitable years: {(by_year.pnl > 0).sum()}/{len(by_year)}")

        # risk of ruin on Rs10K and Rs50K
        worst = abs(c.pnl.min())
        print(f"\n  RISK: worst trade Rs{worst:,.0f}")
        print(f"    Rs10K capital: {'RUINED' if worst > 10000 else 'survives'}")
        print(f"    Rs50K capital: {'RUINED' if worst > 50000 else 'survives'}")
        print()

    # compare: no-cost vs with-cost (how much do costs eat?)
    print("=" * 55)
    print("COST IMPACT (wing 1000): with vs without realistic costs")
    print("=" * 55)
    for use in [False, True]:
        all_t = []
        for f in files:
            try:
                opt = pd.read_parquet(f)
            except Exception:  # noqa: BLE001
                continue
            t = backtest_condor(opt, sig, wing=1000, use_costs=use)
            if not t.empty:
                all_t.append(t)
        if all_t:
            c = pd.concat(all_t)
            label = "WITH costs" if use else "NO costs "
            print(f"  {label}: total Rs{c.pnl.sum():,.0f}, "
                  f"avg Rs{c.pnl.mean():,.0f}, win {(c.pnl>0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
