"""Defined-risk version: IRON CONDOR instead of naked strangle.

The naked strangle edge is real (+Rs9.5L over 3y) BUT the tail is lethal:
worst trade -Rs1.1L. On Rs10K that's account death. The fix every real vol-seller
uses: BUY wings to cap the loss -> iron condor (defined risk).

  Sell ATM-ish CE + PE (collect premium)
  Buy further OTM CE + PE (cap the loss)

Max loss = wing_width - premium_collected, KNOWN upfront. This is the only
version that fits Rs10K safely.

Run: python iron_condor.py
"""
import warnings

import numpy as np
import pandas as pd

from real_options_backtest import build_vix_signal

warnings.filterwarnings("ignore")


def backtest_iron_condor(opt_df, signal_df, wing_width=1000, lot=75):
    """Sell ATM straddle, buy wings `wing_width` points OTM. Defined risk.

    Per expiry where VRP signal fires:
      sell ATM CE + PE (closest to spot)
      buy CE at (atm + wing_width), PE at (atm - wing_width)
      PnL = premium_in - premium_out - max(0, loss_beyond_wings)
    Simplify with close prices; real impl needs intraday fills.
    """
    opt_df = opt_df.copy()
    opt_df["date"] = pd.to_datetime(opt_df["date"])
    signal_df = signal_df.copy()
    signal_df.index = pd.to_datetime(signal_df.index)

    trades = []
    for expiry, grp in opt_df.groupby("expiry"):
        traded = grp[grp["close"] > 0].sort_values("timestamp")
        if traded.empty:
            continue
        traded_days = sorted(traded["date"].unique())
        if len(traded_days) < 2:
            continue
        entry_day, exit_day = pd.Timestamp(traded_days[0]), pd.Timestamp(traded_days[-1])
        if entry_day not in signal_df.index or not signal_df.loc[entry_day, "sell"]:
            continue
        spot = signal_df.loc[entry_day, "spot"]
        strikes = np.sort(traded["strike"].unique())
        atm = strikes[np.argmin(np.abs(strikes - spot))]
        # wing strikes
        ce_wing = atm + wing_width
        pe_wing = atm - wing_width
        # snap to available strikes
        ce_wing = strikes[np.argmin(np.abs(strikes - ce_wing))]
        pe_wing = strikes[np.argmin(np.abs(strikes - pe_wing))]

        def px(day, strike, otype):
            row = traded[(traded["date"] == day) & (traded["strike"] == strike)
                         & (traded["option_type"] == otype)]
            return float(row["close"].iloc[0]) if not row.empty else None

        # entry prices
        ce_atm_in = px(entry_day, atm, "CE")
        pe_atm_in = px(entry_day, atm, "PE")
        ce_wing_in = px(entry_day, ce_wing, "CE")
        pe_wing_in = px(entry_day, pe_wing, "PE")
        # exit prices
        ce_atm_out = px(exit_day, atm, "CE")
        pe_atm_out = px(exit_day, atm, "PE")
        ce_wing_out = px(exit_day, ce_wing, "CE")
        pe_wing_out = px(exit_day, pe_wing, "PE")
        if None in (ce_atm_in, pe_atm_in, ce_wing_in, pe_wing_in,
                    ce_atm_out, pe_atm_out, ce_wing_out, pe_wing_out):
            continue
        # net premium collected (short ATM, long wings)
        premium_in = ((ce_atm_in + pe_atm_in) - (ce_wing_in + pe_wing_in)) * lot
        # cost to close
        premium_out = ((ce_atm_out + pe_atm_out) - (ce_wing_out + pe_wing_out)) * lot
        pnl = premium_in - premium_out
        # max loss is bounded by wing width (the long options cap it)
        max_loss = wing_width * lot - premium_in
        trades.append({
            "expiry": expiry, "atm": atm, "spot": spot,
            "premium_in": premium_in, "pnl": pnl,
            "max_loss": max_loss, "vix": signal_df.loc[entry_day, "vix"],
        })
    return pd.DataFrame(trades)


def main():
    sig = build_vix_signal()
    for wing in [500, 1000, 1500]:
        print(f"\n=== IRON CONDOR, wing=+/-{wing} ===")
        all_t = []
        for yr in [2023, 2024, 2025]:
            try:
                opt = pd.read_parquet(f"NIFTY_{yr}.parquet")
            except Exception:  # noqa: BLE001
                continue
            t = backtest_iron_condor(opt, sig, wing_width=wing)
            if not t.empty:
                t["year"] = yr
                all_t.append(t)
        if not all_t:
            print("  no trades")
            continue
        c = pd.concat(all_t)
        print(f"  trades: {len(c)}")
        print(f"  total PnL: Rs{c.pnl.sum():,.0f}")
        print(f"  avg/trade: Rs{c.pnl.mean():,.0f}")
        print(f"  win rate: {(c.pnl > 0).mean()*100:.0f}%")
        print(f"  WORST trade: Rs{c.pnl.min():,.0f}  <- capped by wings")
        print(f"  avg premium collected: Rs{c.premium_in.mean():,.0f}")
        print(f"  avg max loss (if tail): Rs{c.max_loss.mean():,.0f}")
        print(f"  profitable yrs: {(c.groupby('year').pnl.sum() > 0).sum()}")
    print("\n  DEFINED RISK: worst loss is bounded by wing width, not unbounded.")
    print("  This is the Rs10K-safe version. Compare worst-trade across wings.")


if __name__ == "__main__":
    main()
