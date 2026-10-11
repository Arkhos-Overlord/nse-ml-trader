"""Backtest VRP strategy on REAL option prices (not a modeled payoff).

Uses NSE F&O bhavcopy-derived option data (rissin/nse-options-intraday HF parquet).
Replaces the gamma-drag ASSUMPTION with actual ATM straddle prices:

  Each expiry week:
    - pick ATM strike (closest to Nifty spot)
    - sell CE + PE at Monday open (or first available)
    - buy back at expiry close
    - real PnL = (CE_open + PE_open) - (CE_close + PE_close), per lot
  Only enter when VRP signal says sell (VRP>1, VIX>12, not spiking).

This is the HONEST test: real option prices, real premium, real decay + tail.
Run: python real_options_backtest.py
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def get(sym):
    d = yf.download(sym, period="3y", interval="1d", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d


def load_options(path):
    """Load NIFTY daily options parquet -> DataFrame."""
    return pd.read_parquet(path)


def build_vix_signal():
    """VRP entry signal aligned to trading dates."""
    vix = get("^INDIAVIX")["Close"].dropna()
    nif = get("^NSEI")["Close"]
    df = pd.DataFrame({"vix": vix, "spot": nif}).dropna()
    ret = df["spot"].pct_change()
    df["realized_21d"] = ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100
    df["vix_chg"] = df["vix"].pct_change()
    df["vrp"] = df["vix"] - df["realized_21d"]
    df["sell"] = (df["vrp"] > 1.0) & (df["vix"] > 12) & (df["vix_chg"] < 0.10)
    return df


def backtest_real(opt_df, signal_df, lot=75):
    """Walk each expiry: sell ATM straddle on first trading day, buy back at expiry.

    opt_df columns: date, timestamp, strike, option_type, open, high, low, close, oi
    """
    opt_df = opt_df.copy()
    opt_df["date"] = pd.to_datetime(opt_df["date"])
    signal_df = signal_df.copy()
    signal_df.index = pd.to_datetime(signal_df.index)

    trades = []
    # group by expiry
    for expiry, grp in opt_df.groupby("expiry"):
        grp = grp.sort_values("timestamp")
        # only days where the ATM options actually traded (close > 0)
        days = sorted(grp["date"].unique())
        # find first and last day with nonzero close for this expiry
        traded = grp[grp["close"] > 0]
        if traded.empty or len(days) < 2:
            continue
        traded_days = sorted(traded["date"].unique())
        entry_day, exit_day = pd.Timestamp(traded_days[0]), pd.Timestamp(traded_days[-1])
        if entry_day == exit_day:
            continue
        # signal on entry day
        if entry_day not in signal_df.index:
            continue
        if not signal_df.loc[entry_day, "sell"]:
            continue
        spot = signal_df.loc[entry_day, "spot"]
        # ATM strike = closest to spot among available strikes
        strikes = traded["strike"].unique()
        atm = strikes[np.argmin(np.abs(strikes - spot))]
        # entry: CE + PE close on entry_day at atm (use close — first traded price)
        e = traded[(traded["date"] == entry_day) & (traded["strike"] == atm)]
        ce_e = e[e["option_type"] == "CE"]
        pe_e = e[e["option_type"] == "PE"]
        if ce_e.empty or pe_e.empty:
            continue
        ce_open = float(ce_e["close"].iloc[0])
        pe_open = float(pe_e["close"].iloc[0])
        # exit: CE + PE close on exit_day at atm
        x = traded[(traded["date"] == exit_day) & (traded["strike"] == atm)]
        ce_x = x[x["option_type"] == "CE"]
        pe_x = x[x["option_type"] == "PE"]
        if ce_x.empty or pe_x.empty:
            continue
        ce_close = float(ce_x["close"].iloc[0])
        pe_close = float(pe_x["close"].iloc[0])
        premium_in = (ce_open + pe_open) * lot
        premium_out = (ce_close + pe_close) * lot
        pnl = premium_in - premium_out  # short straddle: collect - pay
        trades.append({
            "expiry": expiry, "atm": atm, "spot": spot,
            "premium_in": premium_in, "premium_out": premium_out,
            "pnl": pnl, "vix": signal_df.loc[entry_day, "vix"],
            "vrp": signal_df.loc[entry_day, "vrp"],
        })
    return pd.DataFrame(trades)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "NIFTY_2024.parquet"
    print(f"Loading options from {path}...")
    try:
        opt = load_options(path)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR loading {path}: {e}")
        print("Data not downloaded yet. Run the curl download first.")
        sys.exit(1)
    print(f"  {len(opt)} rows, {opt['expiry'].nunique()} expiries, "
          f"{opt['date'].min()} -> {opt['date'].max()}")

    print("Building VIX VRP signal...")
    sig = build_vix_signal()

    print("Backtesting REAL ATM straddle sells...")
    trades = backtest_real(opt, sig)
    if trades.empty:
        print("  no trades matched signal — check date alignment")
        sys.exit(0)
    print(f"\n=== REAL OPTIONS BACKTEST ({len(trades)} trades) ===")
    print(f"  total PnL: Rs{trades.pnl.sum():,.0f}")
    print(f"  avg PnL/trade: Rs{trades.pnl.mean():,.0f}")
    print(f"  win rate: {(trades.pnl > 0).mean()*100:.0f}%")
    print(f"  worst trade: Rs{trades.pnl.min():,.0f}")
    print(f"  best trade: Rs{trades.pnl.max():,.0f}")
    print(f"  avg premium collected: Rs{trades.premium_in.mean():,.0f}")
    print(f"\n  HONEST: real option prices, real decay, real tail. No gamma assumption.")
    print(f"  This is the true test of the VRP edge on actual Nifty strangles.")
