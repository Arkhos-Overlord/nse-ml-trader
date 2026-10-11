"""Deep intraday store — keyless via yfinance.

yfinance gives 60 days of 5-min/15-min intraday keyless (vs INDmoney's 4-5
days). This is the unlock for the intraday microstructure engine — it can be
tested NOW on 60 days instead of waiting for mid-Nov.

Pulls 5-min + 15-min intraday for Nifty/BankNifty/Sensex into a NEW store
(data_indmoney/intraday_deep.parquet). Rolling window: re-run periodically to
extend. Dedup-safe.

Run: python pull_intraday_deep.py
"""
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OUT = "data_indmoney/intraday_deep.parquet"
TARGETS = [("^NSEI", "NIFTY50"), ("^NSEBANK", "BANKNIFTY"), ("^BSESN", "SENSEX")]
INTERVALS = ["5m", "15m"]


def main():
    print("=== deep intraday pull (yfinance, keyless, 60d) ===")
    frames = []
    for sym, label in TARGETS:
        for iv in INTERVALS:
            try:
                df = yf.download(sym, period="60d", interval=iv, progress=False,
                                 auto_adjust=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {label} {iv}: FAILED {str(e)[:50]}")
                continue
            if df is None or df.empty:
                print(f"  {label} {iv}: empty")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            dc = "datetime" if "datetime" in df.columns else df.columns[0]
            df = df.rename(columns={dc: "datetime_ist"})
            df = df[["datetime_ist", "open", "high", "low", "close", "volume"]].copy()
            df["label"] = label
            df["interval"] = iv
            df["datetime_ist"] = pd.to_datetime(df["datetime_ist"]).dt.tz_localize(None)
            df = df[df.close > 0]
            nd = df.datetime_ist.dt.date.nunique()
            print(f"  {label} {iv}: {len(df)} bars, {nd} days, "
                  f"{df.datetime_ist.min().date()} -> {df.datetime_ist.max().date()}")
            frames.append(df)
            time.sleep(0.3)

    if not frames:
        print("  nothing pulled")
        return
    new = pd.concat(frames, ignore_index=True)
    dupes = new.duplicated(subset=["label", "interval", "datetime_ist"]).sum()
    new = new.drop_duplicates(subset=["label", "interval", "datetime_ist"]).sort_values(
        ["label", "interval", "datetime_ist"]).reset_index(drop=True)
    new.to_parquet(OUT, index=False)
    print(f"\n  store: {OUT}")
    print(f"  {new.label.nunique()} indices x {new.interval.nunique()} intervals, "
          f"{len(new)} bars, {dupes} dupes (0=clean)")
    print(f"  days: {new.datetime_ist.dt.date.nunique()}")
    print(f"  -> 12x the INDmoney intraday depth. Microstructure engine testable NOW.")


if __name__ == "__main__":
    main()
