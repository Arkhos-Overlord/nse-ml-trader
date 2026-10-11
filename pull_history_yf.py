"""Deep daily history backfill — KEYLESS via yfinance.

The liquid-name momentum engine (liquid_momentum.py, Tier-1 #1) needs 19-24
months of daily history to run its literature config (12m formation + 6-12m
hold). INDmoney only gives ~1y (246 days) — too shallow. yfinance is free,
keyless, and gives 5-10 years of adjusted daily OHLCV per NSE ticker.

This pulls deep history for the whole liquid universe already in
daily_1y.parquet, into a NEW store (data_indmoney/daily_history.parquet) so it
never corrupts the INDmoney store. Dedup-safe.

Ticker mapping: NSE stocks -> '<TICKER>.NS'; NIFTY50 -> '^NSEI'.
Special chars pass through (M&M -> 'M&M.NS' works in yfinance).

Run: python pull_history_yf.py
"""
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

UNIVERSE_STORE = "data_indmoney/daily_1y.parquet"
OUT = "data_indmoney/daily_history.parquet"
START = "2018-01-01"   # ~8y back, gives deep history for momentum formation


def to_symbol(label):
    if label == "NIFTY50":
        return "^NSEI"
    return label + ".NS"


def main():
    labels = sorted(pd.read_parquet(UNIVERSE_STORE).label.unique())
    print(f"=== deep history backfill (yfinance, keyless) ===")
    print(f"  universe: {len(labels)} instruments, start={START}\n")

    frames = []
    ok, failed = [], []
    for lb in labels:
        sym = to_symbol(lb)
        try:
            df = yf.download(sym, start=START, progress=False,
                             auto_adjust=True, threads=False)
        except Exception as e:  # noqa: BLE001
            failed.append((lb, f"download error: {str(e)[:50]}"))
            print(f"  {lb:<12} FAILED: {str(e)[:50]}")
            continue
        if df is None or df.empty:
            failed.append((lb, "empty"))
            print(f"  {lb:<12} empty (bad symbol?)")
            continue
        # yfinance returns MultiIndex columns for single ticker sometimes
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        date_col = "date" if "date" in df.columns else df.columns[0]
        df = df.rename(columns={date_col: "datetime_ist"})
        keep = ["datetime_ist", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["label"] = lb
        df["datetime_ist"] = pd.to_datetime(df["datetime_ist"]).dt.tz_localize(None)
        # sanity: drop rows with zero/NaN close
        df = df[df.close > 0]
        n = len(df)
        if n < 400:  # < ~1.6y of daily bars = suspiciously short
            print(f"  {lb:<12} only {n} bars — FLAGGED (short history)")
        else:
            print(f"  {lb:<12} {n} bars, {df.datetime_ist.min().date()} -> "
                  f"{df.datetime_ist.max().date()}")
            ok.append(lb)
        frames.append(df)
        time.sleep(0.2)

    if not frames:
        print("  nothing pulled — yfinance failed for all tickers")
        return

    new = pd.concat(frames, ignore_index=True)
    # dedup audit
    dupes = new.duplicated(subset=["label", "datetime_ist"]).sum()
    new = new.drop_duplicates(subset=["label", "datetime_ist"]).sort_values(
        ["label", "datetime_ist"]).reset_index(drop=True)
    new.to_parquet(OUT, index=False)

    print(f"\n  store: {OUT}")
    print(f"  {new.label.nunique()} instruments, {len(new)} candles, "
          f"{dupes} duplicates (0=clean)")
    print(f"  date span: {new.datetime_ist.min().date()} -> {new.datetime_ist.max().date()}")
    per = new.groupby("label").datetime_ist.agg(["min", "max", "count"])
    yrs = ((per["max"] - per["min"]).dt.days / 365.25)
    print(f"  history depth: median {yrs.median():.1f}y, "
          f"min {yrs.min():.1f}y, max {yrs.max():.1f}y")
    print(f"  succeeded: {len(ok)}/{len(labels)}")
    if failed:
        print(f"  failed: {[f[0] for f in failed]}")

    # sanity checks vs known recent values
    print(f"\n=== SANITY CHECK (recent closes) ===")
    for lb, lo, hi in [("RELIANCE", 1200, 1600), ("HDFCBANK", 700, 1100),
                       ("INFY", 1300, 1900), ("NIFTY50", 20000, 27000)]:
        if lb in per.index:
            c = new[(new.label == lb)].close.iloc[-1]
            ok_ = lo <= c <= hi
            print(f"  {lb:<10} latest close Rs{c:,.1f} "
                  f"(expected ~{lo}-{hi}) {'OK' if ok_ else '<-- CHECK'}")

    met = yrs.median() >= 1.6  # 19-24 months
    print(f"\n  momentum 19-24m requirement satisfiable: "
          f"{'YES' if met else 'NO'} (median {yrs.median():.1f}y)")


if __name__ == "__main__":
    main()
