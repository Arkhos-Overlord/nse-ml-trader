"""Wide liquid F&O intraday pull — grow the gap-fade sample toward significance.

The standing lead (short-side single-stock gap-fade, t=+2.34) is stuck at n=6
trades on 12 stocks. The only way to resolve it is MORE trades = more stocks.
Pull the liquid NSE F&O universe (~45 names) at 5-min, keyless via yfinance.

Run: python pull_intraday_stocks_wide.py
"""
import time
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OUT = "data_indmoney/intraday_stocks_wide.parquet"
# liquid NSE F&O names (yfinance .NS suffix; special formats kept literal)
TICKERS = [
    "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK", "SBIN", "BHARTIARTL",
    "ITC", "LT", "AXISBANK", "KOTAKBANK", "MARUTI", "M&M", "ADANIENT",
    "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AUROPHARMA", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL", "BRITANNIA", "CHOLAFIN",
    "CIPLA", "COALINDIA", "DLF", "DRREDDY", "EICHERMOT", "GAIL", "GRASIM",
    "HCLTECH", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "IOC",
    "JINDALSTEL", "JSWSTEEL", "LICI", "NESTLEIND", "NTPC", "ONGC", "PIDILITIND",
    "SBILIFE", "SHRIRAMFIN", "SIEMENS", "SUNPHARMA", "TATACONSUM", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ZYDUSLIFE",
]


def main():
    print(f"=== wide F&O intraday pull ({len(TICKERS)} names, yfinance keyless) ===")
    frames, ok, fail = [], [], []
    for tk in TICKERS:
        try:
            df = yf.download(f"{tk}.NS", period="60d", interval="5m",
                             progress=False, auto_adjust=True)
        except Exception as e:  # noqa: BLE001
            fail.append((tk, str(e)[:30]))
            continue
        if df is None or df.empty:
            fail.append((tk, "empty"))
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        dc = "datetime" if "datetime" in df.columns else df.columns[0]
        df = df.rename(columns={dc: "datetime_ist"})
        df = df[["datetime_ist", "open", "high", "low", "close", "volume"]].copy()
        df["label"] = tk
        df["datetime_ist"] = pd.to_datetime(df["datetime_ist"]).dt.tz_localize(None)
        df = df[df.close > 0]
        if len(df) < 100:
            fail.append((tk, f"thin {len(df)}"))
            continue
        ok.append(tk)
        frames.append(df)
        time.sleep(0.25)
    if not frames:
        print("  nothing pulled")
        return
    new = pd.concat(frames, ignore_index=True)
    dupes = new.duplicated(subset=["label", "datetime_ist"]).sum()
    new = new.drop_duplicates(subset=["label", "datetime_ist"]).sort_values(
        ["label", "datetime_ist"]).reset_index(drop=True)
    new.to_parquet(OUT, index=False)
    print(f"\n  OK: {len(ok)} stocks | FAIL: {len(fail)}")
    if fail:
        print(f"  failed: {[f[0] for f in fail]}")
    print(f"  store: {OUT}")
    print(f"  {new.label.nunique()} stocks x {new.datetime_ist.dt.date.nunique()} days, "
          f"{len(new)} bars, {dupes} dupes")
    print(f"  -> gap-fade sample grows ~{len(ok)//12}x -> toward significance")


if __name__ == "__main__":
    main()
