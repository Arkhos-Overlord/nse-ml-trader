"""Single-stock intraday pull — where the gap-fade edge actually lives.

Lit (TickerGuard + Zarattini): gap-fade works on SINGLE STOCKS with news
catalysts, NOT on indices. Nifty gap-fade failed (no catalyst -> gaps trend).
yfinance gives 60d of 5-min single-stock intraday keyless. Pull a liquid basket
into intraday_stocks.parquet so we can test gap-fade + 'stock in play' properly.

Run: python pull_intraday_stocks.py
"""
import time
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OUT = "data_indmoney/intraday_stocks.parquet"
# liquid F&O large-caps (news catalysts + liquidity)
TICKERS = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK", "SBIN",
           "BHARTIARTL", "ITC", "LT", "AXISBANK", "KOTAKBANK", "MARUTI"]


def main():
    print("=== single-stock intraday pull (yfinance keyless, 60d) ===")
    frames = []
    for tk in TICKERS:
        try:
            df = yf.download(f"{tk}.NS", period="60d", interval="5m",
                             progress=False, auto_adjust=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {tk}: FAILED {str(e)[:40]}")
            continue
        if df is None or df.empty:
            print(f"  {tk}: empty")
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
        print(f"  {tk}: {len(df)} bars, {df.datetime_ist.dt.date.nunique()} days")
        frames.append(df)
        time.sleep(0.3)
    if not frames:
        print("  nothing pulled")
        return
    new = pd.concat(frames, ignore_index=True)
    dupes = new.duplicated(subset=["label", "datetime_ist"]).sum()
    new = new.drop_duplicates(subset=["label", "datetime_ist"]).sort_values(
        ["label", "datetime_ist"]).reset_index(drop=True)
    new.to_parquet(OUT, index=False)
    print(f"\n  store: {OUT}")
    print(f"  {new.label.nunique()} stocks x {new.datetime_ist.dt.date.nunique()} days, "
          f"{len(new)} bars, {dupes} dupes (0=clean)")
    print(f"  -> gap-fade testable on single stocks (where the lit says it works)")


if __name__ == "__main__":
    main()
