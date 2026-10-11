"""Equity intraday accumulator — the stronger small-capital candidate.

F&O on Nifty futures is low-edge, HFT-dominated. Individual liquid equities
have more retail-exploitable edge (wider spreads, slower mean-reversion,
more idiosyncratic moves). This pulls clean 5-min intraday for the most
liquid Indian stocks, accumulating daily like the futures pull.

Run: python pull_equity.py
"""
import os
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from indmoney_client import Ind  # noqa: E402

STORE = "data_indmoney/equity_5min.parquet"

# Most liquid Indian stocks (F&O names = highest liquidity)
# alias -> the name INDmoney's lookup returns (avoids fuzzy-match misses)
STOCK_ALIASES = {
    "RELIANCE": "Reliance Industries",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "INFY": "Infosys",
    "TCS": "Tata Consultancy Services",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "ITC": "ITC",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro",
    "AXISBANK": "Axis Bank",
    "HINDUNILVR": "Hindustan Unilever",
    "BAJFINANCE": "Bajaj Finance",
    "MARUTI": "Maruti Suzuki",
    "TATAMOTORS": "Tata Motors",
}
STOCKS = list(STOCK_ALIASES.keys())


def resolve(c):
    """Resolve ticker -> ind_key using explicit company names."""
    names = list(set(STOCK_ALIASES.values()))
    found = {}
    for i in range(0, len(names), 10):
        batch = names[i:i + 10]
        r = c.call("lookup_ind_keys", {"names": batch, "filter_type": "IN_STOCKS"})
        for x in (r or []):
            found[x["name"].upper()] = x["ind_key"]
    keys = {}
    for tick, company in STOCK_ALIASES.items():
        k = found.get(company.upper())
        if k:
            keys[tick] = k
    return keys


def main():
    c = Ind()
    c.init()
    print(f"=== equity intraday accumulate {time.strftime('%Y-%m-%d %H:%M')} ===")

    keys = resolve(c)
    print(f"  resolved {len(keys)}/{len(STOCKS)} stocks")

    frames = []
    for name, key in keys.items():
        for iv, lb in [("5minute", "7d"), ("15minute", "7d")]:
            try:
                r = c.call("get_indian_stocks_ohlc",
                           {"ind_key": key, "interval": iv, "lookback": lb})
            except Exception as e:  # noqa: BLE001
                print(f"  {name} {iv}: ERROR {str(e)[:60]}")
                continue
            n = r.get("count", 0) if isinstance(r, dict) else 0
            if not n:
                continue
            df = pd.DataFrame(r["candles"])
            df["ind_key"], df["label"], df["interval"] = key, name, iv
            frames.append(df)
        print(f"  {name}: pulled")

    if not frames:
        print("  nothing pulled (market closed)")
        return

    new = pd.concat(frames, ignore_index=True)
    new["datetime_ist"] = pd.to_datetime(new["datetime_ist"])

    if os.path.exists(STORE):
        old = pd.read_parquet(STORE)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["ind_key", "interval", "datetime_ist"]).sort_values(
        ["ind_key", "interval", "datetime_ist"]).reset_index(drop=True)
    combined.to_parquet(STORE, index=False)

    days = combined["datetime_ist"].dt.date.nunique()
    print(f"\n  store: {len(combined)} candles, {days} trading day(s), "
          f"{combined.label.nunique()} stocks")
    print(f"  intervals: {sorted(combined.interval.unique())}")
    print(f"  TARGET: 30+ days for a clean equity mean-reversion test. Now {days}.")


if __name__ == "__main__":
    main()
