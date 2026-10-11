"""PULL EVERYTHING AVAILABLE NOW — stop waiting, build the dataset.

INDmoney gives: 1day-1month intervals with 1 YEAR lookback. I have only 5 days
of intraday but I can pull a full YEAR of daily candles for all 15 stocks +
Nifty index right now. Daily data supports swing/overnight-edge research that
intraday can't. And 15min/7d gives more intraday days than 5min.

This pulls the maximum available in one shot and reports what we have.
Run: python pull_max.py
"""
import os
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from indmoney_client import Ind  # noqa: E402

OUT = "data_indmoney"
STORE_DAILY = os.path.join(OUT, "daily_1y.parquet")
STORE_15 = os.path.join(OUT, "intraday_15min.parquet")

STOCKS = {
    "RELIANCE": "Reliance Industries", "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank", "INFY": "Infosys",
    "TCS": "Tata Consultancy Services", "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel", "ITC": "ITC",
    "KOTAKBANK": "Kotak Mahindra Bank", "LT": "Larsen & Toubro",
    "AXISBANK": "Axis Bank", "HINDUNILVR": "Hindustan Unilever",
    "BAJFINANCE": "Bajaj Finance", "MARUTI": "Maruti Suzuki",
    "TATAMOTORS": "Tata Motors",
}
NIFTY_INDEX = "INDI00012"
MIN_PRICE = 50.0


def resolve(c):
    names = list(set(STOCKS.values()))
    found = {}
    for i in range(0, len(names), 10):
        r = c.call("lookup_ind_keys", {"names": names[i:i + 10], "filter_type": "IN_STOCKS"})
        for x in (r or []):
            found[x["name"].upper()] = x["ind_key"]
    keys = {t: found[nm.upper()] for t, nm in STOCKS.items() if nm.upper() in found}
    return keys


def verify(c, key):
    try:
        r = c.call("get_indian_stocks_ohlc", {"ind_key": key, "interval": "1day", "lookback": "14d"})
        return r["candles"][-1]["close"] if isinstance(r, dict) and r.get("count") else 0
    except Exception:  # noqa: BLE001
        return 0


def pull(c, key, label, interval, lookback):
    try:
        r = c.call("get_indian_stocks_ohlc",
                   {"ind_key": key, "interval": interval, "lookback": lookback})
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:50]
    if not isinstance(r, dict) or not r.get("count"):
        return None, (r.get("error_info") or {}).get("msg", "empty")[:40]
    df = pd.DataFrame(r["candles"])
    df["ind_key"], df["label"], df["interval"] = key, label, interval
    return df, None


def accumulate(store, new):
    if os.path.exists(store):
        old = pd.read_parquet(store)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined["datetime_ist"] = pd.to_datetime(combined["datetime_ist"])
    combined = combined.drop_duplicates(
        subset=["ind_key", "interval", "datetime_ist"]).sort_values(
        ["ind_key", "interval", "datetime_ist"]).reset_index(drop=True)
    combined.to_parquet(store, index=False)
    return combined


def main():
    c = Ind()
    c.init()
    print(f"=== PULL MAX {time.strftime('%Y-%m-%d %H:%M')} ===\n")

    keys = resolve(c)
    print(f"resolved {len(keys)}/{len(STOCKS)} stocks")

    # verify + filter
    good = {}
    for t, k in keys.items():
        px = verify(c, k)
        if px >= MIN_PRICE:
            good[t] = k
        else:
            print(f"  REJECT {t} (Rs{px})")
    print(f"{len(good)} stocks pass price verification\n")

    # 1 YEAR daily for every stock + Nifty
    print("--- 1y daily candles ---")
    daily_frames = []
    daily_frames.append(pull(c, NIFTY_INDEX, "NIFTY50", "1day", "1y")[0])
    for t, k in good.items():
        df, err = pull(c, k, t, "1day", "1y")
        if df is not None:
            daily_frames.append(df)
            print(f"  {t}: {len(df)} daily candles {df.datetime_ist.iloc[0][:10]} -> {df.datetime_ist.iloc[-1][:10]}")
        else:
            print(f"  {t}: FAILED {err}")
    daily_frames = [d for d in daily_frames if d is not None]
    if daily_frames:
        daily = accumulate(STORE_DAILY, pd.concat(daily_frames, ignore_index=True))
        print(f"\n  DAILY STORE: {len(daily)} candles, "
              f"{daily.label.nunique()} instruments, "
              f"{daily.datetime_ist.dt.date.nunique()} trading days")

    # 15min intraday (7d) — more days than 5min
    print("\n--- 15min intraday (7d) ---")
    m15 = []
    m15.append(pull(c, NIFTY_INDEX, "NIFTY50", "15minute", "7d")[0])
    for t, k in good.items():
        df, err = pull(c, k, t, "15minute", "7d")
        if df is not None:
            m15.append(df)
    m15 = [d for d in m15 if d is not None]
    if m15:
        m15df = accumulate(STORE_15, pd.concat(m15, ignore_index=True))
        print(f"  15MIN STORE: {len(m15df)} candles, "
              f"{m15df.datetime_ist.dt.date.nunique()} days, "
              f"{m15df.label.nunique()} instruments")

    print("\n=== SUMMARY ===")
    for f, name in [(STORE_DAILY, "daily_1y"), (STORE_15, "intraday_15min")]:
        if os.path.exists(f):
            d = pd.read_parquet(f)
            print(f"  {name}: {len(d)} candles, {d.datetime_ist.dt.date.nunique()} days")


if __name__ == "__main__":
    main()
