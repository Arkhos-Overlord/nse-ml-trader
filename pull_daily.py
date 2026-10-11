"""Daily accumulator — build a clean intraday dataset one trading day at a time.

INDmoney returns only the most recent day for 1min/7d. So: run this EVERY
TRADING DAY (during or after market hours) and it appends that day's candles.
After ~30 trading days we have a real 30-day clean intraday dataset — enough
to honestly test mean-reversion (the strategy that died on fake parity data).

Deduplicates by (ind_key, interval, datetime), so re-running is safe.
Logs honestly: what was pulled, what was skipped, why.

Run: python pull_daily.py
"""
import json
import os
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from indmoney_client import Ind  # noqa: E402

STORE = "data_indmoney/clean_1min.parquet"
TARGETS = [
    ("INDI00012", "NIFTY50_INDEX"),
    ("INDFNHTHTSZ9", "NIFTY_Oct_Fut"),
    ("INDFN2S1JBP8", "NIFTY_Nov_Fut"),
]
INTERVALS = ["1minute", "5minute"]


def main():
    c = Ind()
    c.init()
    print(f"=== daily accumulate {time.strftime('%Y-%m-%d %H:%M')} ===")

    frames = []
    for key, label in TARGETS:
        for iv in INTERVALS:
            try:
                r = c.call("get_indian_stocks_ohlc",
                           {"ind_key": key, "interval": iv, "lookback": "7d"})
            except Exception as e:  # noqa: BLE001
                print(f"  {label} {iv}: ERROR {e}")
                continue
            n = r.get("count", 0) if isinstance(r, dict) else 0
            if not n:
                err = (r.get("error_info") or {}).get("msg", "")
                print(f"  {label} {iv}: 0 candles {f'({err[:50]})' if err else ''}")
                continue
            df = pd.DataFrame(r["candles"])
            df["ind_key"], df["label"], df["interval"] = key, label, iv
            days = df["datetime_ist"].str[:10].nunique()
            print(f"  {label} {iv}: {n} candles across {days} day(s), "
                  f"{df.datetime_ist.iloc[0][:16]} -> {df.datetime_ist.iloc[-1][11:16]}")
            frames.append(df)

    if not frames:
        print("  nothing pulled (market closed / illiquid window)")
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

    print(f"\n  store now: {len(combined)} candles "
          f"(+{before - (before - len(new))} attempted, {len(combined)} unique)")
    n_days = combined["datetime_ist"].dt.date.nunique()
    print(f"  trading days covered: {n_days}")
    per = combined.groupby(["label", "interval"]).size()
    for k, v in per.items():
        print(f"    {k[0]} {k[1]}: {v}")
    print(f"\n  TARGET: 30+ trading days for a clean mean-reversion test. "
          f"Currently {n_days}.")


if __name__ == "__main__":
    main()
