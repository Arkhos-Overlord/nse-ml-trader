"""Widen intraday universe: pull 5-min for ALL resolved stocks + Nifty futures.

Currently testing intraday on 13 stocks. I have 41 resolved. Pull 5-min intraday
for every one now so the dataset is maximal when it matures. Also pulls Nifty
index + futures 5-min (different microstructure than equities).

Run: python pull_intraday_all.py
"""
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from indmoney_client import Ind  # noqa: E402

DAILY = "data_indmoney/daily_1y.parquet"
STORE = "data_indmoney/equity_5min.parquet"
FUT_STORE = "data_indmoney/clean_1min.parquet"


def main():
    c = Ind()
    c.init()
    daily = pd.read_parquet(DAILY)
    stocks = sorted(daily[daily.label != "NIFTY50"].label.unique())
    print(f"=== widen intraday: {len(stocks)} stocks + Nifty + futures ===")

    # map ticker -> ind_key from daily store
    keymap = daily.drop_duplicates("label").set_index("label")["ind_key"].to_dict()

    frames = []
    for t in stocks:
        k = keymap.get(t)
        if not k:
            continue
        try:
            r = c.call("get_indian_stocks_ohlc", {"ind_key": k, "interval": "5minute", "lookback": "7d"})
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(r, dict) or not r.get("count"):
            continue
        df = pd.DataFrame(r["candles"])
        df["ind_key"], df["label"], df["interval"] = k, t, "5minute"
        frames.append(df)
    print(f"  pulled {len(frames)} stocks with intraday data")

    if frames:
        new = pd.concat(frames, ignore_index=True)
        if os.path.exists(STORE):
            old = pd.read_parquet(STORE)
            combined = pd.concat([old, new], ignore_index=True)
        else:
            combined = new
        combined["datetime_ist"] = pd.to_datetime(combined["datetime_ist"])
        combined = combined.drop_duplicates(
            subset=["ind_key", "interval", "datetime_ist"]).sort_values(
            ["ind_key", "interval", "datetime_ist"]).reset_index(drop=True)
        combined.to_parquet(STORE, index=False)
        print(f"  equity store: {combined.label.nunique()} stocks, "
              f"{combined.datetime_ist.dt.date.nunique()} days, {len(combined)} bars")

    # Nifty index + futures 5min
    ff = []
    for key, label in [("INDI00012", "NIFTY50"), ("INDFNHTHTSZ9", "NIFTY_Oct_Fut"),
                       ("INDFN2S1JBP8", "NIFTY_Nov_Fut")]:
        try:
            r = c.call("get_indian_stocks_ohlc", {"ind_key": key, "interval": "5minute", "lookback": "7d"})
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, dict) and r.get("count"):
            df = pd.DataFrame(r["candles"])
            df["ind_key"], df["label"], df["interval"] = key, label, "5minute"
            ff.append(df)
    if ff:
        newf = pd.concat(ff, ignore_index=True)
        if os.path.exists(FUT_STORE):
            oldf = pd.read_parquet(FUT_STORE)
            combined = pd.concat([oldf, newf], ignore_index=True)
        else:
            combined = newf
        combined["datetime_ist"] = pd.to_datetime(combined["datetime_ist"])
        combined = combined.drop_duplicates(
            subset=["ind_key", "interval", "datetime_ist"]).sort_values(
            ["ind_key", "interval", "datetime_ist"]).reset_index(drop=True)
        combined.to_parquet(FUT_STORE, index=False)
        print(f"  futures store: {combined.label.nunique()} instruments, "
              f"{combined.datetime_ist.dt.date.nunique()} days")


if __name__ == "__main__":
    main()
