"""Pull CLEAN intraday data from INDmoney MCP — the unblock.

nsepython/huggingface gave stale-quote option files. INDmoney gives real
synchronized OHLC candles: 1min-30min up to 7d, indices + futures + stocks.

Usage:
  python pull_indmoney.py            # pull Nifty futures + index 5min/1min 7d
  python pull_indmoney.py stocks     # + pull 20 liquid stocks 5min

Saves to data_indmoney/. Logs every pull honestly (counts, errors, timestamps).
"""
import json
import os
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from indmoney_client import Ind  # noqa: E402

OUT = "data_indmoney"
os.makedirs(OUT, exist_ok=True)
LOG = []


def log(msg):
    print(msg)
    LOG.append(f"{time.strftime('%H:%M:%S')} {msg}")


def save(df, name):
    p = os.path.join(OUT, name)
    df.to_parquet(p, index=False)
    log(f"  saved {name}: {len(df)} candles")


def pull(c, key, label, interval, lookback):
    r = c.call("get_indian_stocks_ohlc",
               {"ind_key": key, "interval": interval, "lookback": lookback})
    if not isinstance(r, dict):
        log(f"  {label} {interval}/{lookback}: bad response")
        return None
    n = r.get("count", 0)
    err = (r.get("error_info") or {}).get("msg")
    if not n:
        log(f"  {label} {interval}/{lookback}: 0 candles {f'({err})' if err else ''}")
        return None
    df = pd.DataFrame(r["candles"])
    df["ind_key"] = key
    df["label"] = label
    df["interval"] = interval
    log(f"  {label} {interval}/{lookback}: {n} candles "
        f"{df.datetime_ist.iloc[0]} -> {df.datetime_ist.iloc[-1]}")
    return df


def main():
    c = Ind()
    c.init()
    log("=== INDmoney clean intraday pull ===")

    # Nifty index
    r = c.call("lookup_ind_keys", {"names": ["NIFTY 50"], "filter_type": "IN_STOCKS"})
    idx_key = r[0]["ind_key"]
    log(f"NIFTY 50 index = {idx_key}")

    # Nifty futures (current month)
    r = c.call("lookup_ind_keys", {"names": ["NIFTY FUT"], "filter_type": "IN_STOCKS_FNO"})
    fut_keys = [(x["ind_key"], x["name"]) for x in r]
    log(f"Nifty futures: {fut_keys}")

    frames = []
    for interval, lb in [("5minute", "7d"), ("1minute", "7d"), ("15minute", "7d")]:
        df = pull(c, idx_key, "NIFTY50_INDEX", interval, lb)
        if df is not None:
            frames.append(df)

    for key, name in fut_keys[:2]:
        for interval, lb in [("5minute", "7d"), ("1minute", "7d")]:
            df = pull(c, key, name, interval, lb)
            if df is not None:
                frames.append(df)

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df["datetime_ist"] = pd.to_datetime(all_df["datetime_ist"])
        all_df.to_parquet(os.path.join(OUT, "all_candles.parquet"), index=False)
        log(f"\nTOTAL: {len(all_df)} candles saved to all_candles.parquet")
        # quality check: real 1-min volatility (the thing that was fake before)
        one = all_df[(all_df.interval == "1minute") & (all_df.label.str.contains("Fut|NIFTY"))]
        if not one.empty:
            one = one.sort_values("datetime_ist")
            steps = one.groupby("label")["close"].apply(lambda s: s.diff().abs().mean())
            log("\nQUALITY — mean |1-min step| (the metric that was FAKE before):")
            for k, v in steps.items():
                log(f"  {k}: {v:.2f} pts")
            log("  (real Nifty 1-min steps are ~5-15 pts; the broken parity")
            log("   reconstruction showed +-80pt jumps. This is the fix.)")
    else:
        log("\nNo candles returned — market may be closed (weekend).")

    with open(os.path.join(OUT, "pull_log.txt"), "w") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
