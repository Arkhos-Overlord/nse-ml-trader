"""CRYPTO intraday — the asset class where small capital isn't disqualified.

Why crypto is different from the equity dead-end:
  - 24/7: no market-hours gap, more data per day
  - Retail edge EXISTS (documented): momentum, funding-rate carry, vol
  - Fractional sizing: Rs1.5L or Rs10K both tradeable (no 65-lot minimum)
  - Free public data (Binance/Coinbase public APIs, no key for historical)
  - Lower friction at scale (0.1% fees, tight spreads on majors)

Test on real BTC/ETH data: intraday momentum, mean-reversion, funding carry.
Run: python crypto_pull.py   (fetches real klines)
     python crypto_edge.py   (backtests)
"""
import os
import time
import urllib.request
import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

OUT = "data_indmoney/crypto"
os.makedirs(OUT, exist_ok=True)

# Binance public API — no key needed for historical klines
BASE = "https://api.binance.com/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
INTERVAL = "5m"
LIMIT = 1000   # max per request


def fetch(symbol, interval="5m", limit=1000, end_time=None):
    """Fetch klines. Returns DataFrame."""
    url = f"{BASE}?symbol={symbol}&interval={interval}&limit={limit}"
    if end_time:
        url += f"&endTime={end_time}"
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    if not data:
        return None
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbb", "tbq", "ignore"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def fetch_history(symbol, interval="5m", days=90):
    """Paginate back `days` by walking endTime."""
    all_df = []
    end = int(time.time() * 1000)
    target = end - days * 86400 * 1000
    while end > target:
        df = fetch(symbol, interval, LIMIT, end_time=end)
        if df is None or df.empty:
            break
        all_df.append(df)
        end = int(df.datetime.iloc[0].timestamp() * 1000) - 1
        if len(df) < LIMIT:
            break
        time.sleep(0.2)
    if not all_df:
        return None
    full = pd.concat(all_df).drop_duplicates("datetime").sort_values("datetime")
    return full.reset_index(drop=True)


def main():
    print("=== CRYPTO PULL (Binance public, no key) ===")
    for sym in SYMBOLS:
        try:
            df = fetch_history(sym, INTERVAL, days=90)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: ERROR {str(e)[:60]}")
            continue
        if df is None or df.empty:
            print(f"  {sym}: no data")
            continue
        p = os.path.join(OUT, f"{sym}_5m.parquet")
        df.to_parquet(p, index=False)
        steps = df.close.diff().abs().mean()
        print(f"  {sym}: {len(df)} bars {df.datetime.iloc[0].date()} -> "
              f"{df.datetime.iloc[-1].date()}, mean |5m step| ${steps:.1f}")
    print(f"\n  saved to {OUT}/")
    print("  NOTE: crypto trades 24/7, so 90 days = ~26k bars = REAL sample")


if __name__ == "__main__":
    main()
