"""Widen the daily HISTORY net — add liquid Nifty 200 names to the daily store.

Why this matters: the liquid-name momentum engine (liquid_momentum.py, the
research-map Tier-1 build) needs deep DAILY history across a LIQUID universe.
The literature config (12m formation + 6-12m hold) needs 19-24 months per
stock. INDmoney returns up to 1y of daily history in ONE call per instrument,
so each new liquid name we resolve gets ~246 days immediately, then grows via
the scheduled daily task.

This script:
  1. Resolves NEW liquid large/mid-cap names (Nifty 200 not yet in the store)
     via INDmoney lookup, using explicit company-name aliases (L13: fuzzy
     match mixed Reliance Power into RELIANCE) and price-verifying every key.
  2. Pulls daily (1day) history for each, appends to daily_1y.parquet,
     dedup-safe on (ind_key, interval, datetime_ist).
  3. Prints honest before/after counts + duplicate check.

Reuses the exact resolve/verify/dedup pattern proven in pull_equity.py.
Run: python pull_daily_universe.py
"""
import os
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from indmoney_client import Ind  # noqa: E402

STORE = "data_indmoney/daily_1y.parquet"
EXPECT_MIN_PRICE = 50.0   # L13: reject wrong ind_keys priced below Rs50

# Additional liquid Nifty 200 names not yet in the daily store (which has the
# original ~40). Company-name aliases avoid fuzzy-match misses (L13).
NEW_ALIASES = {
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports",
    "APOLLOHOSP": "Apollo Hospitals",
    "ASIANPAINT": "Asian Paints",
    "DMART": "Avenue Supermarts",
    "BAJAJ-AUTO": "Bajaj Auto",
    "BEL": "Bharat Electronics",
    "BPCL": "Bharat Petroleum",
    "CIPLA": "Cipla",
    "COALINDIA": "Coal India",
    "DIVISLAB": "Divi's Laboratories",
    "DRREDDY": "Dr Reddy's Laboratories",
    "EICHERMOT": "Eicher Motors",
    "GRASIM": "Grasim Industries",
    "HCLTECH": "HCL Technologies",
    "HEROMOTOCO": "Hero MotoCorp",
    "HINDALCO": "Hindalco Industries",
    "JSWSTEEL": "JSW Steel",
    "M&M": "Mahindra & Mahindra",
    "MARICO": "Marico",
    "NESTLEIND": "Nestle India",
    "NTPC": "NTPC",
    "ONGC": "Oil & Natural Gas",
    "POWERGRID": "Power Grid Corporation",
    "SUNPHARMA": "Sun Pharmaceutical",
    "TATACONSUM": "Tata Consumer",
    "TATASTEEL": "Tata Steel",
    "TECHM": "Tech Mahindra",
    "TITAN": "Titan Company",
    "ULTRACEMCO": "UltraTech Cement",
    "WIPRO": "Wipro",
}


def existing_labels():
    if not os.path.exists(STORE):
        return set()
    return set(pd.read_parquet(STORE, columns=["label"])["label"].unique())


def resolve(c, have):
    """Resolve NEW aliases -> ind_key, price-verifying each (L13)."""
    todo = {t: n for t, n in NEW_ALIASES.items() if t not in have}
    print(f"  resolving {len(todo)} new names ({len(have)} already in store)")
    names = sorted(set(todo.values()))
    found = {}
    for i in range(0, len(names), 10):
        r = c.call("lookup_ind_keys", {"names": names[i:i + 10], "filter_type": "IN_STOCKS"})
        for x in (r or []):
            found[x["name"].upper()] = x["ind_key"]
    keys = {}
    for tick, company in todo.items():
        k = found.get(company.upper())
        if not k:
            print(f"    {tick}: no ind_key for '{company}'")
            continue
        # VERIFY price (L13): wrong key -> wildly wrong price -> poison dataset
        try:
            r = c.call("get_indian_stocks_ohlc",
                       {"ind_key": k, "interval": "1day", "lookback": "5d"})
            px = r["candles"][-1]["close"] if isinstance(r, dict) and r.get("count") else 0
        except Exception:  # noqa: BLE001
            px = 0
        if 0 < px < EXPECT_MIN_PRICE:
            print(f"    REJECT {tick}: price Rs{px} (< Rs{EXPECT_MIN_PRICE}) — wrong instrument")
            continue
        keys[tick] = k
    return keys


def main():
    have = existing_labels()
    before_n = pd.read_parquet(STORE).label.nunique() if os.path.exists(STORE) else 0
    print(f"=== widen daily history net {time.strftime('%Y-%m-%d %H:%M')} ===")
    print(f"  store before: {before_n} instruments")

    c = Ind()
    c.init()
    keys = resolve(c, have)
    print(f"  resolved {len(keys)} new instruments")

    if not keys:
        print("  nothing new to add")
        return

    frames = []
    for tick, key in keys.items():
        try:
            r = c.call("get_indian_stocks_ohlc",
                       {"ind_key": key, "interval": "1day", "lookback": "1y"})
        except Exception as e:  # noqa: BLE001
            print(f"    {tick}: ERROR {str(e)[:60]}")
            continue
        n = r.get("count", 0) if isinstance(r, dict) else 0
        if not n:
            print(f"    {tick}: 0 candles")
            continue
        df = pd.DataFrame(r["candles"])
        df["ind_key"], df["label"], df["interval"] = key, tick, "1day"
        d = pd.to_datetime(df["datetime_ist"])
        print(f"    {tick}: {n} days, {d.min().date()} -> {d.max().date()}")
        frames.append(df)

    if not frames:
        print("  nothing pulled")
        return

    new = pd.concat(frames, ignore_index=True)
    new["datetime_ist"] = pd.to_datetime(new["datetime_ist"])
    old = pd.read_parquet(STORE) if os.path.exists(STORE) else new.iloc[:0]
    combined = pd.concat([old, new], ignore_index=True)

    # dedup + duplicate audit (the honesty gate)
    dupes_before = combined.duplicated(subset=["ind_key", "interval", "datetime_ist"]).sum()
    combined = combined.drop_duplicates(
        subset=["ind_key", "interval", "datetime_ist"]).sort_values(
        ["label", "datetime_ist"]).reset_index(drop=True)
    combined.to_parquet(STORE, index=False)

    after_n = combined.label.nunique()
    dd = pd.to_datetime(combined["datetime_ist"])
    print(f"\n  store after: {after_n} instruments (+{after_n - before_n}), "
          f"{len(combined)} candles, {dd.dt.date.nunique()} unique days")
    print(f"  duplicates removed: {dupes_before} (0 = clean)")
    print(f"  date span: {dd.min().date()} -> {dd.max().date()}")
    new_names = sorted(set(new.label.unique()) - have)
    print(f"  new labels added: {new_names}")
    print(f"\n  These now accumulate daily via INDmoney_Daily_Pull; momentum")
    print(f"  engine needs ~19-24 months -> matures ~late 2027.")


if __name__ == "__main__":
    main()
