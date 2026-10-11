"""Expand universe to midcaps + test gap/overnight/volatility strategies.

Large-cap swing = no edge (proven). Midcaps/smallcaps are less efficient —
more retail-exploitable edge. Also test strategies I haven't tried on daily
data: overnight gaps, volatility compression, relative strength.

INDmoney lookup works for any listed stock, so pull a broader universe.
Run: python pull_universe.py
"""
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from indmoney_client import Ind  # noqa: E402

STORE = "data_indmoney/daily_1y.parquet"

# Midcap / smallcap F&O names (less efficient than the large-cap 15)
MIDCAPS = {
    "VOLTAS": "Voltas", "CUMMINSIND": "Cummins India",
    "POLYCAB": "Polycab India", "PERSISTENT": "Persistent Systems",
    "COFORGE": "Coforge", "MPHASIS": "Mphasis",
    "FEDERALBNK": "Federal Bank", "IDFCFIRSTB": "IDFC First Bank",
    "BANDHANBNK": "Bandhan Bank", "PAGEIND": "Page Industries",
    "TATAELXSI": "Tata Elxsi", "LTIM": "LTIMindtree",
    "ASHOKLEY": "Ashok Leyland", "TVSMOTOR": "TVS Motor",
    "BATAINDIA": "Bata India", "MUTHOOTFIN": "Muthoot Finance",
    "ESCORTS": "Escorts Kubota", "EXIDEIND": "Exide Industries",
    "AUBANK": "AU Small Finance Bank", "LICHSGFIN": "LIC Housing Finance",
    "MANAPPURAM": "Manappuram Finance", "SUPREMEIND": "Supreme Industries",
    "ASTRAL": "Astral", "DIXON": "Dixon Technologies",
    "KAYNES": "Kaynes Technology", "TATACOMM": "Tata Communications",
    "GODREJPROP": "Godrej Properties", "OBEROIRLTY": "Oberoi Realty",
    "PRESTIGE": "Prestige Estates", "LALPATHLAB": "Dr Lal Path Labs",
}
NIFTY_INDEX = "INDI00012"
MIN_PRICE = 50.0


def resolve(c, names):
    found = {}
    for i in range(0, len(names), 10):
        r = c.call("lookup_ind_keys", {"names": names[i:i + 10], "filter_type": "IN_STOCKS"})
        for x in (r or []):
            found[x["name"].upper()] = x["ind_key"]
    return found


def verify(c, key):
    try:
        r = c.call("get_indian_stocks_ohlc", {"ind_key": key, "interval": "1day", "lookback": "14d"})
        return r["candles"][-1]["close"] if isinstance(r, dict) and r.get("count") else 0
    except Exception:  # noqa: BLE001
        return 0


def main():
    c = Ind()
    c.init()
    print(f"=== pull midcap universe {pd.Timestamp.now():%Y-%m-%d %H:%M} ===")
    found = resolve(c, list(set(MIDCAPS.values())))
    keys = {t: found[nm.upper()] for t, nm in MIDCAPS.items() if nm.upper() in found}
    print(f"resolved {len(keys)}/{len(MIDCAPS)}")

    frames = []
    for t, k in keys.items():
        px = verify(c, k)
        if px < MIN_PRICE:
            print(f"  REJECT {t} (Rs{px})")
            continue
        try:
            r = c.call("get_indian_stocks_ohlc", {"ind_key": k, "interval": "1day", "lookback": "1y"})
        except Exception as e:  # noqa: BLE001
            print(f"  {t}: ERR {str(e)[:40]}")
            continue
        if not isinstance(r, dict) or not r.get("count"):
            continue
        df = pd.DataFrame(r["candles"])
        df["ind_key"], df["label"], df["interval"] = k, t, "1day"
        frames.append(df)
        print(f"  {t}: {len(df)} candles")

    if not frames:
        print("nothing pulled")
        return
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
    print(f"\nDAILY STORE now: {combined.label.nunique()} instruments, "
          f"{len(combined)} candles, {combined.datetime_ist.dt.date.nunique()} days")


if __name__ == "__main__":
    main()
