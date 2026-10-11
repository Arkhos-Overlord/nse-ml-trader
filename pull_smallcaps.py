"""Pull SMALLCAP universe — less efficient, bigger cross-sectional universe.

Factor investing needs a large universe to rank across. Large/mid-caps are
efficient; smallcaps are not. Pull 40+ smallcaps with 1y daily data.

Run: python pull_smallcaps.py
"""
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from indmoney_client import Ind  # noqa: E402

STORE = "data_indmoney/daily_1y.parquet"

SMALLCAPS = {
    "IREDA": "Indian Renewable Energy Development",
    "RVNL": "Rail Vikas Nigam",
    "IRFC": "Indian Railway Finance",
    "SUZLON": "Suzlon Energy",
    "JPPOWER": "Jaiprakash Power Ventures",
    "IDEA": "Vodafone Idea",
    "YESBANK": "Yes Bank",
    "UJJIVANSFB": "Ujjivan Small Finance Bank",
    "EQUITASBNK": "Equitas Small Finance Bank",
    "KFINTECH": "Kfin Technologies",
    "CDSL": "Central Depository Services",
    "BSE": "BSE",
    "MCX": "Multi Commodity Exchange",
    "IEX": "Indian Energy Exchange",
    "DELHIVERY": "Delhivery",
    "ZOMATO": "Eternal",
    "NYKAA": "FSN E-Commerce Ventures",
    "PAYTM": "One 97 Communications",
    "POLICYBZR": "PB Fintech",
    "ANGELONE": "Angel One",
    "IIFL": "IIFL Finance",
    "MANKIND": "Mankind Pharma",
    "LAURUSLABS": "Laurus Labs",
    "NATCOPHARM": "Natco Pharma",
    "AJANTPHARM": "Ajanta Pharma",
    "BIOCON": "Biocon",
    "TORNTPHARM": "Torrent Pharmaceuticals",
    "ABBOTINDIA": "Abbott India",
    "SANOFI": "Sanofi India",
    "GLENMARK": "Glenmark Pharmaceuticals",
    "IPCALAB": "Ipca Laboratories",
    "ABFRL": "Aditya Birla Fashion",
    "TRENT": "Trent",
    "SHOPERSTOP": "Shoppers Stop",
    "VBL": "Varun Beverages",
    "RADICO": "Radico Khaitan",
    "UNIONBANK": "Union Bank of India",
    "BANKBARODA": "Bank of Baroda",
    "CANBK": "Canara Bank",
    "PNB": "Punjab National Bank",
    "IOB": "Indian Overseas Bank",
    "CENTRALBK": "Central Bank of India",
}
MIN_PRICE = 20.0   # smallcaps can be cheaper; still reject sub-Rs20 junk


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
    print(f"=== smallcap universe pull {pd.Timestamp.now():%Y-%m-%d %H:%M} ===")
    found = resolve(c, list(set(SMALLCAPS.values())))
    keys = {t: found[nm.upper()] for t, nm in SMALLCAPS.items() if nm.upper() in found}
    print(f"resolved {len(keys)}/{len(SMALLCAPS)}")

    frames = []
    rejected = 0
    for t, k in keys.items():
        px = verify(c, k)
        if px < MIN_PRICE:
            print(f"  REJECT {t} (Rs{px})")
            rejected += 1
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
        print(f"  {t}: {len(df)} candles Rs{px:.0f}")

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
    print(f"\nDAILY STORE: {combined.label.nunique()} instruments, "
          f"{len(combined)} candles, {combined.datetime_ist.dt.date.nunique()} days")


if __name__ == "__main__":
    main()
