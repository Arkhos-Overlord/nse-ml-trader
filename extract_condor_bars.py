"""Wide-strike extraction for DEFINED-RISK structures (iron condors).

An iron condor needs wings BEYOND the short strikes to cap the loss. The prior
extract kept only ATM+/-2 (5 strikes) — enough for a naked strangle, not a
condor. Re-extract with ATM+/-6 so we can build:

  SELL CE at ATM+2  /  BUY CE at ATM+5   (call spread, capped loss)
  SELL PE at ATM-2  /  BUY PE at ATM-5   (put spread, capped loss)

Max loss per lot = (wing_width - net_premium) x LOT  -- BOUNDED BY CONSTRUCTION.
That is the fix for the unbounded tail that killed the naked strangle at Rs1.5L.

Run: python extract_condor_bars.py
"""
import sys
import warnings

import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

COLS = ["date", "expiry", "strike", "option_type", "open", "close", "oi", "timestamp"]


def extract(src, out, wing=6):
    pf = pq.ParquetFile(src)
    print(f"{src}: {pf.metadata.num_rows:,} rows, {pf.num_row_groups} groups", flush=True)
    keep = []
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=COLS).to_pandas()
        for d, day in df.groupby("date"):
            oi_by_strike = day.groupby("strike").oi.sum()
            atm = oi_by_strike.idxmax()
            strikes = sorted(day.strike.unique())
            if atm not in strikes:
                continue
            ai = strikes.index(atm)
            want = set(strikes[max(0, ai - wing): ai + wing + 1])
            sub = day[day.strike.isin(want)].sort_values("timestamp")
            first_ts, last_ts = sub.timestamp.iloc[0], sub.timestamp.iloc[-1]
            fb = sub[sub.timestamp == first_ts]
            lb = sub[sub.timestamp == last_ts]
            seen = {}
            for _, r in fb.iterrows():
                k = (d, r.expiry, r.strike, r.option_type)
                seen[k] = {"date": d, "expiry": r.expiry, "strike": r.strike,
                           "type": r.option_type, "atm": atm,
                           "first_close": r.close, "last_close": None}
            for _, r in lb.iterrows():
                k = (d, r.expiry, r.strike, r.option_type)
                if k in seen and seen[k]["last_close"] is None:
                    seen[k]["last_close"] = r.close
            keep.extend(seen.values())
        print(f"  group {i+1}/{pf.num_row_groups}, kept {len(keep)}", flush=True)
    df = pd.DataFrame(keep).dropna(subset=["last_close"])
    df = df.drop_duplicates(subset=["date", "expiry", "strike", "type"])
    df.to_parquet(out, index=False)
    print(f"  -> {out}: {len(df)} rows, {df.date.nunique()} days, "
          f"{df.expiry.nunique()} expiries", flush=True)


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
