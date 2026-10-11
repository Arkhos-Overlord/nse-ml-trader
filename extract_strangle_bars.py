"""Extract ATM+wing option bars for VRP-conditional strangle testing.

Source: NIFTY_intraday_2025.parquet — 69.7M real 1-min Nifty option bars
(Jan-Apr 2025), all strikes, CE+PE, with OI. 674MB, so read it in row groups
and filter to what the strangle engine needs.

Extract, per (date, expiry): the ATM strike and +/-2 wing strikes, first and
last bar of the day for each, plus daily Nifty spot (reconstructed from ATM
parity-free: use the ATM CE+PE midpoint... no — use ATM strike itself as the
spot proxy is wrong. Instead store the ATM CE and PE closes; the engine
computes strangle P&L directly from option prices, no spot needed).

Output: strangle_bars.parquet with one row per (date, expiry, strike_offset,
option_type) holding first-bar and last-bar prices.

Run: python extract_strangle_bars.py
"""
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

SRC = "NIFTY_intraday_2025.parquet"
OUT = "strangle_bars.parquet"
COLS = ["date", "expiry", "strike", "option_type", "open", "high", "low",
        "close", "volume", "oi", "timestamp"]


def main():
    pf = pq.ParquetFile(SRC)
    print(f"reading {pf.metadata.num_rows:,} rows in {pf.num_row_groups} groups...")
    keep = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=COLS)
        df = t.to_pandas()
        # daily Nifty spot proxy: median ATM strike. Approximate first: the
        # strike with the highest total OI that day is ~ATM.
        g = df.groupby("date")
        for d, day in g:
            # ATM = strike with max (CE oi + PE oi) summed
            oi_by_strike = day.groupby("strike").oi.sum()
            atm = oi_by_strike.idxmax()
            strikes = sorted(day.strike.unique())
            if atm not in strikes:
                continue
            ai = strikes.index(atm)
            want = set(strikes[max(0, ai - 2): ai + 3])   # ATM +/- 2
            sub = day[day.strike.isin(want)]
            # first and last timestamp of the day
            sub = sub.sort_values("timestamp")
            first_ts = sub.timestamp.iloc[0]
            last_ts = sub.timestamp.iloc[-1]
            fb = sub[sub.timestamp == first_ts]
            lb = sub[sub.timestamp == last_ts]
            for _, r in fb.iterrows():
                keep.append({"date": d, "expiry": r.expiry, "strike": r.strike,
                             "type": r.option_type, "atm": atm,
                             "first_open": r.open, "first_close": r.close,
                             "first_oi": r.oi})
            for _, r in lb.iterrows():
                # match to the first-bar row
                for k in keep:
                    if (k["date"] == d and k["strike"] == r.strike
                            and k["type"] == r.option_type
                            and k["expiry"] == r.expiry):
                        k["last_close"] = r.close
                        k["last_oi"] = r.oi
                        break
        if (i + 1) % 10 == 0:
            print(f"  group {i+1}/{pf.num_row_groups}, rows kept {len(keep)}")
    df = pd.DataFrame(keep).dropna(subset=["last_close"])
    # option premium in rupees: Nifty options are quoted in points, lot 75 (2025)
    df.to_parquet(OUT, index=False)
    print(f"\n  store: {OUT}")
    print(f"  {len(df)} rows, {df.date.nunique()} days, "
          f"{df.expiry.nunique()} expiries")
    print(f"  ATM range: {df.atm.min():.0f} - {df.atm.max():.0f}")
    print(f"  sample:\n{df.head(4).to_string()}")


if __name__ == "__main__":
    main()
