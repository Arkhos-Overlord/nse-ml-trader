"""Extract 2024 strangle bars — the second-regime test.

NIFTY_intraday_2024.parquet: 14.6M rows, Oct-Dec 2024, 19 expiries. This is a
DIFFERENT market regime from 2025 H1 (2024 Q4 was volatile, post-election,
range-bound with sharp swings). If the DTE<=1d 2-lot VRP strangle survives
HERE, it stops being a calm-market artifact.

Run: python extract_strangle_2024.py
"""
import warnings

import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

SRC = "NIFTY_intraday_2024.parquet"
OUT = "strangle_bars_2024.parquet"
COLS = ["date", "expiry", "strike", "option_type", "open", "close", "oi", "timestamp"]


def main():
    pf = pq.ParquetFile(SRC)
    print(f"reading {pf.metadata.num_rows:,} rows in {pf.num_row_groups} groups...")
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
            want = set(strikes[max(0, ai - 2): ai + 3])
            sub = day[day.strike.isin(want)].sort_values("timestamp")
            first_ts, last_ts = sub.timestamp.iloc[0], sub.timestamp.iloc[-1]
            fb = sub[sub.timestamp == first_ts]
            lb = sub[sub.timestamp == last_ts]
            for _, r in fb.iterrows():
                keep.append({"date": d, "expiry": r.expiry, "strike": r.strike,
                             "type": r.option_type, "atm": atm,
                             "first_close": r.close, "last_close": None})
            for _, r in lb.iterrows():
                for k in keep:
                    if (k["date"] == d and k["strike"] == r.strike
                            and k["type"] == r.option_type
                            and k["expiry"] == r.expiry and k["last_close"] is None):
                        k["last_close"] = r.close
                        break
        print(f"  group {i+1}/{pf.num_row_groups}, kept {len(keep)}")
    df = pd.DataFrame(keep).dropna(subset=["last_close"])
    df.to_parquet(OUT, index=False)
    print(f"\n  store: {OUT}")
    print(f"  {len(df)} rows, {df.date.nunique()} days, {df.expiry.nunique()} expiries")
    print(f"  span: {min(df.date)} -> {max(df.date)}")
    print(f"  ATM range: {df.atm.min():.0f} - {df.atm.max():.0f}")


if __name__ == "__main__":
    main()
