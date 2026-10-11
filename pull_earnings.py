"""PEAD feed + test — the anomaly confirmed twice in India, now testable.

Breakthrough: yfinance.Ticker.get_earnings_dates() gives earnings DATES + EPS
estimate + reported EPS + surprise % — the SUE (standardized unexpected
earnings) signal the Indian PEAD papers (Harshita 2018; Das 2026) used. Keyless.

PEAD: stocks with POSITIVE earnings surprise drift UP for weeks after the
announcement; negative surprise drift DOWN. Confirmed significant in India
(2002-2017 robust; 2015-2024 BSE500). Daily holding (not intraday) — fits small
capital, cheap.

Strategy: on each earnings date, compute surprise%. Long positive-surprise
stocks, short negative, hold N days. Test net of costs, holdout, breadth.

Run: python pull_earnings.py  (feed) then pead_test.py
"""
import time
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OUT = "data_indmoney/earnings.parquet"
TICKERS = None  # filled from wide store if present


def main():
    import os
    tk = []
    wide = "data_indmoney/intraday_stocks_wide.parquet"
    if os.path.exists(wide):
        tk = sorted(pd.read_parquet(wide).label.unique())
    if not tk:
        tk = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN",
              "ITC", "LT", "AXISBANK", "KOTAKBANK", "MARUTI"]
    print(f"=== earnings feed ({len(tk)} stocks, yfinance keyless) ===")
    rows = []
    ok = 0
    for t in tk:
        try:
            ed = yf.Ticker(f"{t}.NS").get_earnings_dates(limit=24)
        except Exception as e:  # noqa: BLE001
            continue
        if ed is None or ed.empty:
            continue
        ed = ed.reset_index()
        for _, r in ed.iterrows():
            rows.append({"label": t,
                         "earn_date": pd.to_datetime(r["Earnings Date"]).tz_localize(None)
                         if pd.to_datetime(r["Earnings Date"]).tzinfo else pd.to_datetime(r["Earnings Date"]),
                         "eps_est": r.get("EPS Estimate"),
                         "eps_rep": r.get("Reported EPS"),
                         "surprise": r.get("Surprise(%)")})
        ok += 1
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    # compute surprise% ourselves if missing (eps_rep vs eps_est)
    df["surprise_pct"] = df.apply(
        lambda r: (r["eps_rep"] / r["eps_est"] - 1) * 100
        if pd.notna(r.get("eps_est")) and pd.notna(r.get("eps_rep")) and r.get("eps_est") not in (0, None)
        and abs(r["eps_est"]) > 0 else r.get("surprise"), axis=1)
    df["earn_date"] = pd.to_datetime(df["earn_date"]).dt.tz_localize(None)
    df = df.dropna(subset=["surprise_pct"])
    df = df.drop_duplicates(subset=["label", "earn_date"]).sort_values(
        ["label", "earn_date"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"  OK {ok} stocks, {len(df)} earnings events")
    print(f"  surprise range: {df.surprise_pct.min():.0f}% to {df.surprise_pct.max():.0f}%")
    print(f"  date span: {df.earn_date.min().date()} -> {df.earn_date.max().date()}")
    print(f"  store: {OUT}")
    print(f"  -> PEAD testable: positive surprise should drift up post-announcement")


if __name__ == "__main__":
    main()
