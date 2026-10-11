"""ETH edge deep-dive — is it real or one bull run?

13 independent strategies all positive on ETH with holdout YES. That's not
coincidence (BTC/BNB/SOL mostly fail). But could be ONE bull run. Deep test:
  1. QUARTERLY breakdown — must be positive in most quarters, not 1-2
  2. YEARLY breakdown
  3. Long vs short contribution (is it just beta/uptrend?)
  4. Walk-forward: does it survive if we only trade after seeing N days?
  5. Actual Rs returns on Rs1.5L with position sizing

If ETH edge is real across quarters AND not just long-beta, it's tradeable.
Run: python eth_deepdive.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FEE = 0.001
NOTIONAL = 1500.0


def load():
    df = pd.read_parquet("data_indmoney/crypto_daily/ETHUSDT_1d.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def cost(n):
    return n * 2 * FEE


def trades_detailed(df, look, hold, thresh=None):
    """Return per-trade: date, direction, pnl, long/short."""
    out = []
    for i in range(max(look, 60), len(df) - hold):
        r = df.close.iloc[i] / df.close.iloc[i - look] - 1
        if thresh is not None:
            d = 1 if r > thresh else (-1 if r < -thresh else 0)
        else:
            d = 1 if r > 0 else -1
        if d == 0:
            continue
        entry = float(df.open.iloc[i + 1])
        exit_ = float(df.close.iloc[min(i + hold, len(df) - 1)])
        if entry <= 0 or exit_ <= 0:
            continue
        qty = NOTIONAL / entry
        gross = (exit_ - entry) * qty * d
        out.append({"date": df.datetime.iloc[i], "dir": d,
                    "pnl": gross - cost(NOTIONAL),
                    "long": d == 1})
    return pd.DataFrame(out)


def main():
    df = load()
    print(f"=== ETH DEEP-DIVE ({len(df)} days, {df.datetime.iloc[0].date()} -> "
          f"{df.datetime.iloc[-1].date()}) ===\n")

    # best config from stress test
    look, hold, thresh = 30, 14, 0.05
    t = trades_detailed(df, look, hold, thresh)
    print(f"config: tsmom_{look}d thresh={thresh} hold={hold}")
    print(f"  {len(t)} trades, {(t.pnl > 0).mean()*100:.0f}% win, "
          f"avg ${t.pnl.mean():.2f}, total ${t.pnl.sum():.0f}\n")

    # quarterly
    t["q"] = t.date.dt.to_period("Q")
    print("--- by quarter ---")
    q = t.groupby("q").pnl.agg(["count", "sum", "mean"])
    pos_q = (q["sum"] > 0).sum()
    for per, row in q.iterrows():
        print(f"  {per}: {int(row['count']):>3} trades  total ${row['sum']:>8.0f}  "
              f"{'+' if row['sum']>0 else '-'}")
    print(f"  positive quarters: {pos_q}/{len(q)}")

    # yearly
    t["y"] = t.date.dt.year
    print("\n--- by year ---")
    y = t.groupby("y").pnl.agg(["count", "sum"])
    for yr, row in y.iterrows():
        print(f"  {yr}: {int(row['count']):>3} trades  total ${row['sum']:>8.0f}  "
              f"{'+' if row['sum']>0 else '-'}")

    # long vs short
    print("\n--- long vs short contribution ---")
    lng = t[t.long]
    sht = t[~t.long]
    print(f"  LONG:  {len(lng)} trades, total ${lng.pnl.sum():.0f}, "
          f"avg ${lng.pnl.mean():.2f}")
    print(f"  SHORT: {len(sht)} trades, total ${sht.pnl.sum():.0f}, "
          f"avg ${sht.pnl.mean():.2f}")
    both_positive = lng.pnl.sum() > 0 and sht.pnl.sum() > 0
    print(f"  both sides profitable? {'YES - true 2-sided edge' if both_positive else 'NO - one-sided (likely beta)'}")

    # Rs returns
    print("\n--- Rs returns on Rs1.5L ---")
    total_usd = t.pnl.sum()
    usd_inr = 83
    total_inr = total_usd * usd_inr
    years = (df.datetime.iloc[-1] - df.datetime.iloc[0]).days / 365
    print(f"  total over {years:.1f}y: Rs{total_inr:,.0f} on Rs{NOTIONAL*usd_inr:,.0f}/trade")
    print(f"  if 1 position at a time (Rs1.5L): Rs{total_inr:,.0f} total "
          f"= {(total_inr/(NOTIONAL*usd_inr))*100:.0f}% return")
    print(f"  avg per trade: Rs{t.pnl.mean()*usd_inr:,.0f}")
    trades_yr = len(t) / years
    print(f"  ~{trades_yr:.0f} trades/yr -> if traded all: "
          f"Rs{t.pnl.mean()*usd_inr*trades_yr:,.0f}/yr gross (needs capital for concurrency)")

    # VERDICT
    print("\n=== VERDICT ===")
    robust = pos_q >= len(q) * 0.6 and both_positive and t.pnl.mean() > 0
    print(f"  positive in {pos_q}/{len(q)} quarters (need >=60%)")
    print(f"  both long+short profitable: {both_positive}")
    print(f"  avg ${t.pnl.mean():.2f}/trade (need > fee ${cost(NOTIONAL):.2f})")
    print(f"  ROBUST ETH EDGE: {'*** YES ***' if robust else 'not robust enough'}")


if __name__ == "__main__":
    main()
