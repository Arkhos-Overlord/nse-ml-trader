"""Reversal probe — is there structure in illiquid names before building?

Chuia et al 2023 (3956 stocks, 2000-2021): "short-term reversals are more
profound among the most illiquid stocks." My momentum test just proved I ran
the wrong signal on the wrong bucket. Before building a full engine, CHECK
whether reversal is even measurable here.

Momentum and reversal are OPPOSITES, so:
  - if reversal has structure, LOW-turnover stocks show POSITIVE lag-1
    autocorrelation (losers keep losing short-term = you fade the loser)
  - high-turnover stocks should show NEGATIVE autocorr (momentum)

If the whole panel is ~0, reversal is dead here too -> say so, don't build.

Run: python reversal_probe.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HIST = "data_indmoney/daily_history.parquet"


def load():
    df = pd.read_parquet(HIST)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df[df.label != "NIFTY50"].copy()
    # liquidity proxy: median daily rupee turnover over the WHOLE period
    df["turnover"] = df.close * df.volume
    liq = df.groupby("label").turnover.median().sort_values(ascending=False)
    return df, liq


def lag1_autocorr_by_bucket(df, liq):
    """Lag-1 daily return autocorrelation per stock, then grouped by liquidity.
    Positive = reversal structure (trend-continuation for losers); the sign
    tells us if a fade-the-loser rule could work."""
    print("=== PROBE 1: lag-1 return autocorrelation by liquidity bucket ===")
    rows = []
    for lb, g in df.groupby("label"):
        g = g.sort_values("datetime_ist")
        r = g.close.pct_change().dropna()
        if len(r) < 250:
            continue
        ac = r.autocorr(1)   # lag-1
        rows.append({"label": lb, "autocorr": ac, "turnover": liq[lb]})
    a = pd.DataFrame(rows).sort_values("turnover", ascending=False)
    half = len(a) // 2
    hi = a.head(half)        # most liquid
    lo = a.tail(half)        # most illiquid
    print(f"  most liquid ({len(hi)}): mean autocorr {hi.autocorr.mean():+.4f} "
          f"(median {hi.autocorr.median():+.4f})")
    print(f"  least liquid ({len(lo)}): mean autocorr {lo.autocorr.mean():+.4f} "
          f"(median {lo.autocorr.median():+.4f})")
    print(f"  full panel mean: {a.autocorr.mean():+.4f}")
    # per-stock spread
    print(f"  per-stock range: {a.autocorr.min():+.3f} .. {a.autocorr.max():+.3f}")
    # t-test: is the illiquid bucket's autocorr significantly != 0?
    t = lo.autocorr.mean() / (lo.autocorr.std() / np.sqrt(len(lo)))
    print(f"  illiquid-bucket t-stat vs 0: {t:+.2f} "
          f"({'significant' if abs(t)>2 else 'NOT significant'})")
    print(f"\n  READ: negative autocorr = momentum; positive = reversal structure.")
    print(f"  Chuia predicts ILLIQUID stocks show reversal (fade-the-loser works).")
    return a


def reversal_backtest(df, liq, look=5, hold=5, bucket="illiquid"):
    """Buy last-`look`-day losers, hold `hold` days, equal-weight, non-
    overlapping (step=hold), net of costs. Only the chosen liquidity bucket."""
    print(f"\n=== PROBE 2: {look}-day reversal, {bucket} bucket, {hold}d hold ===")
    a = pd.DataFrame([{"label": lb, "turnover": liq[lb]} for lb in df.label.unique()])
    a = a.sort_values("turnover", ascending=(bucket == "illiquid"))
    names = a.head(len(a) // 2).label.tolist()   # bottom half turnover if illiquid
    close = df[df.label.isin(names)].pivot(
        index="datetime_ist", columns="label", values="close").sort_index()
    rets = close.pct_change()
    COST = 0.003   # 0.3% round-trip delivery
    BROK = 20.0
    NOTIONAL = 15000.0
    trades = []
    dates = list(close.index)
    i = look
    while i + hold < len(dates):
        past = close.iloc[i - look:i]
        perf = close.iloc[i - 1] / past.iloc[0] - 1   # trailing `look`-day return
        losers = perf.nsmallest(max(1, len(perf) // 3)).dropna().index  # bottom third
        if len(losers) >= 2:
            fwd = (close.iloc[i + hold] / close.iloc[i] - 1).reindex(losers).dropna()
            port = fwd.mean() if not fwd.empty else np.nan
            if not np.isnan(port):
                trades.append({"date": dates[i], "ret": port})
        i += hold   # non-overlapping
    if not trades:
        print("  no trades")
        return
    r = np.array([t["ret"] for t in trades])
    net = r - COST
    per_trade_net = net * NOTIONAL - 2 * BROK
    tstat = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else 0
    print(f"  non-overlapping trades: {len(trades)}")
    print(f"  gross avg {r.mean()*100:+.2f}%/trade, win {100*(r>0).mean():.0f}%")
    print(f"  net (after 0.3% RT) avg {net.mean()*100:+.2f}%/trade, "
          f"win {100*(net>0).mean():.0f}%")
    print(f"  t-stat vs 0: {tstat:+.2f} "
          f"({'significant' if abs(tstat)>2 else 'NOT significant'})")
    print(f"  worst trade {net.min()*100:+.1f}%, best {net.max()*100:+.1f}%")


def main():
    df, liq = load()
    print(f"=== REVERSAL PROBE (deep history: {df.datetime_ist.dt.date.nunique()} days, "
          f"{df.label.nunique()} stocks) ===\n")
    a = lag1_autocorr_by_bucket(df, liq)
    reversal_backtest(df, liq, look=5, hold=5, bucket="illiquid")
    reversal_backtest(df, liq, look=5, hold=5, bucket="liquid")
    print(f"\n=== VERDICT INPUT ===")
    print(f"  If illiquid autocorr is significantly POSITIVE AND the illiquid")
    print(f"  reversal backtest is net-positive with |t|>2 -> build the engine.")
    print(f"  Otherwise: reversal is not measurable here; say so, don't build.")


if __name__ == "__main__":
    main()
