"""SWING/MOMENTUM edge on 1 YEAR of daily candles — real sample, no waiting.

Intraday data needs 30 days to accumulate. Daily data has 246 days RIGHT NOW.
So test daily-swing edges that fit small capital:

  1. MOMENTUM: buy stocks above their 20-day high, hold N days (trend following)
  2. MEAN-REVERSION: buy stocks below their 20-day low, hold N days
  3. GAP: overnight gap continuation/reversion
  4. BREAKOUT: volume-confirmed breakouts

Each long-only, sized to Rs50K notional, equity delivery (no margin/futures).
Deliverable: which edge, if any, survives on 246 real days with honest costs.

Data-quality gate + honesty gate enforced (L13, L7).
Run: python swing_edge.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/daily_1y.parquet"
CAPITAL = 150000.0
NOTIONAL = 50000.0
BROKERAGE = 20
STT = 0.001           # 0.1% delivery STT (both sides combined)
EXCH = 0.0000297
GST = 0.18
SLIP = 0.0005         # 0.05% slippage per side
HOLD_DAYS = [1, 3, 5, 10, 20]


def load():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    # quality gate: reject any instrument with >10x price range or dupes
    ok = []
    for lb, g in df.groupby("label"):
        med = g.close.median()
        if med > 0 and g.close.max() / med < 10 and g.datetime_ist.duplicated().sum() == 0:
            ok.append(g)
    return pd.concat(ok)


def costs(notional, sides=2):
    return sides * BROKERAGE + notional * sides * (STT / 2 + EXCH * (1 + GST) + SLIP)


def backtest(df, signal_fn, hold):
    """Long-only: enter next open after signal, exit after `hold` days."""
    trades = []
    for lb, g in df.groupby("label"):
        g = g.sort_values("datetime_ist").reset_index(drop=True)
        if len(g) < hold + 25:
            continue
        for i in range(20, len(g) - hold):
            if not signal_fn(g, i):
                continue
            entry = float(g.open.iloc[i + 1])
            exit_ = float(g.close.iloc[i + hold])
            if entry <= 0 or exit_ <= 0:
                continue
            qty = int(NOTIONAL / entry)
            gross = (exit_ - entry) * qty
            net = gross - costs(entry * qty)
            trades.append({"stock": lb, "hold": hold, "pnl": net,
                           "date": g.datetime_ist.iloc[i]})
    return trades


def sig_momentum(g, i):
    """Close > 20-day high (breakout)."""
    return g.close.iloc[i] >= g.high.iloc[i - 20:i].max()


def sig_reversion(g, i):
    """Close < 20-day low (oversold)."""
    return g.close.iloc[i] <= g.low.iloc[i - 20:i].min()


def sig_uptrend(g, i):
    """Close > 50-day SMA and rising (trend filter)."""
    sma = g.close.iloc[i - 50:i].mean()
    return g.close.iloc[i] > sma and g.close.iloc[i] > g.close.iloc[i - 5]


def sig_downtrend_rev(g, i):
    """In downtrend but 3 down days (pullback buy)."""
    sma = g.close.iloc[i - 50:i].mean()
    three_down = all(g.close.iloc[i - k] < g.close.iloc[i - k - 1] for k in range(3))
    return g.close.iloc[i] < sma and three_down


STRATS = {
    "momentum_20d_high": sig_momentum,
    "reversion_20d_low": sig_reversion,
    "uptrend_50sma": sig_uptrend,
    "pullback_3down": sig_downtrend_rev,
}


def main():
    df = load()
    print("=== SWING EDGE on 1 YEAR daily (246 real days) ===")
    print(f"  {df.label.nunique()} instruments, {len(df)} bars, Rs{NOTIONAL:,.0f}/trade")
    print(f"  costs: Rs{ costs(NOTIONAL):,.0f}/round-trip (brokerage+STT+exch+slippage)\n")

    results = []
    for name, fn in STRATS.items():
        for hold in HOLD_DAYS:
            t = backtest(df, fn, hold)
            if len(t) < 20:
                continue
            arr = np.array([x["pnl"] for x in t])
            win = (arr > 0).mean()
            avg = arr.mean()
            total = arr.sum()
            # expectancy per Rs of capital risked
            results.append({"strat": name, "hold": hold, "n": len(t),
                            "win": win, "avg": avg, "total": total,
                            "worst": arr.min(), "best": arr.max()})
    r = pd.DataFrame(results)
    if r.empty:
        print("  no strategy produced enough trades")
        return
    r = r.sort_values("avg", ascending=False)
    print(f"{'strategy':<22}{'hold':>5}{'n':>6}{'win%':>7}{'avgPnL':>11}{'total':>13}{'worst':>10}")
    for _, x in r.iterrows():
        print(f"{x.strat:<22}{x.hold:>5}{x.n:>6}{x.win*100:>6.0f}%"
              f"{x.avg:>11,.0f}{x.total:>13,.0f}{x.worst:>10,.0f}")

    best = r.iloc[0]
    print(f"\n=== BEST: {best.strat} hold={best.hold}d ===")
    print(f"  {best.n} trades, {best.win*100:.0f}% win, avg Rs{best.avg:,.0f}/trade")
    # annualized estimate on Rs1.5L running 3 concurrent Rs50K positions
    trades_per_year = best.n
    est_annual = best.avg * trades_per_year * (CAPITAL / NOTIONAL)
    print(f"  est annual on Rs1.5L (3 positions, all signals taken): Rs{est_annual:,.0f}")
    print(f"  monthly ~Rs{est_annual/12:,.0f}")

    # honesty checks
    print(f"\n=== HONESTY CHECKS ===")
    print(f"  worst trade Rs{best.worst:,.0f} on Rs{NOTIONAL:,.0f} notional "
          f"= {abs(best.worst)/NOTIONAL*100:.1f}% (plausible: <15%)")
    print(f"  avg trade Rs{best.avg:,.0f} vs cost Rs{costs(NOTIONAL):,.0f} "
          f"(gross edge must exceed costs)")
    dd_ok = abs(best.worst) < NOTIONAL * 0.15
    print(f"  VERDICT: {'EDGE EXISTS' if best.avg > 0 and dd_ok else 'no edge after costs'}"
          f" — {best.win*100:.0f}% win rate")


if __name__ == "__main__":
    main()
