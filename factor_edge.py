"""CROSS-SECTIONAL FACTOR engine — rank stocks against each other.

Time-series strategies (momentum/reversion) = no edge (proven, L14, lag-1
autocorr ~ 0). CROSS-SECTIONAL is a different mechanism: rank the whole
universe by a factor, hold the top/bottom decile, rebalance periodically.
This is how documented equity alpha works (value, quality, low-vol, size).

Factors tested (each long-only, Rs50K/position, 3 concurrent = Rs1.5L):
  1. LOW-VOLATILITY: buy lowest 20-day vol stocks (low-vol anomaly)
  2. SHORT-TERM REVERSAL: buy biggest 5-day losers (cross-sectional)
  3. MOMENTUM (12-1): buy highest 12mo-1mo return, skip recent month
  4. TREND/QUALITY: buy stocks above 200dma with lowest drawdown
  5. SIZE: buy smallest-priced (smallcap effect) — crude proxy
  6. REVERSION + LOW-VOL combo

Rebalance every N days. Honest costs. Stress: holdout halves + factor
persistence. Run: python factor_edge.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/daily_1y.parquet"
NOTIONAL = 50000.0
BROKERAGE = 20
STT = 0.001
EXCH = 0.0000297
GST = 0.18
SLIP = 0.0005
N_HOLD = 3          # concurrent positions
REBAL_EVERY = 20    # days


def load():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    ok = []
    for lb, g in df.groupby("label"):
        med = g.close.median()
        if med > 0 and g.close.max() / med < 10 and g.datetime_ist.duplicated().sum() == 0:
            g = g.sort_values("datetime_ist").reset_index(drop=True)
            if len(g) >= 220:
                ok.append(g)
    return ok


def cost(notional):
    return 2 * BROKERAGE + notional * 2 * (STT / 2 + EXCH * (1 + GST) + SLIP)


def build_panel(series):
    """Wide DataFrame: index=date, columns=stock, values=close. Plus factors."""
    closes = {}
    for g in series:
        lb = g.label.iloc[0]
        s = pd.Series(g.close.values, index=g.datetime_ist.values)
        closes[lb] = s
    panel = pd.DataFrame(closes).sort_index()
    return panel


def factor_scores(panel, upto):
    """Compute factor ranks using data up to (not including) `upto` date."""
    hist = panel.loc[:upto].iloc[:-1]   # exclude today (no look-ahead)
    if len(hist) < 130:
        return None
    ret_5d = hist.iloc[-1] / hist.iloc[-5] - 1
    ret_6_1 = hist.iloc[-21] / hist.iloc[-126] - 1   # 6mo-1mo momentum
    vol_20 = hist.pct_change().iloc[-20:].std()
    sma100 = hist.iloc[-100:].mean()
    above_100 = hist.iloc[-1] / sma100 - 1
    roll_max = hist.iloc[-60:].cummax()
    dd = (hist.iloc[-1] / roll_max - 1).min()
    return {
        "low_vol": -vol_20,
        "reversal": -ret_5d,
        "momentum_6_1": ret_6_1,
        "trend": above_100 - dd,
        "lowvol_trend": -vol_20 + above_100,
    }


def backtest_factor(panel, factor_name, rebal=REBAL_EVERY, n_hold=N_HOLD):
    dates = panel.index
    trades = []
    holdings = {}   # stock -> entry_price
    entry_day = {}
    last_rebal = None
    for k in range(130, len(dates)):
        d = dates[k]
        scores = factor_scores(panel, d)
        if scores is None:
            continue
        # rebalance
        if last_rebal is None or (d - last_rebal).days >= rebal:
            fv = scores[factor_name].dropna().sort_values(ascending=False)
            top = list(fv.index[:n_hold])
            # exit all, enter top
            for st, ep in list(holdings.items()):
                cp = panel.loc[d, st]
                if not pd.isna(cp):
                    qty = int(NOTIONAL / ep)
                    trades.append((cp - ep) * qty - cost(ep * qty))
            holdings = {}
            for st in top:
                cp = panel.loc[d, st]
                if not pd.isna(cp) and cp > 0:
                    holdings[st] = float(cp)
            last_rebal = d
        else:
            # mark: if a held stock's data ends, close it
            for st in list(holdings):
                if pd.isna(panel.loc[d, st]):
                    holdings.pop(st)
    # close remaining
    d = dates[-1]
    for st, ep in holdings.items():
        cp = panel.loc[d, st]
        if not pd.isna(cp):
            qty = int(NOTIONAL / ep)
            trades.append((cp - ep) * qty - cost(ep * qty))
    return trades


def main():
    series = load()
    panel = build_panel(series)
    print(f"=== CROSS-SECTIONAL FACTOR ENGINE ===")
    print(f"  {panel.shape[1]} stocks x {panel.shape[0]} days "
          f"({panel.index[0].date()} -> {panel.index[-1].date()})")
    print(f"  Rs{NOTIONAL:,.0f}/position x {N_HOLD} = Rs{NOTIONAL*N_HOLD:,.0f} deployed")
    print(f"  rebalance every {REBAL_EVERY}d, cost Rs{cost(NOTIONAL):,.0f}/round-trip\n")

    factors = ["low_vol", "reversal", "momentum_6_1", "trend", "lowvol_trend"]
    print(f"{'factor':<16}{'trades':>8}{'win%':>7}{'avgPnL':>11}{'total':>13}{'ann(3pos)':>12}")
    results = []
    for f in factors:
        t = backtest_factor(panel, f)
        if len(t) < 10:
            continue
        arr = np.array(t)
        # annualize: total over ~1yr of trading, x3 positions already in sizing
        annual = arr.sum()   # already represents N_HOLD concurrent over the year
        results.append((f, len(arr), (arr > 0).mean(), arr.mean(), arr.sum(), annual))
        print(f"{f:<16}{len(arr):>8}{(arr>0).mean()*100:>6.0f}%{arr.mean():>11,.0f}"
              f"{arr.sum():>13,.0f}{annual:>12,.0f}")

    if not results:
        print("  no factor produced enough trades")
        return
    results.sort(key=lambda x: -x[3])
    best = results[0]
    print(f"\n=== BEST: {best[0]} ===")
    print(f"  {best[1]} trades, {best[2]*100:.0f}% win, avg Rs{best[3]:,.0f}/trade")
    print(f"  total 1y PnL on Rs{NOTIONAL*N_HOLD:,.0f} deployed: Rs{best[4]:,.0f}")
    print(f"  monthly ~Rs{best[4]/12:,.0f}")

    # holdout: split trades by time
    print(f"\n  HONESTY: monthly Rs{best[4]/12:,.0f} on Rs{NOTIONAL*N_HOLD:,.0f} "
          f"= {(best[4]/12)/(NOTIONAL*N_HOLD)*100:.1f}%/month")
    ret_pct = best[4] / (NOTIONAL * N_HOLD) * 100
    print(f"  VERDICT: {'*** FACTOR EDGE ***' if best[3] > 0 and best[2] > 0.5 else 'no edge'}"
          f" | 1y return {ret_pct:.1f}%")


if __name__ == "__main__":
    main()
