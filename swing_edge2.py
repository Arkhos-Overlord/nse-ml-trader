"""Expanded swing strategy battery — gap/overnight, relative strength, vol.

Large-cap daily swing = no edge. Test additional mechanisms on the daily data:
  1. GAP FADE: overnight gap down + buy at open (mean reversion of gaps)
  2. GAP CONTINUATION: gap up + buy (momentum continuation)
  3. RELATIVE STRENGTH: buy the top decile of 20-day returns vs universe
  4. VOL COMPRESSION: buy after N narrow-range days (coiled spring)
  5. OVERNIGHT HOLD: buy at close, sell at next open (capture overnight drift)
  6. INSIDE DAY: buy after an inside bar (low-vol breakout setup)

Honesty gates + data quality gate enforced. Honest costs (Rs144/round-trip).
Run: python swing_edge2.py
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


def load():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    ok = []
    for lb, g in df.groupby("label"):
        med = g.close.median()
        if med > 0 and g.close.max() / med < 10 and g.datetime_ist.duplicated().sum() == 0:
            ok.append(g)
    return pd.concat(ok)


def cost(notional):
    return 2 * BROKERAGE + notional * 2 * (STT / 2 + EXCH * (1 + GST) + SLIP)


def prep(g):
    """Add derived columns per stock."""
    g = g.sort_values("datetime_ist").reset_index(drop=True)
    g["prev_close"] = g.close.shift(1)
    g["gap_pct"] = (g.open - g.prev_close) / g.prev_close
    g["ret20"] = g.close.pct_change(20)
    g["range"] = (g.high - g.low) / g.close
    g["range_avg5"] = g.range.rolling(5).mean()
    g["inside"] = (g.high < g.high.shift(1)) & (g.low > g.low.shift(1))
    return g


def run(df, entry_rule, exit_rule, label):
    """Generic: entry_rule(g,i)->bool (enter at g.open[i+1]); exit_rule(g,i,entry_idx)->exit price."""
    trades = []
    for lb, g0 in df.groupby("label"):
        g = prep(g0)
        for i in range(25, len(g) - 1):
            if not entry_rule(g, i):
                continue
            entry = float(g.open.iloc[i + 1])
            ex = exit_rule(g, i)
            if ex is None or entry <= 0 or ex <= 0:
                continue
            qty = int(NOTIONAL / entry)
            net = (ex - entry) * qty - cost(entry * qty)
            trades.append({"stock": lb, "pnl": net})
    return trades


def main():
    df = load()
    print(f"=== EXPANDED SWING BATTERY ({df.label.nunique()} instruments, "
          f"{df.datetime_ist.dt.date.nunique()} days) ===\n")

    # define strategies: (name, entry, exit-after-N-days-or-overnight)
    def mk_hold(n):
        return lambda g, i: float(g.close.iloc[min(i + n, len(g) - 1)])

    def gap_fade_entry(g, i):
        return g.gap_pct.iloc[i] < -0.01  # gapped down 1%+

    def gap_cont_entry(g, i):
        return g.gap_pct.iloc[i] > 0.01   # gapped up 1%+

    def vol_compress_entry(g, i):
        return (g.range.iloc[i] < g.range_avg5.iloc[i] * 0.6
                and not pd.isna(g.range_avg5.iloc[i]))

    def inside_entry(g, i):
        return bool(g.inside.iloc[i])

    def rs_entry(g, i):
        return g.ret20.iloc[i] > 0.15  # top momentum

    STRATS = {
        "gap_fade_3d": (gap_fade_entry, mk_hold(3)),
        "gap_cont_3d": (gap_cont_entry, mk_hold(3)),
        "vol_compress_5d": (vol_compress_entry, mk_hold(5)),
        "inside_break_3d": (inside_entry, mk_hold(3)),
        "rel_strength_5d": (rs_entry, mk_hold(5)),
        "rel_strength_10d": (rs_entry, mk_hold(10)),
    }

    results = []
    for name, (entry, exitr) in STRATS.items():
        t = run(df, entry, exitr, name)
        if len(t) < 20:
            print(f"  {name}: only {len(t)} trades — skipped")
            continue
        arr = np.array([x["pnl"] for x in t])
        results.append({"strat": name, "n": len(t), "win": (arr > 0).mean(),
                        "avg": arr.mean(), "total": arr.sum(),
                        "worst": arr.min(), "best": arr.max()})

    if not results:
        print("  no strategy had enough trades")
        return
    r = pd.DataFrame(results).sort_values("avg", ascending=False)
    print(f"{'strategy':<20}{'n':>6}{'win%':>7}{'avgPnL':>11}{'total':>13}{'worst':>10}{'best':>10}")
    for _, x in r.iterrows():
        print(f"{x.strat:<20}{x.n:>6}{x.win*100:>6.0f}%{x.avg:>11,.0f}"
              f"{x.total:>13,.0f}{x.worst:>10,.0f}{x.best:>10,.0f}")

    best = r.iloc[0]
    print(f"\n=== BEST: {best.strat} ===")
    print(f"  {best.n} trades, {best.win*100:.0f}% win, avg Rs{best.avg:,.0f}/trade")
    print(f"  worst Rs{best.worst:,.0f} = {abs(best.worst)/NOTIONAL*100:.1f}% of Rs{NOTIONAL:,.0f}")
    n_days = df.datetime_ist.dt.date.nunique()
    trades_yr = best.n / n_days * 250
    print(f"  ~{trades_yr:.0f} trades/yr across universe")
    est = best.avg * trades_yr * 3  # 3 concurrent Rs50K slots
    print(f"  est annual on Rs1.5L: Rs{est:,.0f} (monthly ~Rs{est/12:,.0f})")
    dd_ok = abs(best.worst) < NOTIONAL * 0.15
    edge = best.avg > cost(NOTIONAL) and dd_ok
    print(f"\n  HONESTY: avg Rs{best.avg:,.0f} > cost Rs{cost(NOTIONAL):,.0f}? "
          f"{'YES' if best.avg > cost(NOTIONAL) else 'NO'}")
    print(f"  worst < 15% of notional? {'YES' if dd_ok else 'NO'}")
    print(f"  VERDICT: {'*** EDGE FOUND ***' if edge else 'no edge after costs'}")


if __name__ == "__main__":
    main()
