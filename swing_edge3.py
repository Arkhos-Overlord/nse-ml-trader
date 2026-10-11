"""Short-side + Nifty index + gross-vs-net diagnosis.

Every long-only stock strategy is negative AFTER costs. Key question: is there
gross edge being eaten by friction, or no edge at all? And does the SHORT side
or the INDEX behave differently?

  1. Gross vs net: run each strategy WITHOUT costs to see if raw edge exists
  2. Short side: mirror of gap/reversion (sell strength)
  3. Nifty index: index mean-reversion (different dynamics than stocks)
  4. Only best-signal decile: does selectivity help?

Honesty gates enforced. Run: python swing_edge3.py
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
    g = g.sort_values("datetime_ist").reset_index(drop=True)
    g["prev_close"] = g.close.shift(1)
    g["gap_pct"] = (g.open - g.prev_close) / g.prev_close
    g["ret20"] = g.close.pct_change(20)
    return g


def run(df, entry_rule, direction, hold, apply_cost=True):
    """direction: +1 long, -1 short."""
    trades = []
    for lb, g0 in df.groupby("label"):
        g = prep(g0)
        for i in range(25, len(g) - hold):
            if not entry_rule(g, i):
                continue
            entry = float(g.open.iloc[i + 1])
            ex = float(g.close.iloc[min(i + hold, len(g) - 1)])
            if entry <= 0 or ex <= 0:
                continue
            qty = int(NOTIONAL / entry)
            gross = (ex - entry) * qty * direction
            net = gross - (cost(entry * qty) if apply_cost else 0)
            trades.append(net)
    return np.array(trades) if trades else np.array([])


def main():
    df = load()
    nifty = df[df.label == "NIFTY50"]
    stocks = df[df.label != "NIFTY50"]
    print(f"=== GROSS vs NET + SHORT + INDEX ({df.label.nunique()} instruments) ===")
    print(f"  cost/round-trip: Rs{cost(NOTIONAL):,.0f}\n")

    def gap_down(g, i):
        return g.gap_pct.iloc[i] < -0.01

    def gap_up(g, i):
        return g.gap_pct.iloc[i] > 0.01

    def rev_low(g, i):
        return g.close.iloc[i] <= g.low.iloc[i - 20:i].min()

    def rev_high(g, i):
        return g.close.iloc[i] >= g.high.iloc[i - 20:i].max()

    tests = [
        ("gap_fade LONG", stocks, gap_down, +1, 3),
        ("gap_fade SHORT(mirror)", stocks, gap_up, -1, 3),
        ("20d_low LONG", stocks, rev_low, +1, 5),
        ("20d_high SHORT", stocks, rev_high, -1, 5),
        ("NIFTY gap_fade", nifty, gap_down, +1, 3),
        ("NIFTY 20d_low", nifty, rev_low, +1, 5),
    ]

    print(f"{'test':<26}{'n':>6}{'grossAvg':>11}{'netAvg':>11}{'win%':>7}{'verdict':>16}")
    for name, data, rule, d, hold in tests:
        if data.empty:
            continue
        gross = run(data, rule, d, hold, apply_cost=False)
        net = run(data, rule, d, hold, apply_cost=True)
        if len(net) < 15:
            print(f"{name:<26}{len(net):>6}  (too few trades)")
            continue
        win = (net > 0).mean()
        ga, na = gross.mean(), net.mean()
        v = "GROSS edge only" if ga > 0 and na <= 0 else ("EDGE" if na > 0 else "no edge")
        print(f"{name:<26}{len(net):>6}{ga:>11,.0f}{na:>11,.0f}{win*100:>6.0f}%{v:>16}")

    print("\n=== DIAGNOSIS ===")
    print("  If gross>0 but net<0: edge exists but friction eats it -> need")
    print("    lower costs (discount broker) or longer holds to amortize.")
    print("  If gross<=0: no signal edge at all -> strategy is wrong.")


if __name__ == "__main__":
    main()
