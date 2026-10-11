"""Reversal probe — HONESTY GATES. Is the illiquid-reversal signal real?

Probe 1 found: illiquid-bucket lag-1 autocorr +0.026 (t=+2.65, significant),
5-day reversal net +0.10%/trade (t=+2.28). Interesting BUT +0.1%/trade after
0.3% costs is razor-thin — classic fake-edge territory. Before building, run
the gates that kill fake edges:

GATE 1 — HOLDOUT: does it survive split by TIME (train first half, test second
         half)? A real edge holds in both. Overfit/regime edges don't.
GATE 2 — CAPITAL FIT: the only number that matters is FULL-CAPITAL compounding
         on Rs1.5L with 1 position at a time — not fixed Rs15K notional. A
         +0.1%/trade edge on Rs1.5L with -24% worst trade is likely a loser.
GATE 3 — YEAR-BY-YEAR: is it consistent or does it live/die by regime?
GATE 4 — ROBUSTNESS: does it hold across look/hold combos, or only 5/5?

Run: python reversal_gates.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HIST = "data_indmoney/daily_history.parquet"
CAPITAL = 150000.0
COST = 0.003
BROK = 20.0


def load():
    df = pd.read_parquet(HIST)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df[df.label != "NIFTY50"].copy()
    df["turnover"] = df.close * df.volume
    liq = df.groupby("label").turnover.median().sort_values(ascending=False)
    return df, liq


def reversal_trades(df, names, look, hold):
    close = df[df.label.isin(names)].pivot(
        index="datetime_ist", columns="label", values="close").sort_index()
    dates = list(close.index)
    trades = []
    i = look
    while i + hold < len(dates):
        past = close.iloc[i - look:i]
        perf = close.iloc[i - 1] / past.iloc[0] - 1
        losers = perf.nsmallest(max(1, len(perf) // 3)).dropna().index
        if len(losers) >= 2:
            fwd = (close.iloc[i + hold] / close.iloc[i] - 1).reindex(losers).dropna()
            port = fwd.mean() if not fwd.empty else np.nan
            if not np.isnan(port):
                trades.append({"date": dates[i], "ret": port})
        i += hold
    return pd.DataFrame(trades)


def full_capital_sim(trades, capital=CAPITAL):
    """1 position at a time, sequential non-overlapping, compound, net."""
    peak = capital
    maxdd = 0
    last = None
    curve = []
    for _, t in trades.iterrows():
        if last is not None and t["date"] <= last:
            continue
        capital *= (1 + t["ret"] - COST)
        capital -= BROK
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
        last = t["date"]
        curve.append(capital)
    return capital, maxdd, curve


def main():
    df, liq = load()
    # illiquid bucket = bottom half turnover
    ill = liq.tail(len(liq) // 2).index.tolist()

    print("=== REVERSAL HONESTY GATES (illiquid bucket) ===")
    tr = reversal_trades(df, ill, 5, 5)
    tr["net"] = tr.ret - COST
    print(f"  {len(tr)} non-overlapping trades, "
          f"{tr.date.min().date()} -> {tr.date.max().date()}\n")

    # GATE 1: time holdout
    print("--- GATE 1: TIME HOLDOUT ---")
    mid = tr.date.iloc[len(tr) // 2]
    for nm, g in [("first half", tr[tr.date < mid]), ("second half", tr[tr.date >= mid])]:
        n = g.net.mean() / (g.net.std() / np.sqrt(len(g))) if g.net.std() > 0 else 0
        print(f"  {nm:<12} n={len(g):<4} net avg {g.net.mean()*100:+.3f}%/trade  "
              f"t-stat {n:+.2f}")
    both = (tr[tr.date < mid].net.mean() > 0) and (tr[tr.date >= mid].net.mean() > 0)
    print(f"  both halves net-positive: {'YES' if both else 'NO'}")

    # GATE 2: full-capital compounding on Rs1.5L
    print(f"\n--- GATE 2: FULL-CAPITAL Rs1.5L (1 position at a time) ---")
    cap, maxdd, _ = full_capital_sim(tr)
    print(f"  Rs1.5L -> Rs{cap:,.0f}  ({(cap/CAPITAL-1)*100:+.1f}% over "
          f"{len(tr)} trades / {tr.date.max().year - tr.date.min().year}y)")
    print(f"  max drawdown: {maxdd*100:.1f}%")
    worst = tr.net.min()
    print(f"  worst trade: {worst*100:+.1f}% "
          f"(= Rs{worst*CAPITAL:,.0f} on full capital)")
    monthly = (cap / CAPITAL - 1) / ((tr.date.max() - tr.date.min()).days / 30)
    print(f"  ~Rs{(monthly)*CAPITAL:,.0f}/month average")

    # GATE 3: year-by-year
    print(f"\n--- GATE 3: YEAR-BY-YEAR consistency ---")
    tr["yr"] = tr.date.dt.year
    yr = tr.groupby("yr").net.agg(["mean", "count"])
    pos_years = (yr["mean"] > 0).sum()
    print(f"  profitable years: {pos_years}/{len(yr)}")
    for y, row in yr.iterrows():
        print(f"    {y}: {row['mean']*100:+.3f}%/trade (n={int(row['count'])})")

    # GATE 4: robustness across look/hold
    print(f"\n--- GATE 4: ROBUSTNESS (look x hold) ---")
    print(f"  {'look':>5}{'hold':>6}{'n':>6}{'net%/trade':>12}{'t-stat':>9}{'Rs1.5L end':>13}")
    for look in [3, 5, 10]:
        for hold in [3, 5, 10]:
            t2 = reversal_trades(df, ill, look, hold)
            if len(t2) < 30:
                continue
            n = t2.ret.mean() - COST
            ts = n / (t2.ret.std() / np.sqrt(len(t2))) if t2.ret.std() > 0 else 0
            c2, _, _ = full_capital_sim(t2)
            print(f"  {look:>4}d{hold:>5}d{len(t2):>6}{n*100:>11.3f}%{ts:>+9.2f}{c2:>13,.0f}")

    print(f"\n=== VERDICT LOGIC ===")
    print(f"  real edge needs: both-holdout YES + Rs1.5L end > Rs1.5L + "
          f"majority of years positive + consistent across look/hold.")
    print(f"  current: both_halves={'YES' if both else 'NO'}, "
          f"Rs1.5L_end=Rs{cap:,.0f}, pos_years={pos_years}/{len(yr)}")


if __name__ == "__main__":
    main()
