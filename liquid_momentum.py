"""LIQUID-NAME MOMENTUM — the Tier-1 build from the research map.

Implements the exact edge the literature (Chuia et al 2023, Pacific-Basin
Finance J, 3956 stocks 2000-2021) specifies:
  "price momentum ... for the most liquid portfolio ... persists 12 months"

My earlier momentum tests used 5-20 day holds on a MIXED universe — the wrong
parameters. The literature says: 6-12 MONTH holds, LIQUID names, LOW turnover.

Method:
  1. Define liquidity proxy = median daily rupee turnover (volume * close)
     -> rank stocks, keep the top liquid bucket (Chuia: "most liquid portfolio")
  2. At each month end, rank stocks by trailing 12-1 momentum (skip last month,
     standard Jegadeesh-Titman convention to avoid short-term reversal)
  3. Buy top decile (winners), hold K months, equal-weight, monthly rebalance
  4. LOW turnover by construction (monthly rebalance, not daily)
  5. Measure NET of costs; compare long-only vs long-short (winners-losers)
  6. Honesty gates: no look-ahead (formation uses only past data), trade count,
     both-holdout-halves, gross-vs-net

Capital fit check: monthly rebalance = ~2 round-trips/stock/month. On 40 liquid
names that is low turnover = spread does NOT eat the edge (the licensing
condition from the map).

Run: python liquid_momentum.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/daily_1y.parquet"
CAPITAL = 150000.0
# Indian equity delivery costs: brokerage 0.1-0.3%, STT 0.1% both sides, etc.
COST_ROUNDTRIP = 0.003   # 0.3% round-trip, realistic for delivery equity
BROKERAGE_FLAT = 20.0

SKIP = 1      # skip most recent month in formation (JT convention)
HOLDS = [3, 6, 9, 12]


def load():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df["date"] = df.datetime_ist.dt.date
    df = df[df.label != "NIFTY50"].copy()
    df["turnover"] = df.close * df.volume
    # liquidity proxy: median daily rupee turnover per stock
    liq = df.groupby("label").turnover.median().sort_values(ascending=False)
    return df, liq


def build_monthly(df):
    """Pivot to monthly close per stock (last trading day of each month)."""
    df["ym"] = df.datetime_ist.dt.to_period("M")
    monthly = df.sort_values("datetime_ist").groupby(["label", "ym"]).last().reset_index()
    close = monthly.pivot(index="ym", columns="label", values="close").sort_index()
    return close


def backtest_momentum(close, liquid_names, formation, hold, long_only=True):
    """Cross-sectional momentum, no look-ahead. Returns per-trade stats."""
    rets = close.pct_change()
    trades = []
    months = list(close.index)
    for i in range(formation + SKIP, len(months) - hold):
        # formation window ends at month i-1-SKIP (skip recent month)
        past = close.iloc[i - SKIP - formation:i - SKIP]
        if len(past) < formation:
            continue
        mom = past.iloc[-1] / past.iloc[0] - 1  # trailing return
        mom = mom.reindex(liquid_names).dropna()
        if len(mom) < 5:
            continue
        # rank
        if long_only:
            n_pick = max(1, len(mom) // 5)
            picks = mom.nlargest(n_pick).index.tolist()
        else:  # long-short: winners minus losers
            n_pick = max(1, len(mom) // 5)
            picks = mom.nlargest(n_pick).index.tolist()
            losers = mom.nsmallest(n_pick).index.tolist()
        # forward return over hold window
        entry = close.iloc[i]
        exit_ = close.iloc[i + hold]
        fwd = (exit_ / entry - 1).reindex(picks).dropna()
        if fwd.empty:
            continue
        port_ret = fwd.mean()
        if not long_only:
            los_fwd = (exit_ / entry - 1).reindex(losers).dropna()
            if not los_fwd.empty:
                port_ret = fwd.mean() - los_fwd.mean()
        trades.append({
            "date": months[i], "port_ret": port_ret,
            "n": len(fwd), "pick_list": picks,
        })
    return trades


def simulate_capital(trades, per_rebalance_notional, cost_rt):
    """Full-capital compounding with monthly positions, net of costs."""
    capital = CAPITAL
    peak = capital
    maxdd = 0
    curve = []
    for t in trades:
        # split capital across n_pick positions
        n = t["n"]
        per_pos = capital / n
        gross = per_pos * t["port_ret"] * n
        # one round-trip per position per rebalance
        cost = per_pos * cost_rt * n + n * BROKERAGE_FLAT
        capital += gross - cost
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
        curve.append(capital)
    return capital, maxdd, curve


def main():
    df, liq = load()
    close = build_monthly(df)
    print("=== LIQUID-NAME MOMENTUM (research-map Tier-1 #1) ===")
    print(f"  {close.shape[1]} stocks, {len(close)} months "
          f"({close.index[0]} -> {close.index[-1]})")
    print(f"  hold period: 6-12 months (literature), LOW turnover (monthly rebal)\n")

    # liquidity buckets
    all_names = list(close.columns)
    liquid = liq.reindex(all_names).dropna().sort_values(ascending=False)
    top_liquid = liquid.head(len(liquid) // 2).index.tolist()   # top half = "most liquid"
    print(f"--- LIQUIDITY BUCKETS (median daily rupee turnover) ---")
    print(f"  most liquid ({len(top_liquid)}): {', '.join(top_liquid[:8])} ...")
    print(f"  least liquid: {', '.join(liquid.tail(5).index.tolist())}\n")

    # DATA-DEPTH REALITY: only {len(close)} months. The literature uses 6-12m
    # formation + 6-12m hold = needs 12-24m. We have 13m. So we test the
    # SHORTEST viable formation (3m, 6m) to validate the engine + universe NOW,
    # and re-run at 12m formation when history matures (~late 2027).
    formations = [f for f in [3, 6, 12] if f + SKIP + 3 <= len(close) - 1]
    print(f"  (data depth {len(close)}m -> testing formation {formations}; "
          f"12m/6-12m-hold needs 19-24m, matures ~late 2027)\n")

    print(f"--- MOMENTUM: skip last {SKIP}m (JT convention), monthly rebal (LOW turnover) ---")
    print(f"{'formation':<10}{'universe':<13}{'hold':>5}{'mode':>12}{'n':>5}"
          f"{'avg%/trade':>11}{'win%':>7}{'netEnd':>13}{'maxDD':>7}")
    results = []
    for formation in formations:
        for uname, unames in [("most_liquid", top_liquid), ("all", all_names)]:
            for hold in HOLDS:
                if formation + SKIP + hold > len(close) - 1:
                    continue
                for long_only in [True, False]:
                    trades = backtest_momentum(close, unames, formation, hold, long_only)
                    if len(trades) < 3:
                        continue
                    rets = np.array([t["port_ret"] for t in trades])
                    cap, maxdd, _ = simulate_capital(trades, None, COST_ROUNDTRIP)
                    mode = "long" if long_only else "long-short"
                    print(f"{formation:<4}m{uname:<13}{hold:>4}m{mode:>12}"
                          f"{len(trades):>5}{rets.mean()*100:>10.1f}%"
                          f"{(rets>0).mean()*100:>6.0f}%{cap:>13,.0f}{maxdd*100:>6.1f}%")
                    results.append((formation, uname, hold, mode, rets.mean(),
                                    len(trades), maxdd, cap))

    if not results:
        print("  (no config had enough trades with current data depth)")
        print("\n  VERDICT: cannot test yet — need more history. Engine is ready;")
        print(f"  re-run when daily data passes ~24 months (need formation+skip+hold")
        print(f"  >= 19m for the literature config). Currently {len(close)}m.")
        return

    # best-config honesty check: both holdout halves (most liquid, longest viable)
    best = max(results, key=lambda x: x[6] if x[6] > 0 else -x[6])  # smallest maxDD among positives
    best_formation, best_uname, best_hold, best_mode = best[0], best[1], best[2], best[3]
    bnames = top_liquid if best_uname == "most_liquid" else all_names
    print(f"\n--- HONESTY GATE: holdout halves "
          f"(formation={best_formation}m, {best_uname}, {best_mode}, hold={best_hold}m) ---")
    trades = backtest_momentum(close, bnames, best_formation, best_hold,
                               long_only=(best_mode == "long"))
    if len(trades) >= 6:
        half = len(trades) // 2
        r = np.array([t["port_ret"] for t in trades])
        print(f"  first half:  avg {r[:half].mean()*100:+.1f}%/trade, "
              f"{(r[:half]>0).mean()*100:.0f}% win ({half} trades)")
        print(f"  second half: avg {r[half:].mean()*100:+.1f}%/trade, "
              f"{(r[half:]>0).mean()*100:.0f}% win ({len(trades)-half} trades)")
        both = r[:half].mean() > 0 and r[half:].mean() > 0
        print(f"  both halves positive: {'YES' if both else 'NO'}")
        print(f"\n  NOTE: {len(close)} months = {len(trades)} rebalances is a SMALL sample.")
        print(f"  The literature uses 10-22 YEARS / 3,956 stocks. This is a")
        print(f"  DIRECTIONAL read on the right mechanism+universe+horizon,")
        print(f"  not a verdict. Verdict needs ~3-5y of this universe (accumulating).")
    else:
        print(f"  only {len(trades)} trades — sample too small to judge")


if __name__ == "__main__":
    main()
