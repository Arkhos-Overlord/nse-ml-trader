"""Microstructure engine v2 — longer holds + portfolio stack.

v1 (5-min, EOD exit) was mostly negative. Two untested angles that could
rescue microstructure:

1. LONGER HOLDS: microstructure effects may need 1-3 days to pay off, not
   same-day EOD. ORB break held overnight, VWAP-stretch held to next-day mean.
   (Structural effects persist; my EOD-exit may have cut them too early.)

2. PORTFOLIO STACK: practitioners (u/wizzer-algotrader, r/NSEAlgoTrading) run
   'a stack of systematic strategies on just Rs25k'. Individually weak/negative
   signals can combine into a positive portfolio if their errors DECORRELATE.
   Test: ORB + close-drift + fade, equal-weight, combined PnL/drawdown.

Futures economics throughout. Honesty gates: holdout, full-capital, net.

Run: python intraday_futures_edge2.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/intraday_deep.parquet"
LOT, BROKERAGE = 65, 20.0
STT_SELL, EXCH_GST, SLIP = 0.0001, 0.00004, 0.0002
CAPITAL = 150000.0


def load(label="NIFTY50"):
    df = pd.read_parquet(STORE)
    df = df[(df.interval == "5m") & (df.label == label)].copy()
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df.sort_values("datetime_ist").reset_index(drop=True)
    df["date"] = df.datetime_ist.dt.date
    df["hm"] = df.datetime_ist.dt.strftime("%H:%M")
    return df


def fut_cost_frac(notional):
    return (2 * BROKERAGE + notional * STT_SELL + notional * EXCH_GST
            + notional * SLIP) / notional


def simulate(trades, capital=CAPITAL):
    peak, maxdd, last = capital, 0, None
    for t in trades:
        if last is not None and t["date"] <= last:
            continue
        lots = max(1, int(capital / (LOT * t["entry"])))
        notional = lots * LOT * t["entry"]
        gross = (t["exit"] - t["entry"]) * lots * LOT * t["dir"]
        capital += gross - fut_cost_frac(notional) * notional
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
        last = t["date"]
    return capital, maxdd


def orb_multiday(df, orb_min=15, hold_days=1):
    """ORB breakout held `hold_days` (overnight) instead of EOD."""
    daily_close = df.groupby("date").close.last()
    daily_idx = {d: i for i, d in enumerate(daily_close.index)}
    trades = []
    for d, g in df.groupby("date"):
        g = g.sort_values("datetime_ist")
        cutoff = (pd.Timestamp(f"2000-01-01 {g.hm.iloc[0]}")
                  + pd.Timedelta(minutes=orb_min)).strftime("%H:%M")
        orb = g[g.hm <= cutoff]
        after = g[g.hm > cutoff]
        if len(orb) < 3 or after.empty:
            continue
        hi, lo = orb.high.max(), orb.low.min()
        i = daily_idx[d]
        if i + hold_days >= len(daily_close):
            continue
        exit_px = float(daily_close.iloc[i + hold_days])
        for _, r in after.iterrows():
            if r.high >= hi:
                trades.append({"date": d, "dir": 1, "entry": hi, "exit": exit_px})
                break
            if r.low <= lo:
                trades.append({"date": d, "dir": -1, "entry": lo, "exit": exit_px})
                break
    return trades


def close_drift_multiday(df, hold_days=1):
    """Close-drift signal, held overnight."""
    daily = df.groupby("date").agg(o=("open", "first"), c=("close", "last"))
    daily_idx = {d: i for i, d in enumerate(daily.index)}
    trades = []
    for d, row in daily.iterrows():
        i = daily_idx[d]
        if i + hold_days >= len(daily):
            continue
        # 15:00 proxy: use daily close as the 'into-close' level
        p1500 = row["c"]
        o = row["o"]
        if abs(p1500 / o - 1) < 0.001:
            continue
        direction = 1 if p1500 > o else -1
        exit_px = float(daily.c.iloc[i + hold_days])
        trades.append({"date": d, "dir": direction, "entry": p1500, "exit": exit_px})
    return trades


def report(name, trades):
    if not trades:
        print(f"  {name:<32} 0 trades")
        return None
    notional = LOT * np.mean([t["entry"] for t in trades])
    cf = fut_cost_frac(notional)
    r = np.array([(t["exit"] / t["entry"] - 1) * t["dir"] for t in trades])
    net = r - cf
    cap, maxdd = simulate(trades)
    n = len(trades)
    ts = net.mean() / (net.std() / np.sqrt(n)) if net.std() > 0 else 0
    print(f"  {name:<32} n={n:<4} win={100*(net>0).mean():3.0f}% "
          f"net={net.mean()*100:+.3f}% t={ts:+.2f} "
          f"Rs{cap:,.0f} DD={maxdd*100:.0f}%")
    return net


def portfolio_stack(df, strategies):
    """Equal-weight combine signals per day; errors decorrelate -> smoother."""
    # build per-day per-strategy return, then average across strategies per day
    by_day = {}
    for sname, trades in strategies.items():
        notional = LOT * np.mean([t["entry"] for t in trades]) if trades else 1
        cf = fut_cost_frac(notional)
        for t in trades:
            r = (t["exit"] / t["entry"] - 1) * t["dir"] - cf
            by_day.setdefault(t["date"], {}).setdefault(sname, []).append(r)
    # per day, average each strategy's trades, then average across strategies
    port = []
    for d, sd in sorted(by_day.items()):
        strat_rets = [np.mean(v) for v in sd.values()]
        port.append({"date": d, "ret": np.mean(strat_rets), "n_strat": len(strat_rets)})
    return pd.DataFrame(port)


def main():
    print("=== MICROSTRUCTURE v2 — longer holds + portfolio stack ===")
    for label in ["NIFTY50", "BANKNIFTY"]:
        df = load(label)
        print(f"\n--- {label}: {df.date.nunique()} days ---")
        for hd in [1, 2, 3]:
            report(f"ORB breakout hold {hd}d", orb_multiday(df, 15, hd))
        for hd in [1, 2]:
            report(f"close-drift hold {hd}d", close_drift_multiday(df, hd))

    # portfolio stack on Nifty
    print(f"\n--- PORTFOLIO STACK (Nifty, equal-weight, errors decorrelate) ---")
    df = load("NIFTY50")
    strats = {
        "ORB_1d": orb_multiday(df, 15, 1),
        "ORB_2d": orb_multiday(df, 15, 2),
        "close_1d": close_drift_multiday(df, 1),
    }
    port = portfolio_stack(df, strats)
    if not port.empty:
        pr = port.ret.values
        ts = pr.mean() / (pr.std() / np.sqrt(len(pr))) if pr.std() > 0 else 0
        print(f"  stack: n={len(pr)} days, avg {pr.mean()*100:+.3f}%/day, "
              f"win {100*(pr>0).mean():.0f}%, t={ts:+.2f}")
        print(f"  vs individual (above): does combining smooth the drawdown?")
        # holdout
        mid = len(pr) // 2
        print(f"  first half {pr[:mid].mean()*100:+.3f}%/day, "
              f"second half {pr[mid:].mean()*100:+.3f}%/day")
    print(f"\n  VERDICT INPUT: does ANY hold length or the stack turn net-positive")
    print(f"  with |t|>2 and consistent holdout? If not, microstructure is closed too.")


if __name__ == "__main__":
    main()
