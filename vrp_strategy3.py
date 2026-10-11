"""VRP weekly strategy — using the PROVEN realized-vol formula from test_vrp.py.

test_vrp.py already computed VRP correctly (mean +1.71, 75.4% of days) with:
    realized_30d = ret.rolling(30).std().shift(-30) * sqrt(252) * 100
i.e. std of DAILY returns over a forward window, annualized. That IS the standard
realized vol and is directly comparable to VIX (also annualized). v2 tried to
"fix" it with 5-day-sum rolling and broke it. Reverting to the proven formula.

Now build the actual SELL strategy on top of the correct VRP, tail-managed.
Run: python vrp_strategy3.py
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def get(sym):
    d = yf.download(sym, period="2y", interval="1d", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d["Close"].dropna()


vix = get("^INDIAVIX")
nifty = get("^NSEI")
df = pd.DataFrame({"vix": vix, "nifty": nifty}).dropna()
ret = df["nifty"].pct_change()

# PROVEN formula (from test_vrp.py): forward 21d realized vol, annualized
df["realized_21d"] = ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100
df["vix_chg"] = df["vix"].pct_change()
d = df.dropna()

print(f"=== VRP STRATEGY v3 (proven formula) ===  n={len(d)}")
print(f"mean VIX: {d.vix.mean():.2f}  mean realized21d: {d.realized_21d.mean():.2f}")
print(f"mean VRP: {(d.vix - d.realized_21d).mean():+.2f} vol pts")
print(f"VRP>0: {((d.vix - d.realized_21d) > 0).mean()*100:.1f}% of days\n")

# SELL premium when VRP rich, not into a spike, VIX not too thin.
# PnL proxy: collect (vix/100), pay realized move beyond premium, tail-capped.
# A short 1-month ATM straddle ~ collects vix/sqrt(12) monthly vol points of spot.
premium = d["vix"] / np.sqrt(12)          # monthly premium in vol pts
move = d["realized_21d"] / np.sqrt(12)    # monthly realized move in vol pts
# short straddle payoff ~ premium - max(0, move - premium) when |move|>premium
pnl = premium - np.maximum(0, move - premium)
pnl = pnl.clip(lower=-1.5 * premium)      # defined-risk cap (buy back at 1.5x loss)

sell = (d["vix"] > d["realized_21d"]) & (d["vix_chg"] < 0.10) & (d["vix"] > 11)
tp = pnl[sell]

print(f"=== STRATEGY (short monthly premium, tail capped -1.5x) ===")
print(f"  days sold: {len(tp)} ({sell.mean()*100:.0f}% of days)")
if len(tp):
    print(f"  win rate: {(tp>0).mean()*100:.1f}%")
    print(f"  avg PnL/trade: {tp.mean():+.2f} vol pts")
    print(f"  total: {tp.sum():+.1f} vol pts")
    print(f"  worst: {tp.min():+.2f}  best: {tp.max():+.2f}")

    # kill-switch: stop after 3 straight losses
    consec, eq = 0, []
    for p in tp.values:
        if consec >= 3:
            eq.append(0.0); consec = 0; continue
        eq.append(p); consec = consec + 1 if p < 0 else 0
    eq = np.array(eq)
    nifty_now = float(df["nifty"].iloc[-1])
    rs = nifty_now / 100
    print(f"\n  with 3-loss kill-switch: {eq.sum():+.1f} vol pts = ~Rs{eq.sum()*rs:.0f}/lot")
    print(f"  avg/trade: ~Rs{tp.mean()*rs:.0f}/lot")
    per_year = len(tp) / (len(df) / 252)
    print(f"  sell-days/yr: {per_year:.0f}  est annual/lot: ~Rs{per_year*tp.mean()*rs:.0f}")

    # compare to naive (no filters)
    naive = pnl.mean()
    print(f"\n  vs NAIVE sell-every-day avg: {naive:+.2f} vol pts/trade")
    print(f"  filters {'HELP' if tp.mean() > naive else 'HURT'} (edge per trade)")
    print(f"\n  HONEST: proxy on delta-hedged short premium. Real impl needs option")
    print(f"  chain (strikes/Greeks) — NSE blocked, needs broker testnet. Direction")
    print(f"  hedged; this isolates the VOL harvest. If avg>0 after tail mgmt, edge real.")
else:
    print("  no sell days — criteria too strict")
