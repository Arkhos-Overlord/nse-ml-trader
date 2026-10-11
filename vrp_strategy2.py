"""VRP strategy — FIXED proxy (prior version had a units bug, doubted & corrected).

BUG in v1: compared WEEKLY premium (vix/sqrt52 ~2pts) against ANNUALIZED realized
(~12pts) -> fake 0.6% win rate. Different time units. Corrected here:

Both sides are the SAME weekly horizon:
  premium collected  = vix / sqrt(52)          (weekly vol the market charges)
  realized move      = rolling 5d std * sqrt(52)  (weekly vol that happened)
Short strangle profits when realized < premium. This is apples-to-apples.
Run: python vrp_strategy2.py
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

# WEEKLY realized vol = std of 5-day SUM returns (the actual weekly move),
# annualized correctly. v1 used rolling(5).std() which is std of DAILY returns
# inside the window (~2%), NOT the weekly move — a units bug that faked a -9.6 VRP.
df["realized_5d"] = (ret.rolling(5).sum().rolling(30).std().shift(-5)
                     * np.sqrt(252 / 5) * 100)
df["vix_chg"] = df["vix"].pct_change()
# weekly premium the market charges (vix is annualized -> weekly = /sqrt(52))
df["weekly_premium"] = df["vix"] / np.sqrt(52)
df["weekly_vrp"] = df["weekly_premium"] - df["realized_5d"]
d = df.dropna()

print(f"=== VRP WEEKLY STRATEGY v2 (units fixed) ===  n={len(d)}")
print(f"mean weekly premium: {d['weekly_premium'].mean():.2f} vol pts")
print(f"mean weekly realized: {d['realized_5d'].mean():.2f} vol pts")
print(f"mean weekly VRP: {d['weekly_vrp'].mean():+.2f} vol pts")
print(f"VRP>0 (premium overpriced): {(d['weekly_vrp']>0).mean()*100:.1f}%\n")

# sell criteria: VRP positive + not selling into a vol spike + VIX not too thin
sell = (d["weekly_vrp"] > 0) & (d["vix_chg"] < 0.08) & (d["vix"] > 11)
# PnL: collect weekly_premium, pay realized move (capped at -2x = defined risk)
pnl = (d["weekly_premium"] - d["realized_5d"]).clip(lower=-2 * d["weekly_premium"])
tp = pnl[sell]

print(f"=== STRATEGY (short weekly premium, tail capped -2x) ===")
print(f"  days sold: {len(tp)} ({sell.mean()*100:.0f}% of days)")
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
rs_per_pt = nifty_now / 100
print(f"\n  with kill-switch: final {eq.sum():+.1f} vol pts = ~Rs{eq.sum()*rs_per_pt:.0f}/lot")
print(f"  avg/trade: ~Rs{tp.mean()*rs_per_pt:.0f}/lot")

# annualize: how many sell-days per year?
per_year = len(tp) / (len(df) / 252)
print(f"\n  sell-days/year: {per_year:.0f}")
print(f"  est. annual per lot: ~Rs{per_year*tp.mean()*rs_per_pt:.0f}")
print(f"\n  HONEST NOTE: proxy on a delta-hedged short strangle. Real PnL needs")
print(f"  strike selection + slippage + the tail days (VIX spikes). Direction")
print(f"  hedged away; pure vol harvest. If avg/trade>0 with kill-switch, the")
print(f"  edge survives tail management. If <0, the tail eats the premium.")
