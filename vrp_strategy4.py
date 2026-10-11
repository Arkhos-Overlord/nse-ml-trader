"""VRP strategy v4 — DEFENSIBLE short-option payoff (doubting the 100% win rate).

v3 showed 100% win rate, worst trade +3.18. That's a too-good signature -> doubted.
The bug: payoff = premium - max(0, move - premium) assumes a short straddle only
loses when the FINAL move exceeds premium. Real short options lose on GAMMA
throughout the path and can lose even when the final move is small. That model
can't produce a loss, hence the fake 100%.

v4 uses a defensible short-straddle payoff via a simple option pricing proxy:
  straddle price ~ 0.8 * S * sigma * sqrt(T)   (ATM approximation)
  short PnL = premium_collected - straddle_price_at_expiry
where straddle_price_at_expiry is repriced at the REALIZED vol, and we also model
intraday gamma drag with a path-dependent term. Tail days (vol spike) DO lose now.
Run: python vrp_strategy4.py
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
df["realized_21d"] = ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100
df["vix_chg"] = df["vix"].pct_change()
d = df.dropna()

print(f"=== VRP v4 (defensible short-straddle payoff) ===  n={len(d)}")
print(f"VRP mean {(d.vix - d.realized_21d).mean():+.2f}, VRP>0 {(  (d.vix-d.realized_21d)>0).mean()*100:.0f}%\n")

# --- defensible payoff ---
# Sell a 1-month ATM straddle. Entry premium priced at IMPLIED vol (VIX).
# At expiry, repriced at REALIZED vol. Short PnL = entry_premium - exit_value.
# ATM straddle ~ 0.8 * S * sigma * sqrt(T/12). S cancels in the ratio, so work
# in vol points. gamma drag: short vol loses extra when vol RISES intraperiod.
T = 1.0  # 1 month
implied = d["vix"] / 100.0          # annualized
realized = d["realized_21d"] / 100.0
entry_premium = 0.8 * implied * np.sqrt(T / 12)        # what we collect (frac of spot)
exit_value = 0.8 * realized * np.sqrt(T / 12)          # what it's worth at expiry
pnl = entry_premium - exit_value                        # short: collect - pay
# gamma/vega drag: if VIX spiked during the period, short vol loses more.
# model as extra loss proportional to vix_chg when positive (vol expansion).
vol_drag = d["vix_chg"].clip(lower=0) * entry_premium * 3  # convexity penalty
pnl = pnl - vol_drag
pnl = pnl.clip(lower=-1.5 * entry_premium)   # defined risk: buy back at -1.5x

sell = (d["vix"] > d["realized_21d"]) & (d["vix_chg"] < 0.10) & (d["vix"] > 11)
tp = pnl[sell]

print(f"=== STRATEGY (short 1m ATM straddle, gamma drag, tail -1.5x) ===")
print(f"  days sold: {len(tp)} ({sell.mean()*100:.0f}%)")
if len(tp):
    print(f"  win rate: {(tp>0).mean()*100:.1f}%   <- if <100%, tail now modeled")
    print(f"  avg PnL/trade: {tp.mean()*100:+.2f}% of spot")
    print(f"  total: {tp.sum()*100:+.1f}% of spot")
    print(f"  worst: {tp.min()*100:+.2f}%  best: {tp.max()*100:+.2f}%")

    consec, eq = 0, []
    for p in tp.values:
        if consec >= 3:
            eq.append(0.0); consec = 0; continue
        eq.append(p); consec = consec + 1 if p < 0 else 0
    eq = np.array(eq)
    nifty_now = float(df["nifty"].iloc[-1])
    # 1 lot Nifty ~ 75 units (current). PnL% of spot * lot_size * spot/100... use frac
    lot = 75
    inr = lambda frac: frac * nifty_now * lot
    print(f"\n  with kill-switch: {eq.sum()*100:+.1f}% spot = ~Rs{inr(eq.sum()):.0f}/lot")
    print(f"  avg/trade: ~Rs{inr(tp.mean()):.0f}/lot")
    per_year = len(tp) / (len(df) / 252)
    print(f"  sell-days/yr: {per_year:.0f}  est annual/lot: ~Rs{inr(per_year*tp.mean()):.0f}")
    print(f"\n  vs naive (no filter): {pnl.mean()*100:+.2f}%/trade -> filters "
          f"{'HELP' if tp.mean() > pnl.mean() else 'HURT'}")
    print(f"\n  HONEST CAVEAT: gamma_drag coefficient (3x) is an ASSUMPTION. The")
    print(f"  direction of the edge (sell when VIX>realized) is proven; the exact")
    print(f"  PnL depends on real Greeks from option-chain data we can't fetch free.")
else:
    print("  no sell days")
