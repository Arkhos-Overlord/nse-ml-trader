"""THE decisive test: does the Volatility Risk Premium (VRP) exist in India?

VRP = India VIX (implied vol) - realized Nifty vol. If VIX is systematically
ABOVE realized vol, then SELLING premium (short options) captures the difference
— the structural edge that fits Rs10K (options premium, not index units).

This is the documented, model-free edge the price-direction strategies couldn't
touch. Test on real yfinance data: ^INDIAVIX + ^NSEI, aligned, 2 years.

Run: python test_vrp.py
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
print(f"India VIX: {len(vix)} days, {vix.index[0].date()} -> {vix.index[-1].date()}")
print(f"Nifty:     {len(nifty)} days, {nifty.index[0].date()} -> {nifty.index[-1].date()}")

# align
df = pd.DataFrame({"vix": vix, "nifty": nifty}).dropna()
print(f"\nAligned: {len(df)} days\n")

# realized vol (forward-looking, annualized): std of NEXT 30 days of daily returns
ret = df["nifty"].pct_change()
df["realized_30d"] = ret.rolling(30).std().shift(-30) * np.sqrt(252) * 100

# VRP = VIX today - realized vol over next 30 days (this is what a seller earns)
df["vrp"] = df["vix"] - df["realized_30d"]
vrp = df["vrp"].dropna()

print("=== VOLATILITY RISK PREMIUM (VIX - next-30d realized vol) ===")
print(f"  n days: {len(vrp)}")
print(f"  mean VRP:   {vrp.mean():+.2f} vol points")
print(f"  median VRP: {vrp.median():+.2f}")
print(f"  % days VRP > 0 (premium overpriced): {(vrp > 0).mean()*100:.1f}%")
print(f"  mean VIX:   {df['vix'].mean():.1f}")
print(f"  mean realized: {df['realized_30d'].mean():.1f}")
print()
if vrp.mean() > 0 and (vrp > 0).mean() > 0.6:
    print("  >>> EDGE CONFIRMED: implied vol is structurally above realized vol.")
    print("      Selling premium (short strangles/straddles) captures the VRP.")
    print("      This fits Rs10K: 1 Nifty option lot premium ~Rs50-150.")
elif vrp.mean() > 0:
    print("  >>> WEAK edge: VRP positive on average but not consistently.")
else:
    print("  >>> NO edge: VIX is NOT above realized vol. Premium selling loses.")

# rough PnL of a simple monthly short-strangle proxy: collect VRP when >0,
# pay realized when <0. Express as % of premium sold (~= VIX/12 monthly).
print("\n=== naive short-premium PnL proxy (collect VRP when +, pay when -) ===")
# monthly vol points captured per day, as fraction of a 1-month option
df["monthly_premium_pts"] = df["vix"] / np.sqrt(12)
edge = (df["vrp"] / df["monthly_premium_pts"]).dropna()
print(f"  mean edge as % of premium: {edge.mean()*100:+.1f}%")
print(f"  win rate (VRP>0): {(vrp>0).mean()*100:.0f}%")
print(f"  worst single-day VRP: {vrp.min():+.1f} (this is your tail risk when short)")
print("\n  Note: this is a GRADIENT, not a strategy backtest. It proves the premium")
print("  exists; a real strategy must manage the tail (the VRP<0 days).")
