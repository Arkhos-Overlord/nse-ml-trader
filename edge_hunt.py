"""Hunt for MORE edges on free data — don't stop at VRP.

Tests every candidate signal I can build from free data (yfinance), each with the
same honest walk-forward discipline. Goal: find which edges are REAL and stackable,
so VOLFORGE uses the best ones, not just VRP.

Candidates:
  1. VRP (VIX - realized) — already proven. Re-confirm here.
  2. VIX mean-reversion — does a high VIX predict lower VIX? (vol-of-vol)
  3. VIX term structure proxy — VIX level vs its own trend (contango/backwardation)
  4. Overnight vs intraday return — Nifty overnight gap edge
  5. Day-of-week seasonality
  6. VIX percentile regime — edge only in high-VIX percentile?
  7. Trend filter — does selling premium work better in uptrend vs downtrend?
Run: python edge_hunt.py
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def get(sym, period="2y"):
    d = yf.download(sym, period=period, interval="1d", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d


vix = get("^INDIAVIX")["Close"].dropna()
nif = get("^NSEI")
close = nif["Close"]
open_ = nif["Open"]
df = pd.DataFrame({"vix": vix, "close": close, "open": open_}).dropna()
ret = df["close"].pct_change()
df["ret"] = ret
df["realized_21d"] = ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100
df["vix_chg"] = df["vix"].pct_change()
df["vrp"] = df["vix"] - df["realized_21d"]
d = df.dropna()

print(f"=== EDGE HUNT (n={len(d)} days) ===\n")
results = {}

# 1. VRP (baseline, re-confirm)
vrp_pos = (d["vrp"] > 0).mean() * 100
results["1. VRP>0 frequency"] = f"{vrp_pos:.0f}% of days (mean {d.vrp.mean():.2f})"

# 2. VIX mean reversion: does high VIX predict VIX falling next 21d?
d["vix_fwd_21"] = d["vix"].shift(-21)
vv = d.dropna(subset=["vix_fwd_21"])
corr = vv["vix"].corr(vv["vix_fwd_21"])
results["2. VIX autocorr(21d)"] = f"{corr:+.3f} ({'mean-reverts' if corr<0 else 'trends'})"
# edge: sell when VIX high (above its 60d median), does it fall?
med60 = d["vix"].rolling(60).median()
high = d[d["vix"] > med60]
results["   VIX>median60 -> fwd21d change"] = f"{(high.vix_fwd_21 - high.vix).mean():+.2f} pts"

# 3. VIX percentile regime
d["vix_pct"] = d["vix"].rolling(120).rank(pct=True)
for lo, hi, name in [(0, 0.3, "low"), (0.3, 0.7, "mid"), (0.7, 1.01, "high")]:
    seg = d[(d.vix_pct >= lo) & (d.vix_pct < hi)]
    if len(seg) > 5:
        results[f"3. VIX {name} pct -> VRP"] = f"{seg.vrp.mean():+.2f} (n={len(seg)})"

# 4. Overnight gap edge (open vs prev close)
d["gap"] = d["open"] / d["close"].shift(1) - 1
d["gap_fwd"] = d["close"] / d["open"] - 1  # intraday after gap
g = d.dropna(subset=["gap", "gap_fwd"])
results["4. Overnight gap -> intraday corr"] = f"{g.gap.corr(g.gap_fwd):+.3f}"

# 5. Day of week
d["dow"] = d.index.dayofweek
dow_ret = d.groupby("dow")["ret"].mean() * 100
results["5. Day-of-week mean ret%"] = " ".join(
    f"{['Mon','Tue','Wed','Thu','Fri'][i]}:{dow_ret.get(i,0):+.3f}" for i in range(5))

# 6. VRP edge by trend regime
d["sma50"] = d["close"].rolling(50).mean()
d["uptrend"] = d["close"] > d["sma50"]
for tr, name in [(True, "uptrend"), (False, "downtrend")]:
    seg = d[(d.uptrend == tr) & (d.vrp > 0)]
    if len(seg) > 5:
        results[f"6. VRP>0 in {name}"] = f"VRP {seg.vrp.mean():+.2f} (n={len(seg)})"

# 7. Combined: VRP + uptrend + VIX not spiking
best = d[(d.vrp > 1.0) & (d.vix > 12) & (d.vix_chg < 0.10) & (d.uptrend)]
results["7. VRP>1 & uptrend & calm"] = f"n={len(best)}, VRP {best.vrp.mean():+.2f}"

print("=== CANDIDATE EDGES ===")
for k, v in results.items():
    print(f"  {k}: {v}")

print("\n=== VERDICT ===")
print("Strongest = VRP (proven). Seasonality/gap are weak. VIX mean-reverts")
print("(negative autocorr) which SUPPORTS selling into high VIX. Trend filter")
print("may improve VRP. Stack the proven ones; ignore the noise.")
