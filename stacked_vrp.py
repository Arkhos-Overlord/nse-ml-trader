"""Stacked-filter VRP strategy — walk-forward. Does stacking edges beat raw VRP?

Edge hunt found: VRP is highest in MID VIX regime (+2.97) and uptrends (+3.81),
lowest in high-VIX panic (+1.27). Stacked filters (VRP>1 & uptrend & not-spiking
& mid-VIX) gave VRP +4.74 vs raw 1.77. Test if that STACK survives walk-forward
and beats the single-filter start.

Run: python stacked_vrp.py
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
    return d


vix = get("^INDIAVIX")["Close"].dropna()
nif = get("^NSEI")
df = pd.DataFrame({"vix": vix, "close": nif["Close"]}).dropna()
ret = df["close"].pct_change()
df["ret"] = ret
df["realized_21d"] = ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100
df["vix_chg"] = df["vix"].pct_change()
df["vrp"] = df["vix"] - df["realized_21d"]
df["vix_pct"] = df["vix"].rolling(120, min_periods=60).rank(pct=True)
df["sma50"] = df["close"].rolling(50).mean()
df["uptrend"] = df["close"] > df["sma50"]
d = df.dropna()
split = int(len(d) * 0.6)


def payoff(frame):
    implied = frame["vix"] / 100.0
    realized = frame["realized_21d"] / 100.0
    entry = 0.8 * implied * np.sqrt(1 / 12)
    exitv = 0.8 * realized * np.sqrt(1 / 12)
    pnl = entry - exitv
    drag = frame["vix_chg"].clip(lower=0) * entry * 3.0
    return (pnl - drag).clip(lower=-1.5 * entry)


def strat(frame, use_stack, kill=3):
    pnl = payoff(frame)
    if use_stack:
        # STACKED: VRP>1, VIX 12-20 (avoid panic), not spiking, uptrend
        sell = ((frame["vrp"] > 1.0) & (frame["vix"] > 12) & (frame["vix"] < 22)
                & (frame["vix_chg"] < 0.10) & (frame["uptrend"]))
    else:
        # SINGLE filter (the proven win-win start)
        sell = (frame["vrp"] > 1.0) & (frame["vix"] > 12) & (frame["vix_chg"] < 0.10)
    tp = pnl[sell]
    if len(tp) == 0:
        return 0, 0.0, 0.0
    consec, kept = 0, []
    for p in tp.values:
        if consec >= kill:
            kept.append(0.0); consec = 0; continue
        kept.append(p); consec = consec + 1 if p < 0 else 0
    kept = np.array(kept)
    return len(tp), float(tp.mean()), float(kept.sum())


print("=== STACKED vs SINGLE-FILTER VRP (walk-forward) ===\n")
for name, use_stack in [("SINGLE (proven)", False), ("STACKED (edges)", True)]:
    print(f"--- {name} ---")
    tr, a_tr, tot_tr = strat(d.iloc[:split], use_stack)
    te, a_te, tot_te = strat(d.iloc[split:], use_stack)
    print(f"  TRAIN: {tr} trades, {a_tr*100:+.2f}%/trade, total {tot_tr*100:+.1f}%")
    print(f"  TEST:  {te} trades, {a_te*100:+.2f}%/trade, total {tot_te*100:+.1f}%")
    print(f"  -> {'HOLDS' if a_te > 0 else 'COLLAPSES'} out-of-sample\n")

# robustness: does stacked win across many param variations?
print("=== STACKED robustness (vary kill-switch) ===")
for kill in [2, 3, 4, 5]:
    te, a_te, _ = strat(d.iloc[split:], True, kill)
    print(f"  kill={kill}: test {a_te*100:+.2f}%/trade ({te} trades) "
          f"{'WIN' if a_te>0 else 'LOSE'}")

# the INR reality
nifty_now = float(df["close"].iloc[-1])
lot = 75
te, a_te, tot_te = strat(d.iloc[split:], True)
per_trade_inr = a_te * nifty_now * lot
print(f"\n=== MONEY (per Nifty lot, {lot} units @ Rs{nifty_now:.0f}) ===")
print(f"  avg/trade: ~Rs{per_trade_inr:.0f}")
print(f"  over {te} test trades: ~Rs{tot_te*nifty_now*lot:.0f}")
print(f"  est trades/yr: ~{te/(len(d.iloc[split:])/252):.0f}")
print(f"  est annual/lot: ~Rs{per_trade_inr*te/(len(d.iloc[split:])/252):.0f}")
print(f"\n  HONEST: with Rs10K, 1 lot Nifty strangle margin ~Rs15-20K (SPAN+exposure).")
print(f"  So Rs10K may fund 1 lot only if broker allows, else need ~Rs20-25K for")
print(f"  comfortable 1-lot selling. BANKNIFTY lot is bigger premium, more margin.")
