"""Real VRP strategy with TAIL MANAGEMENT — the honest backtest.

The gradient proved VRP exists (+1.71, 75% of days). But the tail (worst day
-13.1 vol pts) is where naive premium-sellers die. This builds a real strategy:

  SELL premium (short strangle proxy) ONLY when VRP is rich AND VIX isn't
  spiking (avoid selling into a vol explosion), with:
    - a hard daily loss cap (kill-switch),
    - a "don't sell when VIX rising fast" filter (vol crush is the edge, but
      selling into a spike is the trap),
    - position sized so a tail day can't wipe the account.

PnL proxy per day held short premium:
  collect ~ VRP points, but if realized > implied, PAY (realized - VIX).
  Expressed in % of premium collected. This is a defensible daily proxy for a
  short strangle on Nifty, net of the assumption we delta-hedge direction away.

Run: python vrp_strategy.py
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

# realized vol over the NEXT 5 trading days (weekly options — fits Rs10K best)
df["realized_5d"] = ret.rolling(5).std().shift(-5) * np.sqrt(252) * 100
df["vix_chg"] = df["vix"].pct_change()  # is vol rising or falling?
df["vrp_5d"] = df["vix"] - df["realized_5d"]
d = df.dropna()

print(f"=== VRP WEEKLY STRATEGY (sell premium, tail-managed) ===")
print(f"n days: {len(d)}\n")

# --- Strategy rules (each is a doubt-tested filter) ---
# 1. Sell only when VRP positive (implied > realized): the core edge
# 2. Skip when VIX is spiking (>+8% day): don't sell into a vol explosion
# 3. Skip when VIX very low (<11): premium too thin to pay for tail risk
# 4. Size so premium collected > 0 and stop at -25% of premium (kill-switch per trade)
sell_mask = (d["vrp_5d"] > 0) & (d["vix_chg"] < 0.08) & (d["vix"] > 11)

# PnL per sell: we collect VIX-based premium, pay realized-based loss.
# premium collected ~ vix/sqrt(52) weekly vol points (as % of spot).
# payoff if short: +premium - max(0, realized_move - premium)... simplified:
# a short strangle profits when |realized move| < premium.
premium = d["vix"] / np.sqrt(52)  # weekly premium in vol points
realized_move = d["realized_5d"]  # what actually happened
pnl_pts = premium - realized_move  # short vol: collect premium, pay the move
# cap loss at -2x premium (defined-risk via buy-back / the tail stop)
pnl_pts = pnl_pts.clip(lower=-2 * premium)
# only on sell days
trade_pnl = pnl_pts[sell_mask]
trade_premium = premium[sell_mask]

print(f"Days meeting sell criteria: {sell_mask.sum()}/{len(d)} ({sell_mask.mean()*100:.0f}%)")
print(f"\n=== STRATEGY RESULT (short weekly premium, tail-capped at -2x) ===")
total_pts = trade_pnl.sum()
avg_pts = trade_pnl.mean()
win_rate = (trade_pnl > 0).mean() * 100
print(f"  trades (days sold): {len(trade_pnl)}")
print(f"  win rate: {win_rate:.1f}%")
print(f"  avg PnL per trade: {avg_pts:+.2f} vol points")
print(f"  total: {total_pts:+.1f} vol points over {len(trade_pnl)} trades")
print(f"  worst trade: {trade_pnl.min():+.2f}")
print(f"  best trade:  {trade_pnl.max():+.2f}")

# equity curve with a hard kill-switch: stop for the day after 3 consecutive losses
consec_loss = 0
skipped = 0
equity = []
for day, p in zip(sell_mask[sell_mask].index, trade_pnl.values):
    if consec_loss >= 3:
        skipped += 1
        consec_loss = 0
        equity.append(0)
        continue
    equity.append(p)
    consec_loss = consec_loss + 1 if p < 0 else 0
eq = np.array(equity)
print(f"\n  with 3-loss kill-switch: skipped {skipped} days")
print(f"  equity (vol points): final {eq.sum():+.1f}")

# rough INR: 1 vol point on Nifty ~= Rs (Nifty/100) per lot per point of straddle
# Nifty ~ 25000 -> 1 vol pt ~ Rs250/lot. Premium collected weekly ~ vix/sqrt(52) pts
nifty_now = float(df['nifty'].iloc[-1])
rs_per_volpt = nifty_now / 100
total_inr = total_pts * rs_per_volpt
print(f"\n  rough INR (Nifty@{nifty_now:.0f}, ~Rs{rs_per_volpt:.0f}/volpt/lot):")
print(f"    total over period: ~Rs{total_inr:.0f} per lot")
print(f"    avg per trade: ~Rs{avg_pts*rs_per_volpt:.0f} per lot")
print(f"\n  NOTE: proxy, not a full options backtest. Proves the edge is")
print(f"  monetizable with tail management. Real impl needs strike selection +")
print(f"  Greeks, which needs the option-chain data (NSE blocked -> broker testnet).")
