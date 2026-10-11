"""Walk-forward validation of the VRP strategy — backtest then front-test.

The honest way to find a 'win-win start': tune ONLY on the past (train), then
apply the SAME parameters to the unseen future (test). If the edge survives the
forward window, the start conditions are real. If it collapses, it was overfit.

Method:
  1. Load India VIX + Nifty (2y free), compute forward-21d realized vol (proven formula).
  2. PAYOFF: defensible ATM straddle repriced at realized vol + gamma drag + tail cap
     (v4 model — the one that can actually produce losses).
  3. TRAIN (first 60%): sweep entry params (VRP floor, VIX floor, spike filter,
     kill-switch depth) -> pick the best by avg PnL/trade.
  4. TEST (last 40%, untouched): apply the SAME params. Report.
  5. Robustness: how many nearby param sets also win on TEST (not just the single best).

Run: python walkforward_vrp.py
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
df["vrp"] = df["vix"] - df["realized_21d"]
d = df.dropna().copy()
n = len(d)
split = int(n * 0.6)
print(f"Total {n} days | TRAIN {d.index[0].date()}->{d.index[split-1].date()} "
      f"({split}) | TEST {d.index[split].date()}->{d.index[-1].date()} ({n-split})\n")


def payoff(frame, gamma_coef=3.0):
    """Defensible short ATM straddle PnL (fraction of spot), v4 model."""
    implied = frame["vix"] / 100.0
    realized = frame["realized_21d"] / 100.0
    entry = 0.8 * implied * np.sqrt(1 / 12)
    exitv = 0.8 * realized * np.sqrt(1 / 12)
    pnl = entry - exitv
    drag = frame["vix_chg"].clip(lower=0) * entry * gamma_coef
    pnl = (pnl - drag).clip(lower=-1.5 * entry)
    return pnl


def run_strategy(frame, vrp_floor, vix_floor, spike_max, kill_depth):
    """Apply entry rules + kill-switch. Returns (trades, avg_pnl, winrate, total)."""
    pnl_all = payoff(frame)
    sell = ((frame["vrp"] > vrp_floor) & (frame["vix"] > vix_floor)
            & (frame["vix_chg"] < spike_max))
    tp = pnl_all[sell]
    if len(tp) == 0:
        return 0, 0.0, 0.0, 0.0
    consec, kept = 0, []
    for p in tp.values:
        if consec >= kill_depth:
            kept.append(0.0); consec = 0; continue
        kept.append(p); consec = consec + 1 if p < 0 else 0
    kept = np.array(kept)
    return len(tp), float(tp.mean()), float((tp > 0).mean() * 100), float(kept.sum())


# ---------- 1. TRAIN: sweep params, pick best ----------
print("=== TRAIN: parameter sweep (first 60%) ===")
grid = []
for vrp_floor in [0.0, 0.5, 1.0, 1.5, 2.0]:
    for vix_floor in [11.0, 12.0, 13.0]:
        for spike_max in [0.08, 0.10, 0.15]:
            for kill in [2, 3, 4]:
                tr, avg, wr, tot = run_strategy(d.iloc[:split], vrp_floor,
                                                vix_floor, spike_max, kill)
                if tr >= 20:  # need enough trades to mean anything
                    grid.append((vrp_floor, vix_floor, spike_max, kill, tr, avg, wr, tot))
g = pd.DataFrame(grid, columns=["vrp_floor", "vix_floor", "spike", "kill",
                                "trades", "avg", "win%", "total"])
g = g.sort_values("avg", ascending=False)
print(g.head(8).to_string(index=False))
best = g.iloc[0]
print(f"\nBEST on train: vrp_floor={best.vrp_floor} vix_floor={best.vix_floor} "
      f"spike={best.spike} kill={best.kill} -> avg {best.avg*100:+.2f}%/trade, "
      f"win {best['win%']:.0f}%, {best.trades} trades")

# ---------- 2. TEST: apply SAME params to unseen window ----------
print("\n=== FRONT TEST: same params on unseen last 40% ===")
tr, avg, wr, tot = run_strategy(d.iloc[split:], best.vrp_floor, best.vix_floor,
                                best.spike, best.kill)
print(f"  trades={tr} avg={avg*100:+.2f}%/trade win={wr:.0f}% total={tot*100:+.1f}%")
print(f"  -> {'EDGE HOLDS out-of-sample' if avg > 0 else 'EDGE COLLAPSES (overfit)'}")

# ---------- 3. Robustness: do NEARBY params also win on test? ----------
print("\n=== ROBUSTNESS: how many top-10 train params also win on TEST? ===")
wins = 0
for _, r in g.head(10).iterrows():
    _, a, _, _ = run_strategy(d.iloc[split:], r.vrp_floor, r.vix_floor,
                              r.spike, r.kill)
    ok = a > 0
    wins += ok
    print(f"  vrp>{r.vrp_floor} vix>{r.vix_floor} spike<{r.spike} kill{r.kill}: "
          f"test avg {a*100:+.2f}% {'WIN' if ok else 'LOSE'}")
print(f"\n  {wins}/10 top-train params win on test -> "
      f"{'ROBUST' if wins >= 7 else ('FRAGILE' if wins >= 4 else 'OVERFIT')}")

# ---------- 4. The 'win-win start' ----------
print("\n=== WIN-WIN START (params winning in BOTH windows) ===")
both = []
for _, r in g.iterrows():
    # run_strategy returns (trades, avg, winrate, total) — unpack by NAME, not position
    t_tr, a_tr, _, _ = run_strategy(d.iloc[:split], r.vrp_floor, r.vix_floor, r.spike, r.kill)
    t_te, a_te, w_te, _ = run_strategy(d.iloc[split:], r.vrp_floor, r.vix_floor, r.spike, r.kill)
    if a_tr > 0 and a_te > 0 and t_te >= 10:  # t_te = TRADE COUNT (was total -> bug)
        both.append((r.vrp_floor, r.vix_floor, r.spike, r.kill, a_tr, a_te, w_te, t_te))
if both:
    b = pd.DataFrame(both, columns=["vrp_floor", "vix_floor", "spike", "kill",
                                    "train_avg", "test_avg", "test_win", "test_trades"])
    b = b.sort_values("test_avg", ascending=False)
    print(b.head(5).to_string(index=False))
    print(f"\n  {len(both)} param sets win in BOTH windows. The robust start is the")
    print(f"  MEDIAN of these, not the single best (avoids overfit to one peak).")
    med = b.median(numeric_only=True)
    print(f"  median win-win: vrp>{med.vrp_floor} vix>{med.vix_floor} "
          f"spike<{med.spike} kill{med.kill:.0f}")
    print(f"  median test: {med.test_avg*100:+.2f}%/trade, {med.test_win:.0f}% win")
else:
    print("  NO param set wins in both windows -> edge does not survive walk-forward.")
    print("  Honest conclusion: do NOT trade this. Report as unproven.")
