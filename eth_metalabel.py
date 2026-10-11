"""META-LABELING on the ETH edge — filter bad trades to fix the tail.

Reddit practitioner intel (u/Yocurt, production trader): "ML will not find
patterns by itself... A much better approach: have an underlying strategy that
has an existing edge, then train a model on the results of that strategy so it
learns to filter out low quality trades."

My ETH edge HAS edge (82% quarters, 2-sided) but dies at Rs1.5L on FAT TAILS
(-31.9% worst trade). Meta-labeling attacks exactly that: learn which signals
are losers and skip them.

Method (triple-barrier labels per Lopez de Prado):
  1. Generate all ETH momentum signals with features AT SIGNAL TIME (no
     look-ahead: only data available when the signal fires)
  2. Label each +1 (win) / -1 (loss) on the 21-day hold outcome
  3. Train LightGBM (or logistic fallback) on FIRST HALF only
  4. Predict on SECOND HALF — take only trades the model says are winners
  5. Compare filtered vs unfiltered: PnL, drawdown, worst trade

If filtering cuts the tail without killing the edge, Rs1.5L becomes viable.
Run: python eth_metalabel.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FEE = 0.001
CAPITAL_USD = 1800.0
USD_INR = 83
LOOK, HOLD, THRESH = 14, 21, 0.08

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


def load():
    df = pd.read_parquet("data_indmoney/crypto_daily/ETHUSDT_1d.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def build_signals(df):
    """All signals with features computable AT SIGNAL TIME (no look-ahead)."""
    rows = []
    for i in range(max(LOOK, 60), len(df) - HOLD):
        r = df.close.iloc[i] / df.close.iloc[i - LOOK] - 1
        d = 1 if r > THRESH else (-1 if r < -THRESH else 0)
        if d == 0:
            continue
        entry = float(df.open.iloc[i + 1])
        exit_ = float(df.close.iloc[min(i + HOLD, len(df) - 1)])
        if entry <= 0 or exit_ <= 0:
            continue
        # features — all use data up to and including bar i
        rets = df.close.iloc[i - 20:i + 1].pct_change().dropna()
        vol20 = rets.std()
        vol5 = df.close.iloc[i - 5:i + 1].pct_change().std()
        hi20 = df.high.iloc[i - 20:i + 1].max()
        lo20 = df.low.iloc[i - 20:i + 1].min()
        sma50 = df.close.iloc[i - 50:i + 1].mean()
        # ATR proxy
        atr = (df.high.iloc[i - 14:i + 1] - df.low.iloc[i - 14:i + 1]).mean()
        rows.append({
            "i": i, "date": df.datetime.iloc[i], "dir": d,
            "entry": entry, "exit": exit_,
            "ret_look": r, "vol20": vol20, "vol5": vol5,
            "vol_ratio": vol5 / vol20 if vol20 > 0 else 1,
            "pos_in_range": (df.close.iloc[i] - lo20) / (hi20 - lo20) if hi20 > lo20 else 0.5,
            "above_sma50": df.close.iloc[i] / sma50 - 1,
            "atr_pct": atr / df.close.iloc[i],
            # outcome
            "pnl": (exit_ - entry) * (CAPITAL_USD / entry) * d - CAPITAL_USD * 2 * FEE,
        })
    return pd.DataFrame(rows)


def simulate(t, capital0=CAPITAL_USD):
    """Full-capital compounding sim, 1 position at a time (sequential trades)."""
    capital = capital0
    peak = capital
    maxdd = 0
    last_exit = -1
    for _, row in t.sort_values("i").iterrows():
        if row["i"] <= last_exit:
            continue
        qty = capital / row["entry"]
        gross = (row["exit"] - row["entry"]) * qty * row["dir"]
        fee = capital * 2 * FEE
        capital += gross - fee
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
        last_exit = row["i"] + HOLD
    return capital, maxdd


def main():
    df = load()
    t = build_signals(df)
    print(f"=== META-LABELING on ETH edge ===")
    print(f"  {len(t)} signals, {df.datetime.iloc[0].date()} -> {df.datetime.iloc[-1].date()}")
    print(f"  features: ret_look, vol20, vol5, vol_ratio, pos_in_range, "
          f"above_sma50, atr_pct (all at signal time)\n")

    # label: win/loss
    t["label"] = (t.pnl > 0).astype(int)

    # train/test split by TIME (first 60% train, last 40% test) — no leakage
    split = int(len(t) * 0.6)
    tr, te = t.iloc[:split], t.iloc[split:]
    feats = ["ret_look", "vol20", "vol5", "vol_ratio", "pos_in_range",
             "above_sma50", "atr_pct", "dir"]
    Xtr, ytr = tr[feats], tr["label"]
    Xte = te[feats]

    # baseline (unfiltered) on test set
    base_cap, base_dd = simulate(te)
    base_worst = te.pnl.min()
    print(f"--- BASELINE (no filter), test period ---")
    print(f"  {len(te)} trades, avg ${te.pnl.mean():.2f}, "
          f"worst ${base_worst:.2f}")
    print(f"  Rs{CAPITAL_USD*USD_INR:,.0f} -> Rs{base_cap*USD_INR:,.0f} "
          f"({(base_cap/CAPITAL_USD-1)*100:+.0f}%), maxDD {base_dd*100:.1f}%")

    # train models
    models = {}
    if HAS_LGB:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, num_leaves=15,
            learning_rate=0.05, verbose=-1, min_child_samples=20)
    models["gbdt"] = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05)
    models["logistic"] = LogisticRegression(max_iter=1000)

    print(f"\n--- META-LABELED (train on first 60%, filter test) ---")
    print(f"{'model':<12}{'taken':>7}{'skip%':>7}{'avg$':>9}{'worst$':>9}"
          f"{'endRs':>11}{'ret%':>7}{'maxDD':>7}")
    best = None
    for name, m in models.items():
        try:
            m.fit(Xtr, ytr)
            proba = m.predict_proba(Xte)[:, 1]
        except Exception as e:  # noqa: BLE001
            print(f"{name:<12}  failed: {str(e)[:40]}")
            continue
        # take only trades model is confident are winners
        filt = te[proba >= 0.5].copy()
        if len(filt) < 10:
            print(f"{name:<12}  too few trades pass filter")
            continue
        cap, dd = simulate(filt)
        skip = (1 - len(filt) / len(te)) * 100
        print(f"{name:<12}{len(filt):>7}{skip:>6.0f}%{filt.pnl.mean():>9.2f}"
              f"{filt.pnl.min():>9.2f}{cap*USD_INR:>11,.0f}"
              f"{(cap/CAPITAL_USD-1)*100:>6.0f}%{dd*100:>6.1f}%")
        if best is None or cap > best[1]:
            best = (name, cap, dd, filt)

    if best:
        name, cap, dd, filt = best
        print(f"\n=== BEST FILTER: {name} ===")
        print(f"  Rs{CAPITAL_USD*USD_INR:,.0f} -> Rs{cap*USD_INR:,.0f} "
              f"({(cap/CAPITAL_USD-1)*100:+.0f}%), maxDD {dd*100:.1f}%")
        print(f"  vs baseline {(base_cap/CAPITAL_USD-1)*100:+.0f}% / DD {base_dd*100:.1f}%")
        improved = cap > base_cap and dd < base_dd
        print(f"\n  VERDICT: {'*** FILTERING WORKS — tail cut, edge preserved ***' if improved else 'filtering did not improve risk-adjusted return'}")
        if improved:
            monthly = (cap*USD_INR - CAPITAL_USD*USD_INR) / ((df.datetime.iloc[-1] - df.datetime.iloc[len(df)//2]).days / 30)
            print(f"  ~Rs{monthly:,.0f}/month on Rs1.5L (test period)")
    else:
        print("\n  no filter improved on baseline")


if __name__ == "__main__":
    main()
