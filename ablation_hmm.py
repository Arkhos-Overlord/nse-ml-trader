"""Fill GAP 3: does the HMM regime engine BEAT a naive baseline?

Head-to-head: HMM-driven sizing vs. a simple rolling-volatility threshold (no HMM).
If the HMM can't beat a one-line vol threshold, it's decorative complexity.
Runs on real 15m data, same strategies + risk, only the REGIME/SIZING differs.
Read-only comparison.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from volforge.backtest import _prep
from volforge.regime import HMMRegimeEngine, Regime
from volforge.risk import RiskManager
from volforge.strategies import Signal, best_setup

warnings.filterwarnings("ignore")


def load(sym):
    d = yf.download(sym, period="60d", interval="15m", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return _prep(d.dropna())


def run_with_regime(df, use_hmm: bool, *, min_train=150, refit_every=30,
                    hold_bars=8, min_rr=1.5):
    """Backtest where sizing multiplier comes from HMM (use_hmm) or a naive
    vol threshold. Everything else identical. Returns trade PnLs."""
    engine = HMMRegimeEngine() if use_hmm else None
    risk = RiskManager()
    close = df["close"]
    n = len(df)
    ret = close.pct_change()
    vol = ret.rolling(20).std()
    # naive baseline: trade only when rolling vol is in top half (high vol = our thesis)
    vol_med = vol.median()
    pnls = []
    position = 0
    entry = sl = tgt = 0.0
    bars_held = 0
    fitted_at = -10**9

    for i in range(min_train, n - 1):
        if use_hmm and (i - fitted_at >= refit_every or engine.model is None):
            engine.fit(df.iloc[:i])
            fitted_at = i
        if use_hmm:
            engine.predict(df.iloc[:i])
            mult = engine.position_size_multiplier()
            tradeable = engine.is_tradeable()
        else:
            # naive: high-vol regime -> full size, low-vol -> half, chop -> flat
            v = vol.iloc[i]
            if v > vol_med * 1.2:
                mult, tradeable = 1.0, True
            elif v > vol_med:
                mult, tradeable = 0.7, True
            else:
                mult, tradeable = 0.0, False

        nxt = float(close.iloc[i + 1])
        if position != 0:
            bars_held += 1
            hit_sl = (position > 0 and nxt <= sl) or (position < 0 and nxt >= sl)
            hit_tgt = (position > 0 and nxt >= tgt) or (position < 0 and nxt <= tgt)
            if hit_sl or hit_tgt or bars_held >= hold_bars:
                exit_price = sl if hit_sl else (tgt if hit_tgt else nxt)
                pnl = position * (exit_price - entry) - risk.cost_of(abs(position) * entry)
                pnls.append(pnl)
                risk.on_trade_closed(pnl)
                position = 0
                bars_held = 0
        if position == 0 and tradeable and risk.can_open():
            setup = best_setup(df.iloc[:i + 1], min_rr=min_rr)
            if setup.signal != Signal.NONE and setup.is_valid(min_rr):
                units = risk.position_size(setup.entry, setup.stop_loss, mult)
                if units > 0 and setup.reward / setup.entry * 100 > 0.15:
                    position = units if setup.signal == Signal.BUY else -units
                    entry, sl, tgt = setup.entry, setup.stop_loss, setup.target
                    risk.state.open_positions += 1
                    bars_held = 0
    return pnls


def summarize(pnls, label):
    if not pnls:
        print(f"  {label}: 0 trades")
        return
    arr = np.array(pnls)
    total = arr.sum()
    wins = (arr > 0).sum()
    print(f"  {label}: trades={len(arr)} total=Rs{total:.0f} "
          f"winrate={wins/len(arr)*100:.0f}% avg=Rs{arr.mean():.0f}")


if __name__ == "__main__":
    df = load("^NSEI")
    print(f"Loaded {len(df)} 15m bars\n")
    print("=== GAP 3: HMM vs naive-vol-threshold (same strategies+risk) ===")
    # NOTE: on 15m Nifty both will likely be 0 trades (the ₹10K wall).
    # So we run on a CHEAP instrument where units>0 is possible: a stock ~<500 Rs.
    print("(Nifty index = 0 trades on ₹10K, so compare on RELIANCE where units>0)")
    df2 = load("RELIANCE.NS")
    print(f"RELIANCE: {len(df2)} bars, last close ~Rs{df2['close'].iloc[-1]:.0f}\n")

    for name, d in [("NIFTY", df), ("RELIANCE", df2)]:
        print(f"--- {name} ---")
        summarize(run_with_regime(d, use_hmm=True), "HMM     ")
        summarize(run_with_regime(d, use_hmm=False), "naive-vol")
        print()
    print("VERDICT: if naive-vol >= HMM, the HMM is decorative. If HMM > naive,")
    print("the regime model earns its complexity.")
