"""Trade the MEASURED vol signal directly — the strategy the thesis always implied.

The ablation showed copied strategies (ORB/VWAP/EMA) lose. But the +0.17 vol
clustering and −0.057 lag-1 autocorr were MEASURED and never traded. This builds
the strategy the data actually supports:

  1. VOL FORECAST: EWMA of |returns| → expected next-bar vol (vol clusters, so
     recent |ret| predicts near-term |ret|).
  2. VOL-TARGET ENTRY: enter when forecast vol is in its high regime (the state
     where moves exceed costs) AND price breaks a vol-scaled band.
  3. MEAN-REVERSION TILT (lag-1): after an extreme bar opposite to the short-term
     drift, fade it — the −0.057 says mild reversion follows.
  4. VOL-TARGET SIZING: size = risk_budget / (k × forecast_vol × price), so size
     shrinks when vol is high (survival) and grows when calm.
  5. COST GATE: only trade when expected move > real costs.

Run on 15m RELIANCE (₹1170 → units>0 possible) AND Nifty, walk-forward, vs cash.
Run: python vol_strategy.py
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def load(sym, interval="15m", period="60d"):
    d = yf.download(sym, period=period, interval=interval, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.columns = [c.lower() for c in d.columns]
    return d.dropna()


def ewma_vol(close, span=20):
    ret = close.pct_change()
    return ret.ewm(span=span).std()


def run_vol_strategy(df, *, capital=10_000.0, risk_pct=1.0, cost_bps=25.0,
                     vol_k=1.5, hold_bars=8, warmup=100, spread_bps=5.0):
    """Vol-target breakout + lag-1 mean-reversion. Walk-forward, no look-ahead.

    cost_bps is deliberately CONSERVATIVE (25) to include realistic F&O friction,
    per GAP 5. spread_bps models bid-ask separately from fees.
    """
    close = df["close"]
    n = len(df)
    vol_fcst = ewma_vol(close, 20)
    # rolling percentile of vol (expanding, past only) to know if vol is "high"
    vol_pct = vol_fcst.expanding().rank(pct=True)
    balance = capital
    position = 0
    entry = 0.0
    bars_held = 0
    equity = [capital]
    pnls = []

    for i in range(warmup, n - 1):
        price = float(close.iloc[i])
        nxt = float(close.iloc[i + 1])
        vf = float(vol_fcst.iloc[i]) if np.isfinite(vol_fcst.iloc[i]) else 0.0
        vp = float(vol_pct.iloc[i]) if np.isfinite(vol_pct.iloc[i]) else 0.5

        # --- manage open position ---
        if position != 0:
            bars_held += 1
            # stop = vol-scaled distance
            stop_dist = vol_k * vf * entry
            hit_stop = (position > 0 and nxt <= entry - stop_dist) or \
                       (position < 0 and nxt >= entry + stop_dist)
            if hit_stop or bars_held >= hold_bars:
                exit_price = entry - stop_dist if hit_stop else nxt
                gross = position * (exit_price - entry)
                notional = abs(position) * entry
                cost = notional * (cost_bps + spread_bps) / 10_000
                pnl = gross - cost
                balance += pnl
                pnls.append(pnl)
                position = 0
                bars_held = 0

        # --- try to enter ---
        if position == 0 and vf > 0:
            # vol-target band: entry only in high-vol regime (vp > 0.6)
            band = vol_k * vf * price
            ret1 = price / float(close.iloc[i - 1]) - 1
            signal = 0
            if vp > 0.6:  # high vol = our tradeable state
                # breakout: price moved > 0.5 band on last bar -> continue (momentum)
                if abs(ret1) > 0.5 * vf and ret1 > 0:
                    signal = 1
                elif abs(ret1) > 0.5 * vf and ret1 < 0:
                    signal = -1
                # mean-reversion tilt: extreme bar vs short drift -> fade
                drift = price / float(close.iloc[i - 5]) - 1
                if ret1 < -2 * vf and drift < 0:  # oversold vs drift -> buy fade
                    signal = 1
                elif ret1 > 2 * vf and drift > 0:  # overbought -> sell fade
                    signal = -1
            if signal != 0:
                stop_dist = vol_k * vf * price
                if stop_dist <= 0:
                    continue
                risk_budget = capital * risk_pct / 100
                units = int(risk_budget / stop_dist)
                # notional cap (GAP 4): never exceed 100% of capital notional
                max_units = int(capital / price)
                units = min(units, max_units)
                # cost gate: expected move must exceed round-trip cost
                exp_move = stop_dist  # we risk ~1x stop, expect to capture it
                if units > 0 and exp_move / price * 100 > (cost_bps + spread_bps) / 100:
                    position = units if signal > 0 else -units
                    entry = price
                    bars_held = 0

        equity.append(balance)

    eq = np.array(equity)
    total_ret = (balance / capital - 1) * 100
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak * 100).min()) if len(eq) else 0
    bh = (close.iloc[-1] / close.iloc[warmup] - 1) * 100
    wins = sum(1 for p in pnls if p > 0)
    winrate = wins / len(pnls) * 100 if pnls else 0
    return {
        "trades": len(pnls), "total_return_pct": round(total_ret, 2),
        "win_rate_pct": round(winrate, 1), "max_dd_pct": round(max_dd, 2),
        "buy_hold_pct": round(float(bh), 2), "final": round(balance, 0),
    }


if __name__ == "__main__":
    print("=== VOL-TARGET STRATEGY (trades the MEASURED vol signal) ===")
    print("costs: 25bps fees + 5bps spread (conservative, GAP 5)\n")
    for sym, label in [("RELIANCE.NS", "RELIANCE"), ("^NSEBANK", "BANKNIFTY"),
                       ("^NSEI", "NIFTY")]:
        print(f"--- {label} 15m ---")
        try:
            df = load(sym)
            r = run_vol_strategy(df)
            for k, v in r.items():
                print(f"    {k}: {v}")
            verdict = "BEATS CASH" if r["total_return_pct"] > 0 else "loses to cash"
            print(f"    >>> {verdict}\n")
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR: {e}\n")
