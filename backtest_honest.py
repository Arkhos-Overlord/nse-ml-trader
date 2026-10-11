"""Honest walk-forward backtest — no look-ahead bias.

The original backtester.py trained on a random 80% split then 'backtested' on
rows whose real next-day target it already had, predicting with a model that
saw overlapping data -> a fake 100% win rate. This version:
  - expands a training window day by day (train on past only),
  - predicts tomorrow using ONLY information available today,
  - simulates a simple long/flat strategy with realistic per-trade costs.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["return", "volatility", "rsi", "ema9", "ema21", "ema_signal"]
COST_BPS = 15  # ~0.15% round-trip slippage+brokerage, conservative for F&O-ish


def walk_forward(df: pd.DataFrame, initial_balance: float = 100_000.0,
                 min_train: int = 200) -> dict:
    X = df[FEATURES].values
    y = df["target"].values
    close = df["Close"].values if "Close" in df.columns else None
    balance = initial_balance
    position = 0  # 0 flat, 1 long
    entry = 0.0
    trades = wins = 0
    equity_curve = []

    for i in range(min_train, len(df) - 1):
        model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                       random_state=42)
        model.fit(X[:i], y[:i])                 # PAST ONLY
        pred = int(model.predict(X[i:i + 1])[0])
        # Realized tomorrow: did close actually rise?
        nxt_up = int(y[i])
        # Strategy: go long if predicted up; exit next day regardless.
        pnl = 0.0
        if pred == 1 and position == 0:
            position = 1
            entry = balance
        elif position == 1:
            ret = 0.0 if close is None else (close[i + 1] / close[i] - 1.0)
            gross = entry * (ret if nxt_up == 1 else -abs(ret))
            cost = entry * COST_BPS / 10_000
            pnl = gross - cost
            balance += pnl
            trades += 1
            if pnl > 0:
                wins += 1
            position = 0
        equity_curve.append(balance)

    equity = np.array(equity_curve) if equity_curve else np.array([balance])
    total_ret = (balance / initial_balance - 1) * 100
    peak = np.maximum.accumulate(equity)
    max_dd = ((equity - peak) / peak * 100).min() if len(equity) else 0.0
    win_rate = (wins / trades * 100) if trades else 0.0

    return {
        "initial": initial_balance,
        "final": round(balance, 2),
        "total_return_pct": round(total_ret, 2),
        "trades": trades,
        "win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "buy_hold_return_pct": round(
            (df["Close"].iloc[-1] / df["Close"].iloc[min_train] - 1) * 100, 2)
        if "Close" in df.columns else None,
    }


if __name__ == "__main__":
    import pipeline

    df = pipeline.engineer(pipeline.load_ohlcv("nifty50_2years.csv"))
    r = walk_forward(df)
    print("\n=== HONEST WALK-FORWARD BACKTEST (no look-ahead) ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
    beat = r["total_return_pct"] > (r["buy_hold_return_pct"] or 0)
    print(f"\n  beats buy&hold: {beat}")
