"""VOLFORGE walk-forward backtest — the honest ledger.

Wires HMM regime → strategy signal → risk-sized position → SL/target exit, rolling
forward bar by bar (train on past only, no look-ahead). Reports net-of-cost return
vs cash AND buy&hold. If it can't beat cash, it says so — no fabrication.

This is an event-driven, bar-by-bar simulation (not vectorbt) so the regime model,
sizing, and stops interact exactly as they would live. Slow but honest.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from volforge.regime import HMMRegimeEngine, RegimeConfig
from volforge.risk import RiskManager, RiskConfig
from volforge.strategies import Signal, best_setup

logger = logging.getLogger("volforge.backtest")


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance frame to lowercase OHLCV."""
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.columns = [c.lower() for c in d.columns]
    return d.dropna()


def walk_forward(df: pd.DataFrame, *, min_train: int = 250, refit_every: int = 50,
                 hold_bars: int = 8, min_rr: float = 1.5,
                 risk_cfg: RiskConfig | None = None,
                 regime_cfg: RegimeConfig | None = None) -> dict:
    d = _prep(df)
    close = d["close"]
    n = len(d)
    min_train = min(min_train, max(60, n // 3))  # don't demand more train than data
    if n <= min_train + 1:
        return {"error": f"window too short ({n} bars, need > {min_train})",
                "total_return_pct": 0.0, "trades": 0, "win_rate_pct": 0.0,
                "max_drawdown_pct": 0.0, "sharpe": 0.0, "final": risk.cfg.total_capital,
                "initial": risk.cfg.total_capital,
                "buy_hold_return_pct": 0.0, "cash_return_pct": 0.0,
                "beats_cash": False, "beats_buy_hold": False}
    engine = HMMRegimeEngine(regime_cfg or RegimeConfig())
    risk = RiskManager(risk_cfg or RiskConfig())
    balance = risk.cfg.total_capital
    initial = balance
    equity = []
    position = 0  # units held (signed)
    entry = sl = tgt = 0.0
    bars_held = 0
    fitted_at = -10**9

    for i in range(min_train, n - 1):
        # refit regime periodically on PAST data only
        if i - fitted_at >= refit_every or engine.model is None:
            engine.fit(d.iloc[:i])
            fitted_at = i
        regime, conf = engine.predict(d.iloc[:i])
        mult = engine.position_size_multiplier()

        price = float(close.iloc[i])
        nxt = float(close.iloc[i + 1])

        # manage open position
        if position != 0:
            bars_held += 1
            hit_sl = (position > 0 and nxt <= sl) or (position < 0 and nxt >= sl)
            hit_tgt = (position > 0 and nxt >= tgt) or (position < 0 and nxt <= tgt)
            if hit_sl or hit_tgt or bars_held >= hold_bars:
                exit_price = sl if hit_sl else (tgt if hit_tgt else nxt)
                gross = position * (exit_price - entry)
                pnl = gross - risk.cost_of(abs(position) * entry)
                balance += pnl
                risk.on_trade_closed(pnl)
                position = 0
                bars_held = 0

        # try to open a new position (only flat, only if tradeable)
        if position == 0 and engine.is_tradeable() and risk.can_open():
            setup = best_setup(d.iloc[:i + 1], min_rr=min_rr)
            if setup.signal != Signal.NONE and setup.is_valid(min_rr):
                units = risk.position_size(setup.entry, setup.stop_loss, mult)
                # cost gate: expected move must exceed round-trip cost
                expected = setup.reward
                notional = units * setup.entry
                if units > 0 and expected > 0 and expected / (setup.entry + 1e-9) * 100 > \
                        risk.cfg.round_trip_cost_bps / 100:
                    position = units if setup.signal == Signal.BUY else -units
                    entry, sl, tgt = setup.entry, setup.stop_loss, setup.target
                    risk.state.open_positions += 1
                    bars_held = 0

        equity.append(balance)

    eq = np.array(equity) if equity else np.array([balance])
    total_ret = (balance / initial - 1) * 100
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak * 100).min()) if len(eq) else 0.0
    bh = (close.iloc[-1] / close.iloc[min_train] - 1) * 100
    win_rate = (risk.state.wins / risk.state.trades * 100) if risk.state.trades else 0.0
    sharpe = 0.0
    if len(eq) > 2:
        rets = np.diff(eq) / eq[:-1]
        sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252 * 25))

    return {
        "initial": initial, "final": round(balance, 2),
        "total_return_pct": round(total_ret, 2),
        "trades": risk.state.trades, "win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(max_dd, 2), "sharpe": round(sharpe, 2),
        "buy_hold_return_pct": round(float(bh), 2),
        "cash_return_pct": 0.0,
        "beats_cash": total_ret > 0, "beats_buy_hold": total_ret > float(bh),
    }


if __name__ == "__main__":
    import sys

    import pipeline

    src = sys.argv[1] if len(sys.argv) > 1 else "nifty50_2years.csv"
    df = pipeline.load_ohlcv(src)
    print(f"Loaded {len(df)} bars\n")
    r = walk_forward(df)
    print("=== VOLFORGE WALK-FORWARD (regime + strategy + risk, net of costs) ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
