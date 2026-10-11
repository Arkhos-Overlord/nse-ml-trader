"""VOLFORGE risk overlay — position sizing + kill-switch.

Rails adapted from the field's proven risk manager (algo-fno-trading-bot), scaled to
₹10,000 capital. Enforces: per-trade risk cap, daily loss kill-switch, max open
positions, and regime-aware vol-target sizing. Returns position size in units, or 0
to stay flat.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskConfig:
    total_capital: float = 10_000.0        # ₹10K
    max_risk_per_trade_pct: float = 1.5    # ₹150 risk per trade
    max_daily_loss_pct: float = 3.0        # kill-switch at ₹300 daily loss
    max_open_positions: int = 2
    min_rr: float = 1.5
    round_trip_cost_bps: float = 15.0      # conservative F&O-ish costs


@dataclass
class RiskState:
    day_pnl: float = 0.0
    open_positions: int = 0
    kill_switch: bool = False
    trades: int = 0
    wins: int = 0

    def reset_day(self) -> None:
        self.day_pnl = 0.0
        self.kill_switch = False


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.cfg = config or RiskConfig()
        self.state = RiskState()

    def daily_loss_limit(self) -> float:
        return self.cfg.total_capital * self.cfg.max_daily_loss_pct / 100.0

    def risk_per_trade(self) -> float:
        return self.cfg.total_capital * self.cfg.max_risk_per_trade_pct / 100.0

    def can_open(self) -> bool:
        if self.state.kill_switch:
            return False
        if self.state.open_positions >= self.cfg.max_open_positions:
            return False
        if self.state.day_pnl <= -self.daily_loss_limit():
            self.state.kill_switch = True
            return False
        return True

    def position_size(self, entry: float, stop_loss: float, regime_mult: float,
                      max_notional_pct: float = 20.0) -> int:
        """Units to trade. size = (risk_budget × regime_mult) / stop_distance.

        regime_mult comes from the HMM engine (0 = flat). A hard notional cap
        (max_notional_pct of capital) stops a tiny stop from producing an absurd
        unit count. Costs are NOT subtracted here (handled in the backtest ledger)
        but the caller must still clear the cost gate before trading.
        """
        if not self.can_open() or regime_mult <= 0:
            return 0
        stop_dist = abs(entry - stop_loss)
        if stop_dist <= 0:
            return 0
        budget = self.risk_per_trade() * min(regime_mult, 1.0)
        units = int(budget / stop_dist)
        # hard notional cap so a near-zero stop can't lever up absurdly
        max_units = int((self.cfg.total_capital * max_notional_pct / 100.0) / entry)
        return max(min(units, max_units), 0)

    def on_trade_closed(self, pnl: float) -> None:
        self.state.day_pnl += pnl
        self.state.trades += 1
        if pnl > 0:
            self.state.wins += 1
        self.state.open_positions = max(0, self.state.open_positions - 1)
        if self.state.day_pnl <= -self.daily_loss_limit():
            self.state.kill_switch = True

    def cost_of(self, notional: float) -> float:
        return notional * self.cfg.round_trip_cost_bps / 10_000.0
