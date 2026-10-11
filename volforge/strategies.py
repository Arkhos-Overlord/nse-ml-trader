"""VOLFORGE intraday strategies — ORB / VWAP-reversion / EMA-momentum.

Adapted from the field's proven strategy set (algo-fno-trading-bot). Every setup is
wrapped in a TradeSetup and gated by a minimum reward:risk before it can be traded.
These produce SIGNALS; the engine applies regime-based sizing and the risk overlay.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


@dataclass
class TradeSetup:
    signal: Signal
    entry: float
    stop_loss: float
    target: float
    strategy: str
    confidence: float = 0.5

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def reward(self) -> float:
        return abs(self.target - self.entry)

    @property
    def rr_ratio(self) -> float:
        return self.reward / (self.risk + 1e-9)

    def is_valid(self, min_rr: float = 1.5) -> bool:
        return self.rr_ratio >= min_rr and self.risk > 0


def _vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return (typical * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)


def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def orb_setup(df: pd.DataFrame, orb_candles: int = 3, buffer: float = 5.0,
              reward_mult: float = 2.0) -> TradeSetup:
    """Opening Range Breakout on the last `orb_candles` bars of the frame tail."""
    if len(df) < orb_candles + 1:
        return TradeSetup(Signal.NONE, 0, 0, 0, "ORB", 0.0)
    opening = df.iloc[-(orb_candles + 1):-1]
    hi, lo = opening["high"].max(), opening["low"].min()
    last = float(df["close"].iloc[-1])
    if last > hi + buffer:
        entry = last
        sl = lo
        tgt = entry + reward_mult * (entry - sl)
        return TradeSetup(Signal.BUY, entry, sl, tgt, "ORB", 0.6)
    if last < lo - buffer:
        entry = last
        sl = hi
        tgt = entry - reward_mult * (sl - entry)
        return TradeSetup(Signal.SELL, entry, sl, tgt, "ORB", 0.6)
    return TradeSetup(Signal.NONE, last, last, last, "ORB", 0.0)


def vwap_reversion_setup(df: pd.DataFrame, dev_pct: float = 0.003) -> TradeSetup:
    """Fade a ≥dev_pct deviation from VWAP with RSI confirmation."""
    if len(df) < 20:
        return TradeSetup(Signal.NONE, 0, 0, 0, "VWAP", 0.0)
    vwap = float(_vwap(df).iloc[-1])
    last = float(df["close"].iloc[-1])
    rsi = float(_rsi(df["close"]).iloc[-1])
    dev = (last - vwap) / (vwap + 1e-9)
    if dev <= -dev_pct and rsi < 35:  # oversold vs vwap -> buy fade
        sl = float(df["low"].iloc[-5:].min())
        tgt = vwap
        return TradeSetup(Signal.BUY, last, sl, tgt, "VWAP", 0.55)
    if dev >= dev_pct and rsi > 65:  # overbought vs vwap -> sell fade
        sl = float(df["high"].iloc[-5:].max())
        tgt = vwap
        return TradeSetup(Signal.SELL, last, sl, tgt, "VWAP", 0.55)
    return TradeSetup(Signal.NONE, last, last, last, "VWAP", 0.0)


def ema_momentum_setup(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> TradeSetup:
    """EMA cross with RSI-not-extreme filter."""
    if len(df) < slow + 2:
        return TradeSetup(Signal.NONE, 0, 0, 0, "EMA", 0.0)
    close = df["close"]
    ef, es = _ema(close, fast), _ema(close, slow)
    rsi = float(_rsi(close).iloc[-1])
    last = float(close.iloc[-1])
    cross_up = ef.iloc[-2] <= es.iloc[-2] and ef.iloc[-1] > es.iloc[-1]
    cross_dn = ef.iloc[-2] >= es.iloc[-2] and ef.iloc[-1] < es.iloc[-1]
    atr = float((df["high"] - df["low"]).rolling(14).mean().iloc[-1])
    if cross_up and rsi < 70:
        return TradeSetup(Signal.BUY, last, last - 1.5 * atr, last + 3.0 * atr, "EMA", 0.55)
    if cross_dn and rsi > 30:
        return TradeSetup(Signal.SELL, last, last + 1.5 * atr, last - 3.0 * atr, "EMA", 0.55)
    return TradeSetup(Signal.NONE, last, last, last, "EMA", 0.0)


STRATEGIES = {"ORB": orb_setup, "VWAP": vwap_reversion_setup, "EMA": ema_momentum_setup}


def best_setup(df: pd.DataFrame, min_rr: float = 1.5) -> TradeSetup:
    """Return the highest-R:R valid setup across all strategies, else NONE."""
    candidates = [fn(df) for fn in STRATEGIES.values()]
    valid = [s for s in candidates if s.signal != Signal.NONE and s.is_valid(min_rr)]
    if not valid:
        return TradeSetup(Signal.NONE, 0, 0, 0, "none", 0.0)
    return max(valid, key=lambda s: s.rr_ratio)
