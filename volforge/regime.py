"""VOLFORGE regime engine — HMM market-regime detection.

Adapted from the proven `algo-fno-trading-bot` HMM engine (the field's reference
implementation), tuned for VOLFORGE. Core idea (validated by our Phase-1 diagnosis):
don't predict direction — detect the volatility/return REGIME and size by it.

Fits a GaussianHMM on [returns, volatility, rsi, adx], labels hidden states by
mean return (BULL/BEAR/SIDEWAYS), and exposes a position-size multiplier driven by
regime confidence. Walk-forward safe: train on past only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from volforge.ghmm import GaussianHMM as _GHMM

logger = logging.getLogger("volforge.regime")


class Regime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeConfig:
    n_regimes: int = 3
    lookback_days: int = 60
    retrain_every_days: int = 7
    min_regime_confidence: float = 0.55
    # position-size multiplier by regime (the vol-adaptive sizing)
    size_bull: float = 1.0
    size_sideways: float = 0.7
    size_bear: float = 0.5
    conf_full: float = 0.65  # confidence at which bull reaches full size


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(),
                                           (low - close.shift(1)).abs()))
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = tr.rolling(period).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / (atr + 1e-9)
    mdi = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / (atr + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return dx.rolling(period).mean()


def compute_features(df: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    """Feature matrix [returns, volatility, rsi, adx] from OHLCV (lowercase cols)."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()
    df["rsi"] = _rsi(df["close"])
    df["adx"] = _adx(df)
    feats = df[["returns", "volatility", "rsi", "adx"]].dropna()
    return feats.values, feats.index


class HMMRegimeEngine:
    """GaussianHMM regime detector with confidence-driven position sizing."""

    def __init__(self, config: RegimeConfig | None = None):
        self.cfg = config or RegimeConfig()
        self.model = None
        self.scaler = None
        self.regime_map: dict[int, Regime] = {}
        self.current_regime = Regime.UNKNOWN
        self.regime_confidence = 0.0

    def _normalize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler

        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def fit(self, df: pd.DataFrame) -> bool:
        try:
            X_raw, _ = compute_features(df)
            if len(X_raw) < self.cfg.n_regimes * 10:
                logger.warning("not enough data to fit HMM (%d rows)", len(X_raw))
                return False
            X = self._normalize(X_raw, fit=True)
            self.model = _GHMM(
                n_components=self.cfg.n_regimes, n_iter=100, tol=1e-4,
                random_state=42,
            ).fit(X)
            means = self.model.means_[:, 0]  # col 0 = returns
            order = np.argsort(means)  # low -> high mean return
            n = self.cfg.n_regimes
            if n == 3:
                self.regime_map = {order[0]: Regime.BEAR, order[1]: Regime.SIDEWAYS,
                                   order[2]: Regime.BULL}
            elif n == 2:
                self.regime_map = {order[0]: Regime.BEAR, order[1]: Regime.BULL}
            logger.info("HMM fit | regimes=%s", {k: v.value for k, v in self.regime_map.items()})
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("HMM fit failed: %s", e)
            return False

    def predict(self, df: pd.DataFrame) -> tuple[Regime, float]:
        if self.model is None:
            return Regime.UNKNOWN, 0.0
        try:
            window = df.tail(self.cfg.lookback_days + 20)
            X_raw, _ = compute_features(window)
            X = self._normalize(X_raw, fit=False)
            state = int(self.model.predict(X)[-1])
            regime = self.regime_map.get(state, Regime.UNKNOWN)
            conf = float(self.model.predict_proba(X)[-1, state])
            self.current_regime, self.regime_confidence = regime, conf
            return regime, conf
        except Exception as e:  # noqa: BLE001
            logger.error("HMM predict failed: %s", e)
            return Regime.UNKNOWN, 0.0

    def is_tradeable(self) -> bool:
        return (self.current_regime != Regime.UNKNOWN
                and self.regime_confidence >= self.cfg.min_regime_confidence)

    def position_size_multiplier(self) -> float:
        """Vol-adaptive sizing: scale by regime + confidence. 0 = stay flat."""
        c = self.cfg
        if self.current_regime == Regime.UNKNOWN:
            return 0.0
        if self.current_regime == Regime.BULL:
            return min(c.size_bull, self.regime_confidence / c.conf_full)
        if self.current_regime == Regime.SIDEWAYS:
            return c.size_sideways
        if self.current_regime == Regime.BEAR:
            return c.size_bear
        return 0.0
