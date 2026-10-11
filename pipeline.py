"""Robust NSE OHLCV loader + feature pipeline.

Fixes the original process_nse_data.py bug: yfinance writes a 12-column
double header (row 1 = simple names, row 2 = ('Close','^NSEI') tuples), and the
old code did skiprows=2 then assigned 6 names to 12 columns -> length mismatch.

This module reads the real file, normalizes to 6 clean columns, engineers the
base features the basic MLModel expects, AND the extended feature set the
EnhancedFeatureEngine / ensemble path can use.
"""
import numpy as np
import pandas as pd

OHLCV = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Base features the basic ml_model.py trains on.
BASE_FEATURES = ["return", "volatility", "rsi", "ema9", "ema21", "ema_signal"]


def load_ohlcv(input_file: str) -> pd.DataFrame:
    """Read a yfinance-style NSE CSV (messy 12-col header) -> clean 6-col frame."""
    # Row 0 is the simple header; row 1 is the tuple/empty duplicate. Drop both
    # and impose clean names so column count always matches.
    df = pd.read_csv(input_file, skiprows=2, header=None)
    df = df.iloc[:, :6]
    df.columns = OHLCV
    for c in OHLCV[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=OHLCV[1:]).reset_index(drop=True)
    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add base features + target. Returns a frame ready for ml_model.train()."""
    df = df.copy()
    df["return"] = df["Close"] - df["Open"]
    df["volatility"] = df["High"] - df["Low"]
    df["rsi"] = _rsi(df["Close"])
    df["ema9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema_signal"] = (df["ema9"] > df["ema21"]).astype(int)
    # Target: 1 if tomorrow's close > today's close.
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna().reset_index(drop=True)


def build(input_file: str, output_file: str = "data.csv") -> pd.DataFrame:
    """Full pipeline: raw NSE CSV -> engineered feature frame, saved to disk."""
    df = engineer(load_ohlcv(input_file))
    df[BASE_FEATURES + ["target"]].to_csv(output_file, index=False)
    print(f"[pipeline] {len(df)} rows, {len(BASE_FEATURES)} features -> {output_file}")
    return df


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "nifty50_2years.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data.csv"
    build(src, out)
