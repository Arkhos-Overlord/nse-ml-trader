"""Phase 1 diagnosis: is there ANY exploitable signal in intraday Nifty?

Tests, in order:
  1. Return autocorrelation (lag-1..5) — the fastest tell for predictability.
  2. Trivial baselines: buy&hold, always-long, random. Beat these or don't bother.
  3. Does a simple ML model beat chance on 15m direction, walk-forward?
All read-only, no broker. Run: python diagnose.py
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def load(sym="^NSEI", interval="15m", period="60d"):
    d = yf.download(sym, period=period, interval=interval, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna()


def autocorr_test(df):
    close = df["Close"]
    ret = close.pct_change().dropna()
    print("=== 1. RETURN AUTOCORRELATION (signal = |acf| above noise floor) ===")
    n = len(ret)
    noise = 1.96 / np.sqrt(n)  # 95% band for white noise
    print(f"  n={n} bars, noise floor ±{noise:.4f}")
    for lag in range(1, 6):
        acf = ret.autocorr(lag)
        flag = "  <-- SIGNAL" if abs(acf) > noise else ""
        print(f"  lag-{lag} acf: {acf:+.4f}{flag}")
    # squared-return autocorr = volatility clustering (real, but not direction)
    print("  -- volatility clustering (|ret| acf) --")
    absret = ret.abs()
    for lag in range(1, 4):
        acf = absret.autocorr(lag)
        flag = "  <-- VOL CLUSTERS" if abs(acf) > noise else ""
        print(f"  |ret| lag-{lag} acf: {acf:+.4f}{flag}")


def baselines(df):
    close = df["Close"]
    ret = close.pct_change().dropna()
    print("\n=== 2. TRIVIAL BASELINES (total return over window) ===")
    bh = (close.iloc[-1] / close.iloc[0] - 1) * 100
    print(f"  buy & hold:        {bh:+.2f}%")
    # always long, entered each bar, cost 15bps
    cost = 0.0015
    always_long = ((1 + ret - cost).prod() - 1) * 100
    print(f"  always-long (net): {always_long:+.2f}%")
    print(f"  cash (do nothing):  0.00%")


def ml_edge(df):
    from sklearn.ensemble import RandomForestClassifier

    close = df["Close"]
    f = pd.DataFrame(index=df.index)
    f["ret1"] = close.pct_change()
    f["ret3"] = close.pct_change(3)
    f["ret12"] = close.pct_change(12)
    f["vol"] = close.pct_change().rolling(12).std()
    f["rsi"] = _rsi(close, 14)
    f["ema_fast"] = close.ewm(span=9).mean() / close - 1
    f["ema_slow"] = close.ewm(span=21).mean() / close - 1
    f = f.dropna()
    y = (close.shift(-1) > close).astype(int).reindex(f.index).dropna()
    f = f.loc[y.index]
    X = f.values

    print("\n=== 3. ML WALK-FORWARD (train on past only, predict next bar) ===")
    correct = total = 0
    start = max(300, len(X) // 3)
    for i in range(start, len(X) - 1):
        m = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42,
                                   n_jobs=-1)
        m.fit(X[:i], y.values[:i])
        pred = m.predict(X[i:i + 1])[0]
        correct += int(pred == y.values[i])
        total += 1
    acc = correct / total if total else 0
    print(f"  walk-forward directional accuracy: {acc:.3f}  ({correct}/{total})")
    print(f"  chance = 0.500  ->  {'EDGE' if acc > 0.53 else ('BREAKEVEN' if acc>0.50 else 'NO EDGE')}")


def _rsi(series, period=14):
    d = series.diff()
    gain = d.where(d > 0, 0).rolling(period).mean()
    loss = (-d.where(d < 0, 0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss)


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} 15m bars, {df.index[0]} -> {df.index[-1]}\n")
    autocorr_test(df)
    baselines(df)
    ml_edge(df)
