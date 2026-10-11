"""Where does an edge actually exist? Scan instruments, walk-forward, no bias.

For each symbol: directional accuracy of a simple RF (train past only) and
lag-1 autocorr of 15m returns. Prints a ranked table. Read-only, no broker.
Answers 'point the new machine WHERE' before we build it.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

SYMBOLS = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "BANK NIFTY",
    "^CNXIT": "NIFTY IT",
    "^BSESN": "SENSEX",
    "RELIANCE.NS": "RELIANCE",
    "HDFCBANK.NS": "HDFC BANK",
    "INFY.NS": "INFOSYS",
    "TATAMOTORS.NS": "TATA MOTORS",
}


def feats(close):
    f = pd.DataFrame(index=close.index)
    f["ret1"] = close.pct_change()
    f["ret3"] = close.pct_change(3)
    f["ret12"] = close.pct_change(12)
    f["vol"] = close.pct_change().rolling(12).std()
    f["rsi"] = _rsi(close, 14)
    f["ema_fast"] = close.ewm(span=9).mean() / close - 1
    f["ema_slow"] = close.ewm(span=21).mean() / close - 1
    return f.dropna()


def _rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l)


def scan(sym, label):
    d = yf.download(sym, period="60d", interval="15m", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna()
    if len(d) < 400:
        return None
    close = d["Close"]
    ret = close.pct_change().dropna()
    acf1 = ret.autocorr(1)
    noise = 1.96 / np.sqrt(len(ret))
    X = feats(close).values
    y = (close.shift(-1) > close).astype(int)
    y = y.reindex(feats(close).index).dropna()
    X = feats(close).loc[y.index].values
    start = max(300, len(X) // 3)
    correct = total = 0
    for i in range(start, len(X) - 1):
        m = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1)
        m.fit(X[:i], y.values[:i])
        correct += int(m.predict(X[i:i + 1])[0] == y.values[i])
        total += 1
    acc = correct / total if total else 0
    return {"sym": label, "bars": len(d), "acf1": acf1, "noise": noise, "acc": acc}


if __name__ == "__main__":
    rows = []
    for sym, label in SYMBOLS.items():
        try:
            r = scan(sym, label)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  {label}: skipped ({e})")
    df = pd.DataFrame(rows).sort_values("acc", ascending=False)
    print("\n=== EDGE SCAN (15m, 60d, walk-forward) ===")
    print(f"{'instrument':<14}{'bars':>6}{'acf1':>9}{'noise':>8}{'dir_acc':>9}  verdict")
    for _, r in df.iterrows():
        v = "EDGE" if r.acc > 0.53 else ("weak" if r.acc > 0.51 else "none")
        star = " *acf" if abs(r.acf1) > r.noise else ""
        print(f"{r['sym']:<14}{r.bars:>6}{r.acf1:>+9.3f}{r.noise:>8.3f}{r.acc:>9.3f}  {v}{star}")
    print("\n(*acf = lag-1 autocorr above noise floor; dir_acc>0.53 = real edge)")
