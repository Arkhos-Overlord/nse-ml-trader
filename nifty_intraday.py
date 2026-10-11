"""NIFTY INDEX intraday engine — different microstructure than stocks.

Stocks = efficient daily. The INDEX is different: tighter spreads, stronger
intraday mean-reversion, behaves as its own asset. Test index-specific intraday
edges on clean 5-min Nifty data:

  1. VWAP reversion (index reverts to VWAP harder than stocks)
  2. Opening-drift fade
  3. Band breakout (Bollinger-style on 5-min)
  4. Time-of-day drift (systematic index moves at open/close)

Honesty gate + no-edge-until-30-days. Run: python nifty_intraday.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/clean_1min.parquet"
NOTIONAL = 100000.0   # index futures need more notional; or use options
BROKERAGE = 20
FUT_STT = 0.0001
FUT_EXCH = 0.00002
SLIP = 0.0002


def cost(notional):
    return 2 * BROKERAGE + notional * 2 * (FUT_STT + FUT_EXCH + SLIP)


def load():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df[df.interval == "5minute"]
    df = df[df.label.str.contains("Fut|NIFTY")]
    ok = []
    for lb, g in df.groupby("label"):
        med = g.close.median()
        if med > 0 and g.close.max() / med < 10 and g.datetime_ist.duplicated().sum() == 0:
            ok.append(g)
    return pd.concat(ok)


def vwap_series(day):
    return (day.close * day.volume).cumsum() / day.volume.cumsum().replace(0, np.nan)


def strat_vwap_rev(day, thresh=0.002):
    if len(day) < 20:
        return None
    v = vwap_series(day)
    for i in range(10, len(day)):
        if pd.isna(v.iloc[i]):
            continue
        dev = (day.close.iloc[i] - v.iloc[i]) / v.iloc[i]
        if abs(dev) > thresh:
            direction = -1 if dev > 0 else 1
            entry = float(day.close.iloc[i])
            for j in range(i + 1, len(day)):
                if (direction == 1 and day.close.iloc[j] >= v.iloc[j]) or \
                   (direction == -1 and day.close.iloc[j] <= v.iloc[j]):
                    return direction, entry, float(day.close.iloc[j])
            return direction, entry, float(day.close.iloc[-1])
    return None


def strat_band(day, lookback=20, k=2.0):
    """Bollinger band fade: enter at band, exit at midline."""
    if len(day) < lookback + 10:
        return None
    c = day.close.values
    for i in range(lookback, len(day)):
        win = c[i - lookback:i]
        mu, sd = win.mean(), win.std()
        if sd == 0:
            continue
        z = (c[i] - mu) / sd
        if z > k:
            return -1, float(c[i]), float(mu)      # short rich, exit at mean
        if z < -k:
            return 1, float(c[i]), float(mu)       # long cheap, exit at mean
    return None


def strat_open_drift(day):
    """Fade the first 15-min drift."""
    if len(day) < 12:
        return None
    o = float(day.close.iloc[0])
    p15 = float(day.close.iloc[3])   # ~15 min in
    drift = (p15 - o) / o
    if abs(drift) < 0.001:
        return None
    direction = -1 if drift > 0 else 1
    return direction, p15, float(day.close.iloc[-1])


def run(df, fn):
    trades = []
    for lb, g in df.groupby("label"):
        g = g.copy()
        g["hm"] = g.datetime_ist.dt.strftime("%H:%M")
        for d, day in g.groupby(g.datetime_ist.dt.date):
            day = day.sort_values("datetime_ist")
            r = fn(day)
            if r is None:
                continue
            direction, entry, exit_ = r
            if entry <= 0 or exit_ <= 0:
                continue
            qty = int(NOTIONAL / entry)
            gross = (exit_ - entry) * qty * direction
            trades.append(gross - cost(entry * qty))
    return trades


def main():
    df = load()
    days = df.datetime_ist.dt.date.nunique()
    print(f"=== NIFTY INDEX/FUTURES INTRADAY ({df.label.nunique()} instruments, {days} days) ===")
    print(f"  NOTIONAL Rs{NOTIONAL:,.0f}, cost Rs{cost(NOTIONAL):,.0f}/RT")
    print(f"  *** {days} days = SMOKE, no edge claim until 30+ ***\n")
    for name, fn in [("vwap_reversion", strat_vwap_rev),
                     ("band_fade", strat_band),
                     ("open_drift_fade", strat_open_drift)]:
        t = run(df, fn)
        if len(t) < 8:
            print(f"  {name:<18} {len(t)} trades (too few)")
            continue
        arr = np.array(t)
        print(f"  {name:<18} {len(arr):>4} trades {(arr>0).mean()*100:>4.0f}% win "
              f"avg Rs{arr.mean():>8,.0f} total Rs{arr.sum():>10,.0f}")


if __name__ == "__main__":
    main()
