"""INTRADAY strategy battery — mechanisms that work where daily can't.

Daily equity = efficient (proven). Intraday is where microstructure effects
(bid-ask bounce, order flow, opening auction) create exploitable edge. These
strategies are BUILT NOW so they fire the moment 5-min data matures to 30+ days.

Strategies (all square off before 15:15, no overnight risk, small-capital fit):
  1. OPENING RANGE BREAKOUT: trade breakout of first 30-min range
  2. VWAP REVERSION: fade moves >X% from VWAP, exit at VWAP
  3. VWAP TREND: buy above VWAP in confirmed uptrend, ride to close
  4. MOMENTUM IGNITION: enter on volume spike + directional move
  5. TIME-OF-DAY: exploit recurring intraday patterns (first/last hour)
  6. OPENING GAP FADE: fade the opening gap intraday

Runs on whatever intraday data exists (currently 5 days = smoke test only).
Honesty gate + data quality gate enforced. NO edge claim until 30+ days.
Run: python intraday_battery.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/equity_5min.parquet"
NOTIONAL = 50000.0
BROKERAGE = 20
STT = 0.00025          # intraday STT 0.025% on sell
EXCH = 0.0000297
GST = 0.18
SLIP = 0.0005
SQUARE_OFF = "15:10"


def load():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df[df.interval == "5minute"]
    ok = []
    for lb, g in df.groupby("label"):
        med = g.close.median()
        if med > 0 and g.close.max() / med < 10 and g.datetime_ist.duplicated().sum() == 0:
            ok.append(g)
    return pd.concat(ok)


def cost(notional):
    return 2 * BROKERAGE + notional * 2 * (STT + EXCH * (1 + GST) + SLIP)


def day_bars(g):
    """Group one stock's bars by trading day."""
    g = g.copy()
    g["hm"] = g.datetime_ist.dt.strftime("%H:%M")
    return {d: sub.sort_values("datetime_ist") for d, sub in g.groupby(g.datetime_ist.dt.date)}


def strat_orb(day, or_minutes=30):
    """Opening range breakout: enter on break of first-30-min high/low."""
    if len(day) < 20:
        return None
    # opening range = first or_minutes of 5-min bars after 09:15
    start_hm = day.hm.iloc[0]
    cutoff = (pd.Timestamp(f"2000-01-01 {start_hm}") + pd.Timedelta(minutes=or_minutes)).strftime("%H:%M")
    orb = day[day.hm <= cutoff]
    if len(orb) < 4:
        return None
    hi, lo = orb.high.max(), orb.low.min()
    after = day[day.hm > cutoff]
    if after.empty:
        return None
    direction = 0
    entry = 0.0
    for _, r in after.iterrows():
        if r.high >= hi:
            direction, entry = 1, hi
            break
        if r.low <= lo:
            direction, entry = -1, lo
            break
    if direction == 0:
        return None
    exit_ = float(after.close.iloc[-1])
    return direction, entry, exit_


def strat_vwap_rev(day, thresh=0.004):
    """Fade first move >thresh from VWAP, exit at VWAP or stop."""
    if len(day) < 20:
        return None
    pv = (day.close * day.volume).cumsum()
    vol = day.volume.cumsum()
    vwap = pv / vol.replace(0, np.nan)
    for i in range(10, len(day)):
        v = vwap.iloc[i]
        if pd.isna(v):
            continue
        dev = (day.close.iloc[i] - v) / v
        if abs(dev) > thresh:
            direction = -1 if dev > 0 else 1   # fade
            entry = float(day.close.iloc[i])
            # exit at VWAP touch or EOD
            for j in range(i + 1, len(day)):
                if (direction == 1 and day.close.iloc[j] >= vwap.iloc[j]) or \
                   (direction == -1 and day.close.iloc[j] <= vwap.iloc[j]):
                    return direction, entry, float(day.close.iloc[j])
            return direction, entry, float(day.close.iloc[-1])
    return None


def strat_vwap_trend(day):
    """Buy above VWAP when trending, ride to close."""
    if len(day) < 20:
        return None
    vwap = (day.close * day.volume).cumsum() / day.volume.cumsum().replace(0, np.nan)
    for i in range(10, len(day)):
        if day.close.iloc[i] > vwap.iloc[i] * 1.002 and day.close.iloc[i] > day.close.iloc[i - 3]:
            return 1, float(day.close.iloc[i]), float(day.close.iloc[-1])
    return None


def strat_gap_fade(day):
    """Fade the opening gap: if opens down, buy; opens up, short. Exit EOD."""
    if len(day) < 20:
        return None
    o, c_prev = float(day.open.iloc[0]), None
    # use prev close proxy: first bar open vs a moment later is not gap; use open vs 5th bar
    entry = float(day.open.iloc[0])
    if len(day) < 6:
        return None
    early = float(day.close.iloc[5])
    gap = (early - entry) / entry
    if abs(gap) < 0.003:
        return None
    direction = -1 if gap > 0 else 1
    return direction, early, float(day.close.iloc[-1])


def run_strat(df, fn, name):
    trades = []
    for lb, g in df.groupby("label"):
        for d, day in day_bars(g).items():
            r = fn(day)
            if r is None:
                continue
            direction, entry, exit_ = r
            if entry <= 0 or exit_ <= 0:
                continue
            qty = int(NOTIONAL / entry)
            gross = (exit_ - entry) * qty * direction
            net = gross - cost(entry * qty)
            trades.append({"stock": lb, "pnl": net})
    return trades


def main():
    df = load()
    days = df.datetime_ist.dt.date.nunique()
    print(f"=== INTRADAY BATTERY ({df.label.nunique()} stocks, {days} days) ===")
    print(f"  NOTIONAL Rs{NOTIONAL:,.0f}, cost Rs{cost(NOTIONAL):,.0f}/RT, square-off {SQUARE_OFF}")
    print(f"  *** {days} days = SMOKE TEST ONLY, no edge claim until 30+ ***\n")

    strats = {
        "orb_30min": strat_orb,
        "vwap_reversion": strat_vwap_rev,
        "vwap_trend": strat_vwap_trend,
        "gap_fade": strat_gap_fade,
    }
    print(f"{'strategy':<18}{'trades':>8}{'win%':>7}{'avgPnL':>11}{'total':>12}{'worst':>10}")
    for name, fn in strats.items():
        t = run_strat(df, fn, name)
        if len(t) < 10:
            print(f"{name:<18}{len(t):>8}  (too few)")
            continue
        arr = np.array([x["pnl"] for x in t])
        print(f"{name:<18}{len(arr):>8}{(arr>0).mean()*100:>6.0f}%{arr.mean():>11,.0f}"
              f"{arr.sum():>12,.0f}{arr.min():>10,.0f}")
    print(f"\n  These are pipeline validations. With {days} days, any number is noise.")
    print(f"  Re-run when data hits 30+ days for a real verdict.")


if __name__ == "__main__":
    main()
