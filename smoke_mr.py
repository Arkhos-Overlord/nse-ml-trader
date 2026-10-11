"""Smoke test: mean-reversion on CLEAN INDmoney 5-min data (4 days).

This is NOT an edge declaration — 4 days is far too small a sample. It's a
pipeline validation: does the strategy run correctly on real synchronized data,
and are the per-trade numbers in the honest range (target ~+Rs1-2K, stop ~-Rs2-4K)?

The previous run showed Rs47Cr because the data was fake. On clean data the
numbers MUST collapse to realistic per-trade economics. If they don't, the sim
is still wrong.

Uses lot 65 (verified from Kite master — L10).
Run: python smoke_mr.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOT = 65                      # verified current lot size (L10)
BROKERAGE = 20
FUT_STT = 0.0001
FUT_EXCH = 0.00002
ENTRY_THRESH = 0.0015
STOP_LOSS = 0.0025
TARGET = 0.0010
SQUARE_OFF = "15:15"
WARMUP = 12                   # 12 x 5min bars = 1 hour warmup
CAPITAL = 150000.0
LOT_MARGIN = 150000.0

STORE = "data_indmoney/clean_1min.parquet"


def sim(df):
    """One-position-at-a-time mean-reversion on 5-min bars. Returns trades."""
    df = df.sort_values("datetime_ist").reset_index(drop=True)
    if len(df) < WARMUP + 5:
        return []
    trades = []
    pos, entry = 0, 0.0
    vw = []
    for i in range(len(df)):
        ts = df.datetime_ist.iloc[i]
        S = float(df.close.iloc[i])
        vw.append(S)
        if len(vw) > 20:
            vw.pop(0)
        if len(vw) < WARMUP:
            continue
        if ts.strftime("%H:%M") > "15:15" or ts.strftime("%H:%M") < "09:20":
            continue
        vwap = float(np.mean(vw))
        dev = (S - vwap) / vwap
        if pos != 0:
            move = (S - entry) / entry * pos
            if move >= TARGET or move <= -STOP_LOSS:
                pnl = (S - entry) * pos * LOT
                pnl -= 2 * BROKERAGE
                pnl -= abs(entry + S) * LOT * (FUT_STT + FUT_EXCH)
                trades.append(pnl)
                pos = 0
        if pos == 0:
            if dev > ENTRY_THRESH:
                pos, entry = -1, S
            elif dev < -ENTRY_THRESH:
                pos, entry = 1, S
        if ts.strftime("%H:%M") >= SQUARE_OFF and pos != 0:
            trades.append((S - entry) * pos * LOT - 2 * BROKERAGE)
            pos = 0
    return trades


def main():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    five = df[(df.interval == "5minute") & (df.label.str.contains("Fut"))]

    print("=== SMOKE TEST: mean-reversion on CLEAN 5-min futures (4 days) ===")
    print(f"  bars: {len(five)} | days: {five.datetime_ist.dt.date.nunique()}")
    print(f"  LOT={LOT} (verified) | target={TARGET*100:.2f}% stop={STOP_LOSS*100:.2f}%\n")

    all_trades = []
    for label, grp in five.groupby("label"):
        for d, g in grp.groupby(five.datetime_ist.dt.date):
            t = sim(g)
            all_trades.extend(t)
            if t:
                print(f"  {label} {d}: {len(t)} trades, Rs{sum(t):,.0f}, "
                      f"worst Rs{min(t):,.0f}, best Rs{max(t):,.0f}")

    if not all_trades:
        print("\n  no trades — thresholds too wide for 4 days")
        return
    arr = np.array(all_trades)
    wins = (arr > 0).sum()
    print(f"\n=== TRADE-LEVEL REALITY CHECK ===")
    print(f"  total trades: {len(arr)}")
    print(f"  win rate: {wins/len(arr)*100:.0f}%")
    print(f"  total PnL: Rs{arr.sum():,.0f}")
    print(f"  avg trade: Rs{arr.mean():,.0f}")
    print(f"  worst trade: Rs{arr.min():,.0f}")
    print(f"  best trade: Rs{arr.max():,.0f}")
    print(f"\n  HONESTY GATE:")
    print(f"    realistic per-trade range? target ~+Rs{65000*TARGET:,.0f}, "
          f"stop ~-Rs{65000*STOP_LOSS:,.0f}")
    realistic = abs(arr.max()) < 20000 and abs(arr.min()) < 20000
    print(f"    all trades within +-Rs20K? {'YES - data is clean' if realistic else 'NO - STILL FAKE'}")
    if not realistic:
        print("    => ABORT. Numbers outside realistic bounds means the data")
        print("       or sim is still wrong. Do NOT trust this result.")
        return
    print(f"\n  VERDICT: pipeline VALIDATED on clean data. Per-trade economics")
    print(f"  are realistic. But {len(five.datetime_ist.dt.date.unique())} days is")
    print(f"  NOT a sample — no edge claim until 30+ days.")


if __name__ == "__main__":
    main()
