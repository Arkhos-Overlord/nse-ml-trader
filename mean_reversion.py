"""INTRADAY NIFTY MEAN-REVERSION SCALPER — futures, no theta, fits small capital.

The options-selling path is dead (proven: trend days bleed -Rs18K to -Rs48K).
This is a DIFFERENT mechanism: intraday mean-reversion on Nifty itself via
futures. No gamma, no theta — tail is a single tick stop-loss, not a gap.

Logic:
  Reconstruct Nifty spot each minute from put-call parity (S ≈ K + CE - PE at ATM).
  Track rolling VWAP. If spot deviates > threshold, take a futures position
  betting on reversion. Exit on reversion, stop-loss, or square-off 15:15.
  No overnight exposure = no gap risk.

Tested on real 1-min option data. Futures costs applied.
Run: python mean_reversion.py
"""
import math
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

LOT = 75
BROKERAGE = 20
FUTURE_STT = 0.0001      # 0.01% STT on sell side (futures)
FUTURE_EXCH = 0.00002    # ~0.002% exchange txn charge
ENTRY_THRESH = 0.0012    # 0.12% deviation from VWAP to enter
STOP_LOSS = 0.0020       # 0.20% stop-loss
TARGET = 0.0008          # 0.08% reversion target
SQUARE_OFF = "15:15"
WARMUP_MINS = 15


def load_day(pf, target_date):
    frames = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["date", "timestamp", "strike",
                                           "option_type", "close", "volume"])
        d = t.to_pandas()
        d = d[d["date"] == target_date]
        if not d.empty:
            frames.append(d)
    return pd.concat(frames) if frames else None


def reconstruct_spot(day):
    """Per-minute Nifty spot from ATM put-call parity: S ≈ K + (CE - PE)."""
    day = day[day["volume"] > 0]
    if day.empty:
        return None
    piv = day.pivot_table(index="timestamp", columns=["strike", "option_type"],
                          values="close", aggfunc="last")
    t0 = piv.index[0]
    ce = {s: piv.loc[t0, (s, "CE")] for s in day["strike"].unique()
          if (s, "CE") in piv.columns}
    pe = {s: piv.loc[t0, (s, "PE")] for s in day["strike"].unique()
          if (s, "PE") in piv.columns}
    common = [s for s in ce if s in pe and ce[s] > 0 and pe[s] > 0]
    if not common:
        return None
    atm = min(common, key=lambda s: ce[s] + pe[s])
    rows = []
    for ts in piv.index:
        try:
            c = piv.loc[ts, (atm, "CE")]
            p = piv.loc[ts, (atm, "PE")]
        except KeyError:
            continue
        if pd.isna(c) or pd.isna(p):
            continue
        S = atm + (c - p)  # parity: S ≈ K + C - P
        if S > 0:
            rows.append((ts, S))
    if len(rows) < 30:
        return None
    ser = pd.DataFrame(rows, columns=["ts", "S"]).set_index("ts").sort_index()
    ser = ser[ser.index.time >= pd.Timestamp("09:15").time()]
    ser = ser[ser.index.time <= pd.Timestamp(SQUARE_OFF).time()]
    return ser


def sim_day(ser):
    """Mean-reversion scalper. Returns (pnl, n_trades)."""
    if len(ser) < WARMUP_MINS + 10:
        return 0.0, 0
    pnl = 0.0
    n_trades = 0
    pos = 0          # +1 long, -1 short, 0 flat
    entry_price = 0.0
    vwap_window = []
    for i, (ts, S) in enumerate(zip(ser.index, ser["S"])):
        vwap_window.append(S)
        if len(vwap_window) > 30:
            vwap_window.pop(0)
        if len(vwap_window) < WARMUP_MINS:
            continue
        vwap = np.mean(vwap_window)
        dev = (S - vwap) / vwap

        # manage open position
        if pos != 0:
            move = (S - entry_price) / entry_price * pos
            # target hit -> take profit
            if move >= TARGET:
                pnl += (S - entry_price) * pos * LOT
                n_trades += 1
                pos = 0
                continue
            # stop hit
            if move <= -STOP_LOSS:
                pnl += (S - entry_price) * pos * LOT
                n_trades += 1
                pos = 0
                continue
        # entry signals (only if flat)
        if pos == 0:
            if dev > ENTRY_THRESH:      # above VWAP -> short, expect reversion down
                pos = -1
                entry_price = S
            elif dev < -ENTRY_THRESH:   # below VWAP -> long, expect reversion up
                pos = 1
                entry_price = S
        # square off at close
        if ts.strftime("%H:%M") >= SQUARE_OFF and pos != 0:
            pnl += (S - entry_price) * pos * LOT
            n_trades += 1
            pos = 0

    # costs: ~2 legs per trade (entry + exit) on futures turnover
    turnover = abs(n_trades) * entry_price * LOT * 2
    pnl -= n_trades * 2 * BROKERAGE
    pnl -= turnover * (FUTURE_STT + FUTURE_EXCH)
    return pnl, n_trades


def main():
    import sys
    year = sys.argv[1] if len(sys.argv) > 1 else "2024"
    f = f"NIFTY_intraday_{year}.parquet"
    pf = pq.ParquetFile(f)
    dates = set()
    for rg in range(min(5, pf.metadata.num_row_groups)):
        t = pf.read_row_group(rg, columns=["date"])
        dates.update(t.to_pandas()["date"].unique())
    dates = sorted(dates)
    print(f"=== {year}: {len(dates)} expiry days ===")

    results = []
    for dt in dates:
        day = load_day(pf, dt)
        if day is None:
            continue
        ser = reconstruct_spot(day)
        if ser is None:
            continue
        pnl, n = sim_day(ser)
        results.append({"date": dt, "pnl": pnl, "trades": n})

    if results:
        t = pd.DataFrame(results)
        print(f"  days traded: {len(t)}")
        print(f"  total PnL: Rs{t.pnl.sum():,.0f}")
        print(f"  avg/day: Rs{t.pnl.mean():,.0f}")
        print(f"  win days: {(t.pnl > 0).mean()*100:.0f}%")
        print(f"  worst day: Rs{t.pnl.min():,.0f}")
        print(f"  best day: Rs{t.pnl.max():,.0f}")
        print(f"  avg trades/day: {t.trades.mean():.1f}")
        worst = abs(t.pnl.min())
        print(f"\n  Rs10K: worst = {worst/10000*100:.0f}% "
              f"({'SURVIVES' if worst < 10000 else 'RUINED'})")
        print(f"  Rs1.5L: worst = {worst/150000*100:.1f}% "
              f"({'SURVIVES' if worst < 150000 else 'RUINED'})")
        print(f"  monthly (20d): Rs{t.pnl.mean()*20:,.0f}")
        print(f"  annualized: Rs{t.pnl.mean()*250:,.0f}")
        # equity curve on 1.5L
        cap = 150000
        peak = cap
        maxdd = 0
        for p in t.pnl:
            cap += p
            peak = max(peak, cap)
            maxdd = max(maxdd, (peak - cap) / peak * 100)
        print(f"  Rs1.5L equity curve: Rs{cap:,.0f} ({(cap/150000-1)*100:+.1f}%), maxDD {maxdd:.1f}%")
        print(f"\n  VERDICT: {'PROFITABLE + FITS' if t.pnl.sum()>0 and worst<150000 else 'not yet'}")


if __name__ == "__main__":
    main()
