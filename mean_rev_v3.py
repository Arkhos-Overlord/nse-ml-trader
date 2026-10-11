"""CAPITAL-CONSTRAINED mean-reversion — HONEST version, minute-resolution.

Bugs fixed from v2 (which falsely showed Rs47 crore):
  1. Ran on EVERY parquet row (many per minute) -> sampled now to ONE bar/minute.
  2. Re-entered same-signal repeatedly -> now ONE position at a time.
  3. This matches reality: Nifty intraday trades in minutes; you cannot catch a
     0.08% move more than once per minute.

Reality check on ONE trade: target 0.08% of Nifty ~23000 = ~18 pts * 75 = Rs~1350.
Stop 0.20% = ~46 pts * 75 = Rs~3450. So a trade is +Rs1350 or -Rs3450.
That is the honest per-trade economics. Run: python mean_rev_v3.py 2024
"""
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

LOT = 75
BROKERAGE = 20
FUTURE_STT = 0.0001
FUTURE_EXCH = 0.00002
ENTRY_THRESH = 0.0012
STOP_LOSS = 0.0020
TARGET = 0.0008
SQUARE_OFF = "15:15"
WARMUP_MINS = 15
CAPITAL = 150000.0
LOT_MARGIN = 150000.0


def load_day(pf, target_date):
    frames = []
    for rg in range(pf.metadata.num_row_groups):
        d = pf.read_row_group(rg, columns=["date", "timestamp", "strike",
                                           "option_type", "close", "volume"]).to_pandas()
        d = d[d["date"] == target_date]
        if not d.empty:
            frames.append(d)
    return pd.concat(frames) if frames else None


def reconstruct_spot(day):
    day = day[day["volume"] > 0]
    if day.empty:
        return None
    piv = day.pivot_table(index="timestamp", columns=["strike", "option_type"],
                          values="close", aggfunc="last")
    t0 = piv.index[0]
    ce = {s: piv.loc[t0, (s, "CE")] for s in day["strike"].unique() if (s, "CE") in piv.columns}
    pe = {s: piv.loc[t0, (s, "PE")] for s in day["strike"].unique() if (s, "PE") in piv.columns}
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
        S = atm + (c - p)
        if S > 0:
            rows.append((ts, S))
    if len(rows) < 30:
        return None
    ser = pd.DataFrame(rows, columns=["ts", "S"]).set_index("ts").sort_index()
    ser = ser[ser.index.time >= pd.Timestamp("09:15").time()]
    ser = ser[ser.index.time <= pd.Timestamp(SQUARE_OFF).time()]
    # SAMPLE to one bar per minute — the honest resolution
    ser = ser.resample("1min").last().dropna()
    return ser


def sim_day(ser):
    if len(ser) < WARMUP_MINS + 10:
        return []
    trades = []
    pos = 0
    entry = 0.0
    vwap_window = []
    for ts, S in zip(ser.index, ser["S"]):
        vwap_window.append(S)
        if len(vwap_window) > 30:
            vwap_window.pop(0)
        if len(vwap_window) < WARMUP_MINS:
            continue
        vwap = np.mean(vwap_window)
        dev = (S - vwap) / vwap
        if pos != 0:
            move = (S - entry) / entry * pos
            if move >= TARGET or move <= -STOP_LOSS:
                pnl = (S - entry) * pos * LOT
                pnl -= 2 * BROKERAGE
                pnl -= abs(entry + S) * LOT * FUTURE_STT
                pnl -= abs(entry + S) * LOT * FUTURE_EXCH
                trades.append(pnl)
                pos = 0
        if pos == 0:
            if dev > ENTRY_THRESH:
                pos, entry = -1, S
            elif dev < -ENTRY_THRESH:
                pos, entry = 1, S
        if ts.strftime("%H:%M") >= SQUARE_OFF and pos != 0:
            pnl = (S - entry) * pos * LOT - 2 * BROKERAGE
            trades.append(pnl)
            pos = 0
    return trades


def main():
    import sys
    year = sys.argv[1] if len(sys.argv) > 1 else "2024"
    pf = pq.ParquetFile(f"NIFTY_intraday_{year}.parquet")
    dates = set()
    for rg in range(min(5, pf.metadata.num_row_groups)):
        dates.update(pf.read_row_group(rg, columns=["date"]).to_pandas()["date"].unique())
    dates = sorted(dates)

    capital = CAPITAL
    peak = CAPITAL
    maxdd = 0
    all_trades = []
    daily = []
    for dt in dates:
        day = load_day(pf, dt)
        if day is None:
            continue
        ser = reconstruct_spot(day)
        if ser is None:
            continue
        trades = sim_day(ser)
        day_pnl = sum(trades)
        capital += day_pnl
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak * 100)
        all_trades.extend(trades)
        daily.append({"date": dt, "pnl": day_pnl, "capital": capital})

    t = pd.DataFrame(daily)
    if t.empty:
        print("no data")
        return
    wins = sum(1 for x in all_trades if x > 0)
    print(f"=== HONEST MEAN-REVERSION ({year}, 1 lot, 1-min resolution) ===")
    print(f"  days: {len(t)} | total trades: {len(all_trades)} "
          f"({len(all_trades)/len(t):.1f}/day)")
    print(f"  trade win rate: {wins/len(all_trades)*100:.0f}%")
    print(f"  start capital: Rs{CAPITAL:,.0f}")
    print(f"  end capital:   Rs{capital:,.0f}")
    print(f"  total return:  {(capital/CAPITAL-1)*100:+.1f}%")
    print(f"  avg/day: Rs{t.pnl.mean():,.0f}")
    print(f"  day win rate: {(t.pnl > 0).mean()*100:.0f}%")
    print(f"  worst day: Rs{t.pnl.min():,.0f} | best day: Rs{t.pnl.max():,.0f}")
    print(f"  max drawdown: {maxdd:.1f}%")
    print(f"  MIN capital: Rs{t.capital.min():,.0f}")
    ruined = (t.capital < LOT_MARGIN).any()
    print(f"  RUINED (< 1 lot margin)? {'YES' if ruined else 'NO'}")
    print(f"  monthly (20d): Rs{t.pnl.mean()*20:,.0f} | annualized: Rs{t.pnl.mean()*250:,.0f}")
    print(f"\n  per-trade reality: target ~+Rs1350, stop ~-Rs3450 (0.08%/0.20% of 23000)")
    print(f"  VERDICT: {'PROFITABLE + SURVIVES Rs1.5L' if capital>CAPITAL and not ruined and maxdd<30 else 'NOT SAFE/UNPROFITABLE'}")


if __name__ == "__main__":
    main()
