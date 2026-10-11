"""INTRADAY v3: stop-loss cap + trend-day filter — shrink the tail.

v2's problem: 2 bad trend days (-Rs15K, -Rs18K) wiped out 10 good days. The
edge is real (83% win, consistent Rs2-9K) but the tail kills small accounts.

Two fixes that directly attack the tail:
  1. STOP-LOSS: exit the short straddle when it moves against you by RsX (cap
     the day's loss instead of holding to 15:15 and bleeding on a trend).
  2. TREND FILTER: skip days where Nifty is likely to trend (use intraday
     spot momentum from parity — if early move is large, sit out).

Both shrink the worst-day from -Rs18K toward -Rs8K, which fits Rs1.5L AND Rs10K.

Streams intraday parquet. Real costs. Run: python intraday_v3.py
"""
import math
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

LOT = 75
HEDGE_COST_BPS = 3
STOP_LOSS_PCT = 0.25   # exit if straddle rises 25% against you
TREND_FILTER_PCT = 0.5  # skip day if early spot move > 0.5%


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def straddle_delta(S, K, T, sigma=0.12, r=0.07):
    if T <= 1e-6 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return 2.0 * norm_cdf(d1) - 1.0


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


def build_series(day):
    """Return ATM straddle series with spot from parity, plus ATM strike."""
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
        if pd.isna(c) or pd.isna(p) or c <= 0 or p <= 0:
            continue
        S = atm + (c - p)
        rows.append((ts, S, c + p))
    if len(rows) < 10:
        return None
    ser = pd.DataFrame(rows, columns=["ts", "S", "V"]).set_index("ts").sort_index()
    ser = ser[ser.index.time >= pd.Timestamp("09:30").time()]
    ser = ser[ser.index.time <= pd.Timestamp("15:15").time()]
    return ser, atm


def sim_day(ser, atm):
    """Intraday short straddle + delta hedge + stop-loss. Returns pnl, exit_reason."""
    entry_V = float(ser["V"].iloc[0])
    entry_S = float(ser["S"].iloc[0])
    stop_level = entry_V * (1 + STOP_LOSS_PCT)
    pnl = 0.0
    hedge = 0.0
    prev_S, prev_V = entry_S, entry_V
    last_rebal = ser.index[0]
    exit_reason = "square_off"

    for i in range(1, len(ser)):
        ts = ser.index[i]
        S, V = float(ser["S"].iloc[i]), float(ser["V"].iloc[i])
        pnl += -(V - prev_V) * LOT + hedge * (S - prev_S)
        if (ts - last_rebal).seconds / 60 >= 30:
            nd = straddle_delta(S, atm, T=0.02)
            pnl -= abs(-nd - hedge) * S * HEDGE_COST_BPS / 10000
            hedge = -nd
            last_rebal = ts
        # STOP-LOSS: straddle moved against us too much
        if V >= stop_level:
            exit_reason = "stop_loss"
            break
        prev_S, prev_V = S, V

    from precise_costs import exact_costs
    exit_V = V
    pnl -= exact_costs(entry_V * LOT, exit_V * LOT)["TOTAL"]
    pnl -= abs(hedge * S) * HEDGE_COST_BPS / 10000
    return pnl, exit_reason


def trend_filter(ser):
    """Skip if early (first 30 min) spot move is large (trend day)."""
    if len(ser) < 20:
        return False
    early = ser[ser.index.time <= pd.Timestamp("10:00").time()]
    if early.empty:
        return False
    move_pct = abs(float(early["S"].iloc[-1]) - float(early["S"].iloc[0])) / float(early["S"].iloc[0]) * 100
    return move_pct > TREND_FILTER_PCT  # True = skip


def main():
    import sys
    years = [2025, 2024, 2023]
    trades = []
    for yr in years:
        f = f"NIFTY_intraday_{yr}.parquet"
        try:
            pf = pq.ParquetFile(f)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f}: {e}")
            continue
        dates = set()
        for rg in range(min(5, pf.metadata.num_row_groups)):
            t = pf.read_row_group(rg, columns=["date"])
            dates.update(t.to_pandas()["date"].unique())
        dates = sorted(dates)
        print(f"\n=== {yr}: {len(dates)} expiry days ===")
        for dt in dates[:20]:
            day = load_day(pf, dt)
            if day is None:
                continue
            built = build_series(day)
            if built is None:
                continue
            ser, atm = built
            if trend_filter(ser):
                continue  # skip trend days
            pnl, reason = sim_day(ser, atm)
            trades.append({"date": dt, "year": yr, "atm": atm, "pnl": pnl,
                           "exit": reason})
            print(f"  {dt} ATM {atm:.0f}: Rs{pnl:,.0f} ({reason})")

    if trades:
        t = pd.DataFrame(trades)
        print(f"\n=== INTRADAY v3: stop-loss + trend filter ({len(t)} days) ===")
        print(f"  total PnL: Rs{t.pnl.sum():,.0f}")
        print(f"  avg/day: Rs{t.pnl.mean():,.0f}")
        print(f"  win rate: {(t.pnl > 0).mean()*100:.0f}%")
        print(f"  worst day: Rs{t.pnl.min():,.0f}")
        print(f"  best day: Rs{t.pnl.max():,.0f}")
        print(f"  stop-losses: {(t.exit=='stop_loss').sum()}, squared off: {(t.exit=='square_off').sum()}")
        worst = abs(t.pnl.min())
        print(f"\n  Rs10K: worst = {worst/10000*100:.0f}% "
              f"({'SURVIVES' if worst < 10000 else 'RUINED'})")
        print(f"  Rs1.5L: worst = {worst/150000*100:.1f}% "
              f"({'SURVIVES' if worst < 150000 else 'RUINED'})")
        print(f"  monthly (20d): Rs{t.pnl.mean()*20:,.0f}/lot")
        print(f"  annualized: Rs{t.pnl.mean()*250:,.0f}/lot")
        print(f"\n  VERDICT: {'EDGE + FITS SMALL CAP' if worst<10000 and t.pnl.sum()>0 else 'still tailed or unprofitable'}")


if __name__ == "__main__":
    main()
