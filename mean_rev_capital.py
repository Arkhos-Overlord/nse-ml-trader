"""CAPITAL-CONSTRAINED mean-reversion — the honest simulator.

v1 claimed +Rs2.99Cr on Rs1.5L. That's FALSE: it ignored that Rs1.5L cannot
fund 44 trades/day of Rs15L notional. The signal is real (77% win) but the
equity curve was fabricated by ignoring lot size + margin + compounding.

This version is the honest one:
  - One Nifty futures lot = Rs22500 * 75 = Rs16.9L notional. Margin ~Rs1.5L/lot.
  - Rs1.5L = EXACTLY ONE lot. You cannot scale up. No compounding to 2 lots.
  - So max realistic = 1 lot, and returns are capped at 1-lot economics.
  - Track equity honestly: every win/loss moves Rs1.5L by the 1-lot PnL.

Run: python mean_rev_capital.py 2024
"""
import math
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

# REAL capital constraint
CAPITAL = 150000.0
LOT_NOTIONAL = 22500 * LOT   # ~Rs16.9L per lot
LOT_MARGIN = 150000.0        # ~Rs1.5L SPAN+exposure margin per lot (approx)


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
    return ser


def sim_day(ser):
    """One-lot mean-reversion. ONE position at a time (no overlapping entries).
    Returns list of trade PnLs. This is the honest version — v2 double-counted
    by re-entering every minute the deviation persisted."""
    if len(ser) < WARMUP_MINS + 10:
        return []
    trades = []
    pos = 0          # +1 long, -1 short, 0 flat
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
        # manage the SINGLE open position first
        if pos != 0:
            move = (S - entry) / entry * pos
            if move >= TARGET or move <= -STOP_LOSS:
                pnl = (S - entry) * pos * LOT
                pnl -= 2 * BROKERAGE
                pnl -= abs(entry + S) * LOT * FUTURE_STT
                pnl -= abs(entry + S) * LOT * FUTURE_EXCH
                trades.append(pnl)
                pos = 0
        # only enter if FLAT (one position at a time)
        if pos == 0:
            if dev > ENTRY_THRESH:
                pos, entry = -1, S
            elif dev < -ENTRY_THRESH:
                pos, entry = 1, S
        # square off at close
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
        t = pf.read_row_group(rg, columns=["date"])
        dates.update(t.to_pandas()["date"].unique())
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
        # margin call: if capital drops below one lot margin, ruined
        all_trades.extend(trades)
        daily.append({"date": dt, "pnl": day_pnl, "capital": capital})

    t = pd.DataFrame(daily)
    if t.empty:
        print("no data")
        return
    print(f"=== CAPITAL-CONSTRAINED MEAN-REVERSION ({year}, 1 lot) ===")
    print(f"  days traded: {len(t)}")
    print(f"  total trades: {len(all_trades)}")
    print(f"  start capital: Rs{CAPITAL:,.0f}")
    print(f"  end capital:   Rs{capital:,.0f}")
    print(f"  total return:  {(capital/CAPITAL-1)*100:+.1f}%")
    print(f"  avg/day: Rs{t.pnl.mean():,.0f}")
    print(f"  win days: {(t.pnl > 0).mean()*100:.0f}%")
    print(f"  worst day: Rs{t.pnl.min():,.0f}")
    print(f"  best day: Rs{t.pnl.max():,.0f}")
    print(f"  max drawdown: {maxdd:.1f}%")
    print(f"  MIN capital hit: Rs{t.capital.min():,.0f}")
    ruined = (t.capital < LOT_MARGIN).any()
    print(f"  RUINED (capital < 1 lot margin)? {'YES' if ruined else 'NO'}")
    print(f"\n  monthly (20d): Rs{t.pnl.mean()*20:,.0f}")
    print(f"  annualized: Rs{t.pnl.mean()*250:,.0f}")
    print(f"\n  HONEST VERDICT:")
    if capital > CAPITAL and maxdd < 30 and not ruined:
        print(f"    PROFITABLE on 1 lot, maxDD {maxdd:.1f}%, survives Rs1.5L.")
        print(f"    This is a real Rs1.5L candidate. Monthly ~Rs{t.pnl.mean()*20:,.0f}/lot.")
    elif ruined:
        print(f"    RUINED — drawdown dips below 1-lot margin. Not safe at Rs1.5L.")
    else:
        print(f"    Not profitable or too risky at 1 lot.")


if __name__ == "__main__":
    main()
