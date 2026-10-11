"""INTRADAY delta-hedged harvest v2 — self-contained spot from option data.

v1 failed: yfinance 5m Nifty spot only goes back 60 days, not 2025. Fix: the ATM
straddle price ITSELF tracks spot (delta ~ 2*|moneyness|). Use the ATM CE-PE
asymmetry to infer intraday spot direction, or simpler: hedge using the straddle's
own delta via the CE/PE premium ratio. Actually cleanest: reconstruct spot from
put-call parity using ATM CE and PE prices at each timestamp:
    S ≈ K + (CE - PE) * ... (parity: C - P ≈ S - K for small T)
So S ≈ K + (CE_price - PE_price). Intraday spot from real option prices.

No external spot needed. Real 1-min option data. Run: python intraday_v2.py
"""
import math
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

LOT = 75
HEDGE_COST_BPS = 3


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


def main():
    pf = pq.ParquetFile("NIFTY_intraday_2025.parquet")
    dates = set()
    for rg in range(min(5, pf.metadata.num_row_groups)):
        t = pf.read_row_group(rg, columns=["date"])
        dates.update(t.to_pandas()["date"].unique())
    dates = sorted(dates)

    from precise_costs import exact_costs
    trades = []
    for dt in dates[:12]:
        day = load_day(pf, dt)
        if day is None or day.empty:
            continue
        day = day[day["volume"] > 0]
        if day.empty:
            continue
        # pivot CE/PE close by timestamp
        piv = day.pivot_table(index="timestamp", columns=["strike", "option_type"],
                              values="close", aggfunc="last")
        # find ATM strike: min CE+PE at first timestamp
        t0 = piv.index[0]
        ce = {s: piv.loc[t0, (s, "CE")] for s in day["strike"].unique()
              if (s, "CE") in piv.columns}
        pe = {s: piv.loc[t0, (s, "PE")] for s in day["strike"].unique()
              if (s, "PE") in piv.columns}
        common = [s for s in ce if s in pe and ce[s] > 0 and pe[s] > 0]
        if not common:
            continue
        atm = min(common, key=lambda s: ce[s] + pe[s])

        # build series: spot (from parity), straddle value
        rows = []
        for ts in piv.index:
            try:
                c = piv.loc[ts, (atm, "CE")]
                p = piv.loc[ts, (atm, "PE")]
            except KeyError:
                continue
            if pd.isna(c) or pd.isna(p) or c <= 0 or p <= 0:
                continue
            S = atm + (c - p)  # put-call parity: S ≈ K + (C - P)
            rows.append((ts, S, c + p))
        if len(rows) < 10:
            continue
        ser = pd.DataFrame(rows, columns=["ts", "S", "V"]).set_index("ts").sort_index()
        ser = ser[ser.index.time >= pd.Timestamp("09:30").time()]
        ser = ser[ser.index.time <= pd.Timestamp("15:15").time()]
        if len(ser) < 10:
            continue

        # intraday delta-hedged short straddle, rebalance every 30 min
        pnl = 0.0
        hedge = 0.0
        prev_S, prev_V = float(ser["S"].iloc[0]), float(ser["V"].iloc[0])
        entry_V = prev_V
        last_rebal = ser.index[0]
        for i in range(1, len(ser)):
            ts = ser.index[i]
            S, V = float(ser["S"].iloc[i]), float(ser["V"].iloc[i])
            pnl += -(V - prev_V) * LOT + hedge * (S - prev_S)
            if (ts - last_rebal).seconds / 60 >= 30:
                nd = straddle_delta(S, atm, T=0.02)
                pnl -= abs(-nd - hedge) * S * HEDGE_COST_BPS / 10000
                hedge = -nd
                last_rebal = ts
            prev_S, prev_V = S, V

        pnl -= exact_costs(entry_V * LOT, float(ser["V"].iloc[-1]) * LOT)["TOTAL"]
        pnl -= abs(hedge * prev_S) * HEDGE_COST_BPS / 10000
        trades.append({"date": dt, "atm": atm, "pnl": pnl,
                       "entry_prem": entry_V * LOT})

    if trades:
        t = pd.DataFrame(trades)
        print(f"=== INTRADAY DELTA-HEDGED v2 ({len(t)} days, spot from parity) ===")
        print(f"  total PnL: Rs{t.pnl.sum():,.0f}")
        print(f"  avg/day: Rs{t.pnl.mean():,.0f}")
        print(f"  win rate: {(t.pnl > 0).mean()*100:.0f}%")
        print(f"  worst day: Rs{t.pnl.min():,.0f}")
        print(f"  best day: Rs{t.pnl.max():,.0f}")
        worst = abs(t.pnl.min())
        print(f"\n  Rs1.5L: worst = {worst/150000*100:.1f}% "
              f"({'SURVIVES' if worst < 150000 else 'RUINED'})")
        print(f"  Rs10K: worst = {worst/10000*100:.0f}% "
              f"({'SURVIVES' if worst < 10000 else 'RUINED'})")
        print(f"  monthly (20d): Rs{t.pnl.mean()*20:,.0f}/lot")
        print(f"  annualized: Rs{t.pnl.mean()*250:,.0f}/lot")
        print(f"\n  for _, r in t.iterrows(): print day pnl for transparency")
        for _, r in t.iterrows():
            print(f"    {r.date}: Rs{r.pnl:,.0f}")
    else:
        print("no trades")


if __name__ == "__main__":
    main()
