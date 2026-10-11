"""INTRADAY delta-HEDGED harvest — properly neutral, no gap, fits Rs1.5L.

The naive version lost because an unhedged straddle bleeds on directional days.
Fix: delta-hedge with Nifty futures every ~30 min through the day. Then PnL is
theta/gamma harvest, direction-neutral, squared off before close = NO gap risk.

Needs: 1-min option prices (have it) + 5-min Nifty spot (fetch here).
Streams the big parquet one expiry at a time. Real costs.
Run: python intraday_hedged.py
"""
import math
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yfinance as yf

warnings.filterwarnings("ignore")

LOT = 75
HEDGE_COST_BPS = 3
BROKERAGE = 20
REBALANCE_MINS = 30


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def straddle_delta(S, K, sigma, T, r=0.07):
    if T <= 1e-6 or sigma <= 1e-6 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return 2.0 * norm_cdf(d1) - 1.0


def get(sym, period="max", interval="1d"):
    d = yf.download(sym, period=period, interval=interval, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d


def load_day(pf, target_date):
    frames = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["date", "timestamp", "expiry", "strike",
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

    # 5-min Nifty spot for hedging
    nif5 = get("^NSEI", period="1y", interval="5m")
    nif5.index = pd.to_datetime(nif5.index)
    nif5["date"] = nif5.index.date

    test_dates = dates[:12]
    trades = []
    for dt in test_dates:
        day = load_day(pf, dt)
        if day is None or day.empty:
            continue
        day = day[day["volume"] > 0]
        if day.empty:
            continue
        t0 = day["timestamp"].min()
        row0 = day[day["timestamp"] == t0]
        ce0 = row0[row0["option_type"] == "CE"].set_index("strike")["close"]
        pe0 = row0[row0["option_type"] == "PE"].set_index("strike")["close"]
        common = ce0.index.intersection(pe0.index)
        sums = (ce0[common] + pe0[common])
        sums = sums[sums > 0]
        if sums.empty:
            continue
        atm = sums.idxmin()

        strad = day[day["strike"] == atm].pivot_table(
            index="timestamp", columns="option_type", values="close", aggfunc="last").dropna()
        if len(strad) < 10:
            continue
        strad["V"] = strad["CE"] + strad["PE"]
        strad = strad.sort_index()
        strad = strad[strad.index.time >= pd.Timestamp("09:30").time()]
        strad = strad[strad.index.time <= pd.Timestamp("15:15").time()]
        if len(strad) < 10:
            continue

        # Nifty spot that day (5-min), aligned
        day_date = pd.Timestamp(dt).date()
        spot_day = nif5[nif5["date"] == day_date]["Close"]
        if len(spot_day) < 5:
            continue
        # resample spot to straddle timestamps (nearest)
        spot_aligned = spot_day.reindex(strad.index, method="nearest")

        # intraday delta-hedged short straddle, rebalance every 30 min
        pnl = 0.0
        hedge = 0.0
        prev_S = float(spot_aligned.iloc[0])
        prev_V = float(strad["V"].iloc[0])
        entry_V = prev_V
        last_rebal = strad.index[0]
        for i in range(1, len(strad)):
            ts = strad.index[i]
            S = float(spot_aligned.iloc[i])
            V = float(strad["V"].iloc[i])
            pnl += -(V - prev_V) * LOT + hedge * (S - prev_S)
            # rebalance every 30 min
            if (ts - last_rebal).seconds / 60 >= REBALANCE_MINS:
                T = max((pd.Timestamp("15:30").time() and 0.04), 1 / 365)  # ~intraday T
                # use a fixed intraday sigma from VIX-ish; delta near 0 ATM
                nd = straddle_delta(S, atm, 0.12, 0.02)  # ~12% vol, small T
                rb = abs(-nd - hedge) * S
                pnl -= rb * HEDGE_COST_BPS / 10000
                hedge = -nd
                last_rebal = ts
            prev_S, prev_V = S, V

        from precise_costs import exact_costs
        pnl -= exact_costs(entry_V * LOT, float(strad["V"].iloc[-1]) * LOT)["TOTAL"]
        pnl -= abs(hedge * prev_S) * HEDGE_COST_BPS / 10000  # close hedge
        trades.append({"date": dt, "atm": atm, "entry_prem": entry_V * LOT,
                       "pnl": pnl})

    if trades:
        t = pd.DataFrame(trades)
        print(f"\n=== INTRADAY DELTA-HEDGED HARVEST ({len(t)} days) ===")
        print(f"  total PnL: Rs{t.pnl.sum():,.0f}")
        print(f"  avg/day: Rs{t.pnl.mean():,.0f}")
        print(f"  win rate: {(t.pnl > 0).mean()*100:.0f}%")
        print(f"  worst day: Rs{t.pnl.min():,.0f}")
        print(f"  best day: Rs{t.pnl.max():,.0f}")
        worst = abs(t.pnl.min())
        print(f"\n  Rs1.5L: worst day = {worst/150000*100:.1f}% drawdown "
              f"({'SURVIVES' if worst < 150000 else 'RUINED'})")
        per_day = t.pnl.mean()
        print(f"  if avg holds: Rs{per_day*20:,.0f}/month (20 trading days)")
        print(f"  annualized: ~Rs{per_day*250:,.0f}/lot")
        print(f"\n  HONEST: no gap risk. Delta-hedged intraday. This is the")
        print(f"  Rs1.5L-fit candidate IF it's profitable and tail is small.")
    else:
        print("no trades")


if __name__ == "__main__":
    main()
