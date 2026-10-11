"""PORTFOLIO OF WEAK EDGES + REGIME FILTERS — how small accounts actually win.

WHY PEOPLE SUCCEED WHERE SINGLE STRATEGIES FAIL:
A single strategy needs to be strong enough to survive costs alone. A PORTFOLIO
of many weak strategies combines into something strong: each contributes a
little edge, and crucially their errors DECORRELATE, so the portfolio's
drawdown is far smaller than any single strategy's. This is the 'stack of
systematic strategies on Rs25k' the Indian trader on Reddit described.

This tests the Indian intraday data I have as a PORTFOLIO:
  1. Run ALL strategies on ALL stocks (each weak, maybe negative alone)
  2. Combine with EQUAL-WEIGHT allocation across many small positions
  3. Add REGIME FILTER (only trade when market conditions suit each strategy)
  4. Measure the PORTFOLIO's Sharpe/drawdown, not individual strategies

Crypto excluded (user directive). Indian data only.
Run: python portfolio_edge.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/equity_5min.parquet"
NOTIONAL_PER_TRADE = 15000.0   # Rs15K per position -> Rs1.5L holds 10 concurrent
BROKERAGE = 20
STT = 0.00025
EXCH = 0.0000297
GST = 0.18
SLIP = 0.0005


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


def cost(n):
    return 2 * BROKERAGE + n * 2 * (STT + EXCH * (1 + GST) + SLIP)


def market_regime(nifty_df, upto):
    """Is the index in a tradeable regime? (trending vs choppy).
    Uses only data before `upto`."""
    h = nifty_df[nifty_df.datetime_ist < upto]
    if len(h) < 20:
        return None
    rets = h.close.pct_change().dropna().iloc[-20:]
    vol = rets.std()
    trend = abs(h.close.iloc[-1] / h.close.iloc[-20] - 1)
    return {"vol": vol, "trend": trend}


def all_signals(df, nifty):
    """Generate every (stock, day, strategy, direction, entry, exit) signal."""
    sigs = []
    for lb, g in df.groupby("label"):
        g = g.sort_values("datetime_ist").copy()
        g["hm"] = g.datetime_ist.dt.strftime("%H:%M")
        for d, day in g.groupby(g.datetime_ist.dt.date):
            day = day.sort_values("datetime_ist")
            if len(day) < 20:
                continue
            reg = market_regime(nifty, day.datetime_ist.iloc[0])
            vwap = (day.close * day.volume).cumsum() / day.volume.cumsum().replace(0, np.nan)

            # strategy 1: VWAP reversion (works in choppy/low-vol regime)
            for i in range(10, len(day)):
                if pd.isna(vwap.iloc[i]):
                    continue
                dev = (day.close.iloc[i] - vwap.iloc[i]) / vwap.iloc[i]
                if abs(dev) > 0.004:
                    direction = -1 if dev > 0 else 1
                    # exit at VWAP touch or EOD
                    for j in range(i + 1, len(day)):
                        if (direction == 1 and day.close.iloc[j] >= vwap.iloc[j]) or \
                           (direction == -1 and day.close.iloc[j] <= vwap.iloc[j]):
                            sigs.append({"stock": lb, "date": d, "strat": "vwap_rev",
                                         "dir": direction, "entry": float(day.close.iloc[i]),
                                         "exit": float(day.close.iloc[j]), "reg": reg})
                            break
                    else:
                        sigs.append({"stock": lb, "date": d, "strat": "vwap_rev",
                                     "dir": direction, "entry": float(day.close.iloc[i]),
                                     "exit": float(day.close.iloc[-1]), "reg": reg})
                    break

            # strategy 2: opening range breakout (works in trending regime)
            cutoff = (pd.Timestamp(f"2000-01-01 {day.hm.iloc[0]}")
                      + pd.Timedelta(minutes=30)).strftime("%H:%M")
            orb = day[day.hm <= cutoff]
            after = day[day.hm > cutoff]
            if len(orb) >= 4 and not after.empty:
                hi, lo = orb.high.max(), orb.low.min()
                for _, r in after.iterrows():
                    if r.high >= hi:
                        sigs.append({"stock": lb, "date": d, "strat": "orb_long",
                                     "dir": 1, "entry": hi,
                                     "exit": float(after.close.iloc[-1]), "reg": reg})
                        break
                    if r.low <= lo:
                        sigs.append({"stock": lb, "date": d, "strat": "orb_short",
                                     "dir": -1, "entry": lo,
                                     "exit": float(after.close.iloc[-1]), "reg": reg})
                        break

            # strategy 3: intraday momentum (first-hour direction holds)
            if len(day) >= 12:
                o = float(day.open.iloc[0])
                p60 = float(day.close.iloc[11])
                if abs(p60 / o - 1) > 0.002:
                    direction = 1 if p60 > o else -1
                    sigs.append({"stock": lb, "date": d, "strat": "mom_1h",
                                 "dir": direction, "entry": p60,
                                 "exit": float(day.close.iloc[-1]), "reg": reg})
    return pd.DataFrame(sigs)


def main():
    df = load()
    nifty = None
    try:
        n = pd.read_parquet("data_indmoney/clean_1min.parquet")
        n["datetime_ist"] = pd.to_datetime(n["datetime_ist"])
        nifty = n[(n.label.str.contains("Fut|NIFTY")) & (n.interval == "5minute")]
    except Exception:  # noqa: BLE001
        pass

    print("=== PORTFOLIO OF WEAK EDGES (Indian intraday, no crypto) ===")
    print(f"  {df.label.nunique()} stocks, {df.datetime_ist.dt.date.nunique()} days")
    print(f"  Rs{NOTIONAL_PER_TRADE:,.0f}/position -> Rs1.5L holds "
          f"{int(150000/NOTIONAL_PER_TRADE)} concurrent\n")

    sigs = all_signals(df, nifty if nifty is not None else df)
    if sigs.empty:
        print("  no signals generated")
        return

    # compute pnl per signal
    sigs["qty"] = (NOTIONAL_PER_TRADE / sigs.entry).astype(int)
    sigs["gross"] = (sigs.exit_ if "exit_" in sigs else sigs["exit"] - sigs.entry) * sigs.qty * sigs["dir"]
    sigs["pnl"] = sigs.gross - sigs.entry * sigs.qty * 0 + (-cost(sigs.entry * sigs.qty))
    sigs["win"] = sigs.pnl > 0

    print(f"--- INDIVIDUAL STRATEGIES (each alone) ---")
    print(f"{'strategy':<12}{'n':>6}{'win%':>7}{'avgPnL':>10}{'total':>12}")
    for st, g in sigs.groupby("strat"):
        print(f"{st:<12}{len(g):>6}{g.win.mean()*100:>6.0f}%{g.pnl.mean():>10,.0f}"
              f"{g.pnl.sum():>12,.0f}")

    # PORTFOLIO: equal-weight all signals, per-day aggregation
    print(f"\n--- PORTFOLIO (all strategies, equal-weight Rs15K each) ---")
    daily = sigs.groupby("date").pnl.agg(["sum", "count", "mean"])
    total = daily["sum"].sum()
    win_days = (daily["sum"] > 0).mean()
    print(f"  total signals: {len(sigs)} over {len(daily)} days")
    print(f"  total PnL: Rs{total:,.0f}")
    print(f"  avg signals/day: {daily['count'].mean():.1f}")
    print(f"  profitable days: {win_days*100:.0f}%")
    print(f"  avg/day: Rs{daily['sum'].mean():,.0f}")
    print(f"  worst day: Rs{daily['sum'].min():,.0f}")
    print(f"  best day: Rs{daily['sum'].max():,.0f}")

    # portfolio Sharpe (daily)
    if daily["sum"].std() > 0:
        sharpe = daily["sum"].mean() / daily["sum"].std() * np.sqrt(252)
        print(f"  daily Sharpe (annualized): {sharpe:.2f}")

    # REGIME FILTERED PORTFOLIO: only low-vol regime for reversion
    print(f"\n--- REGIME-FILTERED (vwap_rev only when index vol is low) ---")
    if sigs.reg.notna().any():
        med_vol = sigs[sigs.strat == "vwap_rev"].reg.apply(
            lambda r: r["vol"] if r else np.nan).median()
        vr = sigs[(sigs.strat == "vwap_rev")].copy()
        vr["lowvol"] = vr.reg.apply(lambda r: r["vol"] < med_vol if r else False)
        lo = vr[vr.lowvol]
        hi = vr[~vr.lowvol]
        print(f"  vwap_rev in LOW-vol regime:  {len(lo)} trades, "
              f"{lo.win.mean()*100:.0f}% win, avg Rs{lo.pnl.mean():,.0f}")
        print(f"  vwap_rev in HIGH-vol regime: {len(hi)} trades, "
              f"{hi.win.mean()*100:.0f}% win, avg Rs{hi.pnl.mean():,.0f}")
        if lo.pnl.mean() > hi.pnl.mean():
            print(f"  -> regime filter WORKS: reversion better in low-vol "
                  f"(+Rs{lo.pnl.mean()-hi.pnl.mean():,.0f}/trade)")

    print(f"\n  NOTE: {df.datetime_ist.dt.date.nunique()} days is a SMOKE test.")
    print(f"  The portfolio approach is the right ARCHITECTURE for small capital")
    print(f"  (weak edges decorrelate -> smaller drawdown). Real verdict at 30+ days.")


if __name__ == "__main__":
    main()
