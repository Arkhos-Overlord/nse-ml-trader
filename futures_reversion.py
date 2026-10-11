"""FUTURES mean-reversion — the cost lever that killed equity reversal.

Why futures: equity reversal died because 0.3% round-trip STT + brokerage ate a
+0.1%/trade edge -> Rs99/month. FUTURES have a completely different cost
structure:
  - brokerage flat Rs20/order (NOT % of notional)
  - STT 0.01% on SELL only (vs 0.1% both sides delivery)
  - no delivery STT, no stamp duty edge
So the SAME gross edge that made Rs99/mo on equity can clear costs on futures
because the fixed cost is Rs40 round-trip regardless of notional.

BUT: Nifty index is efficient at daily resolution (my autocorr test: ~0). So
mean-reversion needs INTRADAY or a multi-day oversold signal, not daily. yfinance
is daily-only. So test what daily CAN show: multi-day oversold bounce (RSI-style
/ z-score) on deep index history, with FUTURES economics.

Signal (no look-ahead): at day t close, if z-score of last N-day return is very
negative (oversold), go LONG futures; if very positive, SHORT. Hold H days.
Costs: Rs40 round-trip + 0.01% sell STT + 0.003% exchange/GST, per futures trade.

Honesty gates: holdout split, full-capital Rs1.5L compounding, year-by-year,
robustness across N/H.

Run: python futures_reversion.py
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CAPITAL = 150000.0
LOT = 65                 # Nifty lot size (current, per Kite master)
BROKERAGE = 20.0         # per order
STT_SELL = 0.0001        # 0.01% on sell
EXCH_GST = 0.00004       # exchange + GST approx
SLIP = 0.0002            # slippage on index


def load(sym="^NSEI"):
    df = yf.download(sym, start="2018-01-01", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"date": "datetime_ist"})
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"]).dt.tz_localize(None)
    return df[["datetime_ist", "close"]].dropna().reset_index(drop=True)


def gen_trades(df, N, H, z_thr):
    """z-score reversion: oversold -> long, overbought -> short. No look-ahead
    (z uses only past N days). Hold H days."""
    close = df.close.values
    dates = df.datetime_ist.values
    trades = []
    i = N
    while i + H < len(close):
        window = close[i - N:i]
        mu, sd = window.mean(), window.std()
        if sd == 0:
            i += 1
            continue
        z = (close[i - 1] - mu) / sd
        if z <= -z_thr:
            direction = 1     # oversold -> long
        elif z >= z_thr:
            direction = -1    # overbought -> short
        else:
            i += 1
            continue
        entry = close[i]
        exit_ = close[i + H]
        ret = (exit_ / entry - 1) * direction
        trades.append({"date": pd.Timestamp(dates[i]), "entry": entry,
                       "exit": exit_, "dir": direction, "ret": ret})
        i += H   # non-overlapping
    return pd.DataFrame(trades)


def futures_cost(notional):
    """Round-trip futures cost: 2x brokerage + sell STT + exch/GST, in Rs."""
    return 2 * BROKERAGE + notional * STT_SELL + notional * EXCH_GST


def full_capital_sim(trades, capital=CAPITAL):
    """1 futures position at a time (sequential non-overlapping), compound.
    Position sized to capital / (lot * index)."""
    peak = capital
    maxdd = 0
    last = None
    for _, t in trades.iterrows():
        if last is not None and t["date"] <= last:
            continue
        lots = max(1, int(capital / (LOT * t["entry"])))
        notional = lots * LOT * t["entry"]
        gross = (t["exit"] - t["entry"]) * lots * LOT * t["dir"]
        cost = futures_cost(notional)
        capital += gross - cost
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
        last = t["date"]
    return capital, maxdd


def main():
    df = load("^NSEI")
    print(f"=== FUTURES MEAN-REVERSION (Nifty, deep history) ===")
    print(f"  {len(df)} days, {df.datetime_ist.min().date()} -> {df.datetime_ist.max().date()}")
    print(f"  lot={LOT}, Rs{BROKERAGE}/order, STT 0.01% sell, no delivery STT")
    print(f"  capital Rs{ CAPITAL:,.0f} -> ~{int(CAPITAL/(LOT*df.close.iloc[-1]))} lots\n")

    print(f"{'N':>3}{'H':>4}{'z':>5}{'n':>6}{'gross%/tr':>10}{'net%/tr':>9}"
          f"{'t-stat':>8}{'win%':>6}{'Rs1.5L end':>13}{'maxDD':>7}")
    best = []
    for N in [5, 10, 20]:
        for H in [3, 5, 10]:
            for z_thr in [1.0, 1.5, 2.0]:
                tr = gen_trades(df, N, H, z_thr)
                if len(tr) < 40:
                    continue
                # net per trade on futures economics (scaled to ~1 lot notional)
                notional = LOT * tr.entry.mean()
                cost_frac = futures_cost(notional) / notional
                net = tr.ret - cost_frac
                ts = net.mean() / (net.std() / np.sqrt(len(net))) if net.std() > 0 else 0
                cap, maxdd = full_capital_sim(tr)
                print(f"{N:>3}{H:>4}{z_thr:>5.1f}{len(tr):>6}"
                      f"{tr.ret.mean()*100:>9.3f}%{net.mean()*100:>8.3f}%"
                      f"{ts:>+8.2f}{100*(net>0).mean():>5.0f}%{cap:>13,.0f}{maxdd*100:>6.1f}%")
                best.append((ts, N, H, z_thr, len(tr), cap, maxdd, net))

    if not best:
        print("  no config had enough trades")
        return
    # honesty gates on the best-by-tstat config
    best.sort(key=lambda x: -x[0])
    ts, N, H, z_thr, n, cap, maxdd, net = best[0]
    print(f"\n=== HONESTY GATES (best by t-stat: N={N} H={H} z={z_thr}) ===")
    tr = gen_trades(df, N, H, z_thr)
    mid = tr.date.iloc[len(tr) // 2]
    for nm, g in [("first half", tr[tr.date < mid]), ("second half", tr[tr.date >= mid])]:
        notional = LOT * g.entry.mean()
        gn = g.ret - futures_cost(notional) / notional
        gts = gn.mean() / (gn.std() / np.sqrt(len(g))) if gn.std() > 0 else 0
        print(f"  {nm:<12} n={len(g):<4} net {gn.mean()*100:+.3f}%/tr  t={gts:+.2f}")
    both = (tr[tr.date<mid].ret.mean() > 0) and (tr[tr.date>=mid].ret.mean() > 0)
    print(f"  both halves positive: {'YES' if both else 'NO'}")

    tr["yr"] = tr.date.dt.year
    notional = LOT * tr.entry.mean()
    tr["net"] = tr.ret - futures_cost(notional) / notional
    yr = tr.groupby("yr").net.mean()
    print(f"  profitable years: {(yr>0).sum()}/{len(yr)}")
    print(f"  full-capital Rs1.5L -> Rs{cap:,.0f} ({(cap/CAPITAL-1)*100:+.0f}%), "
          f"maxDD {maxdd*100:.0f}%")
    print(f"\n  VERDICT: {'futures economics help but need both-halves + Rs1.5L>start + consistent years' if not (both and cap>CAPITAL and (yr>0).mean()>0.5) else 'PROMISING — investigate'}")


if __name__ == "__main__":
    main()
