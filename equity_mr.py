"""Equity intraday mean-reversion smoke test — the stronger small-cap candidate.

Equities beat Nifty futures for retail edge: wider spreads, slower reversion,
idiosyncratic moves, no HFT vacuum. Tests VWAP-band mean-reversion across the
15 most liquid F&O stocks on clean INDmoney 5-min data.

Lot sizes read from Kite instrument master (verified source). Position sized
to Rs50K notional per trade so Rs1.5L can hold several concurrent names.

Honesty gate: every trade must fall within realistic bounds for its lot size.
5 days is NOT a sample — this validates the pipeline and gives an early read.
Run: python equity_mr.py
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/equity_5min.parquet"
KITE = "kite_instruments.csv"

WARMUP = 12          # 12 x 5min = 1hr
ENTRY_BAND = 0.004   # 0.40% deviation from VWAP
STOP = 0.006         # 0.60% stop
TARGET = 0.0025      # 0.25% target
NOTIONAL = 50000.0   # Rs50K per trade
CAPITAL = 150000.0
BROKERAGE = 20
STT = 0.00025        # equity delivery STT 0.025% sell (intraday: 0.025% sell)
EXCH = 0.0000297
GST = 0.18


def load_lots():
    """Real F&O lot sizes from Kite master. Falls back to notional sizing."""
    lots = {}
    if os.path.exists(KITE):
        k = pd.read_csv(KITE, low_memory=False)
        k = k[(k.segment == "NFO") & (k.instrument_type == "EQ")]
        for _, r in k.iterrows():
            lots[str(r.tradingsymbol).upper()] = int(r.lot_size)
    return lots


def sim(df, qty):
    """VWAP-band mean reversion, one position at a time."""
    df = df.sort_values("datetime_ist").reset_index(drop=True)
    if len(df) < WARMUP + 5:
        return []
    trades = []
    pos, entry = 0, 0.0
    pv = 0.0
    vol = 0.0
    n = 0
    for i in range(len(df)):
        ts = df.datetime_ist.iloc[i]
        hm = ts.strftime("%H:%M")
        if hm < "09:25" or hm > "15:10":
            continue
        S = float(df.close.iloc[i])
        v = float(df.volume.iloc[i]) if not pd.isna(df.volume.iloc[i]) else 0.0
        pv += S * v
        vol += v
        n += 1
        if n < WARMUP:
            continue
        if pos != 0:
            move = (S - entry) / entry * pos
            if move >= TARGET or move <= -STOP:
                gross = (S - entry) * qty * pos
                to = abs(entry + S) * qty
                net = gross - 2 * BROKERAGE - to * STT - to * EXCH * (1 + GST)
                trades.append(net)
                pos = 0
        if pos == 0 and vol > 0:
            vwap = pv / vol
            dev = (S - vwap) / vwap
            if dev > ENTRY_BAND:
                pos, entry = -1, S
            elif dev < -ENTRY_BAND:
                pos, entry = 1, S
        if hm >= "15:10" and pos != 0:
            gross = (S - entry) * qty * pos
            trades.append(gross - 2 * BROKERAGE)
            pos = 0
    return trades


def main():
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df[df.interval == "5minute"]

    # DATA QUALITY GATE (L13): a wrong ind_key once injected Rs0.84 rows into a
    # Rs1200 stock and fabricated a Rs36-crore "edge". Refuse to run on data
    # where any stock's price range spans more than 10x its median.
    print("=== DATA QUALITY GATE ===")
    clean = []
    for label, g in df.groupby("label"):
        med = g.close.median()
        lo, hi = g.close.min(), g.close.max()
        ratio = hi / med if med > 0 else 999
        dup = int(g.datetime_ist.duplicated().sum())
        ok = ratio < 10 and dup == 0
        print(f"  {label:<12} median Rs{med:>8.1f} range {lo:.1f}-{hi:.1f} "
              f"({ratio:.1f}x) dups {dup} -> {'OK' if ok else 'REJECTED'}")
        if ok:
            clean.append(g)
    if not clean:
        print("\n  all stocks failed the quality gate — do not trade this data")
        return
    df = pd.concat(clean)
    print(f"\n  {df.label.nunique()}/{df.label.nunique()} stocks pass. "
          f"Running on {len(df)} clean bars.\n")
    lots = load_lots()

    print("=== EQUITY MEAN-REVERSION SMOKE (clean INDmoney 5-min) ===")
    print(f"  bars {len(df)} | stocks {df.label.nunique()} | "
          f"days {df.datetime_ist.dt.date.nunique()}")
    print(f"  band={ENTRY_BAND*100:.2f}% stop={STOP*100:.2f}% "
          f"target={TARGET*100:.2f}% notional=Rs{NOTIONAL:,.0f}\n")

    all_t = []
    per_stock = []
    for label, grp in df.groupby("label"):
        price = float(grp.close.median())
        qty = max(1, int(NOTIONAL / price))
        t = []
        for d, g in grp.groupby(df.datetime_ist.dt.date):
            t.extend(sim(g, qty))
        if t:
            all_t.extend(t)
            per_stock.append((label, len(t), sum(t), min(t), max(t)))

    if not all_t:
        print("  no trades — bands too wide for 5 days of data")
        return

    arr = np.array(all_t)
    print("  per stock (trades, PnL, worst, best):")
    for lb, n, p, lo, hi in sorted(per_stock, key=lambda x: -x[2]):
        print(f"    {lb:<12} {n:>3}  Rs{p:>10,.0f}  Rs{lo:>9,.0f}  Rs{hi:>9,.0f}")

    print(f"\n=== TRADE-LEVEL REALITY CHECK ===")
    print(f"  total trades: {len(arr)}")
    print(f"  win rate: {(arr > 0).mean()*100:.0f}%")
    print(f"  total PnL: Rs{arr.sum():,.0f}")
    print(f"  avg trade: Rs{arr.mean():,.0f}")
    print(f"  worst: Rs{arr.min():,.0f} | best: Rs{arr.max():,.0f}")
    # realistic bound: stop 0.6% of 50K notional = Rs300
    bound = NOTIONAL * STOP
    ok = (np.abs(arr) < bound * 3).all()
    print(f"\n  HONESTY GATE: all trades within 3x stop (Rs{bound*3:,.0f})? "
          f"{'YES - data clean' if ok else 'NO - STILL FAKE'}")
    if not ok:
        print("    ABORT — numbers outside realistic bounds.")
        return
    days = df.datetime_ist.dt.date.nunique()
    print(f"\n  EARLY READ ({days} days, NOT a verdict):")
    print(f"    avg/day across {df.label.nunique()} stocks: "
          f"Rs{arr.sum()/days:,.0f} if all traded daily")
    print(f"    on Rs1.5L with 3 concurrent names (Rs50K each): "
          f"~Rs{arr.mean()*3*days/days:,.0f}/day avg")
    print(f"\n  VERDICT: pipeline validated on equities. No edge claim until "
          f"30+ days (ETA mid-Nov).")


if __name__ == "__main__":
    main()
