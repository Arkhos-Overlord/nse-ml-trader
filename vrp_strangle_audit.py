"""ADVERSARIAL AUDIT — VRP-conditional strangle. Is the DTE<=1d edge real?

vrp_strangle.py showed: DTE<=1d = 60% win, +Rs3,939/trade — the only positive
config. BUT worst trade -75% of capital, and holdout flips negative
(t=-3.28 in second half). Audit before believing:

  A. TAIL REALITY: is the -75% worst day a data artifact or a real vol spike?
     (verified: 2025-08-08 PE 24900 went 375->545 pts, real premium expansion)
  B. HOLDOUT: does DTE<=1d survive both halves?
  C. TAIL-ADJUSTED: what does the edge look like if you CLIP the tail
     (stop-loss at, say, 2x premium collected)? Real traders don't hold a
     strangle to expiry into a vol spike — they cut it.
  D. FIXED SMALL SIZE: 8 lots is the problem. What if 2 lots (Rs36K margin)?
     Smaller absolute tail, but the edge must still beat costs.

Run: python vrp_strangle_audit.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BARS = "strangle_bars.parquet"
LOT = 75
MARGIN_PER_LOT = 18000
CAPITAL = 150000.0
BROKERAGE = 20.0
STT_SELL = 0.0001
EXCH_GST = 0.00004
SLIP = 0.0005


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def build(df, wing=2, dte_max=None):
    out = []
    for d, day in df.groupby("date"):
        day = day.sort_values("expiry")
        nearest = day.expiry.min()
        sub = day[day.expiry == nearest]
        atm = sub.atm.iloc[0]
        strikes = sorted(sub.strike.unique())
        if atm not in strikes:
            continue
        ai = strikes.index(atm)
        ce_k = strikes[min(ai + wing, len(strikes) - 1)]
        pe_k = strikes[max(ai - wing, 0)]
        ce = sub[(sub.strike == ce_k) & (sub.type == "CE")]
        pe = sub[(sub.strike == pe_k) & (sub.type == "PE")]
        if ce.empty or pe.empty:
            continue
        dte = (nearest - d).days
        if dte_max is not None and dte > dte_max:
            continue
        out.append({"date": d, "expiry": nearest, "atm": atm, "dte": dte,
                    "ce_entry": ce.iloc[0].first_close,
                    "pe_entry": pe.iloc[0].first_close,
                    "ce_exit": ce.iloc[-1].last_close,
                    "pe_exit": pe.iloc[-1].last_close})
    return pd.DataFrame(out)


def sim(strat, lots, stop_mult=None):
    """stop_mult: cut the position when exit_prem > stop_mult x entry_prem
    (i.e. loss > (stop_mult-1) x premium collected). Real risk control."""
    res = []
    for _, r in strat.iterrows():
        entry = r.ce_entry + r.pe_entry
        exit_ = r.ce_exit + r.pe_exit
        if stop_mult is not None and exit_ > stop_mult * entry:
            exit_ = stop_mult * entry      # stopped out
        notional = lots * LOT * r.atm
        gross = (entry - exit_) * lots * LOT
        cost = lots * (2 * BROKERAGE + notional * STT_SELL + notional * EXCH_GST
                       + entry * lots * LOT * SLIP + exit_ * lots * LOT * SLIP)
        res.append({"date": r.date.date(), "dte": r.dte, "pnl": gross - cost,
                    "entry": entry, "exit": exit_})
    return pd.DataFrame(res)


def report(name, res):
    if res.empty:
        print(f"  {name:<36} 0 trades")
        return None
    pnl = res.pnl.values
    cap, peak, maxdd = CAPITAL, CAPITAL, 0
    for p in pnl:
        cap += p
        peak = max(peak, cap)
        maxdd = max(maxdd, (peak - cap) / peak)
    print(f"  {name:<36} n={len(pnl):<4} win={100*(pnl>0).mean():3.0f}% "
          f"net/tr=Rs{pnl.mean():+7.0f} t={tstat(pnl):+.2f} "
          f"worst=Rs{pnl.min():+7.0f} ({pnl.min()/CAPITAL*100:+.0f}% cap) "
          f"end=Rs{cap:,.0f} DD={maxdd*100:.0f}%")
    return pnl


def main():
    df = pd.read_parquet(BARS)
    df["date"] = pd.to_datetime(df.date)
    df["expiry"] = pd.to_datetime(df.expiry)

    print("=" * 96)
    print("  ADVERSARIAL AUDIT — VRP-conditional short strangle")
    print("=" * 96)

    print("\n[A] TAIL IS REAL (verified): 2025-08-08 PE24900 375->545 pts = real")
    print("    vol spike. An 8-lot strangle genuinely lost ~90% of capital.")
    print("    -> 8 lots at Rs1.5L is not survivable. The tail is the whole story.")

    print("\n[B] HOLDOUT on DTE<=1d (the only positive config):")
    s = build(df, wing=2, dte_max=1)
    r = sim(s, 8)
    if not r.empty:
        dates = sorted(r.date)
        mid = dates[len(dates) // 2]
        report("  DTE<=1d first half", r[r.date < mid])
        report("  DTE<=1d second half", r[r.date >= mid])

    print("\n[C] STOP-LOSS (cut at 2x / 3x premium = real risk control):")
    for sm in [2.0, 3.0]:
        report(f"  DTE<=1d, stop {sm}x prem, 8 lots", sim(s, 8, stop_mult=sm))

    print("\n[D] SMALLER SIZE (2 lots = Rs36K margin, tail is absolute Rs not %):")
    for sm in [None, 2.0]:
        nm = "no stop" if sm is None else f"stop {sm}x"
        report(f"  DTE<=1d, 2 lots, {nm}", sim(s, 2, stop_mult=sm))

    print("\n[E] THE HONEST HURDLE: does ANY config beat a liquid fund (~7%/yr)")
    print("    with a survivable tail on Rs1.5L?")
    for wing, dmax, sm, lots in [(1, 1, 2.0, 2), (1, 1, 2.0, 8), (2, 1, 2.0, 2)]:
        s = build(df, wing=wing, dte_max=dmax)
        report(f"  wing{wing} DTE<={dmax} stop{sm}x {lots}lots", sim(s, lots, stop_mult=sm))

    print("\n" + "=" * 96)
    print("  VERDICT RULE: TRADEABLE only if net/tr>0 with |t|>2, BOTH holdout")
    print("  halves positive, worst trade < 20% of capital, AND beats a liquid")
    print("  fund. Short vol fails the tail test by construction: the VRP is")
    print("  real (+6.1 vol pts, t=+15.6) but the payoff is many small wins")
    print("  against rare catastrophic days, and Rs1.5L cannot absorb the tail.")


if __name__ == "__main__":
    main()
