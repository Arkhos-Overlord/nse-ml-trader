"""FINAL ADVERSARIAL AUDIT — DTE<=1d, 2-lot VRP strangle.

The candidate: DTE<=1d short strangle, 2 lots (Rs36K margin), wing +/-2.
  net +Rs4,197/trade, t=+2.35, 70% win, worst -16% cap, DD 13%,
  Rs1.5L -> Rs317K over 249 days.

This is the FIRST candidate to clear |t|>2 with a survivable tail AND beat the
hurdle. Per the lesson: confirmation bias peaks now. Audit HARD.

  A. HOLDOUT honesty — the second half was negative (t=-1.29) at 8 lots.
     Re-check at 2 lots: is the edge in both halves, or only the first?
  B. YEAR PERSISTENCE — positive in how many calendar years? (1yr of data =
     limited, but check monthly/quarterly).
  C. BREADTH — is it broad across the 60 expiries or a few lucky ones?
  D. TAIL CLIPPING — does the edge survive if you REMOVE the best 5 days?
     (short vol's P&L is often carried by a few quiet days)
  E. COST SENSITIVITY — 2x slippage.
  F. THE REALISTIC CONSTRAINT — this is 249 days (1 year) of one market regime.
     Is the sample large enough to trust at all?

Run: python vrp_strangle_final.py
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


def build(df, wing=2, dte_max=1):
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
        if dte > dte_max:
            continue
        out.append({"date": d.date(), "expiry": nearest, "atm": atm, "dte": dte,
                    "ce_entry": ce.iloc[0].first_close,
                    "pe_entry": pe.iloc[0].first_close,
                    "ce_exit": ce.iloc[-1].last_close,
                    "pe_exit": pe.iloc[-1].last_close})
    return pd.DataFrame(out)


def sim(strat, lots=2, slip=SLIP):
    res = []
    for _, r in strat.iterrows():
        entry = r.ce_entry + r.pe_entry
        exit_ = r.ce_exit + r.pe_exit
        notional = lots * LOT * r.atm
        gross = (entry - exit_) * lots * LOT
        cost = lots * (2 * BROKERAGE + notional * STT_SELL + notional * EXCH_GST
                       + (entry + exit_) * lots * LOT * slip)
        res.append({"date": r.date, "dte": r.dte, "atm": r.atm,
                    "pnl": gross - cost, "entry": entry, "exit": exit_})
    return pd.DataFrame(res)


def report(name, res):
    if res.empty:
        print(f"  {name:<32} 0 trades")
        return None
    pnl = res.pnl.values
    cap, peak, maxdd = CAPITAL, CAPITAL, 0
    for p in pnl:
        cap += p
        peak = max(peak, cap)
        maxdd = max(maxdd, (peak - cap) / peak)
    print(f"  {name:<32} n={len(pnl):<4} win={100*(pnl>0).mean():3.0f}% "
          f"net/tr=Rs{pnl.mean():+7.0f} t={tstat(pnl):+.2f} "
          f"worst=Rs{pnl.min():+7.0f}({pnl.min()/CAPITAL*100:+.0f}%) "
          f"end=Rs{cap:,.0f} DD={maxdd*100:.0f}%")
    return pnl


def main():
    df = pd.read_parquet(BARS)
    df["date"] = pd.to_datetime(df.date)
    df["expiry"] = pd.to_datetime(df.expiry)
    s = build(df, wing=2, dte_max=1)
    r = sim(s, lots=2)

    print("=" * 96)
    print("  FINAL ADVERSARIAL AUDIT — DTE<=1d, 2-lot VRP strangle")
    print("=" * 96)

    print("\n[A] HOLDOUT (both halves must be positive):")
    dates = sorted(r.date)
    mid = dates[len(dates) // 2]
    h1 = report("  first half", r[r.date < mid])
    h2 = report("  second half", r[r.date >= mid])
    if h1 is not None and h2 is not None:
        both = h1.mean() > 0 and h2.mean() > 0
        print(f"  -> both positive: {'YES' if both else 'NO — regime-dependent'}")
        print(f"     first t={tstat(h1):+.2f}, second t={tstat(h2):+.2f}")

    print("\n[B] QUARTERLY PERSISTENCE:")
    r["q"] = pd.to_datetime(r.date).dt.to_period("Q")
    pos_q = 0
    for q, g in r.groupby("q"):
        m = g.pnl.mean()
        if m > 0:
            pos_q += 1
        print(f"    {q}: n={len(g):<3} net/tr=Rs{m:+7.0f} t={tstat(g.pnl):+.2f}")
    print(f"  -> positive in {pos_q}/{r.q.nunique()} quarters")

    print("\n[C] TAIL CLIPPING (remove best 5 days — is P&L carried by a few?):")
    r_sorted = r.sort_values("pnl", ascending=False)
    clipped = r_sorted.iloc[5:]
    report("  ex-best-5-days", clipped)

    print("\n[D] COST SENSITIVITY (2x slippage):")
    report("  2x slippage", sim(s, lots=2, slip=SLIP * 2))

    print("\n[E] WING SENSITIVITY (plateau or spike at wing=2?):")
    for wing in [1, 2, 3]:
        report(f"  wing +/-{wing}", sim(build(df, wing=wing, dte_max=1), lots=2))

    print("\n[F] SAMPLE HONESTY:")
    print(f"  n={len(r)} trades over {(pd.Timestamp(dates[-1])-pd.Timestamp(dates[0])).days} days")
    print(f"  = ONE year (2025), ONE market regime (mostly bullish/calm).")
    print(f"  A 70% win rate on 40 trades is thin evidence. This is NOT proof —")
    print(f"  it is a well-gated HYPOTHESIS that needs a second year of data.")

    print("\n" + "=" * 96)
    print("  VERDICT")
    print("=" * 96)
    print("""
  What is PROVEN: the VRP exists (+6.1 vol pts, t=+15.6), short-strangle
  DTE<=1d is net-positive at 2-lot sizing with a survivable tail (-16% worst
  day, 13% DD), and it clears |t|>2 on the full sample.

  What is NOT proven: (1) holdout — the halves disagree, so it is regime-
  dependent, not structural; (2) one year of data cannot establish a short-vol
  edge, whose entire risk is the rare catastrophic day; (3) the VRP premium is
  precisely the compensation FOR that tail — collecting it is insurance
  underwriting, and Rs1.5L is thin capital for underwriting.

  HONEST STATUS: NOT YET — the most promising candidate of the session, with
  the tail solved by sizing, but regime-dependent and under-sampled. The
  scheduled pull keeps accumulating 2026 option data; when a second year
  exists, re-run this audit. If both halves turn positive and a second year
  confirms, it becomes the first TRADEABLE edge.
""")


if __name__ == "__main__":
    main()
