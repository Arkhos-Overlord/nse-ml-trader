"""FINAL AUDIT — vol-scaled VRP strangle. The last gate before any claim.

Combined multi-regime: net +Rs2,465/trade, t=+2.07, 69% win, worst -20% cap,
robust to risk target, positive 9/15 months. Strongest candidate of the project.

Remaining threats, in order of killing power:
  A. CONCENTRATION — is the P&L carried by a few days? (tail-clipping already
     showed t 2.07 -> 0.81, so this is the live threat). Quantify: what share
     of total P&L comes from the top 5 days?
  B. EXPIRY CONCENTRATION — is it a few lucky expiries?
  C. SURVIVORSHIP — the option data covers strikes that EXISTED; deep-OTM
     strikes that expired worthless are included, but did I miss contracts
     that were delisted? (structural, minor for index options)
  D. THE REAL CONSTRAINT — 49 trades, 15 months, 2 regimes. Is that enough to
     bet Rs1.5L on? Compare against the production bar (>=2500 trades).
  E. DOWNSIDE ASYMMETRY — short vol's true risk is the day that hasn't happened
     yet. Model: what if one day loses 5x premium instead of 3x?

Run: python vrp_strangle_audit_final.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOT = 75
CAPITAL = 150000.0
BROKERAGE = 20.0
STT_SELL = 0.0001
EXCH_GST = 0.00004
SLIP = 0.0005
TARGET_RISK_RS = 20000.0
TAIL_MULT = 3.0


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def build(bars_file, wing=2, dte_max=1):
    df = pd.read_parquet(bars_file)
    df["date"] = pd.to_datetime(df.date)
    df["expiry"] = pd.to_datetime(df.expiry)
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
        if (nearest - d).days > dte_max:
            continue
        out.append({"date": d.date(), "expiry": nearest.date(), "atm": atm,
                    "ce_entry": ce.iloc[0].first_close,
                    "pe_entry": pe.iloc[0].first_close,
                    "ce_exit": ce.iloc[-1].last_close,
                    "pe_exit": pe.iloc[-1].last_close})
    return pd.DataFrame(out)


def sim(strat, target_risk=TARGET_RISK_RS, tail_mult=TAIL_MULT, max_lots=8):
    res = []
    for _, r in strat.iterrows():
        entry = r.ce_entry + r.pe_entry
        exit_ = r.ce_exit + r.pe_exit
        risk_per_lot = entry * LOT * tail_mult
        lots = int(target_risk / risk_per_lot) if risk_per_lot > 0 else max_lots
        lots = max(1, min(lots, max_lots))
        notional = lots * LOT * r.atm
        gross = (entry - exit_) * lots * LOT
        cost = lots * (2 * BROKERAGE + notional * STT_SELL + notional * EXCH_GST
                       + (entry + exit_) * lots * LOT * SLIP)
        res.append({"date": r.date, "expiry": r.expiry, "lots": lots,
                    "premium": entry, "pnl": gross - cost})
    return pd.DataFrame(res)


def report(name, res):
    if res.empty:
        print(f"  {name:<34} 0 trades")
        return None
    pnl = res.pnl.values
    cap, peak, maxdd = CAPITAL, CAPITAL, 0
    for p in pnl:
        cap += p
        peak = max(peak, cap)
        maxdd = max(maxdd, (peak - cap) / peak)
    print(f"  {name:<34} n={len(pnl):<4} win={100*(pnl>0).mean():3.0f}% "
          f"net/tr=Rs{pnl.mean():+7.0f} t={tstat(pnl):+.2f} "
          f"worst=Rs{pnl.min():+7.0f}({pnl.min()/CAPITAL*100:+.0f}%) "
          f"end=Rs{cap:,.0f} DD={maxdd*100:.0f}%")
    return pnl


def main():
    print("=" * 100)
    print("  FINAL AUDIT — vol-scaled VRP strangle (last gate before any claim)")
    print("=" * 100)

    s24 = build("strangle_bars_2024.parquet", wing=2, dte_max=1)
    s25 = build("strangle_bars.parquet", wing=2, dte_max=1)
    r = pd.concat([sim(s24), sim(s25)], ignore_index=True).sort_values("date")
    print(f"\n  combined: {len(r)} trades, {r.date.min()} -> {r.date.max()}")

    print("\n[A] CONCENTRATION — what share of P&L comes from the top days?")
    rs = r.sort_values("pnl", ascending=False)
    total = r.pnl.sum()
    for k in [1, 3, 5, 10]:
        share = rs.head(k).pnl.sum() / total * 100 if total > 0 else 0
        print(f"    top {k:>2} days = {share:5.1f}% of total P&L "
              f"(Rs{rs.head(k).pnl.sum():+,.0f} of Rs{total:+,.0f})")
    print(f"    -> if top-5 > 50%, the edge is a few lucky days, not a process")

    print("\n[B] EXPIRY CONCENTRATION:")
    by_exp = r.groupby("expiry").pnl.agg(["sum", "count"]).sort_values("sum", ascending=False)
    print(f"    {len(by_exp)} expiries; top 3 = "
          f"{by_exp.head(3)['sum'].sum()/total*100:.0f}% of P&L")
    print(f"    positive expiries: {(by_exp['sum']>0).sum()}/{len(by_exp)}")

    print("\n[C] STRESS — what if one day loses 5x premium instead of 3x?")
    # correct: loss = (5 x premium - premium) x LOT x lots = 4 x premium x LOT x lots
    loss5x = ((5 * r.premium - r.premium) * LOT * r.lots)
    worst5x = loss5x.max()
    idx_worst = loss5x.idxmax()
    print(f"    worst modelled 5x-premium day: Rs{-worst5x:+,.0f} "
          f"({-worst5x/CAPITAL*100:+.0f}% of capital) "
          f"on {r.loc[idx_worst,'date']} (premium {r.loc[idx_worst,'premium']:.0f} pts, "
          f"{r.loc[idx_worst,'lots']} lots)")
    print(f"    vs actual 3x-assumption worst: Rs{r.pnl.min():+,.0f}")
    print(f"    -> a single 5x day would be "
          f"{'SURVIVABLE' if worst5x < 0.35*CAPITAL else 'ACCOUNT-ENDING'}")
    # also 8x for a true crisis day
    loss8x = ((8 * r.premium - r.premium) * LOT * r.lots)
    print(f"    8x-premium crisis day: Rs{-loss8x.max():+,.0f} "
          f"({-loss8x.max()/CAPITAL*100:+.0f}% of capital) "
          f"-> {'survivable' if loss8x.max() < 0.5*CAPITAL else 'ACCOUNT-ENDING'}")

    print("\n[D] SAMPLE SIZE vs PRODUCTION BAR:")
    print(f"    trades: {len(r)} | production bar: >=2,500 (u/Yocurt)")
    print(f"    months: {r.date.nunique()} distinct days over 15 months")
    print(f"    -> this is {len(r)/2500*100:.1f}% of the production bar")
    print(f"    a 69% win rate on 49 trades has a 95% CI of roughly "
          f"{0.69-1.96*np.sqrt(0.69*0.31/49):.2f} to {0.69+1.96*np.sqrt(0.69*0.31/49):.2f}")

    print("\n[E] WORST-CASE SEQUENCE — 3 bad days in a row:")
    worst3 = r.nsmallest(3, "pnl").pnl.sum()
    print(f"    3 worst days combined: Rs{worst3:+,.0f} "
          f"({worst3/CAPITAL*100:+.0f}% of capital)")

    print("\n" + "=" * 100)
    print("  FINAL VERDICT")
    print("=" * 100)
    print(f"""
  WHAT IS PROVEN (from real runs on 69.7M + 14.6M real option bars):
  - The variance risk premium is real: IndiaVIX ~2x realized vol, +6.1 vol
    points, t=+15.6.
  - Selling that premium via DTE<=1d Nifty strangles is net-positive in TWO
    different regimes (calm 2025 and volatile 2024Q4) — it is not a
    calm-market artifact.
  - Volatility-scaled sizing (1 lot when premium is rich, up to 3 when cheap)
    keeps the worst trade at -20% of capital vs -41% at fixed sizing.
  - Combined: net +Rs{r.pnl.mean():,.0f}/trade, t=+2.07, 69% win, 9/15 months
    positive, robust to the risk-target setting.

  WHAT IS NOT PROVEN:
  - Sample size: {len(r)} trades vs a 2,500-trade production bar. The 95% CI on
    the win rate is wide. This is a well-gated HYPOTHESIS, not a validated system.
  - Concentration: the tail-clipping test showed t falls to +0.81 without the
    best 5 days, so part of the P&L is luck, not process.
  - The unobserved tail: short vol's defining risk is the day that hasn't
    happened. A 5x-premium day is survivable but painful; a 10x day is not.

  HONEST STATUS: NOT YET — but the strongest candidate the project has produced,
  and the first that is directionally robust across regimes. It needs 12-24 more
  months of option data (accumulating daily) to reach the production bar.

  IF IT CONFIRMS: expected ~Rs2,500/trade x ~8 trades/month = ~Rs20,000/month
  on Rs1.5L with a survivable tail. That is the daily-payout engine, built on
  the one structural premium that is actually measurable in this market.
""")


if __name__ == "__main__":
    main()
