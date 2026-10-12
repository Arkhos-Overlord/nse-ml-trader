"""VOLATILITY-SCALED SIZING — the fix the two-regime test demands.

The finding: DTE<=1d 2-lot strangle is positive in BOTH regimes (2025 +Rs4,194/
trade t=+2.35; 2024Q4 +Rs3,243 t=+0.30, 10/15 months positive) — so the edge is
NOT a calm-market artifact. BUT the tail scales with vol: worst trade -16% of
capital in calm 2025 vs -41% in volatile 2024Q4. Fixed 2-lot sizing is
unsurvivable in a vol spike.

THE FIX: size inversely to the premium being collected. Premium IS the vol
signal — high premium = high vol = size down; low premium = calm = size up.
This keeps the ABSOLUTE rupee tail roughly constant across regimes.

  lots = clamp(TARGET_RISK_RS / (premium_collected * LOT * tail_mult), 1, 8)

Then re-run the full audit on the COMBINED 2024Q4+2025 sample — the true
multi-regime test.

Run: python vrp_strangle_volscaled.py
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
TARGET_RISK_RS = 20000.0     # target max loss per trade in rupees (~13% of capital)
TAIL_MULT = 3.0              # assume worst case = 3x premium collected


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
        out.append({"date": d.date(), "atm": atm,
                    "ce_entry": ce.iloc[0].first_close,
                    "pe_entry": pe.iloc[0].first_close,
                    "ce_exit": ce.iloc[-1].last_close,
                    "pe_exit": pe.iloc[-1].last_close})
    return pd.DataFrame(out)


def sim_volscaled(strat, target_risk=TARGET_RISK_RS, tail_mult=TAIL_MULT,
                  max_lots=8, min_lots=1):
    res = []
    for _, r in strat.iterrows():
        entry = r.ce_entry + r.pe_entry
        exit_ = r.ce_exit + r.pe_exit
        # size so that a tail_mult x premium move loses ~target_risk rupees
        risk_per_lot = entry * LOT * tail_mult
        lots = int(target_risk / risk_per_lot) if risk_per_lot > 0 else max_lots
        lots = max(min_lots, min(lots, max_lots))
        notional = lots * LOT * r.atm
        gross = (entry - exit_) * lots * LOT
        cost = lots * (2 * BROKERAGE + notional * STT_SELL + notional * EXCH_GST
                       + (entry + exit_) * lots * LOT * SLIP)
        res.append({"date": r.date, "lots": lots, "premium": entry,
                    "pnl": gross - cost})
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
          f"worst=Rs{pnl.min():+7.0f}({pnl.min()/CAPITAL*100:+.0f}%) "
          f"end=Rs{cap:,.0f} DD={maxdd*100:.0f}%")
    return pnl


def main():
    print("=" * 100)
    print("  VOLATILITY-SCALED SIZING — the fix the two-regime test demands")
    print("=" * 100)
    print(f"  target risk Rs{TARGET_RISK_RS:,.0f}/trade, tail assumption {TAIL_MULT}x premium")
    print(f"  lots scale DOWN when premium (vol) is high — keeps the rupee tail flat\n")

    s24 = build("strangle_bars_2024.parquet", wing=2, dte_max=1)
    s25 = build("strangle_bars.parquet", wing=2, dte_max=1)

    print("--- SIZING BEHAVIOUR (does it actually scale?) ---")
    for nm, s in [("2024Q4 (volatile)", s24), ("2025 (calm)", s25)]:
        r = sim_volscaled(s)
        if not r.empty:
            print(f"  {nm}: avg lots {r.lots.mean():.1f}, "
                  f"avg premium {r.premium.mean():.0f} pts, "
                  f"lots range {r.lots.min()}-{r.lots.max()}")

    print("\n--- 2024 Q4 with vol-scaled sizing (was -41% worst at fixed 2 lots) ---")
    report("2024Q4 vol-scaled", sim_volscaled(s24))

    print("\n--- 2025 with vol-scaled sizing ---")
    report("2025 vol-scaled", sim_volscaled(s25))

    print("\n--- COMBINED multi-regime (the real test) ---")
    both = pd.concat([sim_volscaled(s24), sim_volscaled(s25)],
                     ignore_index=True).sort_values("date")
    report("2024Q4+2025 vol-scaled", both)
    if not both.empty:
        both["ym"] = pd.to_datetime(both.date).dt.to_period("M")
        pos_m = 0
        print("\n  monthly:")
        for ym, g in both.groupby("ym"):
            m = g.pnl.mean()
            if m > 0:
                pos_m += 1
            print(f"    {ym}: n={len(g):<3} lots={g.lots.mean():.1f} "
                  f"net/tr=Rs{m:+7.0f} t={tstat(g.pnl):+.2f}")
        print(f"  -> positive in {pos_m}/{both.ym.nunique()} months")
        # tail check by regime
        both["yr"] = pd.to_datetime(both.date).dt.year
        for yr, g in both.groupby("yr"):
            print(f"    {yr}: worst Rs{g.pnl.min():+,.0f} "
                  f"({g.pnl.min()/CAPITAL*100:+.0f}% cap), DD-relevant")

    print("\n--- RISK TARGET SWEEP (is the result robust to the risk setting?) ---")
    for tr in [12000, 20000, 30000]:
        b = pd.concat([sim_volscaled(s24, target_risk=tr),
                       sim_volscaled(s25, target_risk=tr)], ignore_index=True)
        report(f"  target risk Rs{tr:,}", b)

    print("\n--- TAIL CLIPPING (remove best 5 days) ---")
    b = pd.concat([sim_volscaled(s24), sim_volscaled(s25)], ignore_index=True)
    report("  ex-best-5-days", b.nlargest(len(b) - 5, "pnl").tail(len(b) - 5)
           if False else b.sort_values("pnl", ascending=False).iloc[5:])

    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)
    print("""
  TRADEABLE requires: combined multi-regime net/tr > 0 with |t| > 2, worst
  trade under ~20% of capital in BOTH regimes, positive in most months, and
  robustness to the risk-target setting.

  The edge is directionally robust (positive in both regimes, 10/15 months) —
  it is NOT a calm-market artifact. The question is whether vol-scaled sizing
  keeps the tail survivable across regimes while retaining significance.
""")


if __name__ == "__main__":
    main()
