"""THE DECISIVE TEST — DTE<=1d 2-lot VRP strangle on 2024 Q4 (different regime).

2025 H1 (calm/bullish): net +Rs4,194/trade, t=+2.35, 70% win, but Q3/Q4 2025
were negative -> calm-market artifact suspected.

2024 Q4 is a DIFFERENT regime: post-election volatility, sharp swings, Nifty
21600->27950 range. If the edge survives here, it is not a calm-market artifact.

Same config, same costs, same sizing. Also: COMBINED 2024Q4 + 2025 full-year,
which is the true multi-regime sample.

Run: python vrp_strangle_2024.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
        out.append({"date": d.date(), "dte": (nearest - d).days, "atm": atm,
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
        res.append({"date": r.date, "pnl": gross - cost})
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
    print("=" * 96)
    print("  THE DECISIVE TEST — VRP strangle on 2024 Q4 (different regime)")
    print("=" * 96)

    s24 = build("strangle_bars_2024.parquet", wing=2, dte_max=1)
    s25 = build("strangle_bars.parquet", wing=2, dte_max=1)
    print(f"\n  2024 Q4: {len(s24)} tradeable days | 2025 full: {len(s25)}")

    print("\n--- 2024 Q4 ALONE (the new, different regime) ---")
    r24 = sim(s24, lots=2)
    report("2024 Q4, 2 lots", r24)
    r24_8 = sim(s24, lots=8)
    report("2024 Q4, 8 lots", r24_8)

    print("\n--- 2025 ALONE (for comparison, same code) ---")
    r25 = sim(s25, lots=2)
    report("2025 full, 2 lots", r25)

    print("\n--- COMBINED (the true multi-regime sample) ---")
    both = pd.concat([r24, r25], ignore_index=True).sort_values("date")
    report("2024Q4 + 2025, 2 lots", both)
    if not both.empty:
        both["yr"] = pd.to_datetime(both.date).dt.year
        for yr, g in both.groupby("yr"):
            print(f"      {yr}: n={len(g):<3} net/tr=Rs{g.pnl.mean():+7.0f} "
                  f"t={tstat(g.pnl):+.2f} win={100*(g.pnl>0).mean():.0f}%")

    print("\n--- MONTHLY PERSISTENCE (2024Q4 + 2025) ---")
    if not both.empty:
        both["ym"] = pd.to_datetime(both.date).dt.to_period("M")
        pos_m = 0
        for ym, g in both.groupby("ym"):
            m = g.pnl.mean()
            if m > 0:
                pos_m += 1
            print(f"    {ym}: n={len(g):<3} net/tr=Rs{m:+7.0f} t={tstat(g.pnl):+.2f}")
        print(f"  -> positive in {pos_m}/{both.ym.nunique()} months")

    print("\n--- WING PLATEAU on 2024 Q4 (is it real here too?) ---")
    for wing in [1, 2, 3]:
        report(f"  2024Q4 wing +/-{wing}", sim(build("strangle_bars_2024.parquet",
                                                    wing=wing, dte_max=1), lots=2))

    print("\n" + "=" * 96)
    print("  VERDICT")
    print("=" * 96)
    print("""
  If 2024 Q4 (a volatile, different regime) is ALSO net-positive with |t|>2,
  the edge is not a calm-market artifact and the candidate is real.
  If 2024 Q4 is negative, the edge is confirmed as calm-market-only and DEAD.

  Either way this is the answer the audit demanded — and I have the data.
""")


if __name__ == "__main__":
    main()
