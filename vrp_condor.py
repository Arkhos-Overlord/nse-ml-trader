"""DEFINED-RISK VRP — iron condor. The tail is capped BY CONSTRUCTION.

Why this and not the naked strangle: the naked strangle's tail is unbounded
(modelled 5x-premium day = -376% of capital = account-ending at Rs1.5L). An
iron condor buys the wings, so the maximum loss per lot is FIXED at
(wing_width - net_premium) x LOT. That is the whole point.

Structure (0-DTE, nearest expiry):
  SELL CE at ATM+2   BUY CE at ATM+5     (call spread)
  SELL PE at ATM-2   BUY PE at ATM-5     (put spread)
  wing width = 3 strikes x 50pts = 150 pts (approx, varies with strike spacing)
  max loss/lot = (150 - net_credit) x 75  -- bounded, known before entry

VRP condition: only trade when premium is RICH (implied > realized). Proxy:
skip days when total credit is below the median (cheap vol = not worth the risk).

Vol-scaled sizing: lots sized so max loss per trade ~ Rs20K (~13% of capital).

Costs: Indian 2024-25 options (STT 0.01% sell... actually options STT is 0.0625%
on sell premium; use that), exchange, GST, slippage.

Run: python vrp_condor.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOT = 75
CAPITAL = 150000.0
BROKERAGE = 20.0
STT_SELL = 0.000625      # options STT on sell premium
EXCH_GST = 0.00004
SLIP = 0.0005
TARGET_RISK_RS = 20000.0
SHORT_WING = 2           # strikes out for the short leg
LONG_WING = 5            # strikes out for the long (protective) leg


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def build(bars_file, dte_max=1, short_wing=SHORT_WING, long_wing=LONG_WING):
    df = pd.read_parquet(bars_file)
    df["date"] = pd.to_datetime(df.date)
    df["expiry"] = pd.to_datetime(df.expiry)
    out = []
    for d, day in df.groupby("date"):
        day = day.sort_values("expiry")
        nearest = day.expiry.min()
        sub = day[day.expiry == nearest]
        if (nearest - d).days > dte_max:
            continue
        atm = sub.atm.iloc[0]
        strikes = sorted(sub.strike.unique())
        if atm not in strikes:
            continue
        ai = strikes.index(atm)
        def get(k, typ, col):
            if not (0 <= k < len(strikes)):
                return None
            r = sub[(sub.strike == strikes[k]) & (sub.type == typ)]
            if r.empty:
                return None
            return float(r.iloc[0][col])
        ce_s = get(ai + short_wing, "CE", "first_close")
        ce_l = get(ai + long_wing, "CE", "first_close")
        pe_s = get(ai - short_wing, "PE", "first_close")
        pe_l = get(ai - long_wing, "PE", "first_close")
        ce_s_x = get(ai + short_wing, "CE", "last_close")
        ce_l_x = get(ai + long_wing, "CE", "last_close")
        pe_s_x = get(ai - short_wing, "PE", "last_close")
        pe_l_x = get(ai - long_wing, "PE", "last_close")
        if any(v is None for v in [ce_s, ce_l, pe_s, pe_l, ce_s_x, ce_l_x, pe_s_x, pe_l_x]):
            continue
        out.append({"date": d.date(), "expiry": nearest.date(), "atm": atm,
                    "ce_s": ce_s, "ce_l": ce_l, "pe_s": pe_s, "pe_l": pe_l,
                    "ce_s_x": ce_s_x, "ce_l_x": ce_l_x,
                    "pe_s_x": pe_s_x, "pe_l_x": pe_l_x})
    return pd.DataFrame(out)


def condor_pnl(r, lots):
    """Iron condor P&L per lot, in points, then x LOT x lots.
    credit = (ce_s - ce_l) + (pe_s - pe_l)     [sell short, buy long]
    debit  = (ce_s_x - ce_l_x) + (pe_s_x - pe_l_x)   [cost to close]
    Max loss is bounded: the long legs cap it."""
    credit = (r.ce_s - r.ce_l) + (r.pe_s - r.pe_l)
    debit = (r.ce_s_x - r.ce_l_x) + (r.pe_s_x - r.pe_l_x)
    pnl_pts = credit - debit
    # max loss per lot in points = wing_width - max(0, credit)
    wing_pts = 3 * 50.0    # approx; refine with actual strike spacing
    max_loss_pts = max(wing_pts - credit, 0)
    return pnl_pts, credit, max_loss_pts


def sim(strat, lots, vrp_min_credit=None):
    res = []
    for _, r in strat.iterrows():
        pnl_pts, credit, max_loss_pts = condor_pnl(r, lots)
        if vrp_min_credit is not None and credit < vrp_min_credit:
            continue
        notional = lots * LOT * r.atm
        # costs: STT on sell premium (both short legs), exchange, GST, slip
        sell_prem = (r.ce_s + r.pe_s) * lots * LOT
        cost = lots * (2 * BROKERAGE + notional * EXCH_GST
                       + sell_prem * STT_SELL
                       + (r.ce_s + r.pe_s + r.ce_s_x + r.pe_s_x) * lots * LOT * SLIP)
        pnl = pnl_pts * LOT * lots - cost
        res.append({"date": r.date, "lots": lots, "credit": credit,
                    "max_loss_pts": max_loss_pts,
                    "pnl": pnl, "max_loss_rs": max_loss_pts * LOT * lots})
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
    import os
    print("=" * 100)
    print("  DEFINED-RISK VRP — IRON CONDOR (tail capped by construction)")
    print("=" * 100)
    files = [("2024Q4", "condor_bars_2024.parquet"),
             ("2025", "condor_bars_2025.parquet")]
    for nm, f in files:
        if not os.path.exists(f):
            print(f"\n  {f} not ready yet")
            continue
        s = build(f, dte_max=1)
        print(f"\n--- {nm}: {len(s)} tradeable days ---")
        if s.empty:
            continue
        # sizing: 1 lot to start (max loss bounded)
        r = sim(s, lots=1)
        report(f"{nm} 1 lot", r)
        if not r.empty:
            print(f"    avg credit {r.credit.mean():.0f} pts, "
                  f"avg max-loss/lot Rs{r.max_loss_rs.mean():,.0f}, "
                  f"worst actual Rs{r.pnl.min():+,.0f}")

    # combined if both ready
    if all(os.path.exists(f) for _, f in files):
        s24, s25 = build("condor_bars_2024.parquet"), build("condor_bars_2025.parquet")
        print(f"\n--- COMBINED multi-regime ---")
        for lots in [1, 2]:
            r = pd.concat([sim(s24, lots), sim(s25, lots)], ignore_index=True)
            report(f"combined {lots} lots", r)
        print("\n--- VRP GATE (only rich-credit days) ---")
        r_all = pd.concat([sim(s24, 1), sim(s25, 1)], ignore_index=True)
        if not r_all.empty:
            med = r_all.credit.median()
            report(f"credit >= median ({med:.0f}pts)", r_all[r_all.credit >= med])
        print("\n--- TAIL: is max loss actually bounded? ---")
        r = pd.concat([sim(s24, 1), sim(s25, 1)], ignore_index=True)
        if not r.empty:
            print(f"    modelled max loss/lot: Rs{r.max_loss_rs.mean():,.0f} avg, "
                  f"Rs{r.max_loss_rs.max():,.0f} worst")
            print(f"    ACTUAL worst trade:    Rs{r.pnl.min():+,.0f} "
                  f"({r.pnl.min()/CAPITAL*100:+.0f}% of capital)")
            print(f"    -> actual stayed within the bound: "
                  f"{'YES' if r.pnl.min() > -r.max_loss_rs.max() else 'NO — CHECK'}")


if __name__ == "__main__":
    main()
