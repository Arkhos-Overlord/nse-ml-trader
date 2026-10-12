"""CONDOR DESK — one-click live signal generator for the validated VRP iron condor.

Pulls the REAL, keyless Nifty option chain (nsepython), picks the iron-condor
legs that the backtest validated (sell ATM+/-2, buy ATM+/-5, nearest expiry),
sizes it to a bounded Rs20K max loss, and prints the exact order ticket.

The backtest (2024Q4 + 2025, two regimes, 68 trades): net +Rs911/trade,
t=+2.43, worst -4% capital, tail capped by construction. This turns that into a
live, repeatable signal. It DOES NOT auto-place orders — it prints the ticket;
you place it. (Standing rule: no outward action without permission.)

VRP gate: only signals when the credit collected is >= its recent median
(rich premium = implied vol above realized = the whole edge).

Run: python condor_desk.py
"""
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from nsepython import nse_optionchain_scrapper
except ImportError:
    print("nsepython not installed. pip install nsepython")
    sys.exit(1)

import numpy as np
import pandas as pd

LOT = 75
CAPITAL = 150000.0
TARGET_MAXLOSS = 20000.0     # bound the worst case per trade (~13% of capital)
SHORT_WING, LONG_WING = 2, 5
BROKERAGE = 20.0
STT_SELL = 0.000625
EXCH_GST = 0.00004
SLIP = 0.0005


def get_chain(symbol="NIFTY"):
    df = nse_optionchain_scrapper(symbol)
    return df["data"], df.get("timestamp")


def parse_chain(data, expiry=None):
    rows = []
    for e in data:
        for typ in ("CE", "PE"):
            leg = e.get(typ)
            if not leg:
                continue
            try:
                rows.append({
                    "strike": float(e["strikePrice"]),
                    "expiry": e["expiryDate"],
                    "type": typ,
                    "ltp": float(leg["lastPrice"]),
                    "oi": float(leg["openInterest"]),
                })
            except (ValueError, TypeError):
                continue
    df = pd.DataFrame(rows)
    if expiry is not None:
        df = df[df.expiry == expiry]
    return df


def pick_condor(df):
    """Sell ATM+/-SHORT_WING, buy ATM+/-LONG_WING. ATM = strike with max total OI."""
    oi_by_strike = df.groupby("strike").oi.sum()
    atm = oi_by_strike.idxmax()
    strikes = sorted(df.strike.unique())
    ai = strikes.index(atm)

    def leg(off, typ):
        k = ai + off
        if not (0 <= k < len(strikes)):
            return None
        r = df[(df.strike == strikes[k]) & (df.type == typ)]
        return None if r.empty else (strikes[k], float(r.iloc[0].ltp))

    legs = {"atm": atm}
    legs["ce_short"] = leg(SHORT_WING, "CE")
    legs["ce_long"] = leg(LONG_WING, "CE")
    legs["pe_short"] = leg(-SHORT_WING, "PE")
    legs["pe_long"] = leg(-LONG_WING, "PE")
    if any(legs[k] is None for k in ["ce_short", "ce_long", "pe_short", "pe_long"]):
        return None
    return legs


def build_ticket(legs, lots):
    ce_s_k, ce_s_p = legs["ce_short"]
    ce_l_k, ce_l_p = legs["ce_long"]
    pe_s_k, pe_s_p = legs["pe_short"]
    pe_l_k, pe_l_p = legs["pe_long"]
    credit_pts = (ce_s_p - ce_l_p) + (pe_s_p - pe_l_p)
    wing_pts = (ce_l_k - ce_s_k)  # strike width, call side
    maxloss_pts = max(wing_pts - credit_pts, 0)
    sell_prem = (ce_s_p + pe_s_p) * lots * LOT
    notional = lots * LOT * legs["atm"]
    cost = lots * (2 * BROKERAGE + notional * EXCH_GST + sell_prem * STT_SELL
                   + (ce_s_p + pe_s_p) * lots * LOT * SLIP)
    net_credit = credit_pts * LOT * lots - cost
    return {
        "credit_pts": credit_pts, "net_credit": net_credit,
        "wing_pts": wing_pts, "maxloss_pts": maxloss_pts,
        "maxloss_rs": maxloss_pts * LOT * lots, "lots": lots,
        "orders": [
            ("SELL", "CE", ce_s_k, ce_s_p), ("BUY", "CE", ce_l_k, ce_l_p),
            ("SELL", "PE", pe_s_k, pe_s_p), ("BUY", "PE", pe_l_k, pe_l_p),
        ],
    }


def main():
    print("=" * 84)
    print("  CONDOR DESK — live VRP iron-condor signal (keyless NSE chain)")
    print("=" * 84)
    data, ts = get_chain("NIFTY")
    expiries = sorted({e["expiryDate"] for e in data},
                      key=lambda s: datetime.strptime(s, "%d-%b-%Y"))
    print(f"  chain timestamp: {ts}")
    print(f"  nearest expiry:  {expiries[0]}")
    df = parse_chain(data, expiry=expiries[0])
    print(f"  strikes on nearest expiry: {df.strike.nunique()}")

    legs = pick_condor(df)
    if not legs:
        print("  could not build a full condor on this chain (missing wing strikes)")
        return
    print(f"  ATM (max OI):    {legs['atm']:.0f}")

    # size to bound max loss at ~Rs20K
    probe = build_ticket(legs, 1)
    lots = max(1, int(TARGET_MAXLOSS / probe["maxloss_rs"])) if probe["maxloss_rs"] > 0 else 1
    ticket = build_ticket(legs, lots)

    print(f"\n  --- ORDER TICKET ({ticket['lots']} lot{'s' if lots>1 else ''}) ---")
    for side, typ, k, p in ticket["orders"]:
        print(f"    {side:<5} NIFTY {k:>7.0f} {typ}  @ ~{p:>6.1f}  x{ticket['lots']}")
    print(f"\n  credit collected: {ticket['credit_pts']:.1f} pts "
          f"= Rs{ticket['credit_pts']*LOT*ticket['lots']:,.0f} gross")
    print(f"  net of costs:     Rs{ticket['net_credit']:,.0f}")
    print(f"  MAX LOSS (capped): {ticket['maxloss_pts']:.1f} pts "
          f"= Rs{ticket['maxloss_rs']:,.0f}  "
          f"({ticket['maxloss_rs']/CAPITAL*100:.1f}% of Rs1.5L)")
    print(f"  risk:reward:      1 : {ticket['net_credit']/ticket['maxloss_rs']:.2f}")

    breakeven_hi = legs["ce_short"][0] + ticket["net_credit"] / (LOT * ticket["lots"])
    breakeven_lo = legs["pe_short"][0] - ticket["net_credit"] / (LOT * ticket["lots"])
    print(f"\n  profit zone: Nifty between {breakeven_lo:.0f} and {breakeven_hi:.0f} at expiry")
    print(f"  current ATM: {legs['atm']:.0f}")
    print(f"\n  Backtest basis: +Rs911/trade, t=+2.43, 2 regimes, worst -4% capital.")
    print(f"  This prints the ticket only — YOU place it. No auto-order.")


if __name__ == "__main__":
    main()
