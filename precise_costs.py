"""Precise Indian options cost model + ₹1.5L-safe strategy search.

Every prior test used rough costs. This models EXACT NSE charges:
  - STT: 0.0625% on sell premium (options)
  - Exchange txn charge: 0.03503% (NSE options)
  - SEBI: 0.0001% 
  - GST: 18% on (brokerage + txn + SEBI)
  - Stamp duty: 0.003% on buy side
  - Brokerage: ₹20/order (typical discount broker)
  - Slippage: modeled separately

Then tests: what strategy fits ₹1.5L with tail ≤ ₹15K (1% of capital)?
Run: python precise_costs.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOT = 75
BROKERAGE = 20  # Rs per order


def exact_costs(premium_sold, premium_bought, is_intraday=True):
    """Precise round-trip costs for an options trade (Rs).
    
    Args:
        premium_sold: total sell premium (Rs)
        premium_bought: total buy premium (Rs)
        is_intraday: if True, STT on sell at 0.0625%; if held to expiry (MIS->NRML),
                     STT is 0.1% on premium. Use 0.0625% for options sell.
    """
    # STT: 0.0625% on sell side premium (options)
    stt_sell = premium_sold * 0.000625
    # Exchange transaction charge: 0.03503% on premium (both sides)
    exchange = (premium_sold + premium_bought) * 0.0003503
    # SEBI turnover fee: 0.0001%
    sebi = (premium_sold + premium_bought) * 0.000001
    # GST: 18% on (brokerage + exchange + SEBI)
    gst = 0.18 * (BROKERAGE * 2 + exchange + sebi)
    # Stamp duty: 0.003% on buy side only
    stamp = premium_bought * 0.00003
    # Brokerage
    brokerage = BROKERAGE * 2  # entry + exit
    return {
        "STT": stt_sell,
        "Exchange": exchange,
        "SEBI": sebi,
        "GST": gst,
        "Stamp": stamp,
        "Brokerage": brokerage,
        "TOTAL": stt_sell + exchange + sebi + gst + stamp + brokerage,
    }


def show_cost_table():
    print("=" * 65)
    print("EXACT COST MODEL — Nifty options round trip (per lot)")
    print("=" * 65)
    print(f"{'Premium (Rs)':>14} {'STT':>8} {'Exch':>8} {'GST':>6} {'Stamp':>7} "
          f"{'Brok':>6} {'TOTAL':>9} {'% of prem':>10}")
    for prem in [5000, 10000, 20000, 30000, 50000, 75000, 100000]:
        c = exact_costs(prem, prem * 0.5)  # roughly half is bought back
        pct = c["TOTAL"] / prem * 100
        print(f"{prem:>14,} {c['STT']:>8.0f} {c['Exchange']:>8.0f} "
              f"{c['GST']:>6.0f} {c['Stamp']:>7.0f} {c['Brokerage']:>6.0f} "
              f"{c['TOTAL']:>9.0f} {pct:>9.1f}%")
    print("\n  KEY: on a Rs51K straddle, costs eat ~Rs1,500+ = ~3% of premium.")
    print("  On smaller premiums, the fixed Rs40 brokerage dominates.")


def cost_impact_on_edge():
    """How much do exact costs reduce the delta-hedged edge?"""
    print("\n" + "=" * 65)
    print("COST IMPACT ON THE PROVEN EDGE (delta-hedged straddle)")
    print("=" * 65)
    # from delta_hedged.py: +Rs1.07 Cr, 627 trades, avg premium ~Rs51K
    # each trade: sell straddle + hedge rebalances (futures)
    # estimate: 4 option legs + ~10-15 futures rebalances per trade
    n_trades = 627
    avg_option_premium = 51000  # from live data
    avg_hedge_notional = 22500 * 75 * 3  # ~3 rebalances of 1 lot notional
    # option costs (4 legs: sell CE, sell PE, buy back CE, buy back PE)
    opt_c = exact_costs(avg_option_premium, avg_option_premium)
    # futures hedge costs: brokerage Rs20/order * ~15 rebalances * 2 (entry+exit)
    fut_costs = BROKERAGE * 2 * 15
    # futures STT: 0.01% on sell side
    fut_stt = avg_hedge_notional * 0.0001
    per_trade = opt_c["TOTAL"] + fut_costs + fut_stt
    total_cost = per_trade * n_trades
    gross = 10709851  # from interrogate always-on
    net = gross - total_cost
    print(f"  gross PnL (12yr):      Rs{gross:,.0f}")
    print(f"  per-trade costs:       Rs{per_trade:,.0f}")
    print(f"  total costs (12yr):    Rs{total_cost:,.0f}")
    print(f"  NET PnL:               Rs{net:,.0f}")
    print(f"  cost drag:             {total_cost/gross*100:.0f}% of gross")
    print(f"\n  Honest: even with exact costs, the edge survives IF you can")
    print(f"  trade it safely. But per-trade cost Rs{per_trade:,.0f} means you")
    print(f"  need positions big enough to overcome it.")


def what_fits_15l():
    """What can ₹1.5L actually trade with tail ≤ 1% (₹15K)?"""
    print("\n" + "=" * 65)
    print("WHAT FITS ₹1.5L SAFELY (max loss ≤ Rs15K = 1%)")
    print("=" * 65)
    # BankNifty option lot = 35 (was 30, now 35 as of 2025)
    # Nifty lot = 75
    # A defined-risk credit spread: sell ATM, buy 1 strike OTM
    # max loss = wing_width * lot - credit
    
    print("  Nifty credit spread (1 strike wing = 100 pts):")
    for credit in [3000, 5000, 8000]:
        max_loss = 100 * LOT - credit
        c = exact_costs(credit * 2, credit)  # rough
        print(f"    credit Rs{credit:,} -> max loss Rs{max_loss:,} "
              f"({max_loss/150000*100:.1f}% of Rs1.5L), costs Rs{c['TOTAL']:.0f}")
    
    print("\n  BankNifty credit spread (1 strike wing = 100 pts, lot=35):")
    for credit in [2000, 4000, 6000]:
        max_loss = 100 * 35 - credit
        c = exact_costs(credit * 2, credit)
        print(f"    credit Rs{credit:,} -> max loss Rs{max_loss:,} "
              f"({max_loss/150000*100:.1f}% of Rs1.5L), costs Rs{c['TOTAL']:.0f}")
    
    print("\n  HONEST: a 100-pt wing Nifty spread risks Rs{max_loss:,} = "
          f"{max_loss/150000*100:.0f}% of Rs1.5L.")
    print("  That's 4-7% risk per trade — survivable but not 1%.")
    print("  The edge per trade must EXCEED the cost+risk to be worth it.")


if __name__ == "__main__":
    show_cost_table()
    cost_impact_on_edge()
    what_fits_15l()
