"""Reconcile the engine's MTM P&L against an independent recompute.

Two independent methods must AGREE, or we have a bug:
  A) engine:      P&L = (entry_credit_pts - close_cost_pts) * lot * lots
  B) per-leg:     P&L = sum over legs of signed_qty * (current - entry) * lot * lots
       signed_qty: short legs = -1, long legs = +1
       (P&L direction is (current - entry): a long gains when price rises,
        a short gains when price falls.)

Run: python reconcile_pnl.py
"""
import json
import auto_condor_v2 as ac

fy = ac.get_fyers()
st = json.load(open(ac.STATE))
o = st["open"]
if not o:
    print("no open position to reconcile.")
    raise SystemExit

legs = o["legs"]
qty = o["lot"] * o["lots"]
ltp = ac.quotes(fy, [legs[n]["symbol"] for n in
                     ("ce_short", "ce_long", "pe_short", "pe_long")])

# Method A: engine
pnl_engine = ac.condor_pnl(fy, o)

# Method B: per-leg, correct sign convention (current - entry)
signs = {"ce_short": -1, "ce_long": 1, "pe_short": -1, "pe_long": 1}
total = 0.0
for n, s in signs.items():
    cur = ltp.get(legs[n]["symbol"])
    if cur is None:
        print(f"  {n}: no quote; cannot reconcile.")
        raise SystemExit
    contrib = s * (cur - legs[n]["ltp"]) * qty
    total += contrib
    print(f"  {n:8} {s:+d} x ({cur} - {legs[n]['ltp']}) x {qty} = Rs{contrib:+.2f}")

print(f"\n  engine P&L (A):      Rs{pnl_engine:+.2f}" if pnl_engine is not None else "")
print(f"  per-leg P&L (B):     Rs{total:+.2f}")
if pnl_engine is not None:
    diff = abs(pnl_engine - total)
    print(f"  difference:          Rs{diff:.2f}  -> {'AGREE' if diff < 1 else 'MISMATCH (BUG)'}")
