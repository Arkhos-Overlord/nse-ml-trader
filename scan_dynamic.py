"""DYNAMIC condor scanner — finds the BEST trade, not the first that fits.

Unlike scan_small_condor.py (16 hardcoded names, ranked by raw credit), this:
  1. DISCOVERS liquid underlyings from the Fyers symbol master (any stock with
     an active near-expiry option chain), capped to a scan budget for speed.
  2. Builds a condor per underlying with a SPREAD filter (ask-bid on the short
     legs) so illiquid strikes are rejected — liquidity protects the edge.
  3. Scores by REWARD PER RISK = credit_1lot / maxloss_1lot — "more profit for
     less risk", which is what 'better trades' actually means. Raw credit favors
     risky names; per-risk favors efficiency.
  4. Returns the ranked field; the engine takes the top pick that fits capital.

Run:  python scan_dynamic.py            # ranked table
      python scan_dynamic.py --top 3    # only top 3
"""
import csv
import io
import json
import math
import re
import urllib.request
from datetime import datetime

from fyers_apiv3 import fyersModel

import auto_condor_v2 as ac

INDEX_BAD = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX")

SCAN_BUDGET = 40          # max underlyings to probe (keeps the scan ~1-2 min)
MAX_SHORT_SPREAD_PCT = 0.12   # short-leg bid/ask spread <= 12% of price
MIN_CREDIT_RS_1LOT = 60.0     # ignore traps: Rs12/lot needs 100s of lots, cost eats it
MAX_LOTS = 12                 # each lot = 8 orders (4 legs in+out); cap cost drag
S_SHORT, S_LONG = 1, 2        # narrow wings for cheap stocks

MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"


def get_master():
    data = urllib.request.urlopen(MASTER_URL, timeout=30).read().decode(
        "utf-8", errors="ignore")
    return list(csv.reader(io.StringIO(data)))


def discover_underlyings(rows):
    """Pull distinct stock underlyings that have a near option chain."""
    pat = re.compile(r"^NSE:([A-Z&]+)\d{2}[A-Z]{3}\d+(?:\.\d+)?(CE|PE)$")
    seen = {}
    for r in rows:
        sym = r[9] if len(r) > 9 else ""
        m = pat.match(sym)
        if not m:
            continue
        name = m.group(1)
        if any(b in name for b in INDEX_BAD):
            continue
        seen[name] = seen.get(name, 0) + 1
    # most-listed names first = most liquid option chains
    names = sorted(seen, key=lambda n: -seen[n])
    return names[:SCAN_BUDGET], seen


def chain_for(fy, rows, underlying):
    """{expiry: {strike: {CE: sym, PE: sym}}} for one underlying."""
    pat = re.compile(rf"^NSE:{underlying}(\d{{2}}[A-Z]{{3}})(\d+(?:\.\d+)?)(CE|PE)$")
    out = {}
    for r in rows:
        sym = r[9] if len(r) > 9 else ""
        m = pat.match(sym)
        if not m:
            continue
        exp, strike, typ = m.group(1), float(m.group(2)), m.group(3)
        out.setdefault(exp, {}).setdefault(strike, {})[typ] = sym
    return out


def _expdt(e):
    s = e.upper() + " 2026"
    for fmt in ("%d%b %Y", "%d%B %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime.max


def quote_depth(fy, symbol):
    """Return (ltp, spread_pct) using bid/ask if available, else spread=None."""
    q = fy.quotes(data={"symbols": symbol})
    if q.get("s") != "ok" or not q.get("d"):
        return None, None
    v = q["d"][0].get("v", {})
    ltp = v.get("lp")
    bid, ask = v.get("bid"), v.get("ask")
    if ltp is None:
        return None, None
    if bid and ask and ask > 0:
        return ltp, (ask - bid) / ask
    return ltp, None


def build_condor(fy, rows, underlying):
    """Best-scoring condor for one underlying, or None."""
    opts = chain_for(fy, rows, underlying)
    if not opts:
        return None
    today = ac.ist_now() or datetime.now()
    floor = today.replace(hour=0, minute=0, second=0, microsecond=0)
    future = [e for e in opts if _expdt(e) >= floor]
    if not future:
        return None
    exp = sorted(future, key=_expdt)[0]
    strikes = sorted(opts[exp])
    if len(strikes) < S_LONG + 3:
        return None

    # spot
    sq = fy.quotes(data={"symbols": f"NSE:{underlying}-EQ"})
    if sq.get("s") != "ok" or not sq.get("d"):
        return None
    spot = sq["d"][0].get("v", {}).get("lp")
    if not spot:
        return None
    atm = min(strikes, key=lambda k: abs(k - spot))
    ai = strikes.index(atm)

    def near(off, typ):
        for d in range(0, 8):
            for sgn in ((0,) if d == 0 else (d, -d)):
                k = ai + off + sgn
                if not (0 <= k < len(strikes)):
                    continue
                sym = opts[exp][strikes[k]].get(typ)
                if sym:
                    return (strikes[k], sym)
        return None

    legs = {"ce_short": near(S_SHORT, "CE"), "ce_long": near(S_LONG, "CE"),
            "pe_short": near(-S_SHORT, "PE"), "pe_long": near(-S_LONG, "PE")}
    if any(v is None for v in legs.values()):
        return None

    # live prices + spread check on the SHORT legs (the ones we sell)
    px, sp = {}, {}
    for name, (strike, sym) in legs.items():
        ltp, spread = quote_depth(fy, sym)
        if ltp is None:
            return None
        px[name] = ltp
        sp[name] = spread
    for name in ("ce_short", "pe_short"):
        s = sp[name]
        if s is not None and s > MAX_SHORT_SPREAD_PCT:
            return None  # illiquid short leg — the edge leaks through the spread

    # lot size
    lot = 0
    for r in rows:
        if len(r) > 9 and r[9] == legs["ce_short"][1]:
            try:
                lot = int(float(r[2] or 0))
            except (ValueError, TypeError):
                lot = 0
            break
    if lot <= 0:
        return None

    credit = (px["ce_short"] - px["ce_long"]) + (px["pe_short"] - px["pe_long"])
    wing = legs["ce_long"][0] - legs["ce_short"][0]
    maxloss_pts = max(wing - credit, 0)
    if maxloss_pts <= 0:
        return None
    credit_1lot = credit * lot
    maxloss_1lot = maxloss_pts * lot
    if maxloss_1lot <= 0:
        return None
    if credit_1lot < MIN_CREDIT_RS_1LOT:
        return None   # tiny premium: needs too many lots; cost drag eats the edge
    # blended score: reward-per-risk, but require the trade to be worth taking.
    # A Rs12/lot credit at score 4 is a trap (needs 100+ lots of orders).
    score = credit_1lot / maxloss_1lot
    return {"underlying": underlying, "expiry": exp, "lot": lot,
            "legs": {k: {"strike": v[0], "ltp": px[k], "symbol": v[1]}
                     for k, v in legs.items()},
            "credit_pts": credit, "maxloss_pts": maxloss_pts,
            "credit_rs_1lot": credit_1lot, "maxloss_rs_1lot": maxloss_1lot,
            "score": score,
            "spread_pct": {k: sp[k] for k in ("ce_short", "pe_short")}}


def scan(fy, capital, budget=SCAN_BUDGET, top=10):
    rows = get_master()
    names, seen = discover_underlyings(rows)
    print(f"master {len(rows)} rows | {len(names)} liquid underlyings probed "
          f"(of {len(seen)} total)\n")
    cands = []
    for u in names[:budget]:
        try:
            c = build_condor(fy, rows, u)
        except Exception as e:
            print(f"  {u:12} error {repr(e)[:50]}")
            continue
        if not c:
            continue
        fits = c["maxloss_rs_1lot"] <= capital * ac.TARGET_RISK_PCT
        c["fits"] = fits
        cands.append(c)
        tag = "FITS" if fits else "    "
        print(f"  {u:12} {c['expiry']} lot{c['lot']:4}  "
              f"credit/lot Rs{c['credit_rs_1lot']:6.0f}  "
              f"maxloss/lot Rs{c['maxloss_rs_1lot']:6.0f}  "
              f"score {c['score']:.3f}  {tag}")
    # rank by reward-per-risk among those that fit; fall back to all if none fit
    pool = [c for c in cands if c["fits"]] or cands
    pool.sort(key=lambda c: -c["score"])
    return pool[:top]


def show(pool, capital):
    print(f"\n=== TOP {len(pool)} by reward-per-risk (capital Rs{capital:.0f}) ===")
    for c in pool:
        by_risk = max(1, int(capital * ac.TARGET_RISK_PCT // max(c["maxloss_rs_1lot"], 1)))
        lots = max(1, min(by_risk, MAX_LOTS))
        print(f"\n  {c['underlying']}  {c['expiry']}  score {c['score']:.3f}  "
              f"lot {c['lot']}  spread {c['spread_pct']}")
        print(f"    SELL {c['legs']['ce_short']['strike']:.0f} CE "
              f"@{c['legs']['ce_short']['ltp']:.2f}   "
              f"BUY {c['legs']['ce_long']['strike']:.0f} CE "
              f"@{c['legs']['ce_long']['ltp']:.2f}")
        print(f"    SELL {c['legs']['pe_short']['strike']:.0f} PE "
              f"@{c['legs']['pe_short']['ltp']:.2f}   "
              f"BUY {c['legs']['pe_long']['strike']:.0f} PE "
              f"@{c['legs']['pe_long']['ltp']:.2f}")
        print(f"    credit Rs{c['credit_rs_1lot']:.0f}/lot | "
              f"maxloss Rs{c['maxloss_rs_1lot']:.0f}/lot | {lots} lot(s) = "
              f"credit Rs{c['credit_rs_1lot']*lots:.0f}, "
              f"maxloss Rs{c['maxloss_rs_1lot']*lots:.0f} "
              f"({c['maxloss_rs_1lot']*lots/capital*100:.0f}%)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=5000.0)
    ap.add_argument("--budget", type=int, default=SCAN_BUDGET)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    fy = ac.get_fyers()
    pool = scan(fy, args.capital, args.budget, args.top)
    show(pool, args.capital)
