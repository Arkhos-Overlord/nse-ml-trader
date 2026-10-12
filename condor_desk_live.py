"""REAL-TIME CONDOR DESK — authenticated INDmoney feed (SSE JSON-RPC).

The keyless nsepython chain was 3 days stale (last-close, not live). This pulls
the LIVE option chain through the authenticated INDmoney MCP token (market:read
scope, already OAuth'd in Hermes). Two things the earlier client got wrong:
  1. The MCP endpoint returns SSE (text/event-stream), not plain JSON — parse
     the `data:` line.
  2. Token valid until expires_at (verified live, not expired).

Picks the validated iron-condor legs (sell ATM+/-2, buy ATM+/-5, nearest expiry),
sizes to a bounded Rs20K max loss, prints the ticket. Ticket only — no auto-order.

Run: python condor_desk_live.py
"""
import json
import sys
import time
import warnings

import requests

warnings.filterwarnings("ignore")

TOKEN_PATH = r"C:\Users\sydne\AppData\Local\hermes\mcp-tokens\indmoney.json"
URL = "https://mcp.indmoney.com/mcp"
LOT = 75
CAPITAL = 150000.0
TARGET_MAXLOSS = 20000.0
SHORT_WING, LONG_WING = 2, 5
BROKERAGE = 20.0
STT_SELL = 0.000625
EXCH_GST = 0.00004
SLIP = 0.0005
NIFTY_KEY = "INDI00012"


def _auth():
    tok = json.load(open(TOKEN_PATH))
    if tok.get("expires_at", 0) < time.time():
        print("  TOKEN EXPIRED — re-run: hermes mcp login indmoney")
        sys.exit(1)
    return {"Authorization": "Bearer " + tok["access_token"],
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"}


def mcp_call(method, params, headers):
    r = requests.post(URL, headers=headers,
                      json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                      timeout=30)
    for line in r.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise RuntimeError(f"no data line (HTTP {r.status_code}): {r.text[:200]}")


def get_chain(headers):
    res = mcp_call("tools/call",
                   {"name": "get_indian_stocks_option_chain",
                    "arguments": {"ind_key": NIFTY_KEY, "use_expiry_date": True}}, headers)
    for block in res.get("result", {}).get("content", []):
        if block.get("type") == "text":
            txt = block["text"]
            if txt.strip().lower().startswith("error"):
                raise RuntimeError(txt[:300])
            return json.loads(txt)
    raise RuntimeError(f"no text content: {json.dumps(res)[:300]}")


def parse(chain):
    ed = chain["entity_details"]
    spot = float(ed["live_price"])
    lot = int(ed.get("lot_size") or 65)
    price_ts = ed.get("price_updated_at_ist", "unknown")
    rows = []
    for e in chain["option_chain_data"]:
        k = float(e["strike_price"])
        for typ, key in (("CE", "call_data"), ("PE", "put_data")):
            leg = e.get(key)
            if not leg:
                continue
            cp = leg.get("current_price")
            if cp is None:
                continue
            lot = leg.get("lot_size", lot) or lot
            rows.append({"strike": k, "type": typ, "ltp": float(cp),
                         "oi": float(leg.get("oi", 0)), "iv": leg.get("iv", 0)})
    return spot, lot, price_ts, chain.get("selected_expiry", "n/a"), rows


def pick_legs(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty or df.strike.nunique() < LONG_WING + 2:
        return None
    oi = df.groupby("strike").oi.sum()
    atm = oi.idxmax()
    strikes = sorted(df.strike.unique())
    ai = strikes.index(atm)
    def leg(off, typ):
        k = ai + off
        if not (0 <= k < len(strikes)):
            return None
        r = df[(df.strike == strikes[k]) & (df.type == typ)]
        return None if r.empty else (strikes[k], float(r.iloc[0].ltp))
    legs = {"atm": atm,
            "ce_short": leg(SHORT_WING, "CE"), "ce_long": leg(LONG_WING, "CE"),
            "pe_short": leg(-SHORT_WING, "PE"), "pe_long": leg(-LONG_WING, "PE")}
    if any(legs[x] is None for x in ["ce_short", "ce_long", "pe_short", "pe_long"]):
        return None
    return legs


def ticket(legs, lot, lots):
    (cs_k, cs_p), (cl_k, cl_p) = legs["ce_short"], legs["ce_long"]
    (ps_k, ps_p), (pl_k, pl_p) = legs["pe_short"], legs["pe_long"]
    credit = (cs_p - cl_p) + (ps_p - pl_p)
    wing = cl_k - cs_k
    maxloss = max(wing - credit, 0)
    sell_prem = (cs_p + ps_p) * lots * lot
    notional = lots * lot * legs["atm"]
    cost = lots * (2 * BROKERAGE + notional * EXCH_GST + sell_prem * STT_SELL
                   + (cs_p + ps_p) * lots * lot * SLIP)
    return {"credit": credit, "net_credit": credit * lot * lots - cost,
            "wing": wing, "maxloss_pts": maxloss, "maxloss_rs": maxloss * lot * lots,
            "lots": lots,
            "orders": [("SELL", "CE", cs_k, cs_p), ("BUY", "CE", cl_k, cl_p),
                       ("SELL", "PE", ps_k, ps_p), ("BUY", "PE", pl_k, pl_p)]}


def main():
    print("=" * 84)
    print("  CONDOR DESK (LIVE) — authenticated INDmoney feed")
    print("=" * 84)
    headers = _auth()
    try:
        chain = get_chain(headers)
    except Exception as e:
        print(f"  FEED ERROR: {e}")
        return
    spot, lot, price_ts, expiry, rows = parse(chain)
    print(f"  spot: {spot:,.0f} | lot size: {lot} | expiry: {expiry}")
    print(f"  prices as of: {price_ts}   <-- if not today, prices are last-close")
    print(f"  legs parsed: {len(rows)}")
    legs = pick_legs(rows)
    if not legs:
        print("  insufficient strikes for a full condor on this feed")
        print(f"  strikes: {sorted({r['strike'] for r in rows})}")
        return
    print(f"  ATM (max OI): {legs['atm']:.0f}")
    probe = ticket(legs, lot, 1)
    lots = max(1, int(TARGET_MAXLOSS / probe["maxloss_rs"])) if probe["maxloss_rs"] > 0 else 1
    t = ticket(legs, lot, lots)
    print(f"\n  --- ORDER TICKET ({t['lots']} lot{'s' if lots>1 else ''}, lot={lot}) ---")
    for side, typ, k, p in t["orders"]:
        print(f"    {side:<5} NIFTY {k:>8.0f} {typ}  @ ~{p:>7.2f}  x{t['lots']}")
    print(f"\n  net credit:      Rs{t['net_credit']:,.0f}")
    print(f"  MAX LOSS (cap):  Rs{t['maxloss_rs']:,.0f} "
          f"({t['maxloss_rs']/CAPITAL*100:.1f}% of Rs1.5L)")
    print(f"  risk:reward:      1 : {t['net_credit']/t['maxloss_rs']:.2f}")
    print(f"\n  Backtest: +Rs911/trade (at lot 75; rescale x65/75 if comparing),")
    print(f"  t=+2.43, 2 regimes, worst -4% capital. Ticket only — you place it.")


if __name__ == "__main__":
    main()
