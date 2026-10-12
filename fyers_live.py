"""FYERS live condor desk — real-time option chain + one-time OAuth.

Flow (matches fyers-apiv3 3.1.19 exactly):
  1. read credentials from fyers_env.txt
  2. SessionModel(client_id, secret_key, redirect_uri, grant_type) -> auth URL
  3. you open the URL, log in, Fyers redirects to the local callback with
     ?auth_code=...&state=...  -> we capture it with a tiny local server
  4. session.set_token(auth_code) -> generate_token() -> access_token
  5. token cached to fyers_token.json (reused until it expires, ~1 day)
  6. FyersModel pulls the LIVE Nifty option chain -> build the iron condor ->
     print the ticket with real-time prices + bounded max loss.

Ticket only — no auto-order. Run: python fyers_live.py
"""
import http.server
import json
import os
import re
import socketserver
import threading
import urllib.parse
import webbrowser

from fyers_apiv3 import fyersModel

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(HERE, "fyers_env.txt")
TOKEN_FILE = os.path.join(HERE, "fyers_token.json")
CAPITAL = 150000.0
TARGET_MAXLOSS = 20000.0
SHORT_WING, LONG_WING = 2, 5
BROKERAGE = 20.0
STT_SELL = 0.000625
EXCH_GST = 0.00004
SLIP = 0.0005


def read_creds():
    creds = {}
    with open(ENV) as f:
        for line in f:
            m = re.match(r'\s*(\w+)\s*=\s*"([^"]*)"', line)
            if m:
                creds[m.group(1)] = m.group(2)
    for k in ("FYERS_CLIENT_ID", "FYERS_SECRET_KEY", "FYERS_REDIRECT_URI"):
        if not creds.get(k) or "PASTE" in creds[k]:
            raise SystemExit(f"  {k} not set in fyers_env.txt — see the instructions in that file.")
    return creds


def capture_auth_code(redirect_uri):
    """Spin up a local server, open the Fyers login, grab the auth_code."""
    host = urllib.parse.urlparse(redirect_uri).hostname or "127.0.0.1"
    port = urllib.parse.urlparse(redirect_uri).port or 80
    box = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            box["auth_code"] = q.get("auth_code", [None])[0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<h2>Auth captured. You can close this tab and return to the terminal.</h2>")
        def log_message(self, *a):
            pass

    with socketserver.TCPServer((host, port), H) as httpd:
        creds = read_creds()
        s = fyersModel.SessionModel(
            client_id=creds["FYERS_CLIENT_ID"],
            secret_key=creds["FYERS_SECRET_KEY"],
            redirect_uri=creds["FYERS_REDIRECT_URI"],
            response_type="code",
            grant_type="authorization_code",
        )
        url = s.generate_authcode()
        print("\n  Opening Fyers login in your browser...")
        print(f"  (if it doesn't open, paste this URL manually:\n   {url}\n)")
        webbrowser.open(url)
        print("  waiting for you to log in and authorise...")
        httpd.handle_request()   # blocks until the redirect hits us
    if not box.get("auth_code"):
        raise SystemExit("  no auth_code captured — did you complete the login?")
    return box["auth_code"]


def get_fyers():
    creds = read_creds()
    # reuse cached token if still present (Fyers tokens last ~1 trading day)
    if os.path.exists(TOKEN_FILE):
        try:
            t = json.load(open(TOKEN_FILE))
            fy = fyersModel.FyersModel(client_id=creds["FYERS_CLIENT_ID"],
                                       token=t["access_token"], is_async=False, log_path="")
            if fy.get_profile().get("s") == "ok":
                return fy
        except Exception:
            pass
    auth_code = capture_auth_code(creds["FYERS_REDIRECT_URI"])
    s = fyersModel.SessionModel(
        client_id=creds["FYERS_CLIENT_ID"],
        secret_key=creds["FYERS_SECRET_KEY"],
        redirect_uri=creds["FYERS_REDIRECT_URI"],
        response_type="code",
        grant_type="authorization_code",
    )
    s.set_token(auth_code)
    resp = s.generate_token()
    if "access_token" not in resp:
        raise SystemExit(f"  token exchange failed: {resp}")
    json.dump(resp, open(TOKEN_FILE, "w"))
    print(f"  access_token cached to {os.path.basename(TOKEN_FILE)}")
    return fyersModel.FyersModel(client_id=creds["FYERS_CLIENT_ID"],
                                 token=resp["access_token"], is_async=False, log_path="")


def build_ticket(fy):
    """Live Nifty option chain -> iron condor legs."""
    # Fyers symbol for Nifty index; get the nearest weekly expiry via quotes
    prof = fy.get_profile()
    if prof.get("s") != "ok":
        raise SystemExit(f"  auth/profile failed: {prof}")
    # option chain: Fyers uses optionchain endpoint on NSE:NIFTY50-INDEX
    oc = fy.optionchain(data={"symbol": "NSE:NIFTY50-INDEX", "strikecount": 20,
                              "timestamp": ""})
    if oc.get("s") != "ok":
        raise SystemExit(f"  option chain failed: {oc}")
    chain = oc["data"]["optionsChain"]
    rows = []
    lot = 65
    for strike, legs in chain.items():
        for typ in ("CE", "PE"):
            leg = legs.get(typ)
            if not leg:
                continue
            rows.append({"strike": float(strike), "type": typ,
                         "ltp": float(leg.get("ltp", 0)),
                         "oi": float(leg.get("oi", 0)),
                         "lot": int(leg.get("lot_size", lot))})
    return rows


def pick_and_print(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty or df.strike.nunique() < LONG_WING + 2:
        print("  insufficient strikes on the live chain")
        return
    lot = int(df.lot.mode().iloc[0])
    atm = df.groupby("strike").oi.sum().idxmax()
    strikes = sorted(df.strike.unique())
    ai = strikes.index(atm)
    def leg(off, typ):
        k = ai + off
        if not (0 <= k < len(strikes)):
            return None
        r = df[(df.strike == strikes[k]) & (df.type == typ)]
        return None if r.empty else (strikes[k], float(r.iloc[0].ltp))
    legs = {"ce_short": leg(SHORT_WING, "CE"), "ce_long": leg(LONG_WING, "CE"),
            "pe_short": leg(-SHORT_WING, "PE"), "pe_long": leg(-LONG_WING, "PE")}
    if any(v is None for v in legs.values()):
        print("  could not build a full condor")
        return
    (cs_k, cs_p), (cl_k, cl_p) = legs["ce_short"], legs["ce_long"]
    (ps_k, ps_p), (pl_k, pl_p) = legs["pe_short"], legs["pe_long"]
    credit = (cs_p - cl_p) + (ps_p - pl_p)
    maxloss_pts = max((cl_k - cs_k) - credit, 0)
    probe_rs = maxloss_pts * lot
    lots = max(1, int(TARGET_MAXLOSS / probe_rs)) if probe_rs > 0 else 1
    maxloss_rs = maxloss_pts * lot * lots
    sell_prem = (cs_p + ps_p) * lots * lot
    notional = lots * lot * atm
    cost = lots * (2 * BROKERAGE + notional * EXCH_GST + sell_prem * STT_SELL
                   + (cs_p + ps_p) * lots * lot * SLIP)
    net_credit = credit * lot * lots - cost
    print(f"\n  ATM (max OI): {atm:.0f} | lot {lot} | LIVE prices")
    print(f"\n  --- ORDER TICKET ({lots} lot{'s' if lots>1 else ''}) ---")
    for side, typ, k, p in [("SELL", "CE", cs_k, cs_p), ("BUY", "CE", cl_k, cl_p),
                            ("SELL", "PE", ps_k, ps_p), ("BUY", "PE", pl_k, pl_p)]:
        print(f"    {side:<5} NIFTY {k:>8.0f} {typ}  @ {p:>7.2f}  x{lots}")
    print(f"\n  net credit:      Rs{net_credit:,.0f}")
    print(f"  MAX LOSS (cap):  Rs{maxloss_rs:,.0f} ({maxloss_rs/CAPITAL*100:.1f}% of Rs1.5L)")
    print(f"  risk:reward:      1 : {net_credit/maxloss_rs:.2f}")
    print(f"\n  REAL-TIME. Ticket only — you place it.")


def main():
    print("=" * 84)
    print("  CONDOR DESK — FYERS LIVE")
    print("=" * 84)
    fy = get_fyers()
    rows = build_ticket(fy)
    pick_and_print(rows)


if __name__ == "__main__":
    main()
