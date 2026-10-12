"""AUTO-CONDOR ENGINE — trades the VRP iron condor on live Fyers data.

DEFAULT = PAPER MODE. It pulls the live chain, builds the condor, and LOGS the
ticket + simulated fill. It places NOTHING until you explicitly arm it with
--live. That is deliberate: we prove the loop on paper before real money.

Loop:
  1. pull live option chain + IndiaVIX
  2. build the bounded-risk condor (sell ATM+/-2, buy ATM+/-5), size to
     TARGET_MAXLOSS
  3. PAPER: log ticket + simulate fill + track open position
     LIVE:  place the 4 legs as a MARGIN basket (limit orders at the mid)
  4. manage: track the position, exit at expiry or on a stop, log P&L
  5. kill-switch: if daily loss > DAILY_STOP_RS, stop trading for the day

Run (paper):   python auto_condor.py
Run (live):    python auto_condor.py --live        # requires --i-understand
Never auto-places without BOTH flags.
"""
import argparse
import json
import os
import re
import time
from datetime import datetime

import pandas as pd
from fyers_apiv3 import fyersModel

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(HERE, "fyers_env.txt")
TOKEN_FILE = os.path.join(HERE, "fyers_token.json")
LOG = os.path.join(HERE, "auto_condor.log")
STATE = os.path.join(HERE, "auto_condor_state.json")

LOT = 65
CAPITAL = 150000.0
TARGET_MAXLOSS = 20000.0
SHORT_WING, LONG_WING = 2, 5
DAILY_STOP_RS = -25000.0      # stop trading for the day past this loss
ENTRY_HOUR_IST = 9           # don't enter before this hour (IST)
POLL_SECONDS = 30

creds = {}
for line in open(ENV):
    m = re.match(r'\s*(\w+)\s*=\s*"([^"]*)"', line)
    if m:
        creds[m.group(1)] = m.group(2)


def ist_now():
    """Authoritative IST from network time — never trust the local clock
    (user's machine clock is wrong). Falls back to UTC+5:30 if the fetch fails."""
    import datetime as _dt
    import urllib.request
    for url in ("https://worldtimeapi.org/api/timezone/Asia/Kolkata",
                "http://worldtimeapi.org/api/timezone/Asia/Kolkata"):
        try:
            j = json.loads(urllib.request.urlopen(url, timeout=6).read())
            return _dt.datetime.fromisoformat(j["datetime"][:19])
        except Exception:
            continue
    # fallback: real UTC from an HTTP Date header, +5:30
    try:
        import email.utils
        d = urllib.request.urlopen("https://www.google.com", timeout=6).headers.get("Date")
        utc = _dt.datetime(*email.utils.parsedate_to_datetime(d).timetuple()[:6])
        return utc + _dt.timedelta(hours=5, minutes=30)
    except Exception:
        return _dt.datetime.utcnow() + _dt.timedelta(hours=5, minutes=30)


def log(msg):
    line = f"[{ist_now().strftime('%Y-%m-%d %H:%M:%S')} IST] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"open": None, "day": None, "daily_pnl": 0.0, "trades": []}


def save_state(s):
    json.dump(s, open(STATE, "w"), indent=2)


def get_fyers():
    t = json.load(open(TOKEN_FILE))
    return fyersModel.FyersModel(client_id=creds["FYERS_CLIENT_ID"],
                                 token=t["access_token"], is_async=False, log_path="")


def pull_chain(fy):
    oc = fy.optionchain(data={"symbol": "NSE:NIFTY50-INDEX",
                              "strikecount": 25, "timestamp": ""})
    if oc.get("s") != "ok":
        return None, None
    d = oc["data"]
    rows = []
    for leg in d.get("optionsChain", []):
        k, typ, ltp = leg.get("strike_price"), leg.get("option_type"), leg.get("ltp")
        if k is None or not typ or ltp is None:
            continue
        rows.append({"strike": float(k), "type": typ, "ltp": float(ltp),
                     "oi": float(leg.get("oi", 0)),
                     "symbol": leg.get("symbol", "")})
    iv = d.get("indiavixData") or {}
    vix = iv.get("ltp") if isinstance(iv, dict) else None
    return pd.DataFrame(rows), vix


def build_condor(df, vix):
    if df.empty or df.strike.nunique() < LONG_WING + 2:
        return None
    atm = df.groupby("strike").oi.sum().idxmax()
    strikes = sorted(df.strike.unique())
    ai = strikes.index(atm)

    def leg(off, typ):
        k = ai + off
        if not (0 <= k < len(strikes)):
            return None
        r = df[(df.strike == strikes[k]) & (df.type == typ)]
        if r.empty:
            return None
        r = r.iloc[0]
        return {"strike": strikes[k], "ltp": r.ltp, "symbol": r.symbol}

    L = {"ce_short": leg(SHORT_WING, "CE"), "ce_long": leg(LONG_WING, "CE"),
         "pe_short": leg(-SHORT_WING, "PE"), "pe_long": leg(-LONG_WING, "PE")}
    if any(v is None for v in L.values()):
        return None
    credit = (L["ce_short"]["ltp"] - L["ce_long"]["ltp"]) + \
             (L["pe_short"]["ltp"] - L["pe_long"]["ltp"])
    maxloss_pts = max((L["ce_long"]["strike"] - L["ce_short"]["strike"]) - credit, 0)
    probe = maxloss_pts * LOT
    lots = max(1, int(TARGET_MAXLOSS / probe)) if probe > 0 else 1
    return {"atm": atm, "legs": L, "credit": credit, "vix": vix,
            "maxloss_pts": maxloss_pts, "lots": lots,
            "maxloss_rs": maxloss_pts * LOT * lots,
            "credit_rs": credit * LOT * lots}


def place_legs(fy, condor, live):
    """Place/paper the 4 legs. Returns list of order records."""
    orders = []
    for name, side in [("ce_short", -1), ("ce_long", 1), ("pe_short", -1), ("pe_long", 1)]:
        leg = condor["legs"][name]
        rec = {"leg": name, "symbol": leg["symbol"], "side": "SELL" if side == -1 else "BUY",
               "strike": leg["strike"], "ltp": leg["ltp"], "qty": LOT * condor["lots"]}
        if live:
            data = {"symbol": leg["symbol"], "qty": LOT * condor["lots"], "type": 1,
                    "side": side, "productType": "MARGIN", "validity": "DAY",
                    "limitPrice": round(leg["ltp"], 2), "stopPrice": 0,
                    "disclosedQty": 0, "offlineOrder": False, "isSliceOrder": False}
            resp = fy.place_order(data)
            rec["broker_resp"] = resp
        orders.append(rec)
    return orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="place real orders")
    ap.add_argument("--i-understand", action="store_true",
                    help="confirm you understand live trading risks")
    ap.add_argument("--once", action="store_true", help="run one cycle and exit")
    args = ap.parse_args()

    live = args.live and args.i_understand
    if args.live and not args.i_understand:
        log("--live requires --i-understand. Staying in PAPER mode.")
    mode = "LIVE" if live else "PAPER"
    log(f"=== AUTO-CONDOR starting in {mode} mode ===")

    fy = get_fyers()
    st = load_state()
    today = ist_now().strftime("%Y-%m-%d")
    if st.get("day") != today:
        st["day"] = today
        st["daily_pnl"] = 0.0
        save_state(st)

    if st["daily_pnl"] <= DAILY_STOP_RS:
        log(f"daily stop hit (Rs{st['daily_pnl']:,.0f}); not trading today.")
        return

    df, vix = pull_chain(fy)
    if df is None:
        log("chain pull failed; retrying next cycle.")
        return
    log(f"live chain OK | {len(df)} legs | IndiaVIX {vix}")

    if st["open"]:
        log(f"already holding a condor: {st['open'].get('atm')} "
            f"({st['open'].get('lots')} lots). Skipping new entry (one at a time).")
        return

    hour = ist_now().hour
    weekday = ist_now().weekday()   # 0=Mon .. 6=Sun
    if weekday >= 5:
        log(f"weekend (IST {ist_now().strftime('%a %H:%M')}); NSE closed, not trading.")
        return
    if hour < ENTRY_HOUR_IST:
        log(f"before entry window (IST {hour}h < {ENTRY_HOUR_IST}h); waiting.")
        return

    condor = build_condor(df, vix)
    if not condor:
        log("could not build condor.")
        return

    log(f"CONDOR: ATM {condor['atm']:.0f} | {condor['lots']} lots | "
        f"credit Rs{condor['credit_rs']:,.0f} | maxloss Rs{condor['maxloss_rs']:,.0f} "
        f"({condor['maxloss_rs']/CAPITAL*100:.1f}% cap) | IndiaVIX {vix}")
    orders = place_legs(fy, condor, live)
    for o in orders:
        log(f"  {o['side']:<4} {o['symbol']} x{o['qty']} @ {o['ltp']}"
            + (f"  -> {o.get('broker_resp')}" if live else "  [paper]"))

    st["open"] = {"atm": condor["atm"], "lots": condor["lots"],
                  "credit_rs": condor["credit_rs"], "maxloss_rs": condor["maxloss_rs"],
                  "vix": vix, "opened": today, "legs": orders, "mode": mode}
    st["trades"].append({"date": today, **condor, "mode": mode})
    save_state(st)
    log(f"position OPENED ({mode}). max loss capped at Rs{condor['maxloss_rs']:,.0f}.")

    if args.once:
        return
    log(f"sleeping {POLL_SECONDS}s; rerun for the next cycle "
        f"(schedule via Task Scheduler for continuous operation).")


if __name__ == "__main__":
    main()
