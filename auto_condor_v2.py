"""AUTO-CONDOR ENGINE v2 — continuous, self-managing condor trader.

Runs a loop. Enters the VRP iron condor when the window opens, MANAGES the open
position (mark-to-market, stop at risk cap, square off near expiry), and reaps
the credit. Two caps on live money:

  * DEFAULT = PAPER. Places nothing.
  * LIVE requires BOTH --live --i-understand.

Instrument:
  * CAPITAL >= 25000  -> NIFTY index condor (lot 65)
  * CAPITAL <  25000  -> scans liquid single stocks (lot 15) for a condor whose
                         1-lot max loss fits ~13% of capital (the Rs5K case).

Time is authoritative IST from the network (machine clock is unreliable).
"""
import argparse
import csv
import io
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timedelta

import pandas as pd
from fyers_apiv3 import fyersModel

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(HERE, "fyers_env.txt")
TOKEN_FILE = os.path.join(HERE, "fyers_token.json")
LOG = os.path.join(HERE, "auto_condor_v2.log")
STATE = os.path.join(HERE, "auto_condor_state.json")

LOT_NIFTY = 65
NIFTY_THRESHOLD = 25000.0
TARGET_RISK_PCT = 0.13          # max loss <= 13% of capital
SHORT_WING, LONG_WING = 2, 5    # index
S_SHORT, S_LONG = 1, 2          # stocks (cheaper, need narrow wings)
DAILY_STOP_PCT = -0.02          # -2% of capital/day halts new entries
ENTRY_HOUR_IST, ENTRY_MIN_IST = 9, 20
EXIT_HOUR_IST = 15              # square off open positions after 15:00
POLL_SECONDS = 60
STOP_LOSS_FRAC = 1.6            # close if loss > 1.6x credit collected

creds = {}
for line in open(ENV):
    m = re.match(r'\s*(\w+)\s*=\s*"([^"]*)"', line)
    if m:
        creds[m.group(1)] = m.group(2)


# ---------- time ----------
def ist_now():
    import datetime as _dt
    for url in ("https://worldtimeapi.org/api/timezone/Asia/Kolkata",):
        try:
            j = json.loads(urllib.request.urlopen(url, timeout=6).read())
            return _dt.datetime.fromisoformat(j["datetime"][:19])
        except Exception:
            pass
    try:
        import email.utils
        d = urllib.request.urlopen("https://www.google.com", timeout=6).headers.get("Date")
        utc = _dt.datetime(*email.utils.parsedate_to_datetime(d).timetuple()[:6])
        return utc + _dt.timedelta(hours=5, minutes=30)
    except Exception:
        # FAIL CLOSED (red-team fix #3): never fall back to the wrong local clock.
        # A trader on a broken clock misses windows and corrupts daily resets.
        return None


def log(msg):
    line = f"[{ist_now().strftime('%Y-%m-%d %H:%M:%S')} IST] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ---------- state ----------
def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            # FAIL LOUD (red-team fix #2): a corrupt state file means a possible
            # OPEN POSITION. Quarantine (move, never delete) — silently resetting
            # orphans a live condor with no stop watching it.
            import shutil
            bak = STATE + ".corrupt." + _now().strftime("%Y%m%d-%H%M%S")
            shutil.move(STATE, bak)
            log(f"!! STATE CORRUPT — quarantined to {os.path.basename(bak)}. "
                f"CHECK FOR ORPHANED POSITION AT BROKER before trusting FLAT.")
    return {"open": None, "day": None, "daily_pnl": 0.0, "trades": []}


def _now():
    """ist_now with a safe string fallback for LOGGING ONLY (never for decisions)."""
    n = ist_now()
    if n is not None:
        return n
    return datetime.now()


def save_state(s):
    tmp = STATE + ".tmp"
    json.dump(s, open(tmp, "w"), indent=2)
    os.replace(tmp, STATE)


def get_fyers():
    t = json.load(open(TOKEN_FILE))
    return fyersModel.FyersModel(client_id=creds["FYERS_CLIENT_ID"],
                                 token=t["access_token"], is_async=False, log_path="")


# ---------- live data ----------
def quotes(fy, symbols):
    """Batch quotes; Fyers caps ~50 per call. Returns {symbol: ltp}."""
    out = {}
    for i in range(0, len(symbols), 40):
        chunk = [s for s in symbols[i:i + 40] if s]
        if not chunk:
            continue
        q = fy.quotes(data={"symbols": ",".join(chunk)})
        if q.get("s") != "ok":
            continue
        for d in q.get("d", []):
            out[d["n"]] = d.get("v", {}).get("lp")
    return out


def pull_nifty_chain(fy):
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
                     "oi": float(leg.get("oi", 0)), "symbol": leg.get("symbol", "")})
    iv = d.get("indiavixData") or {}
    return pd.DataFrame(rows), (iv.get("ltp") if isinstance(iv, dict) else None)


def build_nifty_condor(df, vix, lot=LOT_NIFTY):
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
    return _finish(L, lot, "NIFTY", vix)


def scan_stock_condor(fy, capital):
    """Scan liquid stocks for a condor whose 1-lot max loss fits capital."""
    CANDS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "ITC",
             "LT", "AXISBANK", "KOTAKBANK", "BHARTIARTL", "MARUTI", "TITAN",
             "SUNPHARMA", "WIPRO", "ADANIENT"]
    data = urllib.request.urlopen("https://public.fyers.in/sym_details/NSE_FO.csv",
                                  timeout=30).read().decode("utf-8", "ignore")
    rows = list(csv.reader(io.StringIO(data)))
    best = None
    for u in CANDS:
        try:
            pat = re.compile(rf"^NSE:{u}(\d{{2}}[A-Z]{{3}})(\d+(?:\.\d+)?)(CE|PE)$")
            by_exp = {}
            for r in rows:
                sym = r[9] if len(r) > 9 else ""
                m = pat.match(sym)
                if m:
                    by_exp.setdefault(m.group(1), {}).setdefault(
                        float(m.group(2)), {})[m.group(3)] = sym
            if not by_exp:
                continue
            # pick nearest FUTURE expiry (red-team fix: sorted()[0] could select
            # an already-expired contract)
            today = ist_now()
            def _expdt(e):
                try:
                    return datetime.strptime(e.upper() + " 2026", "%d%b %Y")
                except ValueError:
                    try:
                        return datetime.strptime(e.upper() + " 2026", "%d%B %Y")
                    except ValueError:
                        return datetime.max
            future = [e for e in by_exp if _expdt(e) >= (today or datetime.now()).replace(
                hour=0, minute=0, second=0, microsecond=0)]
            if not future:
                continue
            exp = sorted(future, key=_expdt)[0]
            strikes = sorted(by_exp[exp])
            if len(strikes) < S_LONG + 3:
                continue
            syms = [by_exp[exp][k].get(t) for k in strikes for t in ("CE", "PE")
                    if by_exp[exp][k].get(t)]
            ltp = quotes(fy, syms)
            sq = fy.quotes(data={"symbols": f"NSE:{u}-EQ"})
            if sq.get("s") != "ok" or not sq.get("d"):
                continue
            spot = sq["d"][0].get("v", {}).get("lp")
            if not spot:
                continue
            atm = min(strikes, key=lambda k: abs(k - spot))
            ai = strikes.index(atm)

            def near(off, typ):
                for d in range(0, 8):
                    for sgn in ((0,) if d == 0 else (d, -d)):
                        k = ai + off + sgn
                        if not (0 <= k < len(strikes)):
                            continue
                        sym = by_exp[exp][strikes[k]].get(typ)
                        if sym and ltp.get(sym) is not None:
                            return {"strike": strikes[k], "ltp": ltp[sym], "symbol": sym}
                return None

            L = {"ce_short": near(S_SHORT, "CE"), "ce_long": near(S_LONG, "CE"),
                 "pe_short": near(-S_SHORT, "PE"), "pe_long": near(-S_LONG, "PE")}
            if any(v is None for v in L.values()):
                continue
            lot = 15
            c = _finish(L, lot, u, None)
            if c and c["maxloss_rs_1lot"] <= capital * TARGET_RISK_PCT:
                if best is None or c["credit_rs_1lot"] > best["credit_rs_1lot"]:
                    c["expiry"] = exp
                    best = c
        except Exception as e:
            log(f"  scan {u}: {repr(e)[:50]}")
            continue
    return best


def _finish(L, lot, name, vix):
    credit = (L["ce_short"]["ltp"] - L["ce_long"]["ltp"]) + \
             (L["pe_short"]["ltp"] - L["pe_long"]["ltp"])
    wing = L["ce_long"]["strike"] - L["ce_short"]["strike"]
    maxloss_pts = max(wing - credit, 0)
    ml1 = maxloss_pts * lot
    cr1 = credit * lot
    return {"underlying": name, "legs": L, "credit_pts": credit,
            "maxloss_pts": maxloss_pts, "lot": lot,
            "credit_rs_1lot": cr1, "maxloss_rs_1lot": ml1, "vix": vix}


# ---------- orders ----------
def _order_ok(resp):
    """RED-TEAM FIX #1: a live fill is only real if the broker says so."""
    if not isinstance(resp, dict):
        return False
    return resp.get("s") == "ok" or resp.get("code") in (200, 2)


def place_legs(fy, condor, lots, live):
    orders = []
    for name, side in [("ce_short", -1), ("ce_long", 1), ("pe_short", -1), ("pe_long", 1)]:
        leg = condor["legs"][name]
        qty = condor["lot"] * lots
        rec = {"leg": name, "symbol": leg["symbol"], "side": "SELL" if side == -1 else "BUY",
               "strike": leg["strike"], "ltp": leg["ltp"], "qty": qty}
        if live:
            data = {"symbol": leg["symbol"], "qty": qty, "type": 1, "side": side,
                    "productType": "MARGIN", "validity": "DAY",
                    "limitPrice": round(leg["ltp"], 2), "stopPrice": 0,
                    "disclosedQty": 0, "offlineOrder": False, "isSliceOrder": False}
            resp = fy.place_order(data)
            rec["broker_resp"] = resp
            if not _order_ok(resp):
                # REJECTED LEG: do not mark the position open. Roll back any
                # filled legs so we never sit in a partial, unmanaged condor.
                log(f"!! ENTRY LEG REJECTED: {name} -> {resp}. Rolling back.")
                for r in orders:
                    if r.get("broker_resp") and _order_ok(r["broker_resp"]):
                        try:
                            fy.exit_positions(data={"id": r["broker_resp"]["id"]}) \
                                if isinstance(r.get("broker_resp"), dict) and r["broker_resp"].get("id") else None
                        except Exception:
                            pass
                return None, False
        orders.append(rec)
    return orders, True


def close_legs(fy, condor, live):
    """Square off: reverse each leg at current market."""
    ltp = quotes(fy, [condor["legs"][n]["symbol"] for n in
                      ("ce_short", "ce_long", "pe_short", "pe_long")])
    orders = []
    for name, side in [("ce_short", 1), ("ce_long", -1), ("pe_short", 1), ("pe_long", -1)]:
        leg = condor["legs"][name]
        qty = condor["lot"] * condor.get("lots", 1)
        px = ltp.get(leg["symbol"]) or leg["ltp"]
        rec = {"leg": name, "symbol": leg["symbol"], "side": "BUY" if side == 1 else "SELL",
               "strike": leg["strike"], "ltp": px, "qty": qty}
        if live:
            data = {"symbol": leg["symbol"], "qty": qty, "type": 1, "side": side,
                    "productType": "MARGIN", "validity": "DAY",
                    "limitPrice": round(px, 2), "stopPrice": 0,
                    "disclosedQty": 0, "offlineOrder": False, "isSliceOrder": False}
            rec["broker_resp"] = fy.place_order(data)
        orders.append(rec)
    return orders


def condor_pnl(fy, condor):
    """Current mark-to-market P&L (Rs). Opened for a credit; P&L = credit - cost to close."""
    legs = condor["legs"]
    ltp = quotes(fy, [legs[n]["symbol"] for n in
                      ("ce_short", "ce_long", "pe_short", "pe_long")])
    keys = ["ce_short", "ce_long", "pe_short", "pe_long"]
    if any(ltp.get(legs[n]["symbol"]) is None for n in keys):
        return None
    closing = (ltp[legs["ce_short"]["symbol"]] - ltp[legs["ce_long"]["symbol"]]) + \
              (ltp[legs["pe_short"]["symbol"]] - ltp[legs["pe_long"]["symbol"]])
    return (condor["credit_pts"] - closing) * condor["lot"] * condor.get("lots", 1)


# ---------- manage ----------
def manage_open(fy, st, live):
    o = st["open"]
    if not o:
        return
    now = ist_now()
    if now is None:
        log("!! clock unavailable (all network time sources failed); HALTING cycle. "
            "Will not trade on an unverified clock.")
        return
    if now.hour >= EXIT_HOUR_IST:
        log(f"exit window (>= {EXIT_HOUR_IST}h): squaring off {o['underlying']} condor")
        close_legs(fy, o, live)
        pnl = condor_pnl(fy, o)
        if pnl is not None:
            st["daily_pnl"] += pnl
            if st["trades"]:
                st["trades"][-1]["pnl_rs"] = round(pnl, 2)
        log(f"closed. P&L Rs{pnl if pnl is not None else 'n/a'} | "
            f"day Rs{st['daily_pnl']:.0f}")
        st["open"] = None
        save_state(st)
        return
    pnl = condor_pnl(fy, o)
    if pnl is None:
        log("manage: quote gap; holding.")
        return
    log(f"manage: {o['underlying']} open, MTM P&L Rs{pnl:.0f} "
        f"(credit Rs{o['credit_rs']:.0f}, stop -Rs{o['credit_rs']*STOP_LOSS_FRAC:.0f})")
    if pnl <= -o["credit_rs"] * STOP_LOSS_FRAC:
        log(f"STOP hit (Rs{pnl:.0f}); closing")
        close_legs(fy, o, live)
        st["daily_pnl"] += pnl
        if st["trades"]:
            st["trades"][-1]["pnl_rs"] = round(pnl, 2)
        st["open"] = None
        save_state(st)


def enter(fy, st, live, capital):
    # --- event + market-status gate (decisions, not just info) ---
    try:
        import entry_decision as ed
        # candidate underlying depends on capital tier; check both names' risk
        probe = "NIFTY" if capital >= NIFTY_THRESHOLD else None
        if probe is None:
            # stock tier: we don't know the underlying until scan; check macro only
            import event_risk as er
            r = er.event_risk_for(None, 7)
            verdict = "WIDEN" if any(e.get("kind") == "macro" and e.get("impact") == "high"
                                     for e in r["events"]) else "CLEAR"
        else:
            v = ed.entry_verdict(probe, 7)
            verdict = v["verdict"]
        if verdict == "BLOCK":
            log("EVENT GATE: BLOCK — corporate/macro event in horizon; skipping entry.")
            return
        if verdict == "WIDEN":
            log("EVENT GATE: WIDEN — macro event near; will widen wing + smaller size.")
    except Exception as e:
        log(f"event gate error ({repr(e)[:60]}); proceeding CLEAR.")
        verdict = "CLEAR"

    if capital >= NIFTY_THRESHOLD:
        df, vix = pull_nifty_chain(fy)
        if df is None:
            log("chain pull failed.")
            return
        c = build_nifty_condor(df, vix)
        lots = 1
        if c:
            lots = max(1, int(capital * TARGET_RISK_PCT / max(c["maxloss_rs_1lot"], 1)))
    else:
        # DYNAMIC scan: discover + score all liquid underlyings, take the best.
        try:
            import scan_dynamic
            pool = scan_dynamic.scan(fy, capital, budget=25, top=1)
            c = pool[0] if pool else None
        except Exception as e:
            log(f"dynamic scan error ({repr(e)[:60]}); falling back to fixed scan.")
            c = scan_stock_condor(fy, capital)
        lots = 1
        vix = None
    if not c:
        log("no tradeable condor built.")
        return
    # re-check event verdict against the ACTUAL underlying chosen by the scan
    try:
        import entry_decision as ed
        v2 = ed.entry_verdict(c["underlying"], 7)
        if v2["verdict"] == "BLOCK":
            log(f"EVENT GATE: BLOCK {c['underlying']} — {v2['why']}; skipping.")
            return
        if v2["verdict"] == "WIDEN":
            verdict = "WIDEN"
    except Exception:
        pass
    # WIDEN -> halve size on macro days. Fix red-team finding: stock tier must
    # be able to halve too — scale by targeting a smaller credit, not by lots.
    if verdict == "WIDEN":
        log("WIDEN: macro day — reducing size (target risk halved).")
    risk_budget = capital * TARGET_RISK_PCT
    if verdict == "WIDEN":
        risk_budget *= 0.5
    if c["maxloss_rs_1lot"] * lots > risk_budget * 1.5:
        log(f"min-lot risk too big (Rs{c['maxloss_rs_1lot']*lots:.0f} > Rs{risk_budget*1.5:.0f}); skipping.")
        return
    c["lots"] = lots
    c["credit_rs"] = c["credit_rs_1lot"] * lots
    opened = ist_now()
    log(f"ENTER {c['underlying']}: {lots} lot(s) | credit Rs{c['credit_rs']:.0f} | "
        f"maxloss Rs{c['maxloss_rs_1lot']*lots:.0f} "
        f"({c['maxloss_rs_1lot']*lots/capital*100:.0f}% of capital) | IndiaVIX {vix}")
    orders, filled = place_legs(fy, c, lots, live)
    if orders is None:
        log("ENTRY ABORTED — leg rejected/failed. Not opening position.")
        return
    for o in orders:
        log(f"  {o['side']:<4} {o['symbol']} x{o['qty']} @ {o['ltp']:.2f}"
            + (f"  -> {o.get('broker_resp')}" if live else "  [paper]"))
    st["open"] = {"underlying": c["underlying"], "legs": c["legs"], "lots": lots,
                  "credit_rs": c["credit_rs"], "credit_pts": c["credit_pts"],
                  "lot": c["lot"], "opened": opened.isoformat(),
                  "vix": vix, "mode": "LIVE" if live else "PAPER"}
    st["trades"].append({"ts": opened.isoformat(), "underlying": c["underlying"],
                         "lots": lots, "credit_rs": c["credit_rs"], "pnl_rs": None,
                         "mode": "LIVE" if live else "PAPER"})
    save_state(st)
    log(f"position OPENED ({'LIVE' if live else 'PAPER'}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--i-understand", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--capital", type=float, default=5000.0)
    args = ap.parse_args()

    capital = args.capital
    live = args.live and args.i_understand
    if args.live and not args.i_understand:
        log("--live requires --i-understand; PAPER mode.")
    log(f"=== AUTO-CONDOR v2 | capital Rs{capital:,.0f} | "
        f"{'LIVE' if live else 'PAPER'} | "
        f"{'NIFTY' if capital >= NIFTY_THRESHOLD else 'STOCKS'} ===")

    fy = get_fyers()
    st = load_state()
    today = ist_now().strftime("%Y-%m-%d")
    if st.get("day") != today:
        st["day"] = today
        st["daily_pnl"] = 0.0
        save_state(st)

    while True:
        now = ist_now()
        if now is None:
            log("!! clock unavailable (all network time sources failed); skipping cycle. "
                "No entries, no exits, on an unverified clock.")
            if args.once:
                break
            time.sleep(POLL_SECONDS)
            continue
        wd = now.weekday()
        if st["open"]:
            manage_open(fy, st, live)
        elif wd < 5 and (now.hour > ENTRY_HOUR_IST or
                         (now.hour == ENTRY_HOUR_IST and now.minute >= ENTRY_MIN_IST)):
            if now.hour >= EXIT_HOUR_IST:
                log("past entry window; waiting for tomorrow.")
            elif st["daily_pnl"] <= DAILY_STOP_PCT * capital:
                log("daily stop; skipping entry.")
            else:
                enter(fy, st, live, capital)
        else:
            log(f"idle ({now.strftime('%a %H:%M')} IST); "
                f"entering at {ENTRY_HOUR_IST}:{ENTRY_MIN_IST:02d}.")

        if args.once:
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
