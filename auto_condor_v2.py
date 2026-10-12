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
        return _dt.datetime.utcnow() + _dt.timedelta(hours=5, minutes=30)


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
            pass
    return {"open": None, "day": None, "daily_pnl": 0.0, "trades": []}


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
            exp = sorted(by_exp.keys(),
                         key=lambda e: datetime.strptime(e.upper() + " 2026", "%d%b %Y"))[0]
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
            rec["broker_resp"] = fy.place_order(data)
        orders.append(rec)
    return orders


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
    if now.hour >= EXIT_HOUR_IST:
        log(f"exit window (>= {EXIT_HOUR_IST}h): squaring off {o['underlying']} condor")
        close_legs(fy, o, live)
        pnl = condor_pnl(fy, o)
        if pnl is not None:
            st["daily_pnl"] += pnl
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
        st["open"] = None
        save_state(st)


def enter(fy, st, live, capital):
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
        c = scan_stock_condor(fy, capital)
        lots = 1
        vix = None
    if not c:
        log("no tradeable condor built.")
        return
    if c["maxloss_rs_1lot"] * lots > capital * TARGET_RISK_PCT * 1.5:
        log(f"min-lot risk too big (Rs{c['maxloss_rs_1lot']*lots:.0f}); skipping.")
        return
    c["lots"] = lots
    c["credit_rs"] = c["credit_rs_1lot"] * lots
    log(f"ENTER {c['underlying']}: {lots} lot(s) | credit Rs{c['credit_rs']:.0f} | "
        f"maxloss Rs{c['maxloss_rs_1lot']*lots:.0f} "
        f"({c['maxloss_rs_1lot']*lots/capital*100:.0f}% of capital) | IndiaVIX {vix}")
    orders = place_legs(fy, c, lots, live)
    for o in orders:
        log(f"  {o['side']:<4} {o['symbol']} x{o['qty']} @ {o['ltp']:.2f}"
            + (f"  -> {o.get('broker_resp')}" if live else "  [paper]"))
    st["open"] = {"underlying": c["underlying"], "legs": c["legs"], "lots": lots,
                  "credit_rs": c["credit_rs"], "credit_pts": c["credit_pts"],
                  "lot": c["lot"], "opened": ist_now().isoformat(),
                  "vix": vix, "mode": "LIVE" if live else "PAPER"}
    st["trades"].append({"ts": ist_now().isoformat(), "underlying": c["underlying"],
                         "lots": lots, "credit_rs": c["credit_rs"],
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
