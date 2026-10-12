"""Scan F&O stock options for a condor that fits Rs5K capital.

Fyers symbol master (NSE_FO.csv) has 84,529 rows. We want single-stock OPTIONS
(near expiry) where 1 lot's max loss is small enough that Rs5K can take a few
lots with a capped tail (~Rs500-1000 risk per lot).

Approach:
 1. parse the master, find stock options (symbol like NSE:NAME<ddmmyy><strike>CE)
    excluding indices (NIFTY/BANKNIFTY/FINNIFTY/MIDCPNIFTY)
 2. pick a shortlist of liquid underlyings (large-cap, known names)
 3. pull the live chain for each via Fyers optionchain, build a condor,
    compute 1-lot max loss, and report the ones that fit Rs5K.
"""
import csv
import io
import json
import re
import urllib.request

from fyers_apiv3 import fyersModel

creds = {}
for line in open("fyers_env.txt"):
    m = re.match(r'\s*(\w+)\s*=\s*"([^"]*)"', line)
    if m:
        creds[m.group(1)] = m.group(2)
t = json.load(open("fyers_token.json"))
FY = fyersModel.FyersModel(client_id=creds["FYERS_CLIENT_ID"],
                           token=t["access_token"], is_async=False, log_path="")

# liquid single-stock underlyings to test (large, tight spreads, small lot values)
CANDIDATES = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN",
              "ITC", "LT", "AXISBANK", "KOTAKBANK", "BHARTIARTL", "MARUTI",
              "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "ADANIENT", "TATAMOTORS"]

CAPITAL = 5000.0
TARGET_RISK = 800.0     # aim <= Rs800 max loss per lot
SHORT_WING, LONG_WING = 1, 2   # tighter: cheap stocks need narrow wings to collect real premium


def get_master():
    url = "https://public.fyers.in/sym_details/NSE_FO.csv"
    data = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", errors="ignore")
    return list(csv.reader(io.StringIO(data)))


def find_stock_option_symbols(rows, underlying):
    """Return {expiry: {strike: {'CE':sym,'PE':sym}}} for one underlying."""
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


def main():
    rows = get_master()
    print(f"master: {len(rows)} rows | scanning {len(CANDIDATES)} underlyings for Rs5K fit\n")
    results = []
    for u in CANDIDATES:
        try:
            opts = find_stock_option_symbols(rows, u)
            if not opts:
                continue
            # nearest expiry (decode ddMmYY; format is like '26OCT' or '03NOV')
            exps = sorted(opts.keys())
            from datetime import datetime

            def keyf(e):
                s = e.upper()
                for fmt in ("%d%b", "%d%B"):
                    try:
                        return datetime.strptime(s, fmt).replace(year=2026)
                    except ValueError:
                        continue
                return datetime.max
            exps = sorted(opts.keys(), key=keyf)
            exp = exps[0]
            strikes = sorted(opts[exp].keys())
            if len(strikes) < 8:
                continue
            # build quotes request for all CE+PE on this expiry (batch)
            syms = []
            for k in strikes:
                for typ in ("CE", "PE"):
                    if typ in opts[exp][k]:
                        syms.append(opts[exp][k][typ])
            # Fyers allows ~50 symbols per quotes call; chunk it
            ltp = {}
            import math
            for i in range(0, len(syms), 40):
                chunk = syms[i:i + 40]
                q = FY.quotes(data={"symbols": ",".join(chunk)})
                if q.get("s") != "ok":
                    continue
                for d in q.get("d", []):
                    v = d.get("v", {})
                    ltp[d["n"]] = v.get("lp")
            # ATM from SPOT (not median strike) — pull underlying equity quote
            sq = FY.quotes(data={"symbols": f"NSE:{u}-EQ"})
            spot = None
            if sq.get("s") == "ok" and sq.get("d"):
                spot = sq["d"][0].get("v", {}).get("lp")
            if not spot:
                continue
            # nearest strike to spot = ATM
            atm = min(strikes, key=lambda k: abs(k - spot))
            ai = strikes.index(atm)

            def nearest(off, typ):
                """Find the strike nearest to ATM+off that actually has a live
                quote for `typ`. Falls back through nearby offsets so a gap in
                the master doesn't kill the wing."""
                for d in range(0, 8):
                    for sgn in ((0,) if d == 0 else (d, -d)):
                        k = ai + off + sgn
                        if not (0 <= k < len(strikes)):
                            continue
                        sym = opts[exp][strikes[k]].get(typ)
                        if sym and ltp.get(sym) is not None:
                            return (strikes[k], ltp[sym], sym)
                return None

            cs = nearest(SHORT_WING, "CE")
            cl = nearest(LONG_WING, "CE")
            ps = nearest(-SHORT_WING, "PE")
            pl = nearest(-LONG_WING, "PE")
            if not all([cs, cl, ps, pl]):
                continue
            # lot size from the master row for this symbol
            lot = 0
            for r in rows:
                if len(r) > 9 and r[9] == cs[2]:
                    try:
                        lot = int(float(r[2] or 0))
                    except (ValueError, TypeError):
                        lot = 0
                    break
            if lot <= 0:
                continue
            credit = (cs[1] - cl[1]) + (ps[1] - pl[1])
            wing = cl[0] - cs[0]
            maxloss_pts = max(wing - credit, 0)
            maxloss_1lot = maxloss_pts * lot
            credit_1lot = credit * lot
            results.append({"u": u, "exp": exp, "atm": atm, "lot": lot,
                            "credit": credit_1lot, "maxloss": maxloss_1lot,
                            "legs": (cs, cl, ps, pl)})
            print(f"{u:12} exp {exp} lot {lot:5}  credit/lot Rs{credit_1lot:7.0f}  "
                  f"maxloss/lot Rs{maxloss_1lot:7.0f}  "
                  f"{'FITS' if maxloss_1lot <= TARGET_RISK else ''}")
        except Exception as e:
            print(f"{u:12} error {repr(e)[:60]}")
    print(f"\n=== FITS Rs5K (maxloss <= Rs{TARGET_RISK:.0f}/lot) ===")
    fits = [r for r in results if r["maxloss"] <= TARGET_RISK]
    fits.sort(key=lambda r: -r["credit"])
    for r in fits[:8]:
        cs, cl, ps, pl = r["legs"]
        print(f"\n  {r['u']}  {r['exp']}  ATM {r['atm']:.0f}  lot {r['lot']}")
        print(f"    SELL {cs[0]:.0f} CE @{cs[1]:.2f}   BUY {cl[0]:.0f} CE @{cl[1]:.2f}")
        print(f"    SELL {ps[0]:.0f} PE @{ps[1]:.2f}   BUY {pl[0]:.0f} PE @{pl[1]:.2f}")
        lots = max(1, int(TARGET_RISK // max(r["maxloss"], 1)))
        lots = min(lots, int(CAPITAL // max(r["maxloss"], 1)))
        print(f"    credit Rs{r['credit']:.0f}/lot | maxloss Rs{r['maxloss']:.0f}/lot | "
              f"{lots} lot(s) = credit Rs{r['credit']*lots:.0f}, maxloss Rs{r['maxloss']*lots:.0f} "
              f"({r['maxloss']*lots/CAPITAL*100:.0f}% of Rs5K)")


if __name__ == "__main__":
    main()
