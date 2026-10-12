"""THE MACHINE — unified condor desk: engine + monitors + live dashboard.

One process that every cycle:
  1. runs the condor engine (enter / manage / square-off) on live Fyers data
  2. composes a consolidated status (position, market, events, whale flows)
  3. writes machine_status.json + regenerates dashboard.html
  4. prints a text dashboard to the log

Paper by default; live requires BOTH --live --i-understand. All safety gates
from auto_condor_v2 apply (one position, daily stop, event BLOCK/WIDEN,
stop-loss, 15:00 IST square-off, authoritative network IST).

Run:   python machine.py                # paper, continuous
       python machine.py --once         # one cycle
       python machine.py --live --i-understand
"""
import argparse
import json
import os
import time
from datetime import datetime

import auto_condor_v2 as ac
import entry_decision as ed

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "machine.log")
STATUS = os.path.join(HERE, "machine_status.json")
DASH = os.path.join(HERE, "dashboard.html")
WHALE_STATE = os.path.join(HERE, "whale_state.json")

POLL_SECONDS = 60
ENTRY_HOUR_IST, ENTRY_MIN_IST = ac.ENTRY_HOUR_IST, ac.ENTRY_MIN_IST
EXIT_HOUR_IST = ac.EXIT_HOUR_IST


def log(msg):
    line = f"[{ac.ist_now().strftime('%Y-%m-%d %H:%M:%S')} IST] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def read_whale():
    try:
        return json.load(open(WHALE_STATE))
    except Exception:
        return None


def build_status(fy, st, capital):
    """Consolidated desk status dict."""
    out = {"ts_ist": ac.ist_now().isoformat(),
           "capital": capital, "mode": "LIVE" if st.get("_live") else "PAPER"}
    # position
    o = st.get("open")
    pos = None
    if o:
        pnl = ac.condor_pnl(fy, o)
        stop = -o["credit_rs"] * ac.STOP_LOSS_FRAC
        pos = {"underlying": o["underlying"], "lots": o["lots"],
               "credit_rs": round(o["credit_rs"]), "mode": o.get("mode"),
               "mtm_pnl": round(pnl) if pnl is not None else None,
               "stop_level": round(stop),
               "stop_breached": pnl is not None and pnl <= stop}
    out["position"] = pos
    # market context
    ctx = {}
    try:
        q = fy.quotes(data={"symbols": "NSE:NIFTY50-INDEX,NSE:INDIAVIX-INDEX,"
                                       "NSE:NIFTYBANK-INDEX"})
        for d in q.get("d", []):
            v = d.get("v", {})
            last, prev = v.get("lp") or 0, v.get("prev_close_price") or 0
            name = d["n"].replace("NSE:", "").replace("-INDEX", "")
            ctx[name] = {"last": round(last, 2),
                         "chg_pct": round((last / prev - 1) * 100, 2) if prev else None}
    except Exception as e:
        ctx["error"] = repr(e)[:60]
    out["context"] = ctx
    # event risk
    try:
        v = ed.entry_verdict((o or {}).get("underlying") or "NIFTY", 7)
        out["event_risk"] = {"verdict": v["verdict"], "why": v["why"]}
    except Exception as e:
        out["event_risk"] = {"verdict": "?", "why": repr(e)[:60]}
    # whale flows
    w = read_whale()
    if w:
        def _wf(x):
            try:
                return float(str(x).replace(",", ""))
            except (ValueError, TypeError):
                return None
        w = {"ts": w.get("ts"), "fii_net": _wf(w.get("fii_net")),
             "dii_net": _wf(w.get("dii_net")), "bulk_deals": w.get("bulk_deals")}
    out["whale"] = w
    # health — measure the log THIS machine writes, not the old engine log
    try:
        out["log_age_s"] = round(time.time() - os.path.getmtime(LOG))
    except Exception:
        out["log_age_s"] = None
    out["engine_age_s"] = out["log_age_s"]
    out["loop_healthy"] = out["log_age_s"] is not None and out["log_age_s"] < 300
    out["daily_pnl"] = round(st.get("daily_pnl", 0))
    return out


CSS = """
*{box-sizing:border-box}
body{font-family:var(--font,ui-sans-serif,system-ui,sans-serif);color:var(--foreground);
 background:transparent;margin:0;padding:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;max-width:1100px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.card h3{margin:0 0 8px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted-foreground);font-weight:600}
.big{font-size:22px;font-weight:700;line-height:1.1}
.sub{font-size:12px;color:var(--muted-foreground);margin-top:3px}
.row{display:flex;justify-content:space-between;font-size:13px;padding:3px 0}
.pos{color:#3fb950}.neg{color:#f85149}.flat{color:var(--muted-foreground)}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;border:1px solid var(--border)}
.b-paper{color:#d29922;border-color:#d29922}
.b-live{color:#3fb950;border-color:#3fb950}
.b-block{color:#f85149;border-color:#f85149}
.b-widen{color:#d29922;border-color:#d29922}
.b-clear{color:#3fb950;border-color:#3fb950}
.bar{height:6px;border-radius:3px;background:var(--border);margin-top:8px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent)}
.head{display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.head h1{font-size:15px;margin:0;font-weight:700}
.btns{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
button{cursor:pointer;font:inherit;font-size:12px;padding:6px 12px;border-radius:8px;
 border:1px solid var(--border);background:var(--card);color:var(--foreground)}
button:hover{border-color:var(--accent)}
.ts{font-size:11px;color:var(--muted-foreground);margin-left:auto}
"""


def _pct_cls(x):
    if x is None:
        return "flat"
    return "pos" if x > 0 else ("neg" if x < 0 else "flat")


def render_dashboard(s):
    """Build dashboard.html from status dict. CSS is plain (no f-string braces)."""
    pos = s.get("position")
    ctx = s.get("context", {})
    ev = s.get("event_risk", {})
    whale = s.get("whale") or {}
    mode = s.get("mode", "PAPER")

    # position card
    if pos:
        pnl = pos["mtm_pnl"]
        pcls = "pos" if (pnl or 0) >= 0 else "neg"
        risk_pct = abs(pos["stop_level"]) / max(s["capital"], 1) * 100
        pos_html = (
            f'<div class="card"><h3>Position</h3>'
            f'<div class="big">{pos["underlying"]} <span class="sub">x{pos["lots"]} lot</span></div>'
            f'<div class="row"><span>MTM P&amp;L</span><b class="{pcls}">'
            f'{"+" if (pnl or 0)>=0 else ""}₹{pnl}</b></div>'
            f'<div class="row"><span>Credit</span><b>₹{pos["credit_rs"]}</b></div>'
            f'<div class="row"><span>Stop</span><b class="neg">₹{pos["stop_level"]}</b></div>'
            f'<div class="sub">max risk {risk_pct:.1f}% of capital · '
            f'{"⚠ STOP BREACHED" if pos["stop_breached"] else "holding"}</div></div>')
    else:
        pos_html = ('<div class="card"><h3>Position</h3><div class="big flat">FLAT</div>'
                    '<div class="sub">no open condor · watching for entry window</div></div>')

    # market card
    def mrow(label, key):
        c = ctx.get(key, {})
        if not c or "last" not in c:
            return ""
        cls = _pct_cls(c.get("chg_pct"))
        arrow = "▲" if (c.get("chg_pct") or 0) > 0 else ("▼" if (c.get("chg_pct") or 0) < 0 else "·")
        return (f'<div class="row"><span>{label}</span><b class="{cls}">'
                f'{c["last"]:,.1f} <span class="sub">{arrow} {c.get("chg_pct")}%</span></b></div>')
    mkt_html = ('<div class="card"><h3>Market</h3>'
                + mrow("Nifty 50", "NIFTY50") + mrow("Bank Nifty", "NIFTYBANK")
                + mrow("India VIX", "INDIAVIX") + '</div>')

    # events card
    verdict = ev.get("verdict", "?")
    vcls = {"BLOCK": "b-block", "WIDEN": "b-widen", "CLEAR": "b-clear"}.get(verdict, "")
    ev_html = (f'<div class="card"><h3>Event Risk</h3>'
               f'<div class="big"><span class="badge {vcls}">{verdict}</span></div>'
               f'<div class="sub" style="margin-top:8px">{ev.get("why","")}</div></div>')

    # flows card
    if whale:
        fii, dii = whale.get("fii_net"), whale.get("dii_net")
        flows_html = (f'<div class="card"><h3>Institutional Flows</h3>'
                      f'<div class="row"><span>FII net</span><b class="{_pct_cls(fii)}">'
                      f'{fii:+,.0f} Cr</b></div>'
                      f'<div class="row"><span>DII net</span><b class="{_pct_cls(dii)}">'
                      f'{dii:+,.0f} Cr</b></div>'
                      f'<div class="sub">as of {whale.get("ts","?")}</div></div>')
    else:
        flows_html = ('<div class="card"><h3>Institutional Flows</h3>'
                      '<div class="big flat">—</div><div class="sub">awaiting whale watcher</div></div>')

    # health card
    health = "healthy" if s.get("loop_healthy") else "STALE"
    hcls = "pos" if s.get("loop_healthy") else "neg"
    health_html = (f'<div class="card"><h3>Engine</h3>'
                   f'<div class="big {hcls}">{health}</div>'
                   f'<div class="sub">cycle {s.get("engine_age_s")}s ago · day P&amp;L ₹{s.get("daily_pnl")}</div>'
                   f'<div class="sub">IST {ac.ist_now().strftime("%a %H:%M")} · entry {ENTRY_HOUR_IST}:{ENTRY_MIN_IST:02d} · exit {EXIT_HOUR_IST}:00</div></div>')

    mb = "b-live" if mode == "LIVE" else "b-paper"
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="30">'
        f'<style>{CSS}</style></head><body>'
        f'<div class="head"><h1>CONDOR DESK</h1>'
        f'<span class="badge {mb}">{mode}</span>'
        f'<span class="ts">{s.get("ts_ist","")}</span></div>'
        f'<div class="grid">{pos_html}{mkt_html}{ev_html}{flows_html}{health_html}</div>'
        '<div class="btns">'
        '<button data-hermes-send="machine status">status</button>'
        '<button data-hermes-send="square off the open position now">square off</button>'
        '<button data-hermes-send="reconcile P&L">reconcile</button>'
        '<button data-hermes-send="arm the machine live">go live</button>'
        '</div></body></html>')
    with open(DASH, "w", encoding="utf-8") as f:
        f.write(html)


def text_dashboard(s):
    pos = s.get("position")
    ctx = s.get("context", {})
    ev = s.get("event_risk", {})
    w = s.get("whale") or {}
    n = ctx.get("NIFTY50", {})
    v = ctx.get("INDIAVIX", {})
    log("─" * 62)
    log(f"MACHINE | {s['mode']} | capital Rs{s['capital']:,.0f} | IST {ac.ist_now().strftime('%a %H:%M')}")
    if pos:
        log(f"  POS  {pos['underlying']} x{pos['lots']} | MTM {pos['mtm_pnl']:+} | "
            f"credit {pos['credit_rs']} | stop {pos['stop_level']} | "
            f"{'BREACH' if pos['stop_breached'] else 'hold'}")
    else:
        log("  POS  FLAT")
    log(f"  MKT  Nifty {n.get('last')} ({n.get('chg_pct')}%) | VIX {v.get('last')} ({v.get('chg_pct')}%)")
    log(f"  EVT  {ev.get('verdict')} — {ev.get('why')}")
    if w:
        log(f"  FLW  FII {w.get('fii_net')} Cr | DII {w.get('dii_net')} Cr ({w.get('ts')})")
    log(f"  HLT  {'ok' if s.get('loop_healthy') else 'STALE'} | day P&L Rs{s.get('daily_pnl')}")
    log("─" * 62)


def run_cycle(fy, st, live, capital):
    """One engine + status cycle."""
    now = ac.ist_now()
    wd = now.weekday()
    if st["open"]:
        ac.manage_open(fy, st, live)
    elif wd < 5 and (now.hour > ENTRY_HOUR_IST or
                     (now.hour == ENTRY_HOUR_IST and now.minute >= ENTRY_MIN_IST)):
        if now.hour >= EXIT_HOUR_IST:
            log("past entry window; waiting for tomorrow.")
        elif st["daily_pnl"] <= ac.DAILY_STOP_PCT * capital:
            log("daily stop; skipping entry.")
        else:
            ac.enter(fy, st, live, capital)
    else:
        log(f"idle ({now.strftime('%a %H:%M')} IST); entry at {ENTRY_HOUR_IST}:{ENTRY_MIN_IST:02d}.")
    status = build_status(fy, st, capital)
    json.dump(status, open(STATUS, "w"), indent=2)
    render_dashboard(status)
    text_dashboard(status)


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

    log("=" * 62)
    log(f"THE MACHINE | Rs{capital:,.0f} | {'LIVE' if live else 'PAPER'} | "
        f"{'NIFTY' if capital >= ac.NIFTY_THRESHOLD else 'STOCKS'} | dashboard -> {DASH}")
    log("=" * 62)

    fy = ac.get_fyers()
    st = ac.load_state()
    st["_live"] = live
    today = ac.ist_now().strftime("%Y-%m-%d")
    if st.get("day") != today:
        st["day"] = today
        st["daily_pnl"] = 0.0
    ac.save_state(st)

    while True:
        try:
            run_cycle(fy, st, live, capital)
        except Exception as e:
            log(f"cycle error: {repr(e)[:120]}")
        if args.once:
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
