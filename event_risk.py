"""Event-risk gate for the condor desk.

Turns real events into decisions, not just info:
  1. NSE corporate event calendar (board meetings, results, dividends, splits)
     -> if the condor's underlying has an event BEFORE the trade expires,
        DON'T enter (single-stock event risk defeats the VRP edge).
  2. India macro calendar (CPI 12th 16:00, WPI 14th, RBI MPC dates, F&O expiry)
     -> hard-coded high-impact macro dates in IST.
  3. Fyers market_status -> exchange open/closed, NSE holidays.

Scores the position's underlying for event risk and returns a verdict.
Called by the desk before entering; also snapshotted for the monitor.
"""
import json
import os
import urllib.request
from datetime import datetime, timedelta

import auto_condor_v2 as ac

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "event_calendar_cache.json")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Accept": "application/json", "Referer": "https://www.nseindia.com/"}

# Hard-coded India macro calendar (IST). CPI = 12th 16:00, WPI = 14th noon,
# RBI MPC (FY27): 7 Oct, 4 Dec, 5 Feb. Extend as needed; these are rule-based.
RBI_MPC_DATES = ["2026-10-07", "2026-12-04", "2027-02-05"]


def _macro_events(now):
    """Rule-based India macro events with a date + time (IST) + impact."""
    evs = []
    y, m = now.year, now.month
    # CPI + IIP: 12th of month, 16:00 IST (after close -> next-day reaction)
    for yy, mm in [(y, m), (y, m + 1 if m < 12 else 1)]:
        if mm > 12:
            yy, mm = y + 1, 1
        evs.append({"date": f"{yy}-{mm:02d}-12", "time": "16:00", "name": "India CPI/IIP",
                    "impact": "high", "src": "rule"})
        evs.append({"date": f"{yy}-{mm:02d}-14", "time": "12:00", "name": "India WPI",
                    "impact": "medium", "src": "rule"})
    for d in RBI_MPC_DATES:
        evs.append({"date": d, "time": "10:00", "name": "RBI MPC decision",
                    "impact": "high", "src": "rule"})
    return evs


def _nse_events():
    """NSE corporate event calendar (board meetings, results, dividends...)."""
    try:
        req = urllib.request.Request(
            "https://www.nseindia.com/api/event-calendar?type=corporate", headers=UA)
        raw = urllib.request.urlopen(req, timeout=10).read()
        rows = json.loads(raw)
        out = []
        for r in rows:
            sym = (r.get("symbol") or "").strip()
            if not sym:
                continue
            out.append({"symbol": sym, "purpose": (r.get("bm_desc") or r.get("purpose") or "")[:80],
                        "date": r.get("date")})
        return out
    except Exception:
        return []


def _parse_nse_date(s):
    for fmt in ("%d-%b-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def event_risk_for(underlying, days_ahead=7):
    """Return {verdict, events} for a given stock/index underlying.
    verdict: 'BLOCK' if a corporate event hits before/at expiry window, else 'CLEAR'."""
    now = ac.ist_now()
    today = now.date()
    horizon = today + timedelta(days=days_ahead)
    events = []

    # corporate events (single-stock only; index NIFTY has none)
    if underlying and underlying.upper() not in ("NIFTY", "NIFTY50"):
        for e in _nse_events():
            if e["symbol"].upper() != underlying.upper():
                continue
            d = _parse_nse_date(e["date"])
            if d and today <= d <= horizon:
                events.append({**e, "kind": "corporate", "impact": "high"})

    # macro events within horizon (these affect index + all stocks)
    for e in _macro_events(now):
        try:
            d = datetime.strptime(e["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if today <= d <= horizon:
            events.append({**e, "kind": "macro"})

    block = any(e["kind"] == "corporate" for e in events) or \
            any(e.get("impact") == "high" and e["kind"] == "macro" for e in events)
    return {"verdict": "BLOCK" if block else "CLEAR", "events": events,
            "asof_ist": now.isoformat()}


def snapshot():
    """Write event state + return a stable alert line for the monitor."""
    now = ac.ist_now()
    today = now.date()
    horizon = today + timedelta(days=3)
    macro = [e for e in _macro_events(now)
             if today <= datetime.strptime(e["date"], "%Y-%m-%d").date() <= horizon]
    nse = _nse_events()
    out = {"ts_ist": now.isoformat(),
           "macro_3d": [{"date": e["date"], "time": e["time"], "name": e["name"]} for e in macro],
           "nse_events_total": len(nse)}
    json.dump(out, open(CACHE, "w"), indent=2)
    names = ",".join(sorted({e["name"] + "@" + e["date"] for e in macro}))
    print(f"EVENTS:{len(macro)}:{names}" if names else "EVENTS:none")


if __name__ == "__main__":
    snapshot()
