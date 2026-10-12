"""Event-aware entry decision for the condor desk.

Distinguishes the two very different risks:
  * CORPORATE event on the underlying (earnings, board meeting, split) -> the
    stock can GAP through the short strike. For a defined-risk condor this is
    the one event that can turn max-loss into a near-certain max-loss. BLOCK.
  * MACRO event (CPI, WPI, RBI MPC) -> moves the whole index, but a WIDE
    defined-risk condor with capped loss survives a normal macro reaction.
    Not a hard block; instead require a WIDER wing / smaller size on macro days.

Verdicts:
  BLOCK  -> do not enter (corporate event within horizon)
  WIDEN  -> enter but widen the long wing (macro day: more room, capped loss)
  CLEAR  -> enter normally
"""
import event_risk as er
import auto_condor_v2 as ac


def entry_verdict(underlying, days_ahead=7):
    r = er.event_risk_for(underlying, days_ahead)
    evs = r["events"]
    corp = [e for e in evs if e.get("kind") == "corporate"]
    macro_hi = [e for e in evs if e.get("kind") == "macro" and e.get("impact") == "high"]

    if corp:
        why = "; ".join(f"{e.get('purpose','event')} on {e.get('date')}" for e in corp[:2])
        return {"verdict": "BLOCK", "why": f"corporate event: {why}",
                "widen": False, "events": evs}
    if macro_hi:
        names = ", ".join(sorted({e["name"] for e in macro_hi}))
        return {"verdict": "WIDEN", "why": f"macro event within {days_ahead}d: {names}",
                "widen": True, "events": evs}
    return {"verdict": "CLEAR", "why": "no events in horizon", "widen": False, "events": evs}


if __name__ == "__main__":
    for u in ["MARUTI", "NIFTY"]:
        v = entry_verdict(u, 7)
        print(f"{u:8} -> {v['verdict']:5} widen={v['widen']}  | {v['why']}")
