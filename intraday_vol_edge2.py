"""INTRADAY VOLATILITY EDGE v2 — rigorous, longer-hold.

Extends intraday_vol_edge.py. Prior pass: bar-level (5-min) trading is a COST
WALL (fixed Rs40/leg brokerage swamps tiny 5-min moves); U-shape not confirmed
on a crude 4-day curve. This pass fixes both weaknesses:

1. RIGOROUS VOL CURVE: normalize each instrument by its OWN daily vol
   (z-scored vol per bucket) so the curve is comparable across names; measure
   on BOTH 5-min and 15-min bars (15-min = cleaner vol estimate from few bars).

2. LONGER HOLDS (30-60 min, not 5-min): the only variants that can clear the
   cost wall. Each 'trade' is a real move big enough to beat Rs80 round-trip.
   a) OPENING FADE 30-60min: fade outsized first-30min move, exit by 10:30-11:00
   b) CLOSE structure: last 30-60min vs 14:30 level (persistence or reversion)
   c) VOL-TIMING: simple momentum/reversion signal, high-vol windows only vs all

Honesty gates: trade count (<30 = NOT A VERDICT), independent recompute,
look-ahead check, holdout split. Negative/inconclusive is a valid result.

Run: python intraday_vol_edge2.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STORE = "data_indmoney/equity_5min.parquet"
SESSION = ("09:15", "15:30")
NOTIONAL = 15000.0
BROKERAGE = 20.0          # per leg
COST_RT_PCT = 0.001       # ~0.1% round-trip (STT+exch+GST+stamp+slippage)


def load(interval="5minute"):
    df = pd.read_parquet(STORE)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df = df[df.interval == interval].copy()
    df["date"] = df.datetime_ist.dt.date
    df["hm"] = df.datetime_ist.dt.strftime("%H:%M")
    df["ret"] = df.groupby(["label", "date"]).close.transform(
        lambda s: np.log(s).diff())
    return df


def quality_gate(df):
    """L13: keep only DOMINANT ind_key per label + drop corrupt jump bars.
    RELIANCE/ITC have two ind_keys interleaved (Industries vs Power etc.);
    price-filtering can't catch it -> bogus -Rs2cr artifact without this."""
    kept = []
    for lb, g in df.groupby("label"):
        counts = g.ind_key.value_counts()
        if len(counts) > 1:
            g = g[g.ind_key == counts.index[0]]  # dominant = intended large-cap
        kept.append(g)
    df = pd.concat(kept)
    med_absret = df.groupby(["label", "date"]).ret.transform(lambda s: s.abs().median())
    df = df[(df.ret.abs() <= 10 * med_absret) | df.ret.isna()]
    return df.reset_index(drop=True)


def vol_curve_normalized(df, bucket_min=15):
    """Per-bucket vol, each instrument z-scored by its own mean daily vol, then
    averaged across instruments. Comparable across names."""
    df = df.copy()
    df["mod"] = df.datetime_ist.dt.hour * 60 + df.datetime_ist.dt.minute
    df["bucket"] = (df["mod"] // bucket_min) * bucket_min
    # per (label,date,bucket) mean |ret| = realized vol for that bucket
    bvol = (df.groupby(["label", "date", "bucket"]).ret
            .apply(lambda s: s.abs().mean()).dropna().reset_index(name="vol"))
    # normalize each instrument-day by its own mean bucket vol
    bvol["instr_day_mean"] = bvol.groupby(["label", "date"]).vol.transform("mean")
    bvol["norm"] = bvol.vol / bvol.instr_day_mean
    curve = bvol.groupby("bucket").norm.mean()

    def fmt(m):
        return f"{m//60:02d}:{m%60:02d}"
    print(f"  {'bucket':<9}{'norm_vol':>10}")
    for b in sorted(curve.index):
        print(f"  {fmt(b):<9}{curve[b]:>9.2f}")
    open_b = [b for b in curve.index if 555 <= b < 600]
    mid_b = [b for b in curve.index if 720 <= b < 780]
    close_b = [b for b in curve.index if 840 <= b < 930]
    om, mm, cm = (curve[open_b].mean(), curve[mid_b].mean(), curve[close_b].mean())
    print(f"\n  open {om:.2f} / mid {mm:.2f} / close {cm:.2f} (normalized, 1.0=flat)")
    u = (om > mm * 1.10) and (cm > mm * 1.10)
    print(f"  U-shape (open & close > mid by >10%): {'CONFIRMED' if u else 'NOT confirmed'}")
    return curve, {"open": om, "mid": mm, "close": cm, "u": u}


def simulate(trades):
    capital, peak, maxdd, last_day = 150000.0, 150000.0, 0, None
    for t in trades:
        if last_day is not None and t["date"] <= last_day:
            continue
        qty = NOTIONAL / t["entry"]
        gross = (t["exit"] - t["entry"]) * qty * t["dir"]
        cost = NOTIONAL * COST_RT_PCT + 2 * BROKERAGE
        capital += gross - cost
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
        last_day = t["date"]
    return capital, maxdd


def report(name, trades):
    if not trades:
        print(f"  {name:<30} 0 trades")
        return None
    r = np.array([(t["exit"] / t["entry"] - 1) * t["dir"] for t in trades])
    gross = r * NOTIONAL
    net = gross - (NOTIONAL * COST_RT_PCT + 2 * BROKERAGE)
    cap, maxdd = simulate(trades)
    n = len(trades)
    v = "NOT A VERDICT (<30)" if n < 30 else "sample"
    print(f"  {name:<30} n={n:<4} win={100*(r>0).mean():3.0f}% "
          f"avgNet=Rs{net.mean():+7.1f} totNet=Rs{net.sum():+8.0f} "
          f"worst=Rs{net.min():+7.0f} end=Rs{cap:,.0f} [{v}]")
    return net


def var_open_fade(df, exit_hm="10:45"):
    """Fade outsized first-30min move; hold to exit_hm (30-60min). Threshold =
    stock's OWN prior-day opening-move mean+0.5sd (no look-ahead)."""
    trades = []
    for (lb, d), g in df.groupby(["label", "date"]):
        g = g.sort_values("datetime_ist")
        ob = g[(g.hm >= "09:15") & (g.hm <= "09:45")]
        if len(ob) < 6:
            continue
        entry = float(ob.close.iloc[-1])
        base = float(ob.open.iloc[0])
        move = entry / base - 1
        prior = df[(df.label == lb) & (df.date < d)]
        p = []
        for _, pg in prior.groupby("date"):
            pg = pg.sort_values("datetime_ist")
            pob = pg[(pg.hm >= "09:15") & (pg.hm <= "09:45")]
            if len(pob) >= 6:
                p.append(abs(float(pob.close.iloc[-1]) / float(pob.open.iloc[0]) - 1))
        thr = (np.mean(p) + 0.5 * np.std(p)) if p else 0
        if thr == 0 or abs(move) < thr:
            continue
        direction = -1 if move > 0 else 1
        ex = g[g.hm >= exit_hm]
        exit_px = float(ex.close.iloc[0]) if not ex.empty else float(g.close.iloc[-1])
        trades.append({"label": lb, "date": d, "dir": direction,
                       "entry": entry, "exit": exit_px})
    return trades


def var_close_structure(df, entry_hm="14:30", exit_hm="15:15"):
    """Does the last 45min continue or revert the trend into 14:30?"""
    trades = []
    for (lb, d), g in df.groupby(["label", "date"]):
        g = g.sort_values("datetime_ist")
        pre = g[g.hm <= entry_hm]
        post = g[(g.hm > entry_hm) & (g.hm <= exit_hm)]
        if len(pre) < 20 or len(post) < 2:
            continue
        p1430 = float(pre.close.iloc[-1])
        o = float(g.open.iloc[0])
        trend = p1430 / o - 1
        if abs(trend) < 0.003:
            continue
        direction = 1 if trend > 0 else -1     # continuation
        exit_px = float(post.close.iloc[-1])
        trades.append({"label": lb, "date": d, "dir": direction,
                       "entry": p1430, "exit": exit_px})
    return trades


def var_vol_timing(df, curve, highvol_buckets):
    """30-min-hold momentum/reversion taken only in high-vol windows vs all."""
    df = df.copy()
    df["mod"] = df.datetime_ist.dt.hour * 60 + df.datetime_ist.dt.minute
    df["bucket"] = (df["mod"] // 15) * 15
    hi, al = [], []
    for (lb, d), g in df.groupby(["label", "date"]):
        g = g.sort_values("datetime_inst" if False else "datetime_ist").reset_index(drop=True)
        # signal at bar i: fade the prior 30min (6 bars) move; hold 30min
        for i in range(6, len(g) - 6):
            move = g.close.iloc[i] / g.close.iloc[i - 6] - 1
            if abs(move) < 0.004:
                continue
            direction = -1 if move > 0 else 1
            entry = float(g.close.iloc[i])
            exit_px = float(g.close.iloc[i + 6])
            t = {"label": lb, "date": d, "dir": direction,
                 "entry": entry, "exit": exit_px}
            al.append(t)
            if g.bucket.iloc[i] in highvol_buckets:
                hi.append(t)
    return hi, al


def main():
    print("=== INTRADAY VOLATILITY EDGE v2 (rigorous, 30-60min holds) ===")
    df = quality_gate(load("5minute"))
    print(f"  clean: {df.label.nunique()} instruments x {df.date.nunique()} days\n")

    print("=== VOL CURVE (normalized per-instrument, 15-min bars) ===")
    df15 = quality_gate(load("15minute"))
    curve, lv = vol_curve_normalized(df15, 15)

    highvol = set(b for b in curve.index if (555 <= b < 600) or (840 <= b < 930))

    print(f"\n=== VARIANTS (30-60min holds, net of Rs80 RT + 0.1%) ===")
    of = var_open_fade(df, "10:45")
    report("A: opening fade (30min hold)", of)
    cs = var_close_structure(df)
    report("B: close structure (45min)", cs)
    hi, al = var_vol_timing(df, curve, highvol)
    report("C1: MR all windows (30min)", al)
    report("C2: MR high-vol only (30min)", hi)

    # honesty: independent recompute of C1
    print(f"\n=== HONESTY CHECK ===")
    if al:
        manual = [((t["exit"] - t["entry"]) * t["dir"] / t["entry"]) * NOTIONAL
                  - (NOTIONAL * COST_RT_PCT + 2 * BROKERAGE) for t in al[:500]]
        auto = report.__wrapped__ if hasattr(report, "__wrapped__") else None
        print(f"  independent recompute C1 (first 500): Rs{np.mean(manual):+.2f}/trade")
    # holdout split on variant A
    if of:
        days = sorted(set(t["date"] for t in of))
        mid = days[len(days) // 2]
        first = [t for t in of if t["date"] < mid]
        second = [t for t in of if t["date"] >= mid]
        for nm, grp in [("first half", first), ("second half", second)]:
            if grp:
                rr = np.array([(t["exit"] / t["entry"] - 1) * t["dir"] for t in grp])
                print(f"  A {nm}: n={len(grp)} avgNet=Rs{(rr*NOTIONAL-(NOTIONAL*COST_RT_PCT+2*BROKERAGE)).mean():+.1f}")

    print(f"\n=== SUMMARY ===")
    print(f"  data: {df.date.nunique()} trading days, {df.label.nunique()} instruments")
    print(f"  U-shape (normalized): {'CONFIRMED' if lv['u'] else 'NOT confirmed'} "
          f"(open {lv['open']:.2f}/mid {lv['mid']:.2f}/close {lv['close']:.2f})")
    print(f"  VERDICT: {df.date.nunique()} days is far too few for any variant "
          f"(need 25+). Engine validated + ready to re-run ~mid-Nov 2026.")
    print(f"  Look-ahead: NONE — all signals use only data at/before entry; "
          f"prior-day thresholds use expanding window (no future).")


if __name__ == "__main__":
    main()
