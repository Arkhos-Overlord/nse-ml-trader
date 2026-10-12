"""BACKTEST / FALSIFICATION of the condor methodology on historical data.

Answers the one question that matters and is currently unproven: does this
approach make money, or is it a regime artifact? The default answer is NO.

Tests, in order of rejection power (per quant-strategy-falsification):
  1. WALK-FORWARD  — train on first 60%, freeze, test on last 40%. Wins BOTH
     windows or it's overfit.
  2. PER-QUARTER   — a real edge is positive most quarters; 2/4 = regime luck.
  3. COST DRAG     — apply realistic stock-option costs; a paper edge that
     needs zero slippage is fiction.
  4. REWARD-RISK SELECTION — does picking the best score beat taking every
     condor? (This is the dynamic-scanner methodology test.)

NOTE: historical bars are Nifty INDEX only (no stock underlyings), so this
tests the STRUCTURE + scoring, not the exact stock names the scanner picks.
Data: condor_bars_2025.parquet (249 days, 61 expiries).
"""
import argparse

import numpy as np
import pandas as pd

# --- strategy params (must match the live engine) ---
SHORT_WING, LONG_WING = 2, 5     # index wings
LOT = 65
COST_PER_LOT = 25.0              # Rs per lot round-trip (brokerage+STT+slippage approx)


def build_condors(df, short_w=SHORT_WING, long_w=LONG_WING):
    """From chain bars, build one condor per (date,expiry) at the OI-max ATM."""
    out = []
    for (d, e), g in df.groupby(["date", "expiry"]):
        if g.empty or g.strike.nunique() < long_w + 2:
            continue
        atm = g.groupby("strike").oi.sum().idxmax() if "oi" in g.columns else g.strike.median()
        strikes = sorted(g.strike.unique())
        try:
            ai = strikes.index(atm)
        except ValueError:
            continue

        def leg(off, typ):
            k = ai + off
            if not (0 <= k < len(strikes)):
                return None
            r = g[(g.strike == strikes[k]) & (g.type == typ)]
            if r.empty:
                return None
            r = r.iloc[0]
            return (strikes[k], r.first_close, r.last_close)

        L = {"cs": leg(short_w, "CE"), "cl": leg(long_w, "CE"),
             "ps": leg(-short_w, "PE"), "pl": leg(-long_w, "PE")}
        if any(v is None or pd.isna(v[1]) or pd.isna(v[2]) for v in L.values()):
            continue
        # entry credit (sell short legs, buy long legs) at first_close
        credit = (L["cs"][1] - L["cl"][1]) + (L["ps"][1] - L["pl"][1])
        # cost to close at last_close
        closing = (L["cs"][2] - L["cl"][2]) + (L["ps"][2] - L["pl"][2])
        wing = L["cl"][0] - L["cs"][0]
        pnl_pts = credit - closing           # >0 = we kept premium
        maxloss_pts = max(wing - credit, 0)
        if maxloss_pts <= 0:
            continue
        out.append({"date": d, "expiry": e,
                    "credit_pts": credit, "pnl_pts": pnl_pts,
                    "maxloss_pts": maxloss_pts, "wing": wing,
                    "score": credit / maxloss_pts,
                    "credit_rs": credit * LOT, "pnl_rs": pnl_pts * LOT,
                    "maxloss_rs": maxloss_pts * LOT})
    return pd.DataFrame(out)


def stats(pnl, label):
    pnl = pnl.dropna()
    if len(pnl) == 0:
        return {"label": label, "n": 0}
    net = pnl - COST_PER_LOT
    t = net.mean() / (net.std(ddof=1) / np.sqrt(len(net))) if len(net) > 1 and net.std(ddof=1) > 0 else 0
    return {"label": label, "n": len(pnl),
            "gross_per_trade": round(pnl.mean(), 1), "net_per_trade": round(net.mean(), 1),
            "win_pct": round((pnl > 0).mean() * 100, 1), "t": round(t, 2),
            "worst_rs": round(pnl.min(), 0), "total_rs": round(pnl.sum(), 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="condor_bars_2025.parquet")
    ap.add_argument("--capital", type=float, default=5000.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.file)
    tr = build_condors(df)
    if tr.empty:
        print("NO CONDORS BUILT — data too shallow. Verdict: NOT TESTABLE.")
        return
    print(f"built {len(tr)} condors over {tr.date.nunique()} days "
          f"({tr.date.min()} -> {tr.date.max()})\n")

    # --- 1. walk-forward: train 60%, test 40% on score selection ---
    tr = tr.sort_values("date").reset_index(drop=True)
    cut = int(len(tr) * 0.6)
    train, test = tr.iloc[:cut], tr.iloc[cut:]
    # "take every condor" baseline vs "take only high-score" selection
    for name, sub in [("TRAIN (first 60%)", train), ("TEST (last 40%)", test)]:
        s_all = stats(sub.pnl_rs, f"{name} | take ALL")
        hi = sub[sub.score >= sub.score.median()]
        s_hi = stats(hi.pnl_rs, f"{name} | take HIGH-score half")
        for s in (s_all, s_hi):
            if s["n"]:
                print(f"  {s['label']:34} n={s['n']:3}  net/trade Rs{s['net_per_trade']:>7}  "
                      f"win {s['win_pct']:5.1f}%  t={s['t']:>6}  worst Rs{s['worst_rs']:>7}")
        print()

    # --- 2. per-quarter persistence (net of cost) ---
    print("PER-QUARTER (net of Rs%.0f/lot cost):" % COST_PER_LOT)
    tr["q"] = pd.to_datetime(tr.date).dt.quarter
    qpos = 0
    for q, g in tr.groupby("q"):
        net = (g.pnl_rs - COST_PER_LOT).sum()
        qpos += net > 0
        print(f"  Q{q}: {len(g):3} condors, net Rs{net:>8.0f}  {'+' if net>0 else '-'}")
    print(f"  -> {qpos}/{tr.q.nunique()} quarters positive "
          f"({'STRUCTURAL' if qpos >= tr.q.nunique()-1 else 'REGIME-LUCKY'})\n")

    # --- 3. survival at this capital (compounding, one at a time) ---
    print(f"SURVIVAL at Rs{args.capital:.0f} (net, one position at a time):")
    eq = args.capital
    peak, maxdd = eq, 0.0
    for p in (tr.pnl_rs - COST_PER_LOT):
        eq += p
        peak = max(peak, eq)
        maxdd = max(maxdd, (peak - eq) / peak if peak else 0)
    ret = (eq / args.capital - 1) * 100
    print(f"  end equity Rs{eq:,.0f} ({ret:+.1f}%) | max drawdown {maxdd*100:.1f}%")
    if eq < args.capital:
        print("  -> LOSES MONEY. The edge does not survive costs at this size.\n")
    else:
        print("  -> survives costs on this sample.\n")

    # --- verdict ---
    test_net = (test.pnl_rs - COST_PER_LOT).mean() if len(test) else 0
    print("=" * 60)
    if test_net <= 0 or qpos < tr.q.nunique() - 1 or eq < args.capital:
        print("VERDICT: NO PROVEN EDGE. The methodology does not clear the gates")
        print("(walk-forward net, quarter persistence, or cost survival) on this")
        print("sample. Default answer stands: unproven. Keep PAPER, do not arm live.")
    else:
        print("VERDICT: PASSES this sample — but 1 regime, index-only. Still a")
        print("hypothesis until a second regime + live fills confirm it.")
    print("=" * 60)


if __name__ == "__main__":
    main()
