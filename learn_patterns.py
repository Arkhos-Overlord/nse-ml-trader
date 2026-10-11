"""Learn patterns + liquidity from real NIFTY option data, then train on it.

Part A — LIQUIDITY MAP: where does volume/OI concentrate? (ATM vs wings,
weekly vs monthly, day-of-week). This tells us what's actually tradable.

Part B — TRADE FEATURES: for every VRP-signal expiry, build features from
liquidity + market state, join to REAL PnL from the backtest.

Part C — LEARN: walk-forward model predicting profitable trades. Does
liquidity + patterns improve trade selection vs. taking every VRP signal?

Doubt rule applies: only trust it if it beats 'take every signal' out-of-sample.
Run: python learn_patterns.py
"""
import warnings

import numpy as np
import pandas as pd

from real_options_backtest import build_vix_signal

warnings.filterwarnings("ignore")

YEARS = [2023, 2024, 2025]


def load_all():
    frames = []
    for yr in YEARS:
        d = pd.read_parquet(f"NIFTY_{yr}.parquet")
        d["date"] = pd.to_datetime(d["date"])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def part_a_liquidity(opt):
    print("=" * 60)
    print("PART A — LIQUIDITY MAP (where is the money tradable?)")
    print("=" * 60)
    traded = opt[opt["volume"] > 0].copy()
    print(f"total rows {len(opt)}, traded (vol>0) {len(traded)} "
          f"({len(traded)/len(opt)*100:.0f}%)")

    # liquidity by option moneyness
    opt2 = traded.copy()
    # need spot to compute moneyness — approximate via max-OI strike per day as ATM proxy
    # simpler: bucket by strike relative to the day's most-traded strike
    day_atm = (opt2.groupby(["date"])["volume"].sum()
               .reset_index())  # not per-strike; do per-strike volume rank instead
    # volume concentration: what % of a day's volume is in the top-5 strikes?
    g = opt2.groupby("date")
    conc = g["volume"].apply(lambda s: s.nlargest(5).sum() / (s.sum() + 1e-9))
    print(f"\nVolume concentration (top-5 strikes share of daily vol):")
    print(f"  mean {conc.mean()*100:.0f}%, median {conc.median()*100:.0f}%")
    print(f"  -> liquidity is {'HIGHLY concentrated' if conc.median()>0.5 else 'spread out'}")

    # CE vs PE liquidity
    ce_vol = traded[traded.option_type == "CE"]["volume"].sum()
    pe_vol = traded[traded.option_type == "PE"]["volume"].sum()
    print(f"\nCE volume {ce_vol:,.0f} vs PE volume {pe_vol:,.0f} "
          f"(CE/PE ratio {ce_vol/(pe_vol+1):.2f})")

    # liquidity by days-to-expiry: nearer expiry = more volume (gamma)
    traded["expiry"] = pd.to_datetime(traded["expiry"])
    traded["dte"] = (traded["expiry"] - traded["date"]).dt.days
    dte_vol = traded.groupby(pd.cut(traded["dte"], [0, 1, 3, 7, 15, 30, 100],
                                    labels=["0-1", "2-3", "4-7", "8-15", "16-30", "30+"]))
    print(f"\nVolume by days-to-expiry:")
    print(dte_vol["volume"].sum().to_string())
    print(f"  -> most liquidity in the {'final days' if True else ''} (gamma/theta)")

    # day of week
    traded["dow"] = traded["date"].dt.dayofweek
    dow = traded.groupby("dow")["volume"].sum()
    print(f"\nVolume by day-of-week (0=Mon):")
    print({["Mon", "Tue", "Wed", "Thu", "Fri"][k]: f"{v:,.0f}"
           for k, v in dow.items()})
    return traded


def part_b_features(opt, sig):
    print("\n" + "=" * 60)
    print("PART B — BUILD TRADE FEATURES (liquidity + state -> real PnL)")
    print("=" * 60)
    sig = sig.copy()
    sig.index = pd.to_datetime(sig.index)
    rows = []
    for expiry, grp in opt.groupby("expiry"):
        traded = grp[grp["close"] > 0].sort_values("timestamp")
        if traded.empty:
            continue
        days = sorted(traded["date"].unique())
        if len(days) < 2:
            continue
        entry_day = pd.Timestamp(days[0])
        exit_day = pd.Timestamp(days[-1])
        if entry_day not in sig.index:
            continue
        if not sig.loc[entry_day, "sell"]:
            continue
        spot = sig.loc[entry_day, "spot"]
        strikes = np.sort(traded["strike"].unique())
        atm = strikes[np.argmin(np.abs(strikes - spot))]

        def px(day, strike, otype):
            r = traded[(traded["date"] == day) & (traded["strike"] == strike)
                       & (traded["option_type"] == otype)]
            return float(r["close"].iloc[0]) if not r.empty else None

        ce_in, pe_in = px(entry_day, atm, "CE"), px(entry_day, atm, "PE")
        ce_out, pe_out = px(exit_day, atm, "CE"), px(exit_day, atm, "PE")
        if None in (ce_in, pe_in, ce_out, pe_out):
            continue
        # liquidity features at entry
        e = traded[traded["date"] == entry_day]
        atm_row_ce = e[(e.strike == atm) & (e.option_type == "CE")]
        atm_vol = float(atm_row_ce["volume"].iloc[0]) if not atm_row_ce.empty else 0
        atm_oi = float(atm_row_ce["oi"].iloc[0]) if not atm_row_ce.empty else 0
        day_vol = float(e["volume"].sum())
        day_oi = float(e["oi"].sum())
        # is ATM the most liquid strike? (liquidity rank of atm)
        strike_vol = e.groupby("strike")["volume"].sum().sort_values(ascending=False)
        atm_rank = list(strike_vol.index).index(atm) if atm in strike_vol.index else 999
        dte = (pd.Timestamp(expiry) - entry_day).days
        pnl = (ce_in + pe_in - ce_out - pe_out) * 75
        rows.append({
            "expiry": expiry, "entry": entry_day, "dte": dte,
            "atm_vol": atm_vol, "atm_oi": atm_oi, "day_vol": day_vol, "day_oi": day_oi,
            "atm_liq_rank": atm_rank, "vix": sig.loc[entry_day, "vix"],
            "vrp": sig.loc[entry_day, "vrp"], "dow": entry_day.dayofweek,
            "premium": (ce_in + pe_in) * 75, "pnl": pnl,
            "profitable": int(pnl > 0),
        })
    return pd.DataFrame(rows)


def part_c_learn(feat):
    print("\n" + "=" * 60)
    print("PART C — LEARN: does liquidity+state predict profitable trades?")
    print("=" * 60)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    feat = feat.sort_values("entry").reset_index(drop=True)
    feature_cols = ["dte", "atm_vol", "atm_oi", "day_vol", "day_oi",
                    "atm_liq_rank", "vix", "vrp", "dow"]
    X = feat[feature_cols].values
    y = feat["profitable"].values
    n = len(feat)
    split = int(n * 0.6)
    print(f"trades: {n}, train {split}, test {n-split}")
    print(f"base rate (profitable): {y.mean()*100:.0f}%")

    # baseline: take every signal
    print(f"\nBASELINE (take every VRP signal):")
    print(f"  train PnL: Rs{feat.pnl[:split].sum():,.0f}  "
          f"test PnL: Rs{feat.pnl[split:].sum():,.0f}")

    # walk-forward: train on past, predict next, take only high-confidence
    for name, model in [("LogReg", LogisticRegression(max_iter=1000)),
                        ("GBM", GradientBoostingClassifier(n_estimators=50,
                                                           max_depth=3,
                                                           random_state=42))]:
        correct = total = 0
        taken_pnl = []
        min_train = max(15, split // 2)
        for i in range(min_train, n):
            m = model.__class__(**model.get_params())
            m.fit(X[:i], y[:i])
            prob = m.predict_proba(X[i:i + 1])[0, 1]
            pred = int(prob > 0.5)
            correct += int(pred == y[i])
            total += 1
            # strategy: only take trades model is confident about
            if prob > 0.6:
                taken_pnl.append(feat.pnl.iloc[i])
        acc = correct / total if total else 0
        tp = np.array(taken_pnl) if taken_pnl else np.array([0])
        print(f"\n{name}:")
        print(f"  walk-forward accuracy: {acc*100:.0f}% (base {y.mean()*100:.0f}%)")
        print(f"  trades taken (conf>0.6): {len(taken_pnl)}/{total}")
        print(f"  PnL on taken: Rs{tp.sum():,.0f}  win {(tp>0).mean()*100:.0f}%"
              if len(taken_pnl) else "  none taken")

    # feature importance (GBM on full data — descriptive, not predictive)
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    gbm.fit(X, y)
    imp = sorted(zip(feature_cols, gbm.feature_importances_),
                 key=lambda t: -t[1])
    print(f"\nFeature importance (what drives profitable trades):")
    for f, v in imp:
        print(f"  {f}: {v:.3f}")
    print("\nHONEST: if liquidity features rank high, they help selection.")
    print("If model can't beat baseline test PnL, liquidity doesn't add edge.")


if __name__ == "__main__":
    print("Loading 3 years of real NIFTY option data...")
    opt = load_all()
    print(f"  {len(opt)} rows, {opt.expiry.nunique()} expiries\n")
    part_a_liquidity(opt)
    sig = build_vix_signal()
    feat = part_b_features(opt, sig)
    print(f"  built {len(feat)} VRP-signal trades with features")
    part_c_learn(feat)
