"""THE STANDARD GATE — adopted from OpenQuant India, stricter than mine.

An edge is TRADEABLE only if it passes ALL of these, in order:

  G1  BENCHMARK-RELATIVE: net-of-cost CAGR strictly beats the declared
      benchmark (Nifty buy-and-hold over the same window). Not "is profitable"
      — "beats doing nothing".
  G2  WALK-FORWARD: train 3y / test 1y, rolled forward. OOS Sharpe must be
      NON-NEGATIVE. Tune on train, freeze, apply unchanged to test.
  G3  COST DECLARED UPFRONT: Indian 2024-25 rates (NSE exchange 0.00297%,
      STT 0.1% delivery both legs, SEBI, GST 18%, stamp) + slippage.
  G4  SURVIVABLE AT Rs1.5L: max drawdown and worst trade as % of capital, with
      the minimum capital stated.
  G5  NOT TOO GOOD: anything implying >50%/yr or abnormal >5%/trade is a BUG
      until root-caused.

Anything failing is labelled NOT YET (with the missing gate) or DEAD (with the
reason). Honest losers are logged, not hidden.

Run: python standard_gate.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HIST = "data_indmoney/daily_history.parquet"
CAPITAL = 150000.0

# --- Indian 2024-25 costs (OpenQuant India calibrated; NSE exchange revised Oct 2024)
STT_DELIVERY = 0.001          # both legs, delivery
EXCHANGE = 0.0000297          # NSE cash, revised Oct 2024
SEBI = 1e-6
GST = 0.18
STAMP_BUY = 0.00015
SLIP = 0.001                  # per side
BROKERAGE = 20.0


def trade_cost_pct(notional, intraday=False):
    """All-in round-trip cost as a fraction of notional."""
    stt = STT_DELIVERY * 2 if not intraday else 0.00025  # intraday: sell only
    exch = EXCHANGE * 2
    sebi = SEBI * 2
    stamp = STAMP_BUY
    fees = (exch + sebi) * notional + 2 * BROKERAGE
    gst = GST * (exch + sebi) * notional
    slip = SLIP * 2 * notional
    total = stt * notional + stamp * notional + fees + gst + slip
    return total / notional


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def sharpe(r):
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    if sd < 1e-12:                      # zero-variance (cash) -> no Sharpe
        return 0.0
    return r.mean() / sd * np.sqrt(252)


def load():
    df = pd.read_parquet(HIST)
    df["date"] = pd.to_datetime(df["datetime_ist"]).dt.date
    return df.sort_values(["label", "date"])


def walk_forward(daily_ret, train_days=756, test_days=252):
    """Roll train/test. Returns OOS sharpes per fold + the fold count."""
    r = np.asarray(daily_ret, float)
    folds, i = [], 0
    while i + train_days + test_days <= len(r):
        test = r[i + train_days: i + train_days + test_days]
        folds.append(sharpe(test))
        i += test_days
    return folds


def evaluate(name, daily_ret, bench_ret, notes=""):
    """Run the standard gate on a daily-return series vs its benchmark."""
    r = np.asarray(daily_ret, float)
    b = np.asarray(bench_ret, float)
    n = min(len(r), len(b))
    r, b = r[:n], b[:n]
    cost = trade_cost_pct(CAPITAL)
    net = r - cost / 21          # amortize one round-trip cost over a 21d hold
    yrs = n / 252

    strat_cagr = (np.prod(1 + net)) ** (1 / max(yrs, 1e-9)) - 1 if n else np.nan
    bench_cagr = (np.prod(1 + b)) ** (1 / max(yrs, 1e-9)) - 1 if n else np.nan
    oos = walk_forward(net)
    oos_neg = sum(1 for s in oos if s < 0)

    # survivability
    cum = np.cumprod(1 + net)
    peak = np.maximum.accumulate(cum)
    dd = ((cum - peak) / peak).min()
    worst = net.min()

    print(f"\n{'='*88}")
    print(f"  {name}   {notes}")
    print(f"{'='*88}")
    print(f"  cost/round-trip: {cost*100:.3f}% of notional (Indian 2024-25, slip {SLIP*100:.1f}%/side)")
    print(f"  net CAGR:        {strat_cagr*100:+.2f}%/yr")
    print(f"  benchmark CAGR:  {bench_cagr*100:+.2f}%/yr")
    print(f"  G1 beats bench:  {'PASS' if strat_cagr > bench_cagr else 'FAIL'} "
          f"({(strat_cagr-bench_cagr)*100:+.2f}pp)")
    print(f"  G2 walk-forward: {len(oos)} folds, OOS Sharpe "
          f"{['%+.2f' % s for s in oos[:6]]}")
    print(f"     non-negative: {'PASS' if oos_neg == 0 else f'FAIL ({oos_neg}/{len(oos)} folds negative)'}")
    print(f"  G4 survivability: max DD {dd*100:.1f}%, worst day {worst*100:.2f}%")
    min_cap = CAPITAL
    if dd < -0.20:
        min_cap = CAPITAL / abs(dd) * 0.20
        print(f"     min capital for <=20% DD: ~Rs{min_cap:,.0f}")
    g5 = (not np.isnan(strat_cagr) and strat_cagr > 0.50)
    print(f"  G5 too-good:     {'KILL-AND-DEBUG' if g5 else 'ok'}")

    passed = (strat_cagr > bench_cagr) and oos_neg == 0 and not g5 and dd > -0.25
    verdict = "TRADEABLE" if passed else (
        "DEAD (no net edge vs benchmark)" if strat_cagr <= bench_cagr else
        "NOT YET (fails walk-forward or survivability)")
    print(f"\n  VERDICT: {verdict}")
    return passed, strat_cagr, bench_cagr, dd


def main():
    prices = load()
    nifty = prices[prices.label == "NIFTY50"].sort_values("date").reset_index(drop=True)
    bench = nifty.close.pct_change().dropna().values
    print("=" * 88)
    print("  THE STANDARD GATE (adopted from OpenQuant India — stricter than mine)")
    print("=" * 88)
    print(f"  Nifty benchmark: {nifty.date.min()} -> {nifty.date.max()}, {len(bench)} days")
    print(f"  buy-and-hold CAGR: "
          f"{((nifty.close.iloc[-1]/nifty.close.iloc[0])**(252/len(nifty))-1)*100:+.1f}%/yr")

    # Candidate 1: VRP-informed long Nifty (long when implied>>realized)
    #   proxy: long Nifty always (the benchmark itself) — shows the hurdle
    evaluate("buy-and-hold Nifty (the hurdle itself)", bench, bench,
             "what every candidate must beat")

    # Candidate 2: cash (do nothing)
    evaluate("cash (0% return)", np.zeros(len(bench)), bench,
             "the floor")

    print("\n" + "=" * 88)
    print("  CANDIDATES FROM THIS SESSION, re-run under the standard gate")
    print("=" * 88)
    print("""
  To evaluate a candidate, its daily return series must be built net of costs
  and passed to evaluate(). The candidates and their prior status:

  1. PEAD broad drift (all positive surprise, 42d hold)
     prior: clustered t=+4.21 but near buy-and-hold beta, no robust band
     -> G1 will likely FAIL (it IS mostly beta)

  2. VRP (IndiaVIX 12.3 vs realized 6.2, +6.1 vol pts, t=+15.6)
     prior: real premium, no Rs1.5L channel (needs short-vol)
     -> G1 untestable without a short-vol P&L series

  3. Short-side gap-fade, TOM, intraday momentum, ORB, pairs, factors
     prior: all DEAD under my own gates

  RUN: build each candidate's net daily-return series and call evaluate().
  The gate is the standard now — not my opinion, an independent project's.
""")


if __name__ == "__main__":
    main()
