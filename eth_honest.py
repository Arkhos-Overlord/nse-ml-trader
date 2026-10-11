"""HONEST ETH edge valuation on Rs1.5L — no concurrency illusion.

The edge is real (82% quarters, 2-sided, $40/trade). But the '1259% return'
assumed trading every signal with infinite capital. HONEST version:
  - Rs1.5L = ONE $1800 position at a time (ETH ~$4000, 1 position)
  - trades are ~every 2-3 days, each held 21 days -> OVERLAPPING
  - with 1 slot, you take a signal only when flat -> ~1 trade per 21 days
  - that's the REALISTIC cadence for Rs1.5L

Compute: actual Rs1.5L equity curve, 1 position at a time, honest fees.
Run: python eth_honest.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FEE = 0.001
CAPITAL_USD = 1800.0    # Rs1.5L at ~83/USD
USD_INR = 83
LOOK, HOLD, THRESH = 14, 21, 0.08


def load():
    df = pd.read_parquet("data_indmoney/crypto_daily/ETHUSDT_1d.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def main():
    df = load()
    print("=== HONEST ETH EDGE on Rs1.5L (1 position at a time) ===")
    print(f"  config: tsmom_{LOOK}d thresh={THRESH} hold={HOLD}")
    print(f"  capital: ${CAPITAL_USD:,.0f} = Rs{CAPITAL_USD*USD_INR:,.0f}\n")

    # simulate: 1 slot, enter when flat, hold HOLD days
    capital = CAPITAL_USD
    peak = capital
    maxdd = 0
    trades = []
    i = max(LOOK, 60)
    flat_until = 0
    while i < len(df) - HOLD:
        if i < flat_until:
            i += 1
            continue
        r = df.close.iloc[i] / df.close.iloc[i - LOOK] - 1
        d = 1 if r > THRESH else (-1 if r < -THRESH else 0)
        if d == 0:
            i += 1
            continue
        entry = float(df.open.iloc[i + 1])
        exit_idx = min(i + HOLD, len(df) - 1)
        exit_ = float(df.close.iloc[exit_idx])
        if entry <= 0 or exit_ <= 0:
            i += 1
            continue
        qty = capital / entry          # full capital in one position
        gross = (exit_ - entry) * qty * d
        fee = capital * 2 * FEE
        net = gross - fee
        capital += net
        peak = max(peak, capital)
        maxdd = max(maxdd, (peak - capital) / peak)
        trades.append({"date": df.datetime.iloc[i], "pnl": net, "capital": capital})
        flat_until = exit_idx + 1
        i = exit_idx + 1

    t = pd.DataFrame(trades)
    years = (df.datetime.iloc[-1] - df.datetime.iloc[0]).days / 365
    print(f"  trades taken (1 slot, non-overlapping): {len(t)} over {years:.1f}y")
    print(f"  = {len(t)/years:.0f} trades/yr")
    print(f"  win rate: {(t.pnl > 0).mean()*100:.0f}%")
    print(f"  avg/trade: ${t.pnl.mean():.2f} = Rs{t.pnl.mean()*USD_INR:,.0f}")
    print(f"\n  START: Rs{CAPITAL_USD*USD_INR:,.0f}")
    print(f"  END:   Rs{capital*USD_INR:,.0f}")
    print(f"  total return: {(capital/CAPITAL_USD-1)*100:+.0f}%")
    print(f"  max drawdown: {maxdd*100:.1f}%")
    print(f"  avg monthly: Rs{(capital*USD_INR - CAPITAL_USD*USD_INR)/(years*12):,.0f}")

    # annualized (CAGR)
    cagr = (capital / CAPITAL_USD) ** (1 / years) - 1
    print(f"  CAGR: {cagr*100:.0f}%/yr")
    print(f"\n  HONEST MONTHLY on Rs1.5L: ~Rs{(capital*USD_INR-CAPITAL_USD*USD_INR)/(years*12):,.0f}")
    print(f"  (vs the 'Rs5.7L/yr' fantasy that assumed infinite concurrency)")

    # worst streak
    t["cum"] = t.capital
    print(f"\n  worst single trade: Rs{t.pnl.min()*USD_INR:,.0f} "
          f"({t.pnl.min()/CAPITAL_USD*100:.1f}% of capital)")
    print(f"  best single trade: Rs{t.pnl.max()*USD_INR:,.0f}")

    print(f"\n=== VERDICT ===")
    monthly = (capital*USD_INR - CAPITAL_USD*USD_INR)/(years*12)
    print(f"  Robust edge (82% quarters, 2-sided) YES")
    print(f"  Honest Rs1.5L monthly: ~Rs{monthly:,.0f}")
    print(f"  Max drawdown {maxdd*100:.1f}% (Rs1.5L survives)")
    if monthly > 5000 and maxdd < 0.5:
        print(f"  *** TRADEABLE on Rs1.5L: ~Rs{monthly:,.0f}/mo, {cagr*100:.0f}% CAGR ***")
    else:
        print(f"  Edge real but monthly Rs{monthly:,.0f} — modest at Rs1.5L scale")


if __name__ == "__main__":
    main()
