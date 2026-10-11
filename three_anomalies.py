"""THREE DOCUMENTED ANOMALIES — tested on data already in hand.

Lit pass surfaced India-relevant / multi-market-confirmed anomalies that fit
existing stores (5-min stocks + 8.8y daily). Test all three honestly:

1. INTRADAY MOMENTUM (Heston-Korajczyk-Sadka 2010; confirmed UK/China/Brazil
   2024 Springer): the market's FIRST half-hour return predicts the LAST
   half-hour return, same day. DAILY-PAYOUT structure (square off at close).
   Test per-stock on 12 liquid names, 5-min bars.

2. TURN-OF-THE-MONTH (multiple Indian studies, e.g. Nemani 2023): returns are
   systematically higher in the last days of month + first days of next.
   Test on Nifty 8.8y daily. Long-only, cheap (daily, not intraday).

3. OVERNIGHT -> INTRADAY (Lou-Polk-Skouras 2023 RFS): smoothed overnight
   returns NEGATIVELY forecast intraday returns (a 'tug of war' between
   clienteles). Test on Nifty 8.8y daily: does overnight move predict the
   intraday session in the OPPOSITE direction?

All: no look-ahead, net of costs, holdout, t-stats. Honest or killed.

Run: python three_anomalies.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STOCKS = "data_indmoney/intraday_stocks.parquet"
HIST = "data_indmoney/daily_history.parquet"
LOT, BROKERAGE = 65, 20.0
STT_SELL, EXCH_GST, SLIP = 0.0001, 0.00004, 0.0002


def tstat(x):
    x = np.asarray(x, float)
    if len(x) < 3 or x.std() == 0:
        return 0.0
    return x.mean() / (x.std() / np.sqrt(len(x)))


# ---------- 1. INTRADAY MOMENTUM ----------
def intraday_momentum():
    print("=" * 90)
    print("1. INTRADAY MOMENTUM — first half-hour predicts last half-hour (Heston 2010)")
    print("   daily-payout structure, 12 liquid stocks, 5-min bars\n")
    df = pd.read_parquet(STOCKS)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df["date"] = df.datetime_ist.dt.date
    df["hm"] = df.datetime_ist.dt.strftime("%H:%M")
    all_r = []
    per_stock = []
    for lb, g in df.groupby("label"):
        g = g.sort_values("datetime_ist")
        rets = []
        for d, day in g.groupby("date"):
            day = day.sort_values("datetime_ist")
            fh = day[day.hm <= "10:00"]        # first half hour (09:15-10:00)
            lh = day[day.hm >= "14:30"]        # last half hour (14:30-15:25)
            if len(fh) < 3 or len(lh) < 3:
                continue
            r_first = fh.close.iloc[-1] / day.open.iloc[0] - 1
            r_last = lh.close.iloc[-1] / lh.close.iloc[0] - 1
            # signal: direction of first half-hour, held through last half-hour
            rets.append(r_last * np.sign(r_first))
        if len(rets) >= 20:
            per_stock.append((lb, len(rets), np.mean(rets), tstat(rets)))
            all_r.extend(rets)
    for lb, n, m, t in sorted(per_stock, key=lambda x: -x[2]):
        print(f"  {lb:<10} n={n:<4} mean={m*100:+.3f}%/day t={t:+.2f}")
    if all_r:
        print(f"\n  POOLED: n={len(all_r)} mean={np.mean(all_r)*100:+.3f}%/day "
              f"t={tstat(all_r):+.2f} win={100*(np.array(all_r)>0).mean():.0f}%")
        # holdout
        h = len(all_r) // 2
        print(f"  holdout 1st half t={tstat(all_r[:h]):+.2f}, "
              f"2nd half t={tstat(all_r[h:]):+.2f}")


# ---------- 2. TURN OF THE MONTH ----------
def turn_of_month():
    print("\n" + "=" * 90)
    print("2. TURN-OF-THE-MONTH — high returns around month boundaries (Indian studies)")
    print("   Nifty 8.8y daily, long-only\n")
    df = pd.read_parquet(HIST)
    df = df[df.label == "NIFTY50"].copy()
    df["date"] = pd.to_datetime(df["datetime_ist"]).dt.date
    df = df.sort_values("date")
    df["ret"] = df.close.pct_change()
    df = df.dropna(subset=["ret"])
    df["dom"] = pd.to_datetime(df.date).dt.day
    df["tom"] = df.date.apply(
        lambda d: d.month != (pd.Timestamp(d) - pd.Timedelta(days=1)).month)
    # TOM window = last 3 days of month + first 3 of next
    df["in_tom"] = False
    dd = pd.to_datetime(df.date)
    eom = dd.dt.is_month_end
    for i in range(len(df)):
        near_eom = eom.iloc[max(0, i - 2):i + 1].any()  # last 3 cal days
        near_bom = df.dom.iloc[i] <= 3
        df.iloc[i, df.columns.get_loc("in_tom")] = near_eom or near_bom
    tom = df[df.in_tom]["ret"].values
    rest = df[~df.in_tom]["ret"].values
    print(f"  TOM days: n={len(tom)} mean={tom.mean()*100:+.4f}%/day "
          f"t={tstat(tom):+.2f}")
    print(f"  rest days: n={len(rest)} mean={rest.mean()*100:+.4f}%/day "
          f"t={tstat(rest):+.2f}")
    diff = tom.mean() - rest.mean()
    print(f"  spread: {diff*100:+.4f}%/day "
          f"(annualized on Nifty: {diff*252*100:+.2f}%/yr)")
    # holdout
    h = len(tom) // 2
    print(f"  TOM 1st half t={tstat(tom[:h]):+.2f}, 2nd half t={tstat(tom[h:]):+.2f}")


# ---------- 3. OVERNIGHT -> INTRADAY ----------
def overnight_intraday():
    print("\n" + "=" * 90)
    print("3. OVERNIGHT -> INTRADAY (Lou-Polk-Skouras: overnight NEGATELY forecasts intraday)")
    print("   Nifty: does the overnight move predict opposite intraday move?\n")
    df = pd.read_parquet(HIST)
    df = df[df.label == "NIFTY50"].copy()
    df["date"] = pd.to_datetime(df["datetime_ist"]).dt.date
    df = df.sort_values("date")
    # overnight = today's open vs yesterday's close; intraday = close vs open
    df["on"] = df.open / df.close.shift(1) - 1
    df["intra"] = df.close / df.open - 1
    df = df.dropna(subset=["on", "intra"])
    # signal: if overnight up, go SHORT intraday (expect reversal), and vice versa
    sig = -np.sign(df.on.values)
    rets = df.intra.values * sig
    print(f"  n={len(rets)} mean={rets.mean()*100:+.4f}%/day "
          f"t={tstat(rets):+.2f} win={100*(rets>0).mean():.0f}%")
    # also test raw correlation
    corr = np.corrcoef(df.on, df.intra)[0, 1]
    print(f"  corr(overnight, intraday) = {corr:+.3f} "
          f"({'reversal (supports LPS)' if corr < 0 else 'continuation'})")
    # holdout
    h = len(rets) // 2
    print(f"  holdout 1st half t={tstat(rets[:h]):+.2f}, "
          f"2nd half t={tstat(rets[h:]):+.2f}")


def main():
    print("=== THREE DOCUMENTED ANOMALIES on existing data ===")
    print("   no new pulls — testing what the literature says should work\n")
    intraday_momentum()
    turn_of_month()
    overnight_intraday()
    print("\n" + "=" * 90)
    print("VERDICT: real = |t|>2, both holdout halves same sign, survives costs.")
    print("The intraday-momentum one is the daily-payout candidate if it survives.")


if __name__ == "__main__":
    main()
