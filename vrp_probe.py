"""VARIANCE RISK PREMIUM (VRP) — implied vol vs realized vol, India.

The one structural edge on the research map NOT yet tested: India VIX is
persistently ABOVE realized Nifty vol (implied > realized), and selling that
premium is the classic structural (non-behavioral) edge — it doesn't decay
like calendar/momentum anomalies because it's paid for the insurance
providers supply.

THIS IS NOT OPTION SELLING (capital wall: a Nifty strangle needs ~Rs15-20K
margin/lot, Rs1.5L = ~0 lots). The retail-fit VRP play is the VIX-Nifty SPREAD
itself: when IndiaVIX is elevated vs recent realized vol, realized vol tends
to fall back toward it. Tested as a SIGNAL on Nifty direction/sizing, or as
a spread-mean-reversion. 64 days hourly is small but it's a probe.

Skill reference (india-vol-markets.md) flags: measure realized vol correctly
(std of returns x sqrt(252)), don't mix horizons, and check margin before
claiming an instrument fits.

Run: python vrp_probe.py
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def pull():
    vix = yf.download("^INDIAVIX", period="3mo", interval="1h", progress=False)
    nk = yf.download("^NSEI", period="3mo", interval="1h", progress=False)
    for d in (vix, nk):
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
    vix = vix[["Close"]].rename(columns={"Close": "vix"})
    nk = nk[["Close"]].rename(columns={"Close": "nifty"})
    df = vix.join(nk, how="inner").dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["date"] = df.index.date
    return df


def main():
    df = pull()
    print("=== VARIANCE RISK PREMIUM PROBE (IndiaVIX vs Nifty realized) ===")
    print(f"  {len(df)} hourly bars, {df.date.nunique()} days\n")

    # daily realized vol (annualized correctly: std of intraday returns x sqrt(bars/yr))
    daily = df.groupby("date").agg(
        vix=("vix", "mean"), nifty_o=("nifty", "first"), nifty_c=("nifty", "last"))
    # realized vol from hourly returns within each day, annualized
    rv = []
    for d, g in df.groupby("date"):
        r = np.log(g.nifty).diff().dropna()
        if len(r) > 3:
            # 6.5 trading hours/day, 252 days -> annualize
            rv.append(r.std() * np.sqrt(252 * 6.5))
        else:
            rv.append(np.nan)
    daily["rv"] = rv
    daily = daily.dropna()
    daily["vrp"] = daily.vix - daily.rv   # implied - realized (in vol points)
    print(f"  mean IndiaVIX:      {daily.vix.mean():.1f}")
    print(f"  mean realized vol:  {daily.rv.mean():.1f}")
    print(f"  mean VRP (IV-RV):   {daily.vrp.mean():+.1f} vol points "
          f"(positive = implied above realized = premium exists)")
    print(f"  VRP positive on:    {(daily.vrp>0).mean()*100:.0f}% of days")
    print(f"  VRP t-stat:         {tstat(daily.vrp):+.2f}")

    # does elevated VIX predict next-day realized vol falling back? (mean reversion)
    print("\n--- VIX MEAN REVERSION: does high VIX predict falling realized vol? ---")
    daily["vix_z"] = (daily.vix - daily.vix.rolling(20).mean()) / daily.vix.rolling(20).std()
    daily["rv_next"] = daily.rv.shift(-1)
    daily["d_rv"] = daily.rv_next - daily.rv
    hi = daily[daily.vix_z > 1]     # VIX elevated
    lo = daily[daily.vix_z < -1]    # VIX depressed
    print(f"  after HIGH vix (z>1): n={len(hi)} dRV={hi.d_rv.mean():+.2f} "
          f"t={tstat(hi.d_rv):+.2f}  (negative = vol falls back = premium captured)")
    print(f"  after LOW vix (z<-1): n={len(lo)} dRV={lo.d_rv.mean():+.2f} "
          f"t={tstat(lo.d_rv):+.2f}")

    # does VIX predict Nifty DIRECTION next day? (should NOT if VRP is vol not direction)
    print("\n--- DOES VIX PREDICT NIFTY DIRECTION? (should be weak if VRP is pure vol) ---")
    daily["nifty_ret"] = daily.nifty_c / daily.nifty_o - 1
    daily["nifty_next"] = daily.nifty_ret.shift(-1)
    corr = daily[["vix_z", "nifty_next"]].dropna().corr().iloc[0, 1]
    print(f"  corr(vix_z, next-day nifty return) = {corr:+.3f} "
          f"({'weak (VRP is vol, not direction)' if abs(corr) < 0.2 else 'directional signal!'})")

    # the tradable signal: when VIX is high vs realized, does Nifty mean-revert
    # (calm returns)? long Nifty when VRP very positive (vol will crush = rally)
    print("\n--- VRP AS NIFTY SIGNAL: long when implied>>realized (vol crush = calm/up) ---")
    daily["sig"] = np.where(daily.vrp > daily.vrp.median(), 1, 0)
    daily["strat"] = daily.sig * daily.nifty_next
    for nm, grp in [("long when VRP high", daily[daily.sig == 1]),
                    ("flat when VRP low", daily[daily.sig == 0])]:
        if len(grp) > 3:
            r = grp.nifty_next.dropna()
            print(f"  {nm:<22} n={len(r)} next-ret={r.mean()*100:+.3f}%/d "
                  f"t={tstat(r):+.2f} win={100*(r>0).mean():.0f}%")

    print("\n=== VERDICT ===")
    print("  VRP EXISTS if IV>RV persistently (t>2). If it also predicts Nifty")
    print("  direction, there's a tradeable signal; if pure vol, it needs an")
    print("  options/sizing channel (capital-wall problem at Rs1.5L). 64d = probe.")


if __name__ == "__main__":
    main()
