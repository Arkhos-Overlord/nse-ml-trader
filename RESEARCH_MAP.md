# RESEARCH MAP — What Actually Works in Indian Markets (sourced, not guessed)

Built 2026-10-11 from academic literature + practitioner sources. This is the
**map** that should have come BEFORE strategy testing. It explains every result
I got, and points at where edge genuinely exists.

---

## THE CENTRAL FINDING — why my tests found "efficient" and traders find edge

**The literature does NOT say Indian equities are efficient. It says edge exists
but is CONDITIONAL — and every condition I failed to test.**

Three sourced papers, one coherent picture:

### 1. Size effect is the STRONGEST anomaly in India (Sehgal, Subramaniam &
   Porteu de La Morandiere, 2012, Int'l J. Economics & Finance v4n4p121)
   - Tested size, value, momentum, liquidity, accruals, profitability, net issues
   - **"Size effect is the strongest with a difference of 4.4% PER MONTH between
     small and big stock returns"** — that is ~53%/year, small minus big
   - **"Size and short term momentum are persistent"** — survive CAPM AND Fama-French
   - Value, accruals, profitability ARE explained by Fama-French (no free lunch)
   - Liquidity anomaly only shows up with a liquidity-augmented model

   → **I tested mid/large-caps almost exclusively. The biggest documented anomaly
     in India is SMALLCAPS, which I never traded.** My "market is efficient"
     conclusion was drawn from the wrong universe.

### 2. Edge lives in ILLIQUID stocks, and it's REVERSAL there (Chuia,
   Ranganathan, Rohit & Veeraraghavan, 2023, Pacific-Basin Finance Journal v82,
   102193) — 3,956 stocks, 2000-2021
   - **"Short-term reversals are more profound among the most illiquid stocks"**
   - **"We find evidence of short-term momentum among the most LIQUID stocks"**
   - Momentum is amplified by liquidity: momentum returns "more significant for the
     most liquid portfolio and exhibit persistence for the next 12 months"
   - Both effects hold after controlling for market, size, value, macro factors
   - Importantly: **the reversal in illiquid stocks "fail[s] to find evidence of
     persistence"** — i.e. it's fragile, regime-dependent

   → **THE RULE: momentum works in LIQUID names, REVERSAL works in ILLIQUID names.
     I ran one strategy type across a mixed bag. That's why reversal showed nothing
     on liquid large-caps — the literature says reversal belongs in illiquid stocks.**

### 3. Expiry-day microstructure dislocation is REAL (arXiv:2507.04859v2, 22yr)
   - Documents **"significant volume spikes and systematic price dips just before
     expiry, often followed by sharp rebounds driven by institutional arbitrage
     and liquidity constraints in the cash market"**
   - Measured timing advantage of expiry-day vs first-day entry over 2003-2024
   - Mechanism named explicitly: **institutional hedging, month-end fund
     rebalancing, systematic liquidity flows creating temporary price pressure**

   → **This is a MICROSTRUCTURE edge — exactly the class I never tested on Indian
     data. It requires intraday data on expiry days. My 5-day sample had ~1 expiry.**

---

## WHY PEOPLE SUCCEED AND I "FAILED" — the honest diagnosis

| What I did | What the literature says works |
|---|---|
| Tested momentum on **large/mid-caps** | Momentum works in **liquid** names ✓ (partial match) |
| Tested reversal on **large/mid-caps** | Reversal works in **ILLIQUID** names ✗ **WRONG UNIVERSE** |
| Never traded **smallcaps** | **Size effect = strongest anomaly in India (4.4%/mo)** ✗ **MISSED ENTIRELY** |
| Tested daily only | **Expiry-day microstructure** needs intraday event data ✗ |
| One strategy across all stocks | **Conditional strategies** — different edge per liquidity bucket ✗ |
| 1 year of data | Literature uses **10-22 years**, 3,956 stocks ✗ **sample too small** |

**I was not failing because the market is efficient. I was failing because I was
testing the wrong universe with the wrong conditioning on too little data.** The
market being efficient at DAILY resolution in LARGE-CAPS is exactly what the
literature predicts — and I wrongly generalized it to "Indian markets."

---

## WHAT THE MAP SAYS TO TEST (ranked by documented strength)

1. **SIZE (smallcap) effect — 4.4%/month documented, strongest in India**
   - Long smallcaps / short largecaps, or simply a smallcap-tilted portfolio
   - Needs: smallcap universe with market-cap data. NOT in my current dataset.
   - Capital fit: excellent — this is a slow, low-turnover, LOW-drawdown strategy
   - **This is the single most promising untested avenue.**

2. **LIQUIDITY-CONDITIONAL REVERSAL — short-term reversal in illiquid stocks**
   - Buy 5-day losers among illiquid names, hold days
   - Needs: turnover/volume data per stock (I have volume!)
   - Caveat from the paper itself: reversal "lacks persistence" → must stress-test
   - Capital fit: moderate (illiquid = slippage risk, cap position size)

3. **EXPIRY-DAY MICROSTRUCTURE — pre-expiry dip, post-expiry rebound**
   - Buy Nifty/index in the pre-expiry window, exit after expiry
   - Needs: index intraday data on expiry days (accumulating)
   - 22 years of documented evidence = most robust of the three

4. **LIQUID-NAME MOMENTUM with 12-month persistence**
   - Already partially tested; literature says hold LONGER (12mo) than my 5-20d

---

## PRACTITIONER CORROBORATION (r/NSEAlgoTrading + r/algotrading)

- **u/wizzer-algotrader: "a stack of systematic strategies running on just ₹25k"**
  → many small edges combined, not one big one. Confirms portfolio architecture.
- **u/Yocurt (production, ES/NQ): minimum bar = >80% profitable years, PF≥1.07,
  ≥2,500 trades, maxDD ≤ avg yearly profit.** My tests were mostly below this bar
  on trade count and history length.
- **u/Thundr3: "market and volatility regime filters" = #1 breakthrough.**
  → matches paper #2's "lacks persistence" caveat.
- **u/Helpful-Medicine-862: short strangle on expiry days, paper-tested 2 weeks,
  "green 5 of 7 expiries."** → independent evidence for the expiry-day edge (#3).

**Note on sources: the Indian sub shows NO audited track records — mostly tool
launches and subscription marketing. Treat those claims as hypotheses, which is
why #1-#4 above are all drawn from peer-reviewed papers with stated samples.**

---

## THE HONEST CONCLUSION

**I was wrong to conclude "Indian markets are efficient."** The correct statement is:

> "Daily-resolution price momentum and reversal in LARGE-CAP Indian equities show
> no exploitable edge after costs. That is consistent with the literature. The
> documented Indian anomalies live in SMALLCAPS (size, 4.4%/mo), in
> LIQUIDITY-CONDITIONED strategies (momentum in liquid, reversal in illiquid),
> and in EXPIRY-DAY MICROSTRUCTURE — none of which I had the data or universe to
> test properly."

**Next action (highest documented value, best capital fit): build the smallcap
universe with market-cap data and test the size effect.** This is the strongest
documented anomaly in the Indian market and I never touched it.
