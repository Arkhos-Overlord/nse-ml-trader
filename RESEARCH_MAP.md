# THE RESEARCH MAP — What Actually Works in Indian Markets

**Version 2 — comprehensive, multi-source, quantified.** Built 2026-10-11.
Supersedes RESEARCH_MAP_v1.md (which overstated size + expiry — corrected below).

Method: 16 systematic queries → 96 sources → pulled full text of the highest-value
papers → resolved contradictions → flagged decay risk per anomaly.

---

## THE CORE INSIGHT (this is the map's spine)

**Every documented Indian anomaly has a LICENSING CONDITION.** It is not
"this market has edge X." It is "edge X works only for agents who clear condition
Y." Almost every failure I had this session was a licensing failure, not a
market-efficiency failure.

| Anomaly | Works for | Fails for |
|---|---|---|
| Size | Long-horizon, low-turnover investors | High-turnover traders (spread eats it) |
| Liquidity momentum | Liquid names, 6-12mo holds | Short horizons, illiquid names |
| Illiquidity reversal | Illiquid names, patient capital | Anyone after it's crowded |
| Expiry structure | **Volatility traders (gamma/theta)** | Directional spot traders |
| Calendar | Tiny edges, low-freq | Anyone after costs |

**The licensing condition that killed almost every strategy I tested: TURNOVER.**
The size paper's own words: *"transaction costs completely erased the return
differential."* My strategies were mostly high-turnover. They were structurally
disqualified before the first backtest.

---

## PART 1 — THE SIZE EFFECT, CORRECTED (this fixes a real error in map v1)

### The 1991-99 claim: 70%+/yr — BUT read the next line
NSE Research Paper 13, "Efficiency of the Market for Small Stocks" (2001):
- Raw small-minus-large return differential **>70% annualized** (1991-99)
- **BUT: "transaction costs completely erased the return differential"**
- Small-stock bid-ask spread **~3%** vs **<1%** for large (Roll's method)
- Size is partly a risk proxy (Fama-French), partly unexplained excess
- Mechanism: institutions won't hold small caps (liquidity + research cost)

**The 2012 figure I quoted earlier (4.4%/month) is the GROSS spread.** The same
literature shows it is not net tradeable at high frequency. Map v1 was wrong to
imply smallcaps were a clean ₹1.5L edge. Corrected here.

### The current-era picture (Incwert 2024 study, 1995-2023)
Size premium over CAPM prediction, by bucket (annual, varies by year):
- Large-cap: ~0.5-1.5% (small, stable)
- Mid-cap: ~0.5-1.2%
- Small-cap: ~1.0-1.8% (peak years 2004-05: 8.6%/8.1%; crash years negative)
- Micro-cap: highest and most volatile

**Crucially: the Incwert series is MONTHLY-CAPM-premium, and it is much smaller
than the raw 70% headline.** The honest reading: the size premium is real but
single-digit %/year after risk adjustment, and **negative in crash years**
(2008-09 small-cap premium went to −8.4%, −5.0%). A 2021 paper is literally
titled *"Size Effect in Indian Equity Market: Myth or Reality?"*

### McLean-Pontiff decay (the meta-fact that governs ALL of this)
McLean & Pontiff (2016), 97 US predictors: **published anomalies weaken ~58%
post-publication** (long-short returns diminish after the paper appears).
International replication ("Once public, no longer existent?"): same pattern.
→ **Any anomaly I can google has already been partially arbitraged out.**
This is the single most important line in the map.

### VERDICT ON SIZE (honest)
- Real as a long-horizon risk premium: yes, but single-digit %/yr net
- **NOT a high-frequency edge** — the spread (~3% round-trip) is the edge
- **NOT a safe ₹1.5L all-weather trade** — goes deeply negative in risk-off
- Good for: patient, diversified, low-turnover holding. Bad for: trading.

---

## PART 2 — LIQUIDITY-CONDITIONAL EDGE (the best-specified, most usable)

Chuia, Ranganathan, Rohit & Veeraraghavan (2023), Pacific-Basin Finance J,
**3,956 stocks, 2000-2021** — the most rigorous Indian anomaly paper I found:
- **Momentum works in LIQUID stocks**, short/intermediate/long term, persists 12mo
- **Reversal works in ILLIQUID stocks** (short-term)
- Momentum is *amplified* by liquidity; reversal is *amplified* by illiquidity
- Both survive market/size/value/macro controls
- **Caveat the authors themselves flag: the illiquid-stock reversal "fail[s] to
  find evidence of persistence"** → regime-dependent, not structural

**Why I failed this:** I ran one strategy type across a mixed universe. The
literature says the edge is CONDITIONAL on liquidity bucket. Correct design:
two separate strategies, two separate universes.

**Capital fit:** GOOD. Liquid-name momentum = low spread, survivable tail.
Illiquid reversal = bad fit (spread + no persistence).

---

## PART 3 — EXPIRY MICROSTRUCTURE, CORRECTED (map v1 was wrong here too)

Map v1 claimed "systematic pre-expiry dip → post-expiry rebound, 22 years."
The deeper sources contradict the *directional* claim:

**Kotha Kiran Kumar & Shankar (ISB/IFMR), high-frequency GARCH, 5- & 10-min
Nifty data:**
> *"while there is NO pressure – downward or upward – on index returns, the
> volatility IS significantly affected by expiration. This effect doesn't show
> up in daily data."*

So on expiry days: **returns ≈ no directional edge. VOLATILITY ≈ significant.**
This is corroborated by Stoll & Whaley (1986/87) and replicated across Japan,
HK, UK, Germany, Australia: **high volume + high volatility + economically
insignificant price pressure.**

**What this means:** the expiry edge is a **VOLATILITY/STRUCTURE edge (gamma,
theta, IV compression, term structure)**, NOT a "buy the dip before expiry"
directional edge. Map v1 misread the arXiv SIP paper as directional alpha; it
measured *timing advantage of entry* (stochastic dominance), which is not a
tradeable directional signal on expiry day itself.

Practitioner corroboration: r/NSEAlgoTrading traders run short strangles/theta
on expiry (u/Helpful-Medicine-862: "green 5 of 7 expiries"); InfiQuant: "104
weekly BankNifty expiries showing statistically significant IV compression."
→ The edge people monetize is **selling vol into expiry**, exactly matching the
"volatility up, direction random" finding.

**Capital fit:** MEDIUM-GOOD but this is OPTIONS, and I already proved short
straddles have a gap tail on ₹1.5L. Needs intraday + vol-surface data.

---

## PART 4 — SECOND-TIER ANOMALIES (ranked, with decay flags)

| Anomaly | Evidence | Decay risk | ₹1.5L fit |
|---|---|---|---|
| Illiquidity premium (Amihud) | Positive illiquidity-return relation confirmed India | HIGH | poor (spread) |
| Turn-of-month | Present in Nifty/Sensex studies | MEDIUM | tiny edge |
| Holiday effect | Pre-holiday returns higher (mixed evidence) | MEDIUM | poor (few events/yr) |
| Idiosyncratic-vol puzzle | Negative IV-return relation in India | MEDIUM | medium |
| Intraday periodicity (U-shape vol) | Well documented in NSE tick data | LOW (structural) | medium (needs intraday) |
| Call-auction price discovery | Opening auction affects discovery | LOW (structural) | poor |
| Cash-futures basis arb | Parity violations exist, more in far months | MEDIUM | good but thin |
| FII/DII flow → returns | Mixed; regime-flipped post-2016 | HIGH | poor |
| IPO underpricing | Listing-day gains documented, long-run mixed | MEDIUM | poor (lottery) |

**Structural (low decay) beats behavioral (high decay).** Microstructure
anomalies (intraday vol periodicity, auction, basis) decay slower because they
come from market mechanics, not psychology that arbitrage removes.

---

## PART 5 — THE FULL DIAGNOSIS: why I "failed" and traders succeed

Ranked by how much damage each did to my results:

1. **Wrong universe (biggest).** Reversal tested on large-caps; literature puts
   it in ILLIQUID names. Size never tested. ~1/3 of documented anomalies were
   never even in scope.
2. **Turnover disqualification (2nd).** Every high-frequency strategy was
   self-defeating — spread (~0.2-3%) exceeds the gross edge. The size paper
   says this outright. I optimized strategies that were structurally unviable.
3. **Sample too short.** 1 year vs literature's 10-22 years / 3,956 stocks.
   I can't distinguish a real anomaly from a regime with 5 days of data.
4. **No conditioning.** Literature edge = conditional on liquidity/vol/regime.
   I ran unconditional strategies. u/Thundr3 (practitioner): "regime filters
   = #1 breakthrough."
5. **Decay blind spot.** Anything I find on Google is 58% weaker than published.
   I treated literature edges as if live at full strength.

**Traders succeed because they trade the LOW-DECAY, STRUCTURAL, LIQUIDITY-MATCHED
slice** (liquid momentum, expiry vol-selling, microstructure) with realistic
turnover — not because they have secret strategies. The edges are all published.
The skill is picking the subset that survives costs and crowding.

---

## PART 6 — THE RANKED SHORTLIST (what to actually build, best-fit-first)

**TIER 1 — build now, good ₹1.5L fit, low decay, documented:**
1. **Liquid-name momentum, 6-12mo hold, low turnover.** (Chuia 2023; survives
   controls; low spread; low turnover = tail-safe.) This is the cleanest fit
   for a small account in the entire map.
2. **Intraday volatility periodicity** (U-shaped vol, NSE tick data). Structural,
   low decay, needs intraday data (accumulating, matures ~mid-Nov).

**TIER 2 — real but conditional / needs care:**
3. **Illiquid reversal** — only with position-size caps and regime filter
   (authors warn: no persistence).
4. **Expiry vol-selling** — real vol edge, but options gap tail on ₹1.5L
   (already proven). Needs vol-surface data.
5. **Cash-futures basis arb** — thin but structural.

**TIER 3 — do NOT chase:**
6. Size as a trade (spread eats it at any real turnover).
7. Calendar (holiday/turn-of-month) — few events, tiny, high decay.

**The single highest-value next build: Tier-1 #1 — liquid-name momentum on the
liquid universe I already have data for, with LOW TURNOVER and a 6-12 month
hold.** It's the only documented Indian edge that is simultaneously strong,
low-decay, and survivable at ₹1.5L. My earlier momentum tests failed because
they used 5-20 day holds on a mixed universe — the literature says 6-12mo on
liquid names. Same idea, wrong parameters.

---

## SOURCES (full text pulled)
- NSE Research Paper 13 (2001) — size, transaction costs
- Sehgal, Subramaniam & Porteu de La Morandiere (2012, IJEF) — 7-anomaly survey
- Chuia et al. (2023, Pacific-Basin Finance J 82:102193) — momentum/liquidity
- Kotha Kiran Kumar & Shankar (ISB/IFMR) — expiry GARCH
- arXiv:2507.04859v2 — 22yr expiry timing / SIP
- Incwert (2024) — India Size Premium 1995-2023
- McLean & Pontiff (2016) — anomaly decay
- Aggarwal & Tandon / NSE seasonality papers — calendar
- Practitioner: r/NSEAlgoTrading, r/algotrading (u/Yocurt, u/wizzer-algotrader,
  u/Thundr3) — production standards

**Raw search results: research_raw.json. Discipline applied: gross-vs-net flagged,
contradictions resolved (expiry direction corrected, size corrected), decay risk
rated per anomaly. Two v1 claims were overturned by deeper sources.**
