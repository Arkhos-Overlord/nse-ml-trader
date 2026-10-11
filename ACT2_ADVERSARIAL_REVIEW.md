# ACT 2 — ADVERSARIAL REVIEW (self-executed, fresh-eyes)

Subagent reviewer failed 3x (runtime pins children to a dead free model despite
per-call override — config issue, not the work). Instead of a 4th dispatch, I
attack my OWN gap analysis from scratch. Devil's advocate against my own verdict.

## Attack 1: "No edge" is OVERSTATED. What did I actually test?
My verdict said "no monetizable edge exists in free data." But what I actually
tested was THREE strategies (ORB/VWAP/EMA) copied from a GitHub repo, plus an HMM.
I did NOT test:
- A strategy that trades VOL CLUSTERING directly (the +0.17 signal). Vol breakout,
  range-expansion entries, vol-target carry — none built.
- Mean-reversion on the measured lag-1 autocorr (−0.057) — the one real signal.
- Options structures (the field monetizes vol via premium, per the REA of
  algo_trading_strategies_india — short straddles/strangles with stops).
- Any breadth beyond Nifty + RELIANCE + BANKNIFTY.
→ **REVISE: my claim is too broad. Correct claim: "the 3 inherited strategies
lose money net of costs on tested instruments." NOT "no edge exists."**
Untested space is real and large. The vol signal was MEASURED but never traded.

## Attack 2: Is the ₹10K wall real, or my own conservative choice?
The 0-units result came from MY config: 20% notional cap + integer flooring +
1.5% risk/trade. On Nifty index at ₹25,000, yes, ₹10K can't buy a unit. BUT:
- BANKNIFTY/Nifty weekly OPTIONS: premium ₹50-300/lot. ₹10K buys MANY lots.
  The wall applies to SPOT/FUTURES, not options.
- My notional cap was arbitrary. A small account legitimately runs 50-100%
  concentration on 1-2 defined-risk positions.
→ **REVISE: the wall is instrument-specific, not absolute. "Impossible on ₹10K"
is true only for index spot/futures. Options change the math entirely.**
And GAP 4's real lesson isn't "give up" — it's "the instrument was wrong for
the capital."

## Attack 3: Does the HMM ablation prove the HMM is useless?
HMM −₹305 vs naive −₹214. Both lose. My verdict: "HMM is decorative." But both
used the SAME broken strategies. If the strategies have no edge, NEITHER regime
model can produce one — that's a strategy problem, not evidence the HMM is bad.
The ablation proved the strategies lose; it did NOT isolate the HMM's value.
→ **PARTIAL REVISE: "HMM decorative" is unproven. Correct: "no regime model can
rescue a no-edge strategy." The HMM question needs re-testing once a real
strategy exists.**

## Attack 4: Is "accept the finding" the right recommendation?
I offered 3 paths and leaned toward accept-and-document. But the REA of the field
showed real bots DO monetize vol via options premium-selling with real risk rails.
My engine never tested that path. Recommending "give up" while the most promising
path (options, vol-selling) is untested is premature.
→ **REVISE: the recommended path should be OPTIONS vol-selling (the field-proven
monetization of the measured vol edge), not accept-and-document.**

## Attacks 5-7 (missing gaps I didn't list):
- **GAP 8: survivorship/execution realism.** Backtest fills at signal-bar close,
  next-bar resolution. No slippage model beyond flat 15bps, no spread, no gap risk
  on overnight holds. Real fills are worse. My losses are OPTIMISTIC.
- **GAP 9: the vol signal was measured but never weaponized.** +0.17 clustering is
  real; not one strategy in the machine uses a vol forecast to size or enter.
- **GAP 10: no walk-forward on RELIANCE ablation.** The ablation ran one window;
  no holdout/sweep. Same "doubt your results" rule applies — untested.

## VERDICT: REVISE — my own analysis was too pessimistic and mis-scoped
Core errors in my prior verdict:
1. Overstated "no edge" (tested 3 copied strategies, not the vol signal itself).
2. Treated the ₹10K wall as absolute (it's spot/futures-only; options fit).
3. Declared HMM decorative without isolating it from the broken strategies.
4. Recommended giving up while the field-proven path (options vol-selling) is untested.

## The hardened plan (post-review)
1. Build a strategy that actually TRADES the measured vol signal (vol-target
   breakout + lag-1 mean-reversion), not the 3 copied ones.
2. Add a real options layer: premium-based sizing (fits ₹10K), defined-risk
   structures, source IV/premium data (NSE unofficial or broker testnet).
3. Re-run the HMM ablation ONLY after a real strategy exists.
4. Add slippage/spread/gap realism to costs (GAP 5 + 8).
5. Keep the doubt-until-hard-test rule on every new result.
