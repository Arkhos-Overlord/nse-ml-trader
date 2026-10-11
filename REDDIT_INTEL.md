# Reddit Practitioner Intel — r/algotrading + r/NSEAlgoTrading

Sourced 2026-10-11 via public Reddit feeds. This is what actual practitioners
(with strategies in production) say — used to calibrate my own standards.

---

## THE GOLD: u/Yocurt's production criteria (r/algotrading, meta-labeling thread)
https://www.reddit.com/r/algotrading/comments/1lnm48w/

A practitioner with **two strategies live in production on ES and NQ futures**
(profit factor ~1.15, Calmar ~2.5). His minimum bar before even considering a
strategy real:

1. **Profitable in >80% of YEARS backtested** (min 5 years)
2. **Profit factor ≥ 1.07** (with slippage + fees modeled accurately)
3. **≥2,500 total trades** (statistical significance)
4. **Max drawdown ≤ average yearly profit**

### His core methodological claims (validates everything I've been doing)
- "**ML will not find patterns by itself** from candlesticks/indicators — too much
  noise, can't generalize. ML is NOT good at creating edge out of nowhere."
- "**A much better approach: have an underlying strategy with an existing edge,
  then train a model to FILTER OUT low-quality trades**" (meta-labeling /
  triple-barrier, per López de Prado)
- "**Finding an edge → ML bad. Improving an existing edge → ML good.**"
- Retrains weekly; uses 6 base models, 20-70 features each, MBO (level-3
  orderbook) data from Databento + FRED macro.
- Warns explicitly about LLMs: "They'll just tell you what it thinks you want
  to hear." ← exactly the failure mode I've been guarding against all session.

### Thread consensus (multiple practitioners)
- u/Phunk_Nugget: success with **LightGBM meta-labeling for filtering** — "made
  me smile the few times I saw my model's PnL go up while trade count went down"
- u/blearx + u/elchulito89: **triple-barrier method + Yang-Zhang volatility**
  targeting works
- u/xTruegloryx: 6-model ensemble (XGB/LightGBM/GBRT/HGB/CatBoost/ADA) with
  LightGBM meta-trainer; stresses **no look-ahead bias, no data leakage**
- u/echoeysaber: triple-barrier labels = +1 (hit TP), 0 (timeout), −1 (hit SL);
  OP simplifies 0 → −1

---

## INDIA-SPECIFIC (r/NSEAlgoTrading)

Observed practitioner activity:
- u/wizzer-algotrader: "**Today's P&L from a stack of systematic strategies
  running on just ₹25k**" — claims a STACK of small strategies on ₹25k
- u/wizzer-algotrader also asks: "**How do you stop good strategies from dying
  to transaction costs?**" ← the exact wall I hit
- u/Jada_Bite: "**13 Mins in the Morning is all it takes**" — a 13-minute
  morning routine claim (likely opening-range based)
- u/Helpful-Medicine-862: "tested a strat on paper for 2 weeks... simple short
  strangle on expiry days... 7 expiries, would've been green 5 of them.
  **paper isn't just for noobs, even decent traders should validate**"
- OpenTerminalUI: free open-source NSE/BSE analytics terminal (F&O analytics,
  quant screeners, backtesting) — https://github.com (built by an Indian trader
  frustrated by paywalled tools)
- Community is heavily options/expiry-focused (strangles, theta, gamma)

### What the India sub does NOT show
- No verified track records. Mostly tool launches, "DM for beta", and
  subscription-service marketing. **No published audited returns.** The ₹25k
  and "13 minutes" posts are claims, not evidence.

---

## HONEST CALIBRATION — where my results stand vs practitioner standards

| Yocurt's production bar | My ETH edge | My best equity result |
|---|---|---|
| Profitable >80% of years | ✅ 82% of quarters (proxy) | ❌ |
| Profit factor ≥1.07 | ✅ ~1.3 (57% win, 2:1 payoff) | ❌ |
| ≥2,500 trades | ❌ 448 trades | ❌ |
| Max DD ≤ avg yearly profit | ❌ 64.7% DD | ❌ |

**My ETH edge actually clears 2 of 4 production bars** — it's closer to real
than I credited. Its failure is (a) too few trades and (b) drawdown too large
for ₹1.5L.

---

## ACTIONABLE TECHNIQUES I HAVEN'T TRIED (from practitioners)

1. **Meta-labeling / triple-barrier filtering** — take my ETH edge (which HAS
   edge) and train a classifier to filter out low-quality signals. This
   AMPLIFIES an existing edge rather than inventing one. Directly applicable.
2. **Yang-Zhang volatility for barrier setting** — better than fixed % targets
3. **Profit factor + Calmar as primary metrics**, not total PnL (PnL hides
   regime luck — same reason I score on quarterly consistency)
4. **Model slippage + fees accurately** — I already do this (₹119-144/RT)
5. **Regime/volatility filters** — u/Thundr3: "market and volatility regime
   filters" was his #1 breakthrough. Could fix my ETH edge's regime dependence.

## THE STRATEGIC IMPLICATION

Practitioners confirm: **edge is scarce, ML doesn't create it, and the winning
move is to find a small real edge then FILTER it to amplify it.** My ETH edge
is exactly that kind of candidate. Meta-labeling on ETH signals is the highest-
value untried technique — it directly attacks the two things killing my ETH
edge at ₹1.5L: too many bad trades (filter them) and fat-tail drawdown
(filter the worst setups).
