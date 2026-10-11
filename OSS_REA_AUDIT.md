# REA: Open-Source Trading Repos — what they actually contain

Method: **REA** (Read source, Examine architecture, Apply to a controlled target).
Rule applied: "verified-open over OSS slop" — a repo is only worth adopting if
its claims survive unpacking the actual code.

---

## 1. OpenQuant India (revorhq/openquant) — the strongest find

**Claim on homepage:** "Same strategy. Two backtests. Gross CAGR +22% / Sharpe
1.8 / DD −12% vs Net-of-everything." Looks like a documented alpha strategy.

**What the source actually contains (PROVEN from clone):**
- Packages: `oq-backtest`, `oq-data`, `oq-broker`, `oq-core`, `oq-mcp`, `oq-zoo`, `oqstack`.
- The strategy zoo (`oq-zoo`) contains **exactly ONE registered strategy**:
  `buy_and_hold_equal_weight` — explicitly tagged `educational`,
  "intentionally **not** alpha".
- `examples/momentum_nifty.py` and `examples/mean_reversion.py` run on
  **`synthetic_universe()`** — generated data, not real prices.
- The "+22% gross / +9% net" table lives in `marketing/talks/*.md` and
  `docs/honest-backtesting.md` as an **illustration of cost drag**, not a
  strategy result. The talk states costs + STCG taxes "eat 13 points of CAGR."

**VERDICT: zero tradeable alpha in this repo. The homepage number is a
methodology demo, not an edge.** Anyone citing it as a strategy result has not
opened the source.

### What IS worth taking (the real asset)

**A. The honesty gate** (`packages/oq-zoo/src/oq_zoo/gate.py`) — stricter than
mine, and worth adopting as the standard:
1. Net-of-cost CAGR must **strictly beat the declared benchmark**.
2. Walk-forward OOS Sharpe must be **non-negative** (train 252×3 / test 252).
3. Cost preset + universe **declared upfront**.
4. Honest losers admitted only if tagged `educational`.

**B. The cost engine** (`packages/oq-backtest/src/oq_backtest/costs.py`) —
broker presets (`zerodha`, `upstox`, `fyers`, `dhan`, `full_service`) with
correct 2024-25 Indian rates: STT 0.1% both legs delivery / 0.025% sell
intraday, NSE exchange 0.00297% (revised Oct 2024), SEBI ₹10/crore, GST 18% on
brokerage+exchange+SEBI, stamp 0.015% delivery buy / 0.003% intraday buy,
"0.03% or ₹20 whichever lower" brokerage convention.

**C. The independent confirmation of this session's central finding.** Their
entire project exists because "a +22% CAGR backtest on Indian equities almost
always becomes a +9% one" — **gross 22% → net 9% is exactly the cost wall every
strategy here died against.** An independent team reached the same conclusion:
the Indian retail edge problem is costs, not signal discovery.

### Apply to a controlled target (the 'A' of REA)
Adopt (A)+(B) into `nse-ml-trader` as the standard gate + cost preset, and
re-run the surviving candidates through it. If nothing survives a
benchmark-relative + walk-forward + non-negative-OOS-Sharpe gate, that is the
final answer for ₹1.5L, arrived at with someone else's stricter standard.

---

## 2. Repos screened and rejected

| Repo | Why rejected |
|---|---|
| `freqtrade/freqtrade-strategies`, `jmrichardson/freqback`, `webclinic017/strategies-freqtrade-` | Crypto-only; user excluded crypto. Also classic overfit-strategy repos. |
| `jailsonduarte/emerging-markets-sentiment-anomalies` | Sentiment/EPU research; needs EPU index data, no keyless source. |
| `miindisponi99/Statistical-Arbitrage-Emerging-Markets` | Pairs trading — already tested and lost (37% win, −₹244/trade). |
| `achntj/statistical-arbitrage` | ML stat-arb, generic; no India-specific licensing conditions documented. |
| `mirzabaig313/PaperTrade-India` | Paper-trading simulator — useful for forward-testing, not an edge source. |
| `virendrakumar93/algo-strategy-testing` | Self-promoted LinkedIn project, no published backtest results to audit. |
| `jayadevrana/supertrend-optimizer-zerodha-backtester` | Supertrend parameter optimizer = in-sample sweep, the exact overfit pattern the falsification skill bans. |
| `mulhamfetna/trading-strategy-finder` | Futures strategy discovery framework; generic, no India costs. |

---

## 3. The pattern across all of them

Every open-source repo promising Indian-market alpha either:
(a) ships **no strategy** and sells infrastructure (OpenQuant India — honest), or
(b) ships strategies with **gross-only** or **in-sample-swept** results
(supertrend optimizers, freqtrade strategy zoos), or
(c) is **crypto** (excluded), or
(d) is **sentiment/pairs** work I already tested and killed.

There is no open-source repo with a documented, net-of-cost, walk-forward-
validated Indian edge for a ₹1.5L account. That is itself the finding: the
absence is consistent with my own ~22 dead families.

## 4. What I'm taking forward
1. Adopt the OpenQuant honesty gate + cost presets as the project standard.
2. Re-run surviving candidates under it (benchmark-relative + walk-forward).
3. Keep the auto-retest loop running on the data side.
