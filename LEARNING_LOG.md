# VOLFORGE Learning Log — Every Strategy Tested, Every Lesson

This is the trace. Every backtest, every result, every hard-won lesson, logged.
No strategy gets built until its predecessor is understood. The log grows only
from real runs on real data — never from hopes.

---

## STRATEGY LEDGER

| # | Strategy | Data | Costs | Result | Edge? | Fits ₹1.5L? | Lesson |
|---|----------|------|-------|--------|-------|-------------|--------|
| 1 | Overnight naked strangle | 12yr daily | exact | 48% win, negative | ❌ | ❌ | Naked premium has no edge; tail −₹2.8L |
| 2 | Overnight regime-gated | 12yr daily | exact | negative | ❌ | ❌ | Regime filter didn't rescue it |
| 3 | Overnight delta-hedged straddle | 12yr daily | exact | **+₹1Cr/12yr** | ✅ | ❌ (needs ~₹75L) | THE proven edge — but gap tail −₹2.4L |
| 4 | Overnight delta-hedged condor | 12yr daily | exact | marginal | ~ | ❌ | Wings don't fix the gap |
| 5 | Overnight credit spread ±500 | 2023-25 | exact | −₹4.2L | ❌ | ❌ | Wings tested too often; hedge can't save |
| 6 | Overnight credit spread ±1000 | 2023-25 | exact | +₹6.5L but tail −₹2.77L | ~ | ❌ | Wings cap at EXPIRY only, not daily gaps |
| 7 | Intraday naive straddle (unhedged) | 2025 1-min | exact | −₹41K/8d, tail −₹38K | ❌ | ⚠️ survives 1.5L | No gap risk = tail is intraday-limited |
| 8 | Intraday delta-hedged (spot from parity) | 2025 1-min | exact | **83% win, ~breakeven**, tail −₹17.9K | ~ | ❌ for ₹10K | 2 bad trend days wipe 10 good days |
| 9 | Intraday v3 (stop-loss + trend filter) | 2025 1-min | exact | **−₹2.97L, 29% win** | ❌ | ❌ | **Stop-loss HURTS**: cuts winners that recover. Short premium structurally dead intraday. |
| 10 | ~~Intraday mean-reversion scalper~~ | 2024-25 1-min | futures | **+₹47Cr → DISCARDED** | ❌ ARTIFACT | ❌ | **Fake data**: parity spot (S=K+CE−PE) fabricated ±80pt 1-min swings from stale CE/PE. Built a ₹47-crore illusion on noise. Caught by arithmetic audit (max real trade ₹1350, sim showed ₹12K avg = 9× inflated). |

---

## KEY LESSONS (the distillation)

### L1 — The gap is the killer, not the cost
Costs are only ~₹110-173 (0.2%) per option leg. I initially overclaimed "₹1500+ = 3%" — **corrected**: costs are 7% of gross, not a killer. The real killer is the **overnight gap tail** (−₹1.5L to −₹2.8L) that no hedge survives on a daily-rebalanced small account.

### L2 — Overnight premium-selling has ONE real edge, and it's capital-hungry
Delta-hedged ATM straddle: **+₹1Cr over 12 years** on real data, exact costs. But its worst gap (−₹2.4L) demands **~₹75L** to run safely. ₹1.5L is 50× too small. The edge and the tail are the same coin.

### L3 — Intraday kills the gap, which is the ONE thing that fits small capital
No overnight exposure = no gap. Worst intraday day is intraday-range-limited (₹15-48K), not the ₹1.5L gap. This is structurally the small-account path.

### L4 — Short premium is structurally dead intraday (proven, not assumed)
Raw data: Nifty intraday range was often >1% (₹110→₹1,281 on one day). Short straddles win on flat days, bleed 2-4× on trend days. Net negative. Stop-losses make it *worse* (v3: cuts recovering winners → 29% win). **Directional premium-selling is a loser at small capital.**

### L5 — The surviving edge needs a mechanism WITHOUT theta/gamma
Options-selling dies on trend days because of gamma. The candidate: **intraday mean-reversion on Nifty futures** — no theta, no gamma, tail is a single tick stop-loss. Different animal entirely. (Tested as #10 — see L7/L8: it was a data artifact, DISCARDED. Need real synchronized spot to retry honestly.)

### L6 — Capital floor is arithmetic, not preference
One Nifty lot margin ≈ ₹50K. Worst day to survive at 1% risk sets the floor:
- intraday straddle: −₹18K → needs ~₹18L
- proven overnight edge: −₹2.4L → needs ~₹75L
₹10K cannot trade even ONE lot. No position-sizing trick fixes this.

### L7 — A too-good result is a bug until proven otherwise (the ₹47-crore lesson)
Mean-reversion on reconstructed spot showed **+₹47Cr on ₹1.5L** (+31,372%), 89% win, 2.8% DD. Impossible. Arithmetic audit exposed it: max possible trade = target 0.08% of ₹23,000 = ₹1,350, but sim's implied avg was ₹12,242 = **9× inflated**. Root cause: spot reconstructed as **S = K + (CE − PE)** from stale last-trade CE/PE prices fabricated **±80pt single-minute swings** real Nifty never makes. The "edge" was pure noise in broken data. Saved only by refusing to believe a too-good number and recomputing the arithmetic.

### L8 — Reconstructed spot from option parity is NOT trustworthy for scalping
Put-call parity S ≈ K + (CE − PE) holds at one instant with *synchronized* quotes. Intraday option files store last-trade prices at staggered times, so CE−PE embeds stale-quote noise, not real spot motion. Deep-ITM-call proxy (S ≈ K + C) is ALSO contaminated (+101pt jump then flat = stale-trade dead time). **Both reconstructions unusable as 1-min price series.** Fine for rough delta only. Need real synchronized spot.

### L9 — Clean intraday futures data is broker-gated (no keyless source)
Verified: Kite instrument master (9MB) downloads keyless and confirms Nifty futures (lot now **65**, not 75), but historical 1-min candles require api_key+access_token (HTTP 400 without). NSE bhavcopy = EOD only. yfinance caps intraday at 60 days. **No keyless 1-min intraday spot/futures source exists.** The mean-reversion path is blocked by data access, not effort — a broker account (Zerodha/Dhan) is the unblock.

### L10 — Contract specs drift; re-verify lot size
Nifty futures/options lot was **75** in the 2024-2025 datasets but is **65** in current Kite master. Any PnL using the old lot overstates by 75/65 = 1.15×. Always re-pull lot size from the live instrument file before sizing.

### L11 — UNBLOCK: INDmoney MCP provides clean synchronized intraday data (₹0)
Connected INDmoney MCP (Indian app, free, **read-only**, OAuth 2.1+PKCE — no keys to manage). 22 tools incl. `lookup_ind_keys` + `get_indian_stocks_ohlc` (1min–30min up to 7d, 1day–1month up to 1y) covering indices, futures, options, stocks. Built `indmoney_client.py` (JSON-RPC over Hermes's cached OAuth token). **Verified clean:** Nifty futures 1-min mean |step| = 4.9–5.2 pts (vs ±80pt fake jumps from parity reconstruction). Daily OHLC cross-checks exactly against nsepython. Futures directly available (NIFTY Oct/Nov/Dec). This replaces the contaminated HuggingFace option files for anything needing real synchronized spot. Constraint: 1min/7d returns only the most recent trading day → needs a **daily accumulator** (`pull_daily.py`, Windows scheduled task 16:35 IST weekdays, dedup by key+interval+datetime).

### L12 — Smoke test passed the honesty gate; naive mean-reversion is flat
Mean-reversion on 4 days of clean 5-min Nifty futures (lot **65**, verified): 37 trades, 62% win, **−₹18,361**, worst trade −₹7,282. **Honesty gate passed** — all trades within ±₹20K (vs the old fake run's ₹12K avg trades on parity noise). Pipeline validated end-to-end on real data; the naive config is slightly negative, as expected. **4 days is not a sample — no edge claim until 30+ days.** Equity track added: 15 most-liquid F&O stocks via explicit ticker→company alias map (fuzzy matching missed HDFC Bank / Infosys / TCS), 6,375 clean candles. Daily accumulation via Windows scheduled tasks: `INDmoney_Daily_Pull` (futures, 16:35) and `INDmoney_Equity_Pull` (futures+equity, 16:45) IST weekdays.

---

## DATA ASSETS ACQUIRED
- `NIFTY_2023-2025.parquet` — daily EOD option prices (rissin/nse-options)
- `NIFTY_intraday_2024.parquet` (140MB) — **1-min intraday, full 2024**
- `NIFTY_intraday_2025.parquet` (674MB) — 1-min intraday, 2025 (69.7M bars)
- Live NSE chain via nsepython (working — ATM 22750, CE ₹314 + PE ₹375)
- Reconstructed intraday spot from put-call parity: **S ≈ K + (CE − PE)** — removes the yfinance 60-day 5-min limit

---

## NEXT (do not stop)
- [ ] #10 mean-reversion scalper — full 2024 result (running)
- [ ] If profitable: tune thresholds, test 2025, stress-test tail
- [ ] If dead: try long-premium on range days, or equity intraday
- [ ] Build the learning loop: every run appends here automatically
- [ ] Paper-trade the winner live on the harness (₹10K virtual)

**Rule: the log only grows from real runs. No entry without a number.**
