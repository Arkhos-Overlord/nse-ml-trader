# nse-ml-trader — Upgrade Roadmap

> Rewritten 2026-10-11 after a full code + runtime audit. The old README/roadmap
> described an aspirational "production-ready autonomous bot." This reflects what
> the code **actually does** when run, verified locally on 2 years of Nifty data.

## Honest current state (verified, not claimed)

**What exists and works:**
- `enhanced_ml_model.py` — real 4-model voting ensemble (XGBoost/LightGBM/RF/GB) + scaling.
- `enhanced_strategy.py` — Kelly criterion, ATR stops, pyramid sizing, trailing stops.
- `enhanced_features.py` — 40+ indicator *functions* (MACD, Stochastic, Bollinger, ...).
- `daily_data_updater.py` / `weekly_retrainer.py` — scheduling scaffolding.
- `exchange.py` — IndMoney/INDstocks API client (untested live).
- `pipeline.py` (NEW) — fixed the CSV loader; produces clean features.
- `backtest_honest.py` (NEW) — leak-free walk-forward backtest.

**What was actually broken (now fixed):**
- CSV loader crashed on yfinance's 12-col double header → `pipeline.py` normalizes it.
- Feature modules were never wired to the model → pipeline connects them.
- Old `backtester.py` had **look-ahead bias**: trained on a random split then
  "backtested" rows whose true next-day target it already held.

**The uncomfortable result — honest walk-forward (train on past only):**

| Metric | Old (leaky) | Honest (walk-forward) |
|---|---|---|
| Win rate | 100% | **37.6%** |
| Return | +45% | **−13.6%** |
| Max drawdown | — | **−16.8%** |
| vs Buy & Hold | "wins" | **loses** (B&H +6.0%) |

**Verdict:** the current model has **no edge**. It loses money and underperforms
holding Nifty. Daily direction prediction on a highly efficient index with 6 basic
features + 2 years of data is not enough. Everything below is about finding a
*real, verified* edge — or honestly concluding there isn't one on this target.

## Ground rules for this project (non-negotiable)

1. **No look-ahead bias, ever.** Every backtest is walk-forward / purged. Any
   strategy that shows >60% win rate on daily Nifty is leaking until proven otherwise.
2. **Every claim is recomputed.** Numbers come from `backtest_honest.py` output,
   not the README. Drop false precision.
3. **Paper before live.** No broker order until a strategy beats buy&hold net of
   costs for a sustained walk-forward window.
4. **Costs are real.** ≥15 bps round-trip modeled in every test.

## Phase 0 — Foundation (DONE)
- [x] Fix CSV loader (`pipeline.py`)
- [x] Wire features → model (`pipeline.build` → `ml_model`)
- [x] Leak-free walk-forward backtest (`backtest_honest.py`)
- [x] Honest baseline: −13.6% / 37.6% win / loses to B&H

## Phase 1 — Diagnose the edge problem (NEXT)
The model predicts "up tomorrow" on an efficient index. Before adding complexity,
find out *if any signal exists*:
- [ ] **Baseline the trivial strategies**: buy&hold, always-long, random. Is the
      index even predictable day-to-day? (autocorrelation of daily returns)
- [ ] **Feature importance + ablation**: which of the 6 features carry signal?
      (`ml_model` feature_importances_) Drop the noise.
- [ ] **Class balance / drift**: is `target` ~50/50? Does accuracy hold across
      market regimes (trend vs chop)?
- [ ] **Probability calibration**: is the model's 0.5 threshold even meaningful,
      or is it noise around chance?
- [ ] **Longer horizon**: does predicting 5-day / 20-day direction work better
      than next-day? (efficiency hurts most at 1-day)

**Exit criteria:** a written diagnosis — either "edge exists in feature X at
horizon H" or "no exploitable edge at daily frequency on ^NSEI; change target."

## Phase 2 — Fix the target if Phase 1 says so
Options, in order of likely payoff (your domain: FNO/intraday):
- [ ] **Trade something less efficient than the index**: single mid-cap stocks,
      sectoral indices, or **options** (vol > direction predictability).
- [ ] **Intraday data** (15m/5m) instead of daily — intraday is your stated edge zone.
- [ ] **Volatility/regime features**: India VIX, ATR regime, gap stats.
- [ ] **More history**: 5-7 years, not 2.

## Phase 3 — Integrate what's already written
- [ ] Wire `enhanced_ml_model` (ensemble) into the walk-forward loop, compare to
      the single RF. Keep whichever wins **honestly**.
- [ ] Wire `enhanced_strategy` risk sizing into the sim (ATR stops, Kelly).
- [ ] Add the full `enhanced_features` indicator set, then ablate — keep only
      features that improve walk-forward return.
- [ ] Replace `backtester.py` usage with `backtest_honest.py` everywhere.

## Phase 4 — Validate before any live capital
- [ ] 30-day paper trading via `exchange.py` against IndMoney test/paper.
- [ ] Walk-forward on out-of-sample data the model never saw.
- [ ] Only then: small capital, hard stop-loss, per your risk rules.

## What NOT to do
- Don't trust the old README's "70-75% accuracy, 10-15% monthly ROI" — that was
  aspiration with a leaky backtest behind it.
- Don't add more indicators hoping accuracy rises — ablate first.
- Don't go live on a strategy that doesn't beat buy&hold net of costs.

## Repo hygiene
- Renamed `Kimi-Auto-trader` → `nse-ml-trader` (drops the session-artifact name).
- TODO: dedupe `strategy.py` vs `enhanced_strategy.py`; remove `.venv` from tracking;
  fix the "production-ready" README badge (this roadmap supersedes it).
