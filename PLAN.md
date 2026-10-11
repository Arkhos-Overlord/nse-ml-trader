# Plan: VOLFORGE — Volatility-Adaptive Trading Machine
_Locked via grill — by Hermes + user (user delegated decisions to agent)_

## Goal
Build a self-contained, honest quant machine that trades the ONE signal the data
supports — **volatility, not direction** — via an adaptive engine (vol-target
sizing, ATR stops, cost-gated mean-reversion fades, cash-when-no-edge), validated
by leak-free walk-forward backtesting. It runs as a paper/research platform first
and graduates to live defined-risk trading only when a strategy clears a hard,
pre-declared profit bar net of costs. No fabricated edge, ever.

## Background (measured, not assumed)
Phase-1 diagnosis on real yfinance data (walk-forward, no look-ahead):
- ML directional accuracy 15m ^NSEI: **50.7%** → no directional edge.
- Lag-1 return autocorr: −0.057 (floor ±0.051) → whisper of mean-reversion.
- Volatility clustering (|ret| acf lag-1): **+0.17** → the one real, strong signal.
- Always-long net of costs: −90%; buy&hold: −6.7%; cash: 0% → cash wins.
Conclusion: predict variance, not direction. Monetize vol via sizing/stops/regime.

## Approach
1. **Data layer** (`volforge/data.py`): yfinance intraday (15m/5m, 60d free) +
   daily/hourly (3y) via `pipeline.load_ohlcv`. Normalize, feature-engineer.
2. **Vol forecaster** (`volforge/forecast.py`): regressor predicting next-bar σ
   from recent |ret|, ATR, range, vol-of-vol. Walk-forward. Must beat naive
   rolling-std baseline (rank corr / Diebold-Mariano) or admit no edge.
3. **Regime + cost gate** (`volforge/regime.py`): classify vol regime; ENTER only
   when expected move > round-trip cost floor (≥15 bps). Else cash.
4. **Execution engine** (`volforge/engine.py`): mean-reversion fade in high-vol
   regime; size = vol-target / σ_fcst; ATR stop = k×σ_fcst; time-stop = N bars.
5. **Risk overlay** (`volforge/risk.py`): Kelly-capped sizing, daily loss cap,
   max-drawdown halt, exposure caps. Defined-risk only.
6. **Backtest ledger** (`volforge/backtest.py`): leak-free walk-forward PnL,
   hit-rate, Sharpe, max DD, cost drag; compared vs cash AND buy&hold.
7. **App/dashboard** (`app/`): equity curve, live regime, open risk, kill-switch;
   one-click launch (`.cmd`). Paper mode default.
8. **Options upgrade (Phase B, documented)**: defined-risk index option structures
   IF/once IV-chain data is sourced; not required for v1 to be honest.

## Key decisions & tradeoffs (the contestable ones — reviewer: bite here)
- **Vol over direction**: because direction tested at 50.7%. Risk: vol edge may be
  too thin to monetize net of costs. Mitigation: cost gate + cash default.
- **Free data only (yfinance)**: no keys, matches user pref. Risk: 60-day intraday
  cap = small sample. Mitigation: also train vol model on 3y daily/hourly; report
  confidence honestly.
- **Index (NIFTY/BANKNIFTY) primary**: liquid, user's domain. Risk: most efficient,
  least predictable. Mitigation: edge-scan stocks; if a stock is more tradable,
  the engine is instrument-agnostic.
- **Cash is a valid outcome**: if nothing beats cash net of costs, we ship that
  finding, not a fake backtest. Risk: user may find "no edge" unsatisfying. Accepted.
- **Reuse old repo pieces** (Kelly sizing, ATR stops, IndMoney client, fixed
  pipeline/backtester) rather than rewrite — they're sound once wired correctly.

## Risks / open questions
- Monetizing vol net of 15bps on an efficient index is genuinely hard; the vol
  gate + cash discipline is the survival mechanism, not a magic predictor.
- Overfitting is the default failure; walk-forward + ablation is the defense.
- Option IV data sourcing (Phase B) is unsolved — flagged, not faked.
- Small intraday sample (60d) limits statistical power.

## Out of scope
- Live auto-trading without explicit user go.
- High-frequency / colocation / market-making.
- Undefined-risk option selling.
- Any claim of recovered "original source" or guaranteed returns.

## Definition of done
A walk-forward backtest on backtest_honest-grade code that net of costs beats cash
AND buy&hold, <15% max DD, ≥30 trades, over ≥60 days — plus a dashboard with equity
curve and kill-switch. Anything less is reported as "no edge found," honestly.
