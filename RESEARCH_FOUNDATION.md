# VOLFORGE — Research Foundation (REA synthesis)

Learned by reverse-engineering the top algos/apps in the Indian intraday F&O field.
These are proven architectures, not my guesses. VOLFORGE is built on top of them.

## What the field's best bots actually do (verified by reading their source)

### 1. `algo-fno-trading-bot` (nagnathsherikar) — the closest match to our thesis
A production-shaped NIFTY/BANKNIFTY/FINNIFTY F&O bot (Dhan API). Its crown jewel:

**HMM Regime Engine** (`core/hmm_engine.py`) — this is the algo that vindicates our
Phase-1 finding. It does NOT predict direction. It:
- Fits a `GaussianHMM` (3 states) on **[returns, volatility, RSI, ADX]**.
- Labels hidden states by mean return → BULL / BEAR / SIDEWAYS.
- Outputs `regime_confidence` (state posterior prob).
- **Adapts position size to regime** (`get_position_size_multiplier`):
  high-conf bull → 1.0x, sideways → 0.7x, bear/low-conf → 0.5x, unknown → 0 (flat).
- `is_tradeable_regime()` gates entries by confidence.

→ This is *exactly* "predict the regime/volatility, size by it, don't call direction."
It's a proven, running implementation of our core insight. **We adopt this pattern.**

**Risk Manager** (`risk/risk_manager.py`) — production discipline:
- Per-trade risk cap (1.5% of capital), daily loss kill-switch (3%), max open
  positions (3), max lots (2), min R:R (2.0), trailing SL after +1% (trail 0.5%).
- Full P&L ledger, drawdown tracking, kill-switch state.
→ We adopt these exact risk rails, scaled to ₹10K capital.

**Intraday strategies** (`strategies/intraday.py`):
- ORB (Opening Range Breakout): first 3×5min range, CE on breakout / PE on breakdown,
  SL at opposite range end, 2:1 target.
- VWAP mean-reversion: entry at ≥0.3% deviation from VWAP + RSI confirm, 50% reversion target.
- EMA momentum: 9/21 cross + ADX>25 + SuperTrend + RSI-not-extreme.
- All wrapped in a `TradeSetup` with `is_valid(min_rr=1.5)` — R:R enforced before entry.

### 2. `vectorbt` (polakowi) — the research-engine standard
- Vectorized backtesting on NumPy/numba (+ optional Rust). 100x faster than
  event-driven engines for parameter sweeps.
- `vectorbt.portfolio` = the backtest core (orders, trades, logs, from_signals).
→ This is how we run thousands of strategy variants fast. We use it for the
research/optimization layer, not our own slow loop.

### 3. Landscape lessons (from search)
- **vectorbt** = best for fast research/sweeps. **backtrader** = learning/event-driven.
  **NautilusTrader** = live intraday/order-book. **LEAN/QuantConnect** = hosted+data.
- Option-selling bots (`algo_trading_strategies_india`) confirm the field monetizes
  **premium/vol**, not direction — short straddles/strangles/iron-flies with fixed,
  %, and trailing stops. Reinforces our vol thesis.

## The VOLFORGE architecture (upgraded with these learnings)

```
DATA (yfinance 15m, free)
   │
   ▼
HMM REGIME ENGINE  ← adopted from algo-fno-trading-bot
(GaussianHMM on ret/vol/rsi/adx → BULL/BEAR/SIDEWAYS + confidence)
   │  regime + confidence
   ▼
STRATEGY LAYER  ← ORB / VWAP-reversion / EMA-momentum, R:R-gated (≥1.5-2.0)
   │  TradeSetup{entry, SL, target, rr}
   ▼
VOL-ADAPTIVE SIZING  ← size = regime_multiplier × risk-per-trade / stop-distance
   │  sized order
   ▼
RISK OVERLAY  ← per-trade 1.5%, daily kill-switch 3%, max positions, trailing SL
   │
   ▼
BACKTEST (vectorbt, walk-forward)  ← beat cash & buy&hold net of costs, or admit no edge
   │
   ▼
PAPER → (on proven edge + user go) → LIVE via Dhan/IndMoney, kill-switch armed
```

## What changed vs. the original VOLFORGE plan
- **Regime engine is now HMM, not just a vol forecaster** — proven, adopt wholesale.
- **Strategies are the field's proven intraday set** (ORB/VWAP/EMA), not invented ones.
- **Risk rails are the field's real numbers** (1.5%/3%/2:1/trailing), scaled to ₹10K.
- **Backtest via vectorbt** for speed, still walk-forward, still must beat cash.
- **Broker path**: Dhan (what the reference bot uses) or IndMoney (what our repo has).

## Honest status
The reference bot is a solid *architecture* but its README does not publish a
verified net-of-cost walk-forward P&L — so we still must prove the edge ourselves on
our own data with our own honest backtester. We inherit the *design*, not the returns.
