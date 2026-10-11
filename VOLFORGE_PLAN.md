# VOLFORGE — Volatility-Adaptive Trading Machine

**A new machine, built on measured evidence — not on the hope that ML predicts direction.**

Codename: **VOLFORGE** (VOLatility FORcasting + Gradient-scaled Execution).
Working repo: `nse-ml-trader` (renamed from Kimi-Auto-trader).

---

## 1. Why this machine exists (the evidence, not vibes)

I ran a Phase-1 diagnosis and an instrument edge-scan. Conclusions, all recomputed
locally on real yfinance data, walk-forward, no look-ahead:

| Test | Result | Meaning |
|---|---|---|
| ML directional accuracy, 15m ^NSEI | **50.7%** | No directional edge. Noise. |
| Lag-1 return autocorr, 15m | **−0.057** (floor ±0.051) | A *whisper* of mean-reversion. Not tradeable alone. |
| Volatility clustering (\|ret\| acf lag-1) | **+0.17** | **The one real, strong signal.** Volatility is predictable. |
| Always-long (net of costs) | **−90%** | Direction-long bleeds to costs. Dead. |
| Buy & hold over window | −6.7% | Even holding lost in this chop. |
| Cash | 0% | Beat everything. Respect that. |

**The insight:** the market's *direction* is unpredictable, but its *volatility*
is strongly predictable. A machine that predicts direction is building on sand.
A machine that predicts **volatility** and adapts **sizing, stops, and hold-times**
to it is building on the one signal the data actually supports.

VOLFORGE is that machine. Its alpha is **risk calibration + mean-reversion fades
in high-vol regimes + cost discipline** — not "calling the next candle."

---

## 2. Design principles (non-negotiable)

1. **Predict variance, not direction.** The core model forecasts next-bar
   volatility (a regression), not up/down. Direction is handled by a mean-reversion
   rule that only activates when vol is high enough to pay for costs.
2. **Volatility-gated entries.** Only trade when forecast vol exceeds a threshold
   that makes the expected move > round-trip cost (≥15 bps). Otherwise stay in cash.
3. **Adaptive position sizing.** Size ∝ 1/forecast_vol (vol targeting). Bigger bets
   in calm, smaller in chaos — the opposite of gamblers, the same as professionals.
4. **ATR-based stops from the forecast.** Stop distance = k × forecast_vol, not a
   fixed %.
5. **Walk-forward everything.** No random train/test split. Train on the past,
   predict the next bar, roll forward. Any strategy that can't beat cash net of
   costs in walk-forward does not ship.
6. **Cash is a position.** If no setup clears the cost bar, the machine holds cash
   and that counts as a win vs. the −90% always-long baseline.

---

## 3. Architecture (the machine)

```
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER        yfinance 15m/5m (free, 60d) + daily (long) │
│                    normalize → features → labels              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  VOL FORECASTER    GBM/RNN regressor → next-bar σ            │
│  (the alpha)       Inputs: recent |ret|, ATR, range, vol-of-vol│
│                    Walk-forward. Target: realized next-bar σ.  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  REGIME + GATE     vol regime (low/normal/high/spike)         │
│                    ENTER only if expected move > cost floor   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  EXECUTION ENGINE  mean-reversion fade (lag-1 tilt) in high   │
│                    vol; size = vol-target / σ_fcst; ATR stop  │
│                    time-stop = N bars; hard daily loss cap    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  RISK OVERLAY      Kelly-capped sizing, max drawdown halt,    │
│                    correlation-aware exposure, ₹ risk/trade   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  BACKTEST LEDGER   walk-forward PnL, hit-rate, Sharpe,        │
│                    max DD, cost drag, vs cash & vs buy&hold   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  APP / DASHBOARD   live signals, regime, equity curve,        │
│                    kill-switch. (Phase 4)                     │
└─────────────────────────────────────────────────────────────┘
```

Reuse from the old repo (already written, now wired correctly):
- `enhanced_strategy.py` → Kelly sizing, ATR stops, pyramid, trailing (keep).
- `enhanced_ml_model.py` → the regressor/classifier toolbox (repurpose for σ).
- `exchange.py` → IndMoney client (paper only until validated).
- `pipeline.py`, `backtest_honest.py` → fixed loader + leak-free backtester (keep).

New modules to build:
- `vol forecast` (`volforge/forecast.py`) — the σ regressor, walk-forward.
- `regime` (`volforge/regime.py`) — vol-gate + regime classifier.
- `engine` (`volforge/engine.py`) — signal → sized order → stop → exit.
- `risk` (`volforge/risk.py`) — drawdown halt, exposure caps, Kelly cap.
- `backtest` (`volforge/backtest.py`) — the honest ledger (supersedes old).
- `app` — dashboard + live loop (Phase 4).

---

## 4. Thorough plan (phased, each phase has an exit gate)

### Phase 0 — Foundation ✅ DONE
- Fixed CSV loader (`pipeline.py`), leak-free backtest (`backtest_honest.py`).
- Diagnosed: no directional edge; vol clusters; cash wins. **This is the brief.**

### Phase 1 — Instrument edge-scan (RUNNING)
- `edge_scan.py` ranks NIFTY/BANKNIFTY/IT/SENSEX/RELIANCE/HDFC/INFY/TATAMOTORS
  on 15m walk-forward accuracy + autocorr.
- **Exit gate:** pick the 1-2 instruments with the best risk-adjusted tradability
  (not highest accuracy — highest *vol predictability* + liquidity + tight spreads).
  If NOTHING clears the bar, VOLFORGE stays a paper research platform, honestly.

### Phase 2 — Volatility forecaster (the alpha core)
- Build `forecast.py`: predict next-bar σ from recent |ret|, ATR, range, vol-of-vol.
- Metric that matters: not R², but **QICK/realized-vol rank correlation** and
  whether the vol-gate improves walk-forward PnL vs. a fixed-vol baseline.
- **Exit gate:** forecast vol ranks realized vol better than a naive rolling-std
  baseline (Diebold-Mariano style comparison). If it can't beat rolling std, the
  machine has no edge — say so and stop.

### Phase 3 — Execution + risk engine
- Mean-reversion fade only in high-vol regime; vol-target sizing; ATR stops;
  time-stops; daily loss cap; Kelly cap; cash-when-no-edge.
- **Exit gate:** walk-forward net-of-cost return **beats cash AND beats buy&hold**
  over ≥60 days with max drawdown < 15% and ≥30 trades. Else iterate or halt.

### Phase 4 — App + paper trading
- Dashboard (equity curve, regime, open risk, kill-switch) — your one-click preference.
- 30-day paper via `exchange.py`. Compare paper to backtest expectations.
- **Exit gate:** paper tracks backtest within tolerance. No live capital before this.

### Phase 5 — Live (small, guarded) — ONLY with your explicit go
- Small capital, hard stops, per your risk rules. Never auto-scale.

---

## 5. Honest risks / what could kill this
- **60-day intraday limit** (free yfinance) → small sample. Mitigate: also train on
  daily/hourly (3y) for the vol model; be honest about confidence intervals.
- **The edge is thin.** Vol predictability is real but monetizing it net of costs in
  a 15bps-spread index is hard. The vol-gate + cash discipline is what makes it
  survivable, not a magic predictor.
- **Overfitting** is the default failure. Walk-forward + ablation is the defense.
- **I will not fabricate a profitable backtest.** If VOLFORGE can't beat cash
  honestly, the deliverable is that finding, not a pretty lie.

---

## 6. Definition of done (for the whole machine)
A walk-forward backtest, run on `backtest_honest.py`-grade code, that net of costs:
beats cash, beats buy&hold, < 15% max DD, ≥ 30 trades, over ≥ 60 days — plus a
dashboard that shows the equity curve and a kill-switch. Anything less is reported
as "no edge found," not dressed up.

**Next action:** finish `edge_scan.py` → pick instruments → build `forecast.py`.
