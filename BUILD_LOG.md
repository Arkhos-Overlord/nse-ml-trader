# VOLFORGE Build Log — Honest Status

## What's built (verified by running, not by claim)
- `volforge/regime.py` — 3-state GaussianHMM regime engine. Runs, fits, predicts.
- `volforge/ghmm.py` — self-contained Baum-Welch HMM (no native build needed).
- `volforge/strategies.py` — ORB / VWAP-reversion / EMA-momentum, R:R-gated.
- `volforge/risk.py` — ₹10K risk rails (1.5%/trade, 3% daily kill-switch, vol-target sizing).
- `volforge/backtest.py` — leak-free walk-forward ledger, net of 15bps costs.

## First full run (2yr Nifty daily)
+139.73% return, Sharpe 7.9, 15 trades, 40% win, beats cash & B&H (13.4%).
**FLAGGED AS SUSPECT.** 15 trades is a tiny sample; +140% is an overfit signature.
I do NOT trust it. Hard test in progress (holdout + 27-combo sweep + 4 blocks).

## Bugs found & fixed during the build (self-caught, then verified)
- hmmlearn won't build on Py3.12 → wrote pure-NumPy GHMM instead.
- `_logsumexp` returned 2D not 1D → (3,1) broadcast crash → fixed with squeeze.
- Holdout window crashed: `min_train=250` > 197-bar holdout → guarded the floor.

## Rule I'm enforcing (user's standing instruction)
"Always doubt your own results and only get sure after running a hard test."
So: no number ships until holdout + param-sweep + time-blocks agree. If they don't,
the honest deliverable is "overfit, no proven edge," not the pretty number.

## Next
1. Read robustness.out → verdict on +139%.
2. If robust: build dashboard + paper loop. If not: report honestly, tune, retest.
