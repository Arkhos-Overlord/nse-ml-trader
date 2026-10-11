# Plan Review Log: VOLFORGE
Act 1 (grill) complete — plan locked with the user (user delegated decisions to agent). MAX_ROUNDS=5.

## SELF-GRILL RESULT (this session)
Self-grill (user directive: don't ask, grill myself) → 7 gaps in SELF_GRILL.md.
Hard-tested the top gaps rather than trusting them:

- GAP 2 (timeframe): ran on real 15m (1500 bars Nifty+BANKNIFTY). Still 0 trades.
- Traced bar-by-bar: strategies fire, regime tradeable, mult fine — but
  position_size()=0 every bar.
- ROOT CAUSE = GAP 4 (money math), concrete: ₹10K cannot buy 1 unit of Nifty
  (~₹25,000/unit). Notional cap (20% of ₹10K=₹2,000) < Nifty price. Arithmetic.
  The machine is CORRECT to stay flat.

Conclusions:
1. A Nifty-INDEX-SPOT autotrader is mathematically impossible on ₹10K.
2. Viable paths: (a) BankNifty/Nifty weekly OPTIONS (premium ~₹50-300/lot,
   defined risk, fits ₹10K) — also the ONLY way to monetize the vol edge (GAP 1/6);
   or (b) more capital (~₹50-100K+) for index futures.
3. Engine/regime/risk/backtest are built and working; the wall is capital×instrument.

GAP 3 (does HMM beat naive vol threshold?) ablation running.
ACT 2 adversarial reviewer: 3rd dispatch (first 2 died on a dead free model).
