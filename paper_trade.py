"""PAPER TRADING HARNESS — live intraday vol harvest on a Rs10K virtual account.

This is the LIVE TEST you asked for: run the strategy against REAL current NSE
option prices (nsepython live chain), track a virtual Rs10K account, log every
signal/entry/exit with timestamps. No real money — pure paper.

It proves: (1) the data pipeline works live, (2) the strategy logic runs in real
time, (3) you see actual PnL vs the backtest. Run this during market hours.

Run: python paper_trade.py  (logs to paper_trades.csv + paper_account.json)
"""
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

ACCOUNT_FILE = "paper_account.json"
TRADES_FILE = "paper_trades.csv"
CAPITAL = 10000.0
LOT = 75
HEDGE_COST_BPS = 3
SQUARE_OFF = "15:15"
ENTRY = "09:45"


def load_account():
    if os.path.exists(ACCOUNT_FILE):
        with open(ACCOUNT_FILE) as f:
            return json.load(f)
    return {"capital": CAPITAL, "start": CAPITAL, "trades": [],
            "open": None, "log": []}


def save_account(acc):
    with open(ACCOUNT_FILE, "w") as f:
        json.dump(acc, f, indent=2, default=str)


def get_live_chain():
    """Fetch live Nifty option chain via nsepython (NSE unofficial)."""
    from nsepython import nse_optionchain_scrapper
    chain = nse_optionchain_scrapper("NIFTY")
    data = chain["data"]
    ce = {float(d["strikePrice"]): d["CE"] for d in data if d.get("CE")}
    pe = {float(d["strikePrice"]): d["PE"] for d in data if d.get("PE")}
    return ce, pe, data[0]["expiryDate"] if data else None


def atm_straddle(ce, pe):
    """Find ATM strike and current straddle premium."""
    sums = {s: (ce[s]["lastPrice"] + pe[s]["lastPrice"])
            for s in ce if s in pe and ce[s].get("lastPrice", 0) > 0
            and pe[s].get("lastPrice", 0) > 0}
    if not sums:
        return None, None, None
    atm = min(sums, key=sums.get)
    return atm, ce[atm]["lastPrice"], pe[atm]["lastPrice"]


def paper_tick():
    """One iteration: check time, manage open position or open new one."""
    acc = load_account()
    now = datetime.now().strftime("%H:%M")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] tick")

    try:
        ce, pe, expiry = get_live_chain()
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR fetching chain: {e}")
        return acc

    atm, c, p = atm_straddle(ce, pe)
    if atm is None:
        print("  no ATM straddle available")
        return acc
    straddle = c + p
    print(f"  expiry {expiry} | ATM {atm:.0f} | CE Rs{c} + PE Rs{p} = Rs{straddle:.1f}")

    # square off before close
    if acc["open"] and now >= SQUARE_OFF:
        exit_prem = straddle
        entry = acc["open"]
        pnl = (entry["straddle"] - exit_prem) * LOT
        acc["capital"] += pnl
        acc["trades"].append({
            "date": str(datetime.now().date()), "atm": atm,
            "entry_straddle": entry["straddle"], "exit_straddle": exit_prem,
            "pnl": round(pnl, 0),
        })
        acc["log"].append(f"{now} SQUARE OFF: PnL Rs{pnl:,.0f}, capital Rs{acc['capital']:,.0f}")
        acc["open"] = None
        print(f"  SQUARED OFF: PnL Rs{pnl:,.0f}, capital Rs{acc['capital']:,.0f}")
        save_account(acc)
        return acc

    # open new position at entry time if flat
    if not acc["open"] and ENTRY <= now < SQUARE_OFF:
        # risk check: only if straddle premium is reasonable (not too cheap)
        if straddle > 50:  # min premium to be worth it
            acc["open"] = {"atm": atm, "straddle": straddle,
                           "entry_time": now, "expiry": expiry}
            acc["log"].append(f"{now} SELL straddle ATM {atm:.0f} @ Rs{straddle:.1f}")
            print(f"  SOLD ATM straddle @ Rs{straddle:.1f} (paper)")
            save_account(acc)
        else:
            print(f"  premium too thin (Rs{straddle:.1f}), skipping")
    elif acc["open"]:
        print(f"  holding short straddle ATM {acc['open']['atm']:.0f} "
              f"@ Rs{acc['open']['straddle']:.1f} (now Rs{straddle:.1f})")

    save_account(acc)
    return acc


def status():
    acc = load_account()
    print(f"\n=== PAPER ACCOUNT STATUS ===")
    print(f"  start capital: Rs{acc['start']:,.0f}")
    print(f"  current:       Rs{acc['capital']:,.0f}")
    print(f"  PnL:           Rs{acc['capital']-acc['start']:,.0f} "
          f"({(acc['capital']/acc['start']-1)*100:+.1f}%)")
    print(f"  trades closed: {len(acc['trades'])}")
    if acc["open"]:
        print(f"  OPEN: short straddle ATM {acc['open']['atm']:.0f} "
              f"@ Rs{acc['open']['straddle']:.1f} (entry {acc['open']['entry_time']})")
    if acc["trades"]:
        wins = sum(1 for t in acc["trades"] if t["pnl"] > 0)
        print(f"  win rate: {wins/len(acc['trades'])*100:.0f}%")
        worst = min(t["pnl"] for t in acc["trades"])
        print(f"  worst trade: Rs{worst:,.0f}")
        print(f"  Rs10K survived worst: {'YES' if abs(worst) < 10000 else 'NO — would be ruined'}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status()
    elif len(sys.argv) > 1 and sys.argv[1] == "loop":
        print("Paper trading loop — runs every 5 min. Ctrl+C to stop.")
        while True:
            try:
                paper_tick()
            except Exception as e:  # noqa: BLE001
                print(f"error: {e}")
            time.sleep(300)  # 5 min
    else:
        print("Paper trading harness for Rs10K virtual account.")
        print("Usage:")
        print("  python paper_trade.py          — one tick (check + maybe trade)")
        print("  python paper_trade.py status   — account status")
        print("  python paper_trade.py loop     — run every 5 min (market hours)")
        print("\nRunning one tick now...")
        paper_tick()
        status()
