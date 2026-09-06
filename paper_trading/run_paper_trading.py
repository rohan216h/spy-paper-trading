"""
Phase 9, step 3 - THE scheduled paper-trading script.

Run this on a schedule (every 5 min, during market hours) via Windows
Task Scheduler / cron. Each run:
  1. Checks current state (are we already in a position?)
  2. If in a position: increment bars_held; if max_hold reached, close it
     (matches backtest exit logic - TIME exit only, same as Strategy B's
     backtested behavior)
  3. If flat: get the latest signal; if LONG/SHORT, open a position
  4. Logs every decision to paper_trading/trade_log.csv - this log is
     what will feed the Phase 10 dashboard

SAFETY: defaults to DRY RUN (prints what it would do, places no real
paper order). Pass --live to actually submit orders to your Alpaca
paper account. Start with dry runs until you've watched it behave
correctly for a few cycles.

Run with:  python paper_trading/run_paper_trading.py
       or: python paper_trading/run_paper_trading.py --live
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import csv
import argparse
from datetime import datetime, timezone

from dotenv import load_dotenv
import os
load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from paper_trading.generate_signal import get_latest_signal
from configs.config import SYMBOL

STATE_PATH = Path(__file__).resolve().parent / "state.json"
LOG_PATH = Path(__file__).resolve().parent / "trade_log.csv"


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"position": None, "last_processed_bar": None}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def log_decision(row: dict):
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def submit_order(client, side: OrderSide, qty=1):
    order = MarketOrderRequest(
        symbol=SYMBOL, qty=qty, side=side, time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(order)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                         help="Actually submit paper orders. Without this, dry run only.")
    args = parser.parse_args()

    run_time = datetime.now(timezone.utc).isoformat()
    state = load_state()

    trading_client = None
    if args.live:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        trading_client = TradingClient(api_key, secret_key, paper=True)

    latest_time, close, pred, signal, threshold, max_hold = get_latest_signal()
    print(f"[{run_time}] Bar: {latest_time} | Close: {close:.2f} | "
          f"Pred: {pred:.6f} | Signal: {signal}")

    # ---- Skip entirely if this bar was already processed -------------------
    # Prevents double-counting bars_held (or firing duplicate entries) if
    # this script runs more often than new bars form - e.g. outside market
    # hours, on weekends, or if the scheduler fires while the market feed
    # hasn't updated yet. Without this, running the script 5x in a row on
    # the SAME bar could fake-advance a position toward its max_hold exit
    # without any real time/price movement actually happening.
    last_processed = state.get("last_processed_bar")
    if last_processed == str(latest_time):
        print(f"  -> Already processed this bar ({latest_time}). No new action.\n")
        sys.exit(0)
    state["last_processed_bar"] = str(latest_time)

    action = "NONE"
    position = state.get("position")

    # ---- Manage existing position -----------------------------------------
    if position is not None:
        position["bars_held"] += 1
        print(f"  In position: {position['direction']} since {position['entry_time']}, "
              f"bars_held={position['bars_held']}/{max_hold}")

        if position["bars_held"] >= max_hold:
            exit_side = OrderSide.SELL if position["direction"] == "LONG" else OrderSide.BUY
            action = f"CLOSE_{position['direction']}"
            print(f"  -> Max hold reached. Closing {position['direction']} position.")

            if args.live:
                submit_order(trading_client, exit_side, qty=1)
                print("  -> Order submitted (LIVE PAPER).")
            else:
                print("  -> [DRY RUN] Would submit close order here.")

            position = None
        state["position"] = position

    # ---- Consider opening a new position -----------------------------------
    elif signal in ("LONG", "SHORT"):
        action = f"OPEN_{signal}"
        side = OrderSide.BUY if signal == "LONG" else OrderSide.SELL
        print(f"  -> Signal fired: {signal}. Opening position.")

        if args.live:
            submit_order(trading_client, side, qty=1)
            print("  -> Order submitted (LIVE PAPER).")
        else:
            print("  -> [DRY RUN] Would submit open order here.")

        state["position"] = {
            "direction": signal,
            "entry_time": str(latest_time),
            "entry_price": close,
            "bars_held": 0,
        }
    else:
        print("  -> No signal, flat, no action.")

    save_state(state)
    log_decision({
        "run_time_utc": run_time,
        "bar_time": str(latest_time),
        "close": close,
        "pred": pred,
        "signal": signal,
        "action": action,
        "live_mode": args.live,
    })
    print(f"  Logged -> {LOG_PATH}\n")
