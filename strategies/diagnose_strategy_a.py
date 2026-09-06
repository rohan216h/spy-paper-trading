"""
Quick, cheap diagnostic on the Strategy A trade log we already generated -
no new backtest run needed. Checks whether the (currently negative)
reversal edge varies meaningfully by vol regime or time-of-day. This is
diagnostic only, not a new tuned strategy - informs whether Phase 5 (ML)
should weight these as features, and whether Strategy A's "skip_regime"
option is worth using.

Run with:  python strategies/diagnose_strategy_a.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

trade_log_path = Path(__file__).resolve().parent.parent / "notebooks" / "phase4_output" / "strategy_a_train_trades.csv"
tl = pd.read_csv(trade_log_path)
tl["entry_time"] = pd.to_datetime(tl["entry_time"], utc=True)
tl["exit_time"] = pd.to_datetime(tl["exit_time"], utc=True)

print(f"Total trades: {len(tl)}\n")

print("="*55)
print("BY VOL REGIME")
print("="*55)
by_regime = tl.groupby("regime").agg(
    n=("pnl", "count"),
    win_rate=("pnl", lambda x: (x > 0).mean() * 100),
    avg_pnl=("pnl", "mean"),
    total_pnl=("pnl", "sum"),
)
print(by_regime)

print("\n" + "="*55)
print("BY ENTRY HOUR")
print("="*55)
tl["entry_hour"] = tl["entry_time"].dt.tz_convert("US/Eastern").dt.hour
by_hour = tl.groupby("entry_hour").agg(
    n=("pnl", "count"),
    win_rate=("pnl", lambda x: (x > 0).mean() * 100),
    avg_pnl=("pnl", "mean"),
    total_pnl=("pnl", "sum"),
)
print(by_hour)

print("\n" + "="*55)
print("BY DIRECTION")
print("="*55)
by_dir = tl.groupby("direction").agg(
    n=("pnl", "count"),
    win_rate=("pnl", lambda x: (x > 0).mean() * 100),
    avg_pnl=("pnl", "mean"),
    total_pnl=("pnl", "sum"),
)
print(by_dir)
