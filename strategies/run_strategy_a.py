"""
Phase 4 runner - runs Strategy A (reversal rules) through the generalized
engine, on the TRAIN split only (per configs.config.TRAIN_END). Test data
stays untouched until Phase 7/8 - no peeking while we're still designing
the strategy.

Run with:  python strategies/run_strategy_a.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import INSTRUMENTS_DIR, SYMBOL, BAR_SIZE, TRAIN_END

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from engine.backtest_engine import run_backtest, compute_metrics, print_metrics
from strategies.rules import PARAMS, check_signal, check_exit


if __name__ == "__main__":
    path = INSTRUMENTS_DIR / f"{SYMBOL}_{BAR_SIZE}_features.parquet"
    df = pd.read_parquet(path).sort_index()

    train_df = df.loc[:TRAIN_END].copy()
    print(f"Full data : {len(df):,} bars ({df.index.min()} to {df.index.max()})")
    print(f"TRAIN split (<= {TRAIN_END}): {len(train_df):,} bars "
          f"({train_df.index.min()} to {train_df.index.max()})")

    print(f"\nStrategy A params: {PARAMS}\n")

    sig_log, trd_log, equity = run_backtest(
        train_df, PARAMS, check_signal, check_exit, instrument_name=f"{SYMBOL}_reversal"
    )
    m = compute_metrics(trd_log, sig_log, equity, f"{SYMBOL}_reversal")
    print_metrics(m)

    # Gut-check: is the negative result purely a cost problem, or real?
    if not trd_log.empty:
        gross_pnl = trd_log["pnl"] + trd_log["entry_price"] * (PARAMS["cost_bps"] / 10000.0)
        print(f"\n  [Cost check] Gross PnL (before costs): {gross_pnl.sum():.2f}"
              f"  |  Total cost paid: {(trd_log['entry_price'] * (PARAMS['cost_bps']/10000.0)).sum():.2f}"
              f"  |  Net PnL: {trd_log['pnl'].sum():.2f}")

    if not trd_log.empty:
        out_dir = Path(__file__).resolve().parent.parent / "notebooks" / "phase4_output"
        out_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        axes[0].plot(equity.index, equity.values, color="steelblue", lw=0.8)
        axes[0].axhline(0, color="black", lw=0.6, linestyle="--")
        axes[0].set_title(f"Equity Curve - Strategy A (Reversal) - {SYMBOL} TRAIN")
        axes[0].set_ylabel("Cumulative $/share")

        dd = equity - equity.cummax()
        axes[1].fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.5)
        axes[1].set_title("Drawdown")

        pnl = trd_log["pnl"]
        axes[2].hist(pnl[pnl >= 0], bins=60, color="seagreen", alpha=0.7, label="Winners")
        axes[2].hist(pnl[pnl < 0], bins=60, color="tomato", alpha=0.7, label="Losers")
        axes[2].axvline(0, color="black", lw=1)
        axes[2].set_title("PnL Distribution per Trade")
        axes[2].legend()

        plt.tight_layout()
        out_path = out_dir / "strategy_a_train_results.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"\nSaved plot -> {out_path}")

        trd_log.to_csv(out_dir / "strategy_a_train_trades.csv", index=False)
        print(f"Saved trade log -> {out_dir / 'strategy_a_train_trades.csv'}")
    else:
        print("\nNo trades generated - check thresholds in strategies/rules.py")
