"""
THE FINAL CHECK - run once, no tuning after seeing the result.

Extends the walk-forward all the way through the full dataset (instead of
stopping at VAL_END), then evaluates ONLY on the portion after
2025-02-14 - the period no decision, threshold, or feature choice has
ever been influenced by. Configuration is FROZEN at what we already
committed to: Linear Regression, no VIX, threshold=1.0x prediction std,
12-bar hold, 1bps cost. Nothing gets adjusted based on this output.

Run with:  python strategies/final_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import INSTRUMENTS_DIR, SYMBOL, BAR_SIZE

import pandas as pd
from scipy.stats import spearmanr

from strategies.ml import (
    walk_forward, prepare_dataset, FEATURE_COLS, LABEL_COL, MODEL_TYPE
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from engine.backtest_engine import run_backtest, compute_metrics, print_metrics
from strategies.run_strategy_b import check_signal_ml, check_exit_ml

FROZEN_THRESHOLDS = [1.0, 2.0]  # report BOTH, honestly, per prior discussion -
                                  # 1.0x was the original non-cherry-picked choice,
                                  # 2.0x looked best in the noisy sensitivity scan.
                                  # Neither gets silently dropped based on outcome.
FROZEN_MAX_HOLD = 12
FROZEN_COST_BPS = 1.0
TRUE_TEST_START = "2025-02-14"

if __name__ == "__main__":
    print(f"MODEL: {MODEL_TYPE} (no VIX) | FROZEN CONFIG: "
          f"thresholds={FROZEN_THRESHOLDS}x, hold={FROZEN_MAX_HOLD} bars, "
          f"cost={FROZEN_COST_BPS}bps\n")

    path = INSTRUMENTS_DIR / f"{SYMBOL}_{BAR_SIZE}_features.parquet"
    df = pd.read_parquet(path).sort_index()
    df = prepare_dataset(df)

    print("Running walk-forward through the FULL dataset (previous runs stopped at VAL_END)...")
    df_out, reports = walk_forward(df, stop_at=None)

    test_mask = df_out.index >= pd.Timestamp(TRUE_TEST_START, tz=df_out.index.tz)
    test_preds = df_out.loc[test_mask].dropna(subset=["pred", LABEL_COL])

    print(f"\nTrue held-out test period: {TRUE_TEST_START} onward")
    print(f"OOS predictions in this period: {len(test_preds):,}")

    ic, p = spearmanr(test_preds["pred"], test_preds[LABEL_COL])
    print(f"IC on TRUE test period: {ic:.4f} (p={p:.4g})")

    pred_std = df_out.loc[df_out.index < pd.Timestamp(TRUE_TEST_START, tz=df_out.index.tz), "pred"].std()
    print(f"(Using prediction std from BEFORE the test period, {pred_std:.6f}, "
          f"to set thresholds - not computed on test data itself)")

    bt_df = df_out.loc[test_mask].dropna(subset=["pred"])

    for mult in FROZEN_THRESHOLDS:
        params = {
            "threshold": pred_std * mult,
            "max_hold_bars": FROZEN_MAX_HOLD,
            "cost_bps": FROZEN_COST_BPS,
            "skip_open_window": True, "skip_close_window": True, "skip_anomaly": True,
            "skip_regime": None, "entry_on_next": True, "enter_immediately": True,
        }
        sig, trd, eq = run_backtest(bt_df, params, check_signal_ml, check_exit_ml,
                                     instrument_name=f"FINAL TEST - threshold={mult}x")
        m = compute_metrics(trd, sig, eq, f"FINAL TEST - threshold={mult}x")
        print_metrics(m)

    print(f"\n{'='*60}")
    print("BOTH results above are final and reported together. Neither")
    print("gets selected as 'the' result after the fact based on which")
    print("looks better - that would reintroduce the cherry-picking this")
    print("test was designed to avoid.")
    print(f"{'='*60}")
