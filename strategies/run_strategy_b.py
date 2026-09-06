"""
Converts Strategy B (ML) predictions into an actual backtested strategy,
through the SAME engine Strategy A used. Position held for a FIXED
horizon matching the label (12 bars = 1hr), since that's what the model
predicts - a retracement-based exit (like Strategy A) wouldn't make
sense here.

Signal: LONG if predicted fwd_ret > +threshold, SHORT if < -threshold,
flat otherwise. Threshold is a free parameter - we scan a few values
since "how confident must the model be before we act" is a real design
choice, not something to hardcode blindly.

Run with:  python strategies/run_strategy_b.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import INSTRUMENTS_DIR, SYMBOL, BAR_SIZE
from strategies.ml import LABEL_COL, MODEL_TYPE

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from engine.backtest_engine import run_backtest, compute_metrics, print_metrics


def check_signal_ml(row, params):
    pred = row["pred"]
    if pd.isna(pred):
        return None
    if pred > params["threshold"]:
        return "LONG"
    if pred < -params["threshold"]:
        return "SHORT"
    return None


def check_exit_ml(direction, row, bars_held, params):
    if bars_held >= params["max_hold_bars"]:
        return "TIME"
    return None


if __name__ == "__main__":
    feat_path = INSTRUMENTS_DIR / f"{SYMBOL}_{BAR_SIZE}_features.parquet"
    pred_path = Path(__file__).resolve().parent.parent / "notebooks" / "phase5_output" / f"predictions_{MODEL_TYPE}.parquet"

    features = pd.read_parquet(feat_path).sort_index()
    preds = pd.read_parquet(pred_path)[["pred"]]

    df = features.join(preds, how="inner")
    df = df.dropna(subset=["pred"])
    print(f"Backtesting {len(df):,} bars with OOS predictions ({MODEL_TYPE})")
    print(f"Date range: {df.index.min()} to {df.index.max()}\n")

    pred_std = df["pred"].std()
    print(f"Prediction std: {pred_std:.6f} (used to scale thresholds)\n")

    for mult in [0.5, 1.0, 1.5, 2.0]:
        threshold = pred_std * mult
        params = {
            "threshold": threshold,
            "max_hold_bars": 12,   # matches fwd_ret_12 label horizon
            "cost_bps": 1.0,
            "skip_open_window": True,
            "skip_close_window": True,
            "skip_anomaly": True,
            "skip_regime": None,
            "entry_on_next": True,
            "enter_immediately": True,
        }
        sig_log, trd_log, equity = run_backtest(
            df, params, check_signal_ml, check_exit_ml,
            instrument_name=f"{SYMBOL}_ml_{MODEL_TYPE}_thresh{mult}",
        )
        m = compute_metrics(trd_log, sig_log, equity, f"threshold={mult}x std ({threshold:.6f})")
        print_metrics(m)
