"""
Phase 5 - Strategy B: ML model, walk-forward.

Label: fwd_ret_H (continuous forward log-return over H bars), chosen
instead of binary direction since we want position sizing/confidence
proportional to predicted magnitude, not just a direction.

Walk-forward scheme (NOT a single train/test split - a single split
would let the model implicitly "see" one future regime while training,
which defeats the point):
  - Initial training window: first N bars (~2 years)
  - Retrain on an EXPANDING window every `retrain_every_bars` bars
    (~1 quarter), predict the NEXT quarter out-of-sample, then expand
    the training window to include that quarter and repeat.
  - This produces genuine out-of-sample predictions across the whole
    TRAIN+VAL history. The true TEST split (after configs.VAL_END)
    stays completely untouched until Phase 7/8.

This file is model-agnostic: swap `build_model()` to change between
Linear Regression and LightGBM without touching the walk-forward loop,
feature prep, or evaluation code below.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import INSTRUMENTS_DIR, SYMBOL, BAR_SIZE, TRAIN_END, VAL_END

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODEL_TYPE = "linear"   # "linear" or "lightgbm" - switch this to change model
LABEL_COL = "fwd_ret_12"  # 12 bars * 5min = 1 hour ahead
USE_VIX = False  # dropped - testing showed it didn't help (see conversation)

FEATURE_COLS = [
    "ret_1",
    "roll_ret_3", "roll_ret_z_3",
    "roll_ret_6", "roll_ret_z_6",
    "roll_ret_12", "roll_ret_z_12",
    "roll_ret_24", "roll_ret_z_24",
    "realized_vol_12", "realized_vol_48", "realized_vol_288",
    "volume_z",
    "minutes_since_open", "minutes_to_close", "dow",
]
if USE_VIX:
    FEATURE_COLS += ["vix_level", "vix_chg_1d", "vix_zscore"]

INITIAL_TRAIN_BARS = 78 * 252 * 2       # ~2 years of 5-min bars
RETRAIN_EVERY_BARS = 78 * 63            # ~1 quarter


def build_model():
    if MODEL_TYPE == "linear":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    elif MODEL_TYPE == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


def prepare_dataset(df: pd.DataFrame):
    data = df.dropna(subset=FEATURE_COLS + [LABEL_COL]).copy()
    data["is_open_window"] = data["is_open_window"].astype(int)
    data["is_close_window"] = data["is_close_window"].astype(int)
    return data


def walk_forward(df: pd.DataFrame, stop_at: str = None):
    """
    Returns a DataFrame of out-of-sample predictions indexed the same as
    df, covering everything from INITIAL_TRAIN_BARS onward through
    `stop_at` (defaults to VAL_END - used during model development).
    Pass stop_at=None (or a date past the end of the data) to walk all
    the way through the true held-out test period - only do this ONCE,
    as the final honest check, not as another round of tuning.
    """
    n = len(df)
    preds = np.full(n, np.nan)
    window_reports = []
    cutoff = pd.Timestamp(stop_at, tz=df.index.tz) if stop_at else None

    start = INITIAL_TRAIN_BARS
    while start < n:
        train_end_idx = start
        test_end_idx = min(start + RETRAIN_EVERY_BARS, n)

        train = df.iloc[:train_end_idx]
        test = df.iloc[train_end_idx:test_end_idx]

        if len(test) == 0:
            break

        if cutoff is not None and train.index[-1] >= cutoff:
            break

        model = build_model()
        model.fit(train[FEATURE_COLS], train[LABEL_COL])
        pred = model.predict(test[FEATURE_COLS])
        preds[train_end_idx:test_end_idx] = pred

        ic, _ = spearmanr(pred, test[LABEL_COL])
        window_reports.append({
            "train_end": train.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "n_test": len(test),
            "ic": ic,
        })
        print(f"  Trained on {len(train):,} bars (up to {train.index[-1]}), "
              f"predicted {len(test):,} bars ({test.index[0]} to {test.index[-1]}), "
              f"IC={ic:.4f}")

        start += RETRAIN_EVERY_BARS

    df_out = df.copy()
    df_out["pred"] = preds
    return df_out, pd.DataFrame(window_reports)


if __name__ == "__main__":
    suffix = "_vix" if USE_VIX else ""
    path = INSTRUMENTS_DIR / f"{SYMBOL}_{BAR_SIZE}_features{suffix}.parquet"
    df = pd.read_parquet(path).sort_index()
    df = prepare_dataset(df)

    print(f"Dataset: {len(df):,} rows, model = {MODEL_TYPE}, label = {LABEL_COL}, "
          f"vix_features = {USE_VIX}")
    print(f"Initial train window: {INITIAL_TRAIN_BARS:,} bars, "
          f"retrain every {RETRAIN_EVERY_BARS:,} bars\n")

    df_out, reports = walk_forward(df)

    valid = df_out.dropna(subset=["pred", LABEL_COL])
    overall_ic, overall_p = spearmanr(valid["pred"], valid[LABEL_COL])

    print(f"\n{'='*55}")
    print(f"  OVERALL WALK-FORWARD RESULT ({MODEL_TYPE})")
    print(f"{'='*55}")
    print(f"  OOS predictions made : {len(valid):,}")
    print(f"  Overall IC (Spearman): {overall_ic:.4f}  (p={overall_p:.4g})")
    print(f"  Mean per-window IC   : {reports['ic'].mean():.4f}")
    print(f"  Windows with IC > 0  : {(reports['ic'] > 0).mean()*100:.1f}%")

    # If linear, coefficients are directly interpretable - show them.
    # If LightGBM, show feature importance instead (same diagnostic purpose).
    if MODEL_TYPE == "linear":
        final_model = build_model()
        final_model.fit(df.iloc[:INITIAL_TRAIN_BARS][FEATURE_COLS],
                         df.iloc[:INITIAL_TRAIN_BARS][LABEL_COL])
        coefs = pd.Series(final_model.coef_, index=FEATURE_COLS).sort_values()
        print(f"\n  Coefficients (from initial training window):")
        print(coefs.to_string())
    elif MODEL_TYPE == "lightgbm":
        final_model = build_model()
        final_model.fit(df.iloc[:INITIAL_TRAIN_BARS][FEATURE_COLS],
                         df.iloc[:INITIAL_TRAIN_BARS][LABEL_COL])
        importance = pd.Series(final_model.feature_importances_, index=FEATURE_COLS).sort_values()
        print(f"\n  Feature importance (from initial training window):")
        print(importance.to_string())

    out_dir = Path(__file__).resolve().parent.parent / "notebooks" / "phase5_output"
    out_dir.mkdir(exist_ok=True)
    model_tag = f"{MODEL_TYPE}{suffix}"
    df_out[["close", "pred", LABEL_COL]].to_parquet(out_dir / f"predictions_{model_tag}.parquet")
    reports.to_csv(out_dir / f"window_ic_{model_tag}.csv", index=False)
    print(f"\nSaved predictions -> {out_dir / f'predictions_{model_tag}.parquet'}")
    print(f"Saved window IC report -> {out_dir / f'window_ic_{model_tag}.csv'}")
