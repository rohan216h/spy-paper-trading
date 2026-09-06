"""
Phase 9, step 1 - Train and save the FINAL model artifact.

For live/paper trading we retrain on ALL available historical data (not
walk-forward - that was for honest backtesting; in production you always
want the model trained on everything you have up to today). Frozen
config, matching what passed the final test:
  - Linear Regression
  - No VIX features
  - Label: fwd_ret_12 (12 bars = 1hr ahead at 5-min bars)
  - Threshold: 1.0x the prediction std (the non-cherry-picked choice)
  - Max hold: 12 bars

Saves:
  paper_trading/model.joblib      - the fitted sklearn model
  paper_trading/model_meta.json   - feature list, threshold, hold period,
                                     training date, training data range
                                     (so we always know exactly what's
                                     deployed and when it was last trained)

Run with:  python paper_trading/train_final_model.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import INSTRUMENTS_DIR, SYMBOL, BAR_SIZE

import json
import joblib
import pandas as pd
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression

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
LABEL_COL = "fwd_ret_12"
THRESHOLD_MULT = 1.0
MAX_HOLD_BARS = 12
COST_BPS = 1.0

MODEL_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    path = INSTRUMENTS_DIR / f"{SYMBOL}_{BAR_SIZE}_features.parquet"
    df = pd.read_parquet(path).sort_index()
    df["is_open_window"] = df["is_open_window"].astype(int)
    df["is_close_window"] = df["is_close_window"].astype(int)
    data = df.dropna(subset=FEATURE_COLS + [LABEL_COL])

    print(f"Training final model on ALL {len(data):,} available bars "
          f"({data.index.min()} to {data.index.max()})")

    model = LinearRegression()
    model.fit(data[FEATURE_COLS], data[LABEL_COL])

    pred_std = model.predict(data[FEATURE_COLS]).std()
    print(f"In-sample prediction std: {pred_std:.6f} -> "
          f"live threshold = {pred_std * THRESHOLD_MULT:.6f}")

    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)

    meta = {
        "symbol": SYMBOL,
        "bar_size": BAR_SIZE,
        "feature_cols": FEATURE_COLS,
        "label_col": LABEL_COL,
        "threshold_mult": THRESHOLD_MULT,
        "threshold_value": pred_std * THRESHOLD_MULT,
        "pred_std": pred_std,
        "max_hold_bars": MAX_HOLD_BARS,
        "cost_bps": COST_BPS,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "trained_on_data_through": str(data.index.max()),
        "n_training_bars": len(data),
    }
    meta_path = MODEL_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model -> {model_path}")
    print(f"Saved metadata -> {meta_path}")
    print(f"\nCoefficients:")
    coefs = pd.Series(model.coef_, index=FEATURE_COLS).sort_values()
    print(coefs.to_string())
