"""
Phase 3 - Feature engineering.

Builds point-in-time features on top of the clean 5-min SPY bars, designed
around what Phase 2 EDA actually found:
  - No linear autocorrelation in raw returns -> a plain price z-score
    (the FF/SRA approach) has nothing to exploit here.
  - Known equity effect worth testing instead: short-term REVERSAL after
    unusually large moves (not the same as mean-reversion in price level).
  - Strong, clean intraday seasonality (vol/volume U-shape) -> time-of-day
    features and a session filter matter a lot.
  - Vol regime materially changes tail behavior -> keep a regime flag for
    position sizing / filtering later.

Every feature here only uses PAST data (rolling/trailing windows) - no
lookahead. This is the shared feature set both Strategy A (rules) and
Strategy B (ML) will read from.

Reads:  data/interim/SPY_5min_clean.parquet
Writes: data/instruments/SPY_5min_features.parquet

Run with:  python features/build_features.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import INTERIM_DIR, INSTRUMENTS_DIR, SYMBOL, BAR_SIZE

import numpy as np
import pandas as pd


# ── Individual feature blocks ──────────────────────────────────────
def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["log_price"] = np.log(df["close"])
    df["ret_1"] = df["log_price"].diff()
    return df


def add_reversal_features(df: pd.DataFrame, windows=(3, 6, 12, 24)) -> pd.DataFrame:
    """
    Rolling N-bar return, and its z-score relative to its own trailing
    distribution. Large |z| = an unusually big recent move -> the thing
    a reversal strategy would react to.
    """
    for w in windows:
        col_ret = f"roll_ret_{w}"
        df[col_ret] = df["log_price"].diff(w)

        roll_mean = df[col_ret].rolling(500).mean()
        roll_std = df[col_ret].rolling(500).std()
        df[f"roll_ret_z_{w}"] = (df[col_ret] - roll_mean) / roll_std
    return df


def add_volatility_features(df: pd.DataFrame, windows=(12, 48, 288)) -> pd.DataFrame:
    """Realized vol at multiple lookbacks (1hr, 4hr, ~1 day at 5-min bars)."""
    for w in windows:
        df[f"realized_vol_{w}"] = df["ret_1"].rolling(w).std()

    # Vol regime: today's rolling daily vol vs its own trailing median
    daily_vol_proxy = df["realized_vol_288"]
    vol_median = daily_vol_proxy.rolling(500).median()
    df["vol_regime"] = np.where(daily_vol_proxy > vol_median, "HIGH_VOL", "LOW_VOL")
    return df


def add_volume_features(df: pd.DataFrame, window=288) -> pd.DataFrame:
    roll_mean = df["volume"].rolling(window).mean()
    roll_std = df["volume"].rolling(window).std()
    df["volume_z"] = (df["volume"] - roll_mean) / roll_std
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["minute_of_day"] = idx.hour * 60 + idx.minute
    minutes_since_open = df["minute_of_day"] - (9 * 60 + 30)
    df["minutes_since_open"] = minutes_since_open
    df["minutes_to_close"] = (16 * 60) - df["minute_of_day"]
    df["dow"] = idx.dayofweek  # Mon=0 ... Fri=4

    # First/last 15 min flag - the seasonality spike zone from Phase 2
    df["is_open_window"] = df["minutes_since_open"] <= 15
    df["is_close_window"] = df["minutes_to_close"] <= 15
    return df


def add_anomaly_flag(df: pd.DataFrame, z_thresh=6.0) -> pd.DataFrame:
    """
    Flags extreme single-bar returns (likely bad ticks / halts / flash
    moves) using a trailing z-score - same role as the anomaly_flag in
    the old FF/SRA instrument builder. These bars get excluded from
    training/backtest tradeable-bar checks, not deleted from the data.
    """
    ret_std = df["ret_1"].rolling(288).std()
    ret_z = (df["ret_1"] / ret_std).abs()
    raw_flag = ret_z > z_thresh
    df["anomaly_flag"] = raw_flag.rolling(3, center=True, min_periods=1).max().fillna(0).astype(bool)
    return df


def add_forward_labels(df: pd.DataFrame, horizons=(6, 12, 24)) -> pd.DataFrame:
    """
    Forward returns for ML labeling (Phase 5). NOT used by Strategy A.
    These look FORWARD on purpose (labels only, never used as a feature
    at prediction time) - kept in the same file for convenience, but
    any code building "features seen at time t" must exclude these
    columns explicitly.
    """
    for h in horizons:
        df[f"fwd_ret_{h}"] = df["log_price"].shift(-h) - df["log_price"]
    return df


if __name__ == "__main__":
    path = INTERIM_DIR / f"{SYMBOL}_{BAR_SIZE}_clean.parquet"
    df = pd.read_parquet(path)
    df = df.sort_index()
    print(f"Loaded {len(df):,} clean bars")

    df = add_returns(df)
    df = add_reversal_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_time_features(df)
    df = add_anomaly_flag(df)
    df = add_forward_labels(df)

    n_before = len(df)
    # Drop warmup rows where rolling windows aren't full yet (NaN features)
    feature_cols = [c for c in df.columns if not c.startswith("fwd_ret_")]
    df_clean = df.dropna(subset=[c for c in feature_cols if c not in
                                  ("open", "high", "low", "close", "volume", "trade_count")])
    print(f"Dropped {n_before - len(df_clean):,} warmup rows (rolling windows not full yet)")

    out_path = INSTRUMENTS_DIR / f"{SYMBOL}_{BAR_SIZE}_features.parquet"
    df_clean.to_parquet(out_path)
    print(f"Saved -> {out_path}")
    print(f"Final shape: {df_clean.shape}")
    print(f"\nColumns:\n{list(df_clean.columns)}")
    print(f"\nAnomaly bars flagged: {df_clean['anomaly_flag'].sum():,} "
          f"({df_clean['anomaly_flag'].mean()*100:.2f}%)")
    print(f"\nSample:\n{df_clean[['close', 'ret_1', 'roll_ret_z_12', 'realized_vol_288', 'vol_regime', 'anomaly_flag']].tail()}")
