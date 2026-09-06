"""
Merges VIX-derived features onto the SPY 5-min feature set.

Critical point-in-time rule: VIX closes once per day, after the fact.
Today's VIX close isn't known until today's session has ended. So every
bar in TODAY's session uses YESTERDAY's VIX close - that's the most
recent VIX value actually available at any point while today is trading.
Using today's own VIX close as a same-day feature would be lookahead
bias (exactly the trap flagged in the original roadmap: "a daily macro
feature known only after market close can't be used intraday same-day").

New features added:
  vix_level       - yesterday's VIX close
  vix_chg_1d      - yesterday's VIX close minus the day before (1-day change)
  vix_zscore      - yesterday's VIX close, z-scored vs its own trailing 60-day mean/std
  vix_regime      - HIGH_VIX / LOW_VIX vs trailing median (a market-wide fear
                     regime, distinct from the existing realized-vol regime
                     which only looks at SPY's own recent price action)

Run with:  python features/merge_vix_features.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import RAW_DIR, INSTRUMENTS_DIR, SYMBOL, BAR_SIZE, TIMEZONE

import numpy as np
import pandas as pd


if __name__ == "__main__":
    # ---- Load VIX, build lagged daily features ---------------------------
    vix = pd.read_parquet(RAW_DIR / "VIX_daily_raw.parquet")
    vix.index = pd.to_datetime(vix.index)
    vix = vix.sort_index()

    vix_feat = pd.DataFrame(index=vix.index)
    vix_feat["vix_level_raw"] = vix["close"]
    vix_feat["vix_chg_1d_raw"] = vix["close"].diff()
    roll_mean = vix["close"].rolling(60).mean()
    roll_std = vix["close"].rolling(60).std()
    vix_feat["vix_zscore_raw"] = (vix["close"] - roll_mean) / roll_std
    vix_median = vix["close"].rolling(60).median()
    vix_feat["vix_regime_raw"] = np.where(vix["close"] > vix_median, "HIGH_VIX", "LOW_VIX")

    # LAG BY 1 DAY - this is the point-in-time fix. Shifted values are what
    # today's session actually has available.
    vix_feat_lagged = vix_feat.shift(1)
    vix_feat_lagged.columns = ["vix_level", "vix_chg_1d", "vix_zscore", "vix_regime"]
    vix_feat_lagged = vix_feat_lagged.dropna(subset=["vix_level"])

    print(f"VIX lagged features: {len(vix_feat_lagged):,} days, "
          f"{vix_feat_lagged.index.min().date()} to {vix_feat_lagged.index.max().date()}")

    # ---- Load SPY features, join by calendar date -------------------------
    spy_path = INSTRUMENTS_DIR / f"{SYMBOL}_{BAR_SIZE}_features.parquet"
    spy = pd.read_parquet(spy_path).sort_index()

    spy["_date"] = spy.index.tz_convert(TIMEZONE).normalize().tz_localize(None)
    vix_feat_lagged.index = vix_feat_lagged.index.tz_localize(None)

    merged = spy.merge(vix_feat_lagged, left_on="_date", right_index=True, how="left")
    merged = merged.drop(columns=["_date"])

    n_before = len(merged)
    merged = merged.dropna(subset=["vix_level"])
    print(f"Dropped {n_before - len(merged):,} rows with no VIX match "
          f"(e.g. dates before VIX history starts)")

    out_path = INSTRUMENTS_DIR / f"{SYMBOL}_{BAR_SIZE}_features_vix.parquet"
    merged.to_parquet(out_path)
    print(f"Saved -> {out_path}")
    print(f"Final shape: {merged.shape}")
    print(f"\nSample of new columns:")
    print(merged[["close", "vix_level", "vix_chg_1d", "vix_zscore", "vix_regime"]].tail())
