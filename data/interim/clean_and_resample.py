"""
Phase 1 - Data cleaning.

Reads the raw 1-min SPY parquet pulled from Alpaca (data/raw/SPY_1min_raw.parquet)
and produces a clean, resampled 5-min bar file in data/interim/.

Steps:
  1. Load raw parquet (multi-index: symbol, timestamp - UTC)
  2. Drop the symbol level (single-symbol file, don't need it)
  3. Convert timestamp index to US/Eastern (NYSE-native timezone)
  4. Filter to regular session hours only (09:30-16:00 ET) - drops any
     pre/post-market bars IEX might include
  5. Resample 1-min -> 5-min OHLCV (Open=first, High=max, Low=min,
     Close=last, Volume=sum) - standard aggregation, same convention as
     your old FF/SRA resampling
  6. Drop any bar with no trades (NaN close) - holidays/gaps
  7. Save to data/interim/SPY_5min_clean.parquet

Run with:  python data/interim/clean_and_resample.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import RAW_DIR, INTERIM_DIR, SYMBOL, TIMEZONE, BAR_SIZE

import pandas as pd


def load_raw(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / f"{symbol}_1min_raw.parquet"
    df = pd.read_parquet(path)
    # multi-index (symbol, timestamp) -> drop symbol level
    df = df.reset_index(level="symbol", drop=True)
    return df


def to_session_hours(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.tz_convert(tz)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    hour_min = df.index.hour * 60 + df.index.minute
    session_start = 9 * 60 + 30   # 09:30
    session_end   = 16 * 60       # 16:00
    mask = (hour_min >= session_start) & (hour_min < session_end)
    df = df[mask]

    # No weekend session for equities, but drop just in case of bad data
    df = df[df.index.dayofweek < 5]
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "trade_count": "sum",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}

    # Resample per calendar day so we never bleed bars across the
    # overnight gap (group by date, resample within each day, concat)
    out = []
    for _, day_df in df.groupby(df.index.date):
        r = day_df.resample(rule, label="left", closed="left").agg(agg)
        out.append(r)
    result = pd.concat(out)
    result = result.dropna(subset=["close"])
    return result


if __name__ == "__main__":
    print(f"Loading raw {SYMBOL} 1-min data...")
    df = load_raw(SYMBOL)
    print(f"  Raw rows: {len(df):,}")

    print(f"Converting to {TIMEZONE}, filtering to regular session hours...")
    df = to_session_hours(df, TIMEZONE)
    print(f"  After session filter: {len(df):,} rows")

    print(f"Resampling 1-min -> {BAR_SIZE}...")
    df_resampled = resample_ohlcv(df, BAR_SIZE)
    print(f"  After resample: {len(df_resampled):,} rows")

    out_path = INTERIM_DIR / f"{SYMBOL}_{BAR_SIZE}_clean.parquet"
    df_resampled.to_parquet(out_path)
    print(f"Saved -> {out_path}")

    print("\nSanity check:")
    print(df_resampled.head())
    print("...")
    print(df_resampled.tail())
    print(f"\nDate range: {df_resampled.index.min()} to {df_resampled.index.max()}")
