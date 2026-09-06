"""
Phase 9, step 2 - DRY RUN live signal generation. Places NO orders yet -
this only proves the live pipeline reproduces the exact same features
the model was trained on, using freshly pulled data. Get this right
before touching order placement, even paper orders.

Pulls the last ~15 trading days of 1-min bars (comfortably more than
the 288-bar (~1 day) longest rolling window needs), replicates the same
cleaning -> resample -> feature steps as the offline pipeline, takes the
LATEST fully-formed 5-min bar, and runs it through the saved model.

Run with:  python paper_trading/generate_signal.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import SYMBOL, TIMEZONE

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone as dt_timezone

from dotenv import load_dotenv
import os
load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

MODEL_DIR = Path(__file__).resolve().parent


# ---- Reuse the SAME cleaning/feature logic as the offline pipeline --------
# (kept inline here, not imported, since the offline scripts assume a
# parquet file on disk - this operates on a freshly pulled DataFrame
# instead. Logic must stay IDENTICAL to data/interim/clean_and_resample.py
# and features/build_features.py or live predictions won't match backtest.)

def to_session_hours(df, tz):
    df = df.copy()
    df.index = df.index.tz_convert(tz)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    hour_min = df.index.hour * 60 + df.index.minute
    mask = (hour_min >= 9 * 60 + 30) & (hour_min < 16 * 60)
    df = df[mask]
    df = df[df.index.dayofweek < 5]
    return df


def resample_ohlcv(df, rule="5min"):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "trade_count": "sum"}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    out = []
    for _, day_df in df.groupby(df.index.date):
        r = day_df.resample(rule, label="left", closed="left").agg(agg)
        out.append(r)
    result = pd.concat(out)
    return result.dropna(subset=["close"])


def add_features(df):
    df = df.copy()
    df["log_price"] = np.log(df["close"])
    df["ret_1"] = df["log_price"].diff()

    for w in (3, 6, 12, 24):
        df[f"roll_ret_{w}"] = df["log_price"].diff(w)
        roll_mean = df[f"roll_ret_{w}"].rolling(500).mean()
        roll_std = df[f"roll_ret_{w}"].rolling(500).std()
        df[f"roll_ret_z_{w}"] = (df[f"roll_ret_{w}"] - roll_mean) / roll_std

    for w in (12, 48, 288):
        df[f"realized_vol_{w}"] = df["ret_1"].rolling(w).std()

    roll_mean_v = df["volume"].rolling(288).mean()
    roll_std_v = df["volume"].rolling(288).std()
    df["volume_z"] = (df["volume"] - roll_mean_v) / roll_std_v

    idx = df.index
    df["minute_of_day"] = idx.hour * 60 + idx.minute
    df["minutes_since_open"] = df["minute_of_day"] - (9 * 60 + 30)
    df["minutes_to_close"] = (16 * 60) - df["minute_of_day"]
    df["dow"] = idx.dayofweek
    return df


def pull_recent_bars(symbol, lookback_days=15):
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(api_key, secret_key)

    end = datetime.now(dt_timezone.utc)
    start = end - timedelta(days=lookback_days)
    request = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
        start=start, end=end, feed="iex",
    )
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index(level="symbol", drop=True)
    return df


def get_latest_signal():
    """Returns (latest_time, close_price, pred, signal, threshold, max_hold_bars)."""
    model = joblib.load(MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "model_meta.json") as f:
        meta = json.load(f)

    raw = pull_recent_bars(SYMBOL)
    clean = to_session_hours(raw, TIMEZONE)
    resampled = resample_ohlcv(clean, "5min")
    featured = add_features(resampled)

    valid = featured.dropna(subset=meta["feature_cols"])
    latest = valid.iloc[-1]
    latest_time = valid.index[-1]

    X = pd.DataFrame([latest[meta["feature_cols"]]], columns=meta["feature_cols"]).astype(float)
    pred = model.predict(X)[0]

    threshold = meta["threshold_value"]
    if pred > threshold:
        signal = "LONG"
    elif pred < -threshold:
        signal = "SHORT"
    else:
        signal = "FLAT"

    return latest_time, latest["close"], pred, signal, threshold, meta["max_hold_bars"]


if __name__ == "__main__":
    latest_time, close, pred, signal, threshold, max_hold = get_latest_signal()
    print(f"Latest usable bar: {latest_time}")
    print(f"Close price: {close:.2f}")
    print(f"Predicted fwd_ret_12 (next ~1hr): {pred:.6f}")
    print(f"Threshold: +/-{threshold:.6f}")
    print(f"SIGNAL: {signal}")
    print(f"\n[DRY RUN - no order placed. This only checks the live pipeline works.]")
