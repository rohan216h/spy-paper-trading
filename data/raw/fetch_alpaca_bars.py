"""
Phase 1 - Data acquisition.

Pulls raw intraday SPY bars from Alpaca's Market Data API and saves them
untouched (as returned by the API, just written to disk) under data/raw/.

Cleaning, timezone alignment, and session filtering happen in a SEPARATE
step (data/interim/) - keep raw data raw so we can always re-derive
anything downstream without re-hitting the API.

Requires:
  pip install alpaca-py python-dotenv
  A .env file (copied from .env.example) with real ALPACA_API_KEY /
  ALPACA_SECRET_KEY.

NOT YET RUN - waiting on API keys.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import RAW_DIR, SYMBOL

from dotenv import load_dotenv
import os

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def fetch_bars(symbol: str, start: str, end: str, timeframe: TimeFrame):
    """
    Pull raw bars for [start, end) and return the raw DataFrame from Alpaca.
    start/end as 'YYYY-MM-DD' strings.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca API keys. Copy .env.example to .env and fill "
            "in ALPACA_API_KEY / ALPACA_SECRET_KEY."
        )

    client = StockHistoricalDataClient(api_key, secret_key)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed="iex",  # free tier feed
    )
    bars = client.get_stock_bars(request)
    return bars.df


if __name__ == "__main__":
    # First run: figure out how far back the free IEX feed actually goes
    # for 1-minute bars before committing to a full multi-year pull.
    df = fetch_bars(
        symbol=SYMBOL,
        start="2016-01-01",
        end="2026-09-06",
        timeframe=TimeFrame.Minute,
    )
    print(df.shape)
    print(df.head())
    print(df.index.get_level_values("timestamp").min())
    print(df.index.get_level_values("timestamp").max())

    out_path = RAW_DIR / f"{SYMBOL}_1min_raw.parquet"
    df.to_parquet(out_path)
    print(f"Saved -> {out_path}")
