"""
Pulls daily VIX (^VIX) history via yfinance - a free, standard source for
this cash index (VIX isn't a tradable stock, so it's not on Alpaca's
stock bars endpoint).

VIX is a DAILY close - today's value isn't known until after today's
session ends. This file saves the raw daily series; the 1-day lag
(to avoid lookahead) is applied later in merge_vix_features.py, not here,
so this raw file stays an honest, unmodified copy of what yfinance
actually returns.

Run with:  python data/raw/fetch_vix.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.config import RAW_DIR

import yfinance as yf

if __name__ == "__main__":
    vix = yf.download("^VIX", start="2016-01-01", end="2026-09-06", auto_adjust=False)
    vix.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in vix.columns]
    print(vix.shape)
    print(vix.head())
    print(vix.tail())

    out_path = RAW_DIR / "VIX_daily_raw.parquet"
    vix.to_parquet(out_path)
    print(f"Saved -> {out_path}")
