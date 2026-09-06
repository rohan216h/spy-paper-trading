"""
Single source of truth for paths and project-wide constants.
Import this everywhere instead of hardcoding paths/strings again.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT_DIR         = Path(__file__).resolve().parent.parent
DATA_DIR         = ROOT_DIR / "data"
RAW_DIR          = DATA_DIR / "raw"
INTERIM_DIR      = DATA_DIR / "interim"
INSTRUMENTS_DIR  = DATA_DIR / "instruments"

# ── Instrument ─────────────────────────────────────────────────────
SYMBOL   = "SPY"
TIMEZONE = "US/Eastern"          # CME/NYSE-native, one consistent zone

# Regular NYSE session (used for the session-hours filter, same role as
# the trade_hours filter in the old FF/SRA engine)
SESSION_START = "09:30"
SESSION_END   = "16:00"

# ── Train / validation / test split (BY DATE, never shuffled) ──────
TRAIN_END   = "2023-12-31"
VAL_END     = "2024-12-31"
# anything after VAL_END is test

# ── Bar size ─────────────────────────────────────────────────────
# Decided after inspecting real Alpaca data: ~6 years of clean 1-min bars
# available (2020-07-27 to present). We resample 1-min -> 5-min for the
# actual strategy - good noise/signal balance, standard for intraday.
BAR_SIZE = "5min"       # pandas resample rule
RAW_BAR_SIZE = "1min"   # what we pull from Alpaca before resampling
