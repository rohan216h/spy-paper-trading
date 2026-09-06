# SPX Intraday Quant Project

A from-scratch quant research + backtesting project on the S&P 500 (SPY as the
tradable proxy), following the classic pipeline: data → EDA → features →
rules-based strategy → ML strategy → unified backtest engine → performance
evaluation → robustness testing → (later) paper trading + monitoring dashboard.

## Status
Phase 0 — scaffolding. No data pulled yet, no strategies written yet.

## Instrument & horizon
- Instrument: SPY (S&P 500 ETF), intraday bars
- Horizon: not fixed a priori — the model determines how long a position is
  held (max-hold is a safety cap, not a target)
- Data source: Alpaca Market Data API (free tier, IEX feed)

## Folder structure
```
data/
  raw/            # untouched vendor data (Alpaca API responses, as pulled)
  interim/        # cleaned, timezone-aligned, session-filtered bars
  instruments/    # final feature-enriched parquet files, ready for the engine
features/         # point-in-time feature engineering code (no lookahead)
strategies/
  rules.py        # Strategy A - rules-based baseline (to be built)
  ml.py           # Strategy B - ML model (to be built)
engine/           # generalized backtest engine (plug-in strategy interface)
notebooks/        # research/exploration only - nothing here is "the pipeline"
dashboard/        # Streamlit monitoring dashboard (later phase)
paper_trading/    # Alpaca paper-trading execution loop (later phase)
tests/            # unit tests for engine + feature correctness
configs/          # YAML/py config for paths, API keys (never commit secrets)
```

## Rule of thumb
Anything that should be reproducible lives in a `.py` module under
`features/`, `strategies/`, or `engine/`. Notebooks in `notebooks/` are for
one-off exploration and always import from the modules above rather than
duplicating logic.

## Phases (see PHASES.md for detail)
0. Scaffolding (this)
1. Data acquisition & cleaning
2. Exploratory & statistical validation
3. Feature engineering
4. Strategy A - rules-based baseline
5. Strategy B - ML model
6. Unified backtest engine
7. Performance evaluation
8. Robustness testing
9. Paper trading
10. Monitoring dashboard
11. Writeup
