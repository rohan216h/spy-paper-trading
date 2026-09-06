# Roadmap

## Phase 0 — Project scaffolding [DONE]
Repo structure, instrument decision (SPY), horizon decision (model-determined,
capped by a max-hold safety limit).

## Phase 1 — Data acquisition & cleaning [NEXT]
- Pull raw intraday bars (1min/5min, decide after inspecting what Alpaca gives us)
  for as many years as the free tier allows
- Handle: missing bars/holidays, timezone alignment (one consistent zone,
  US/Eastern makes sense for equities), corporate actions (SPY dividends -
  Alpaca adjusted bars handle splits; dividends need explicit handling if we
  want a total-return series)
- Store as parquet: data/raw -> data/interim -> data/instruments
- Train/validation/test split by DATE, never shuffled

## Phase 2 — Exploratory & statistical validation
- Return distribution, ACF/PACF, ADF stationarity test, Hurst exponent
- Intraday seasonality (time-of-day vol/volume patterns)
- Regime check (VIX-conditioned high-vol vs low-vol)
- Goal: know whether mean-reversion or momentum is even plausible BEFORE
  building a strategy around the wrong assumption

## Phase 3 — Feature engineering
- Price/technical: rolling returns, z-scores at multiple lookbacks, realized
  vol, RSI-style momentum, volume features
- Macro overlays: VIX level & term structure, yield curve slope, rate regime
- Time features: time-of-day, day-of-week, proximity to FOMC/CPI
- All features must be point-in-time correct - no lookahead

## Phase 4 — Strategy A: rules-based baseline
- Adapt the z-score engine directly: entry/exit thresholds, trade-hour
  filters, anomaly/vol-regime filters
- This becomes the benchmark everything else must beat

## Phase 5 — Strategy B: ML model
- Label: forward return over a chosen horizon, binarized or continuous
- Model: logistic regression / LightGBM first
- Walk-forward training: retrain on rolling windows, never test on data used
  in training
- Feature importance / SHAP sanity check
- Convert probability/predicted return into a position via threshold rule

## Phase 6 — Unified backtest engine
- Generalize run_backtest to take either rule module as a plug-in
- Realistic costs: commission, spread-based slippage, tick value
- Position sizing: vol-targeting instead of fixed size

## Phase 7 — Performance evaluation
- Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor
- Strategy A vs B side by side, same costs, same period
- Regime breakdown (rate hiking cycle vs easing, vol regimes)

## Phase 8 — Robustness testing
- Walk-forward out-of-sample (not just one split)
- Monte Carlo bootstrap on trade sequence
- Parameter sensitivity

## Phase 9 — Paper trading
- Alpaca paper trading API
- Scheduled script: pull latest bar -> compute features -> get signal -> order
- Log every decision

## Phase 10 — Monitoring dashboard
- Streamlit reading from the paper-trading log

## Phase 11 — Writeup
- Ask -> approach -> findings, per phase
