"""
Generalized backtest engine (Phase 6, built now so Strategy A can use it
immediately, and Strategy B/ML plugs into the exact same code later).

Design: the engine knows nothing about WHAT the strategy is. It just calls
two functions the strategy provides:
    signal_fn(row, params)                       -> "LONG" / "SHORT" / None
    exit_fn(direction, row, bars_held, params)    -> exit_reason str / None

This is the same three-layer structure as the old FF/SRA engine
(Rules -> Engine Core -> Logs/Metrics), just with the rules layer passed
in instead of hardcoded, so it generalizes to any future strategy
(reversal rules now, ML thresholds later) without touching this file.

Costs: modeled as a flat round-trip cost in basis points of the entry
price (spread + commission + slippage combined) - simple but explicit,
easy to tighten later with a real spread model.
"""
import pandas as pd
import numpy as np


def bar_is_tradeable(row, params):
    """Pre-trade filters: session window, anomaly bars, optional regime skip."""
    if params.get("skip_open_window") and row["is_open_window"]:
        return False
    if params.get("skip_close_window") and row["is_close_window"]:
        return False
    if params.get("skip_anomaly", True) and row["anomaly_flag"]:
        return False
    if params.get("skip_regime") is not None and row["vol_regime"] == params["skip_regime"]:
        return False
    return True


def run_backtest(df: pd.DataFrame, params: dict, signal_fn, exit_fn,
                  instrument_name: str = "SPY"):
    """
    Bar-by-bar simulator. df must be sorted ascending by time, indexed by
    timestamp, and contain whatever columns signal_fn/exit_fn need plus
    'open','close','is_open_window','is_close_window','anomaly_flag',
    'vol_regime'.

    Returns: signal_log (DataFrame), trade_log (DataFrame), equity (Series, $ per share cumulative)
    """
    df = df.sort_index()
    cost_bps = params.get("cost_bps", 1.0)  # round-trip cost, bps of entry price

    # ---- Look-ahead fix ------------------------------------------------
    # Signals/exits are decided using information known at bar t's CLOSE
    # (that's when features like roll_ret_12 are actually computable).
    # Execution can therefore only happen at bar t+1's open, never bar
    # t's own open - trading at bar t's open using bar t's close-time
    # information is look-ahead bias. We precompute the next bar's open
    # and timestamp here so every fill in the loop below is honest.
    df["_exec_price"] = df["open"].shift(-1)
    df["_exec_time"] = df.index.to_series().shift(-1)
    df = df.iloc[:-1]  # last row has no next bar to execute on - drop it

    position = None
    bars_held = 0
    cumulative = 0.0
    signal_rows = []
    trade_rows = []
    equity_rows = []

    for ts, row in df.iterrows():
        tradeable = bar_is_tradeable(row, params)

        # ---- In a trade: check exit first ----------------------------
        if position is not None:
            bars_held += 1
            row["_entry_log_price"] = position["entry_log_price"]
            row["_trigger_move"] = position["trigger_move"]
            exit_reason = exit_fn(position["direction"], row, bars_held, params)

            if exit_reason:
                exit_price = row["_exec_price"] if params.get("entry_on_next", True) else row["close"]
                exit_time = row["_exec_time"] if params.get("entry_on_next", True) else ts
                entry_p = position["entry_price"]
                direction = position["direction"]

                raw_pnl = (exit_price - entry_p) if direction == "LONG" else (entry_p - exit_price)
                cost = entry_p * (cost_bps / 10000.0)
                pnl = raw_pnl - cost
                cumulative += pnl

                trade_rows.append({
                    "instrument": instrument_name,
                    "direction": direction,
                    "entry_time": position["entry_time"],
                    "exit_time": exit_time,
                    "entry_price": entry_p,
                    "exit_price": exit_price,
                    "pnl": round(pnl, 5),
                    "exit_reason": exit_reason,
                    "duration_bars": bars_held,
                    "regime": position["regime"],
                })
                position = None
                bars_held = 0

        # ---- Flat: check for entry -------------------------------------
        if position is None and tradeable:
            signal = signal_fn(row, params)
            if signal:
                signal_rows.append({
                    "time": ts, "direction": signal, "regime": row["vol_regime"],
                })
                if params.get("enter_immediately", True):
                    entry_price = row["_exec_price"] if params.get("entry_on_next", True) else row["close"]
                    entry_time = row["_exec_time"] if params.get("entry_on_next", True) else ts
                    position = {
                        "direction": signal,
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "entry_log_price": np.log(entry_price),
                        "trigger_move": row[params.get("move_col", "roll_ret_12")],
                        "regime": row["vol_regime"],
                    }
                    bars_held = 0

        equity_rows.append((ts, cumulative))

    equity = pd.Series(
        [v for _, v in equity_rows],
        index=pd.DatetimeIndex([t for t, _ in equity_rows]),
        name="equity",
    )
    signal_log = pd.DataFrame(signal_rows)
    trade_log = pd.DataFrame(trade_rows)
    return signal_log, trade_log, equity


def compute_metrics(trade_log: pd.DataFrame, signal_log: pd.DataFrame,
                     equity: pd.Series, instrument_name: str = "") -> dict:
    if trade_log.empty:
        return {"instrument": instrument_name, "n_trades": 0}

    tl = trade_log.copy()
    pnl = tl["pnl"]

    n_signals = len(signal_log) if signal_log is not None and not signal_log.empty else 0
    n_trades = len(tl)

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = len(wins) / n_trades if n_trades else 0
    total_pnl = pnl.sum()
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")

    rolling_max = equity.cummax()
    drawdown = equity - rolling_max
    max_drawdown = drawdown.min()

    daily = equity.resample("1D").last().dropna().diff().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() != 0 else 0

    exit_counts = tl["exit_reason"].value_counts(normalize=True) * 100

    return {
        "instrument": instrument_name,
        "n_signals": n_signals,
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(pnl.mean(), 5),
        "profit_factor": round(profit_factor, 4),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_drawdown, 2),
        "avg_duration_bars": round(tl["duration_bars"].mean(), 1),
        "pct_stop": round(exit_counts.get("STOP", 0), 1),
        "pct_time": round(exit_counts.get("TIME", 0), 1),
        "pct_revert": round(exit_counts.get("REVERT", 0), 1),
    }


def print_metrics(m: dict):
    if not m or m.get("n_trades", 0) == 0:
        print(f"No trades for {m.get('instrument', '')}")
        return
    print(f"\n{'='*55}")
    print(f"  PERFORMANCE - {m['instrument']}")
    print(f"{'='*55}")
    print(f"  Signals fired      : {m['n_signals']:>8,}")
    print(f"  Trades taken       : {m['n_trades']:>8,}")
    print(f"  Win rate           : {m['win_rate_pct']:>7.2f}%")
    print(f"  Total PnL ($/share): {m['total_pnl']:>8.2f}")
    print(f"  Avg PnL / trade    : {m['avg_pnl']:>8.5f}")
    print(f"  Profit factor      : {m['profit_factor']:>8.4f}")
    print(f"  Sharpe (annualized): {m['sharpe']:>8.4f}")
    print(f"  Max drawdown       : {m['max_dd']:>8.2f}")
    print(f"  Avg duration (bars): {m['avg_duration_bars']:>8.1f}")
    print(f"  Exit mix - STOP:{m['pct_stop']}% TIME:{m['pct_time']}% REVERT:{m['pct_revert']}%")
