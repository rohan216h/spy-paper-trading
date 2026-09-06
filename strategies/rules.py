"""
Phase 4 - Strategy A: rules-based baseline.

Signal design based on what Phase 2 EDA actually showed (not the FF/SRA
price-z-score approach, which needs structural mean-reversion that raw
SPY price doesn't have):

  Entry:  an unusually large N-bar move just happened (|roll_ret_z| >
          entry_z) -> bet on short-term reversal (a known, documented
          equity effect distinct from price-level mean-reversion).
  Exit:   PRICE has actually retraced a fraction of the triggering move
          (target_retrace), OR max hold bars reached, OR the adverse
          move extends past a stop (in price terms, not z-score terms).

IMPORTANT FIX: the first version of this file exited when the z-score
decayed back toward zero. That's a measurement artifact, not a real
signal - roll_ret_z is computed on a ROLLING window, so it drifts back
toward zero simply because the triggering bar ages out of the window,
regardless of whether price actually reversed. Exiting on that produced
"REVERT" exits that weren't real reversals, which is why win rate was
low despite 79% of exits being classified as REVERT. Exit logic below
is now based on actual price retracement instead.
"""
PARAMS = {
    "z_col"             : "roll_ret_z_12",  # which reversal feature triggers entry
    "move_col"          : "roll_ret_12",    # the actual price move (log-return) behind the z-score
    "entry_z"           : 2.5,
    "target_retrace"    : 0.5,    # exit once 50% of the triggering move is given back in price
    "stop_extra_move"   : 0.5,    # stop if price moves ANOTHER 50%-of-original-move further adverse
    "max_hold_bars"     : 24,     # 24 * 5min = 2 hours
    "cost_bps"          : 1.0,    # round-trip cost, bps of entry price
    "skip_open_window"  : True,
    "skip_close_window" : True,
    "skip_anomaly"      : True,
    "skip_regime"       : None,   # e.g. "HIGH_VOL" to avoid high-vol days
    "entry_on_next"     : True,
    "enter_immediately" : True,
}


def check_signal(row, params):
    z = row[params["z_col"]]
    if z <= -params["entry_z"]:
        return "LONG"   # big drop -> bet on bounce
    if z >= params["entry_z"]:
        return "SHORT"  # big spike -> bet on pullback
    return None


def check_exit(direction, row, bars_held, params):
    """
    direction/row don't carry the entry price or original move magnitude
    by themselves - the engine attaches them onto `row` as
    `_entry_price` and `_trigger_move` before calling this (see engine
    changes). Exit is based on actual log-price retracement of that
    original move, not on the z-score.
    """
    entry_log_price = row["_entry_log_price"]
    trigger_move = row["_trigger_move"]  # log-return magnitude that triggered entry (signed)
    current_log_price = row["log_price"]

    moved = current_log_price - entry_log_price  # signed, in log-return space

    if direction == "LONG":
        # trigger_move was negative (a drop); retracement = positive `moved`
        retrace_frac = moved / (-trigger_move) if trigger_move != 0 else 0
    else:  # SHORT
        # trigger_move was positive (a spike); retracement = negative `moved`
        retrace_frac = -moved / trigger_move if trigger_move != 0 else 0

    if retrace_frac <= -params["stop_extra_move"]:
        return "STOP"
    if bars_held >= params["max_hold_bars"]:
        return "TIME"
    if retrace_frac >= params["target_retrace"]:
        return "REVERT"
    return None

