"""
Phase 7 - Performance evaluation.

Compares Strategy A (reversal rules), Strategy B - Linear, and
Strategy B - LightGBM on the EXACT SAME date range and cost model - the
walk-forward OOS period common to all three (2022-08-08 to 2025-02-14,
the range where ML predictions actually exist out-of-sample).

Strategy A is re-run here (not reusing the earlier train-split run)
specifically to make this a fair, same-period comparison - the earlier
Strategy A numbers were on a different, non-overlapping date range.

Run with:  python strategies/compare_strategies.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import INSTRUMENTS_DIR, SYMBOL, BAR_SIZE

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from engine.backtest_engine import run_backtest, compute_metrics, print_metrics
from strategies.rules import PARAMS as RULES_PARAMS, check_signal as rules_signal, check_exit as rules_exit
from strategies.run_strategy_b import check_signal_ml, check_exit_ml

COMMON_START = "2022-08-08"
COMMON_END = "2025-02-14"
ML_THRESHOLD_MULT = 1.0  # chosen without cherry-picking from the noisy scan - middle of the tested range

if __name__ == "__main__":
    feat_path = INSTRUMENTS_DIR / f"{SYMBOL}_{BAR_SIZE}_features.parquet"
    features = pd.read_parquet(feat_path).sort_index()
    common = features.loc[COMMON_START:COMMON_END].copy()
    print(f"Common comparison period: {common.index.min()} to {common.index.max()} "
          f"({len(common):,} bars)\n")

    results = {}

    # ---- Strategy A: rules-based reversal ------------------------------
    sig_a, trd_a, eq_a = run_backtest(
        common, RULES_PARAMS, rules_signal, rules_exit, instrument_name="Strategy A (Reversal)"
    )
    m_a = compute_metrics(trd_a, sig_a, eq_a, "Strategy A (Reversal)")
    print_metrics(m_a)
    results["Strategy A"] = (m_a, eq_a)

    # ---- Strategy B: Linear ---------------------------------------------
    pred_lin_path = Path(__file__).resolve().parent.parent / "notebooks" / "phase5_output" / "predictions_linear.parquet"
    preds_lin = pd.read_parquet(pred_lin_path)[["pred"]]
    df_lin = common.join(preds_lin, how="inner").dropna(subset=["pred"])
    pred_std_lin = df_lin["pred"].std()
    params_lin = {
        "threshold": pred_std_lin * ML_THRESHOLD_MULT, "max_hold_bars": 12, "cost_bps": 1.0,
        "skip_open_window": True, "skip_close_window": True, "skip_anomaly": True,
        "skip_regime": None, "entry_on_next": True, "enter_immediately": True,
    }
    sig_b, trd_b, eq_b = run_backtest(
        df_lin, params_lin, check_signal_ml, check_exit_ml, instrument_name="Strategy B (Linear)"
    )
    m_b = compute_metrics(trd_b, sig_b, eq_b, "Strategy B (Linear)")
    print_metrics(m_b)
    results["Strategy B (Linear)"] = (m_b, eq_b)

    # ---- Strategy B: LightGBM ---------------------------------------------
    pred_lgb_path = Path(__file__).resolve().parent.parent / "notebooks" / "phase5_output" / "predictions_lightgbm.parquet"
    preds_lgb = pd.read_parquet(pred_lgb_path)[["pred"]]
    df_lgb = common.join(preds_lgb, how="inner").dropna(subset=["pred"])
    pred_std_lgb = df_lgb["pred"].std()
    params_lgb = {
        "threshold": pred_std_lgb * ML_THRESHOLD_MULT, "max_hold_bars": 12, "cost_bps": 1.0,
        "skip_open_window": True, "skip_close_window": True, "skip_anomaly": True,
        "skip_regime": None, "entry_on_next": True, "enter_immediately": True,
    }
    sig_c, trd_c, eq_c = run_backtest(
        df_lgb, params_lgb, check_signal_ml, check_exit_ml, instrument_name="Strategy B (LightGBM)"
    )
    m_c = compute_metrics(trd_c, sig_c, eq_c, "Strategy B (LightGBM)")
    print_metrics(m_c)
    results["Strategy B (LightGBM)"] = (m_c, eq_c)

    # ---- Summary table --------------------------------------------------
    print(f"\n{'='*70}\nSIDE-BY-SIDE SUMMARY (same period, same {params_lin['cost_bps']}bps cost)\n{'='*70}")
    summary = pd.DataFrame({k: v[0] for k, v in results.items()}).T
    print(summary[["n_trades", "win_rate_pct", "total_pnl", "sharpe", "profit_factor", "max_dd"]])

    # ---- Overlaid equity curves ------------------------------------------
    out_dir = Path(__file__).resolve().parent.parent / "notebooks" / "phase7_output"
    out_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(14, 6))
    for name, (m, eq) in results.items():
        plt.plot(eq.index, eq.values, label=f"{name} (Sharpe={m.get('sharpe', 0):.2f})", lw=1.0)
    plt.axhline(0, color="black", lw=0.6, linestyle="--")
    plt.title(f"Strategy Comparison - {SYMBOL} {BAR_SIZE} ({COMMON_START} to {COMMON_END})")
    plt.ylabel("Cumulative $/share")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "strategy_comparison.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved comparison plot -> {out_path}")
