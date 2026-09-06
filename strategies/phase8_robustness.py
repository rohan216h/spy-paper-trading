"""
Phase 8 - Robustness testing, on Strategy B (Linear) - the only strategy
that cleared the bar in Phase 7. No point stress-testing something that
already lost.

Two checks:

1. PARAMETER SENSITIVITY - does performance survive small nudges to the
   confidence threshold and hold period? If Sharpe swings wildly (as we
   already saw hints of: 0.66 -> -0.16 -> 1.59 across just 4 points),
   that's fragility, not a robust edge. Uses the SAME already-computed
   predictions (no retraining) - just re-runs the cheap engine step
   across a finer grid.

2. MONTE CARLO BOOTSTRAP - resample the realized trade-level PnLs with
   replacement (many times) to build a distribution of plausible outcomes
   under the assumption trades are roughly independent draws from the
   same underlying process. Reports what fraction of resamples would
   still show a profit - a crude but honest stand-in for "is this
   noise" without needing a fancier significance test.
   Caveat (stated honestly, not hidden): trades aren't perfectly
   independent - clustering by regime/time could inflate apparent
   significance somewhat. Treat this as a lower bound on confidence,
   not a rigorous p-value.

Run with:  python strategies/phase8_robustness.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import INSTRUMENTS_DIR, SYMBOL, BAR_SIZE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from engine.backtest_engine import run_backtest, compute_metrics
from strategies.run_strategy_b import check_signal_ml, check_exit_ml

COMMON_START = "2022-08-08"
COMMON_END = "2025-02-14"

OUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "phase8_output"
OUT_DIR.mkdir(exist_ok=True)


def load_common_with_preds():
    feat_path = INSTRUMENTS_DIR / f"{SYMBOL}_{BAR_SIZE}_features.parquet"
    features = pd.read_parquet(feat_path).sort_index()
    common = features.loc[COMMON_START:COMMON_END].copy()

    pred_path = Path(__file__).resolve().parent.parent / "notebooks" / "phase5_output" / "predictions_linear.parquet"
    preds = pd.read_parquet(pred_path)[["pred"]]
    df = common.join(preds, how="inner").dropna(subset=["pred"])
    return df


def run_one(df, threshold_mult, max_hold_bars):
    pred_std = df["pred"].std()
    params = {
        "threshold": pred_std * threshold_mult, "max_hold_bars": max_hold_bars, "cost_bps": 1.0,
        "skip_open_window": True, "skip_close_window": True, "skip_anomaly": True,
        "skip_regime": None, "entry_on_next": True, "enter_immediately": True,
    }
    sig, trd, eq = run_backtest(df, params, check_signal_ml, check_exit_ml, instrument_name="probe")
    m = compute_metrics(trd, sig, eq, "probe")
    return m, trd


if __name__ == "__main__":
    df = load_common_with_preds()
    print(f"Loaded {len(df):,} bars with predictions ({COMMON_START} to {COMMON_END})\n")

    # ---- 1a. Threshold sensitivity (fine grid, fixed hold=12) -----------
    print("="*60)
    print("1a. THRESHOLD SENSITIVITY (max_hold_bars fixed at 12)")
    print("="*60)
    thresholds = np.arange(0.25, 3.01, 0.25)
    thresh_results = []
    for t in thresholds:
        m, _ = run_one(df, t, 12)
        sharpe = m.get("sharpe", 0)
        n_trades = m.get("n_trades", 0)
        thresh_results.append({"threshold_mult": t, "sharpe": sharpe, "n_trades": n_trades})
        print(f"  threshold={t:.2f}x  n_trades={n_trades:>5}  sharpe={sharpe:>7.3f}")
    thresh_df = pd.DataFrame(thresh_results)

    # ---- 1b. Hold-period sensitivity (fixed threshold=1.0x) --------------
    print(f"\n{'='*60}")
    print("1b. HOLD-PERIOD SENSITIVITY (threshold fixed at 1.0x)")
    print("="*60)
    holds = [4, 6, 8, 12, 16, 20, 24, 30, 36]
    hold_results = []
    for h in holds:
        m, _ = run_one(df, 1.0, h)
        sharpe = m.get("sharpe", 0)
        n_trades = m.get("n_trades", 0)
        hold_results.append({"max_hold_bars": h, "sharpe": sharpe, "n_trades": n_trades})
        print(f"  hold={h:>3} bars  n_trades={n_trades:>5}  sharpe={sharpe:>7.3f}")
    hold_df = pd.DataFrame(hold_results)

    # ---- Sensitivity plot -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(thresh_df["threshold_mult"], thresh_df["sharpe"], marker="o", color="steelblue")
    axes[0].axhline(0, color="black", lw=0.6, linestyle="--")
    axes[0].set_title("Sharpe vs Confidence Threshold (hold=12 bars)")
    axes[0].set_xlabel("Threshold (x prediction std)")
    axes[0].set_ylabel("Sharpe")

    axes[1].plot(hold_df["max_hold_bars"], hold_df["sharpe"], marker="o", color="darkorange")
    axes[1].axhline(0, color="black", lw=0.6, linestyle="--")
    axes[1].set_title("Sharpe vs Hold Period (threshold=1.0x)")
    axes[1].set_xlabel("Max hold bars")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "parameter_sensitivity.png", dpi=120)
    plt.close()
    print(f"\nSaved -> {OUT_DIR / 'parameter_sensitivity.png'}")

    sign_flips = (np.diff(np.sign(thresh_df["sharpe"])) != 0).sum()
    print(f"\nSign flips across threshold grid: {sign_flips} "
          f"(0 = fully stable direction, more = fragile)")

    # ---- 2. Monte Carlo bootstrap on the Phase 7 configuration -----------
    print(f"\n{'='*60}")
    print("2. MONTE CARLO BOOTSTRAP (threshold=1.0x, hold=12 - the Phase 7 config)")
    print("="*60)
    _, trd = run_one(df, 1.0, 12)
    pnls = trd["pnl"].values
    n_trades = len(pnls)
    actual_mean = pnls.mean()
    actual_total = pnls.sum()

    n_sims = 5000
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(pnls, size=n_trades, replace=True).mean()
        for _ in range(n_sims)
    ])

    pct_negative = (boot_means <= 0).mean() * 100
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print(f"  Trades in this config       : {n_trades}")
    print(f"  Actual mean PnL/trade       : {actual_mean:.5f}")
    print(f"  Actual total PnL            : {actual_total:.2f}")
    print(f"  Bootstrap 95% CI (mean/trade): [{ci_low:.5f}, {ci_high:.5f}]")
    print(f"  % of bootstrap resamples with mean PnL <= 0: {pct_negative:.1f}%")
    print(f"  (Lower is better - this is a rough stand-in for a p-value.")
    print(f"   Caveat: trades aren't perfectly independent, so treat this")
    print(f"   as a lower bound on confidence, not a rigorous test.)")

    plt.figure(figsize=(10, 5))
    plt.hist(boot_means, bins=80, color="steelblue", alpha=0.8)
    plt.axvline(0, color="black", lw=1, linestyle="--", label="Zero")
    plt.axvline(actual_mean, color="crimson", lw=2, label=f"Actual mean ({actual_mean:.5f})")
    plt.title(f"Bootstrap Distribution of Mean PnL/Trade (n_sims={n_sims})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bootstrap_distribution.png", dpi=120)
    plt.close()
    print(f"\nSaved -> {OUT_DIR / 'bootstrap_distribution.png'}")
