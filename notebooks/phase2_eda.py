"""
Phase 2 - Exploratory & statistical validation.

Answers one question before we build anything: is there any statistical
basis for mean-reversion or momentum on SPY 5-min bars? If neither shows
up, a naive z-score strategy (like the old FF/SRA one) has nothing to
exploit and we need a different approach.

Reads: data/interim/SPY_5min_clean.parquet
Writes: PNG plots into notebooks/phase2_output/, prints stats to console.

Run with:  python notebooks/phase2_eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import INTERIM_DIR, SYMBOL, BAR_SIZE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "phase2_output"
OUT_DIR.mkdir(exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────
def hurst_exponent(ts, max_lag=100):
    """H < 0.5 => mean-reverting, H ~ 0.5 => random walk, H > 0.5 => trending"""
    ts = np.array(ts)
    ts = ts[~np.isnan(ts)]
    lags = range(2, min(max_lag, len(ts) // 2))
    tau = []
    for lag in lags:
        chunks = len(ts) // lag
        if chunks < 2:
            continue
        rs_list = []
        for i in range(chunks):
            chunk = ts[i * lag:(i + 1) * lag]
            mean = np.mean(chunk)
            dev = np.cumsum(chunk - mean)
            r = np.max(dev) - np.min(dev)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            tau.append((lag, np.mean(rs_list)))
    if len(tau) < 2:
        return np.nan
    lags_ = np.log([t[0] for t in tau])
    rs_ = np.log([t[1] for t in tau])
    h, _ = np.polyfit(lags_, rs_, 1)
    return h


def variance_ratio_test(ts, k=10):
    """VR < 1 => mean-reverting, VR ~ 1 => random walk, VR > 1 => trending"""
    ts = np.array(ts)
    ts = ts[~np.isnan(ts)]
    n = len(ts)
    if n < k * 2:
        return np.nan
    rets = np.diff(ts)
    var1 = np.var(rets, ddof=1)
    rets_k = ts[k:] - ts[:-k]
    var_k = np.var(rets_k, ddof=1) / k
    if var1 == 0:
        return np.nan
    return var_k / var1


# ── Load ─────────────────────────────────────────────────────────────
path = INTERIM_DIR / f"{SYMBOL}_{BAR_SIZE}_clean.parquet"
df = pd.read_parquet(path)
df = df.sort_index()
print(f"Loaded {len(df):,} bars from {df.index.min()} to {df.index.max()}")

price = df["close"]
log_price = np.log(price)
ret = log_price.diff().dropna()  # log returns, per-bar

print(f"\n{'='*60}\n1. RETURN DISTRIBUTION\n{'='*60}")
print(f"  Mean (per bar)   : {ret.mean():.8f}")
print(f"  Std  (per bar)   : {ret.std():.8f}")
print(f"  Skew             : {ret.skew():.4f}")
print(f"  Kurtosis (excess): {ret.kurtosis():.4f}  (0 = normal, >0 = fat tails)")
print(f"  Min / Max        : {ret.min():.6f} / {ret.max():.6f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(ret, bins=200, color="steelblue", alpha=0.8)
axes[0].set_title(f"{SYMBOL} {BAR_SIZE} Log Return Distribution")
axes[0].set_xlabel("Log return")
from scipy import stats as scistats
scistats.probplot(ret, dist="norm", plot=axes[1])
axes[1].set_title("QQ Plot vs Normal")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_return_distribution.png", dpi=120)
plt.close()
print(f"  Saved -> {OUT_DIR / '01_return_distribution.png'}")

print(f"\n{'='*60}\n2. STATIONARITY (ADF TEST)\n{'='*60}")
adf_price = adfuller(price.dropna())
adf_ret = adfuller(ret.dropna())
print(f"  Outright PRICE  - ADF stat: {adf_price[0]:.4f}, p-value: {adf_price[1]:.6f}"
      f"  {'(NON-stationary, expected)' if adf_price[1] > 0.05 else '(stationary)'}")
print(f"  RETURNS         - ADF stat: {adf_ret[0]:.4f}, p-value: {adf_ret[1]:.6f}"
      f"  {'(stationary, expected)' if adf_ret[1] < 0.05 else '(NON-stationary, unusual)'}")

print(f"\n{'='*60}\n3. AUTOCORRELATION (ACF/PACF on returns)\n{'='*60}")
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ret, lags=50, ax=axes[0])
axes[0].set_title(f"{SYMBOL} {BAR_SIZE} Returns - ACF")
plot_pacf(ret, lags=50, ax=axes[1])
axes[1].set_title(f"{SYMBOL} {BAR_SIZE} Returns - PACF")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_acf_pacf.png", dpi=120)
plt.close()
print(f"  Saved -> {OUT_DIR / '02_acf_pacf.png'}  (look for spikes outside the blue band)")

print(f"\n{'='*60}\n4. HURST EXPONENT & VARIANCE RATIO (on outright price)\n{'='*60}")
h = hurst_exponent(price.values)
vr = variance_ratio_test(price.values, k=10)
print(f"  Hurst exponent   : {h:.4f}  "
      f"({'mean-reverting' if h < 0.48 else 'trending' if h > 0.52 else 'random walk-like'})")
print(f"  Variance ratio(10): {vr:.4f}  "
      f"({'mean-reverting' if vr < 0.9 else 'trending' if vr > 1.1 else 'random walk-like'})")

print(f"\n{'='*60}\n5. INTRADAY SEASONALITY\n{'='*60}")
df_s = df.copy()
df_s["ret"] = log_price.diff()
df_s["abs_ret"] = df_s["ret"].abs()
df_s["time_str"] = df_s.index.strftime("%H:%M")

seasonality = df_s.groupby("time_str").agg(
    avg_abs_ret=("abs_ret", "mean"),
    avg_volume=("volume", "mean"),
).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(range(len(seasonality)), seasonality["avg_abs_ret"], color="darkorange")
axes[0].set_title("Intraday Volatility Pattern (avg |return| by time-of-day)")
axes[0].set_ylabel("Avg |log return|")
axes[1].plot(range(len(seasonality)), seasonality["avg_volume"], color="steelblue")
axes[1].set_title("Intraday Volume Pattern")
axes[1].set_ylabel("Avg volume")
tick_idx = range(0, len(seasonality), max(1, len(seasonality) // 15))
axes[1].set_xticks(list(tick_idx))
axes[1].set_xticklabels([seasonality["time_str"].iloc[i] for i in tick_idx], rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_intraday_seasonality.png", dpi=120)
plt.close()
print(f"  Saved -> {OUT_DIR / '03_intraday_seasonality.png'}")
print(f"  Highest-vol 5-min bucket : {seasonality.loc[seasonality['avg_abs_ret'].idxmax(), 'time_str']}")
print(f"  Lowest-vol  5-min bucket : {seasonality.loc[seasonality['avg_abs_ret'].idxmin(), 'time_str']}")

print(f"\n{'='*60}\n6. VOLATILITY REGIME CHECK (high-vol vs low-vol days)\n{'='*60}")
daily_vol = df_s.groupby(df_s.index.date)["ret"].std()
vol_median = daily_vol.median()
high_vol_days = daily_vol[daily_vol > vol_median].index
low_vol_days = daily_vol[daily_vol <= vol_median].index

df_s["date"] = df_s.index.date
high_mask = df_s["date"].isin(high_vol_days)
ret_high = df_s.loc[high_mask, "ret"]
ret_low = df_s.loc[~high_mask, "ret"]

print(f"  Daily realized vol - median: {vol_median:.6f}")
print(f"  HIGH-vol days -> n_bars={len(ret_high):,}, "
      f"mean_abs_ret={ret_high.abs().mean():.6f}, kurtosis={ret_high.kurtosis():.4f}")
print(f"  LOW-vol  days -> n_bars={len(ret_low):,}, "
      f"mean_abs_ret={ret_low.abs().mean():.6f}, kurtosis={ret_low.kurtosis():.4f}")

h_high = hurst_exponent(np.log(df.loc[high_mask, "close"]).values)
h_low = hurst_exponent(np.log(df.loc[~high_mask, "close"]).values)
print(f"  Hurst (HIGH-vol days) : {h_high:.4f}")
print(f"  Hurst (LOW-vol days)  : {h_low:.4f}")

print(f"\n{'='*60}\nDONE - check {OUT_DIR} for plots\n{'='*60}")
