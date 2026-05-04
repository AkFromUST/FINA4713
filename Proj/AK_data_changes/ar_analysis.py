"""
ar_analysis.py  —  Box-Jenkins workflow for stock return autocorrelation
------------------------------------------------------------------------
Workflow (no pre-judged conclusion):
  Step 1  Visual ACF / PACF
  Step 2  Stationarity (ADF) + raw-residual autocorrelation tests
  Step 3  If autocorrelation detected → fit AR(1), check residuals
  Step 4  If AR(1) residuals still show structure → ARMA(p,q) grid,
          AIC / BIC, residual diagnostics
  Step 5  Summary: what the data actually says

Analysis is run on two levels:
  (a) Equal-weighted market return — the aggregate cross-section
  (b) Sample of 200 individual stocks — where our prediction lives

Run from Proj/:  ../venv/bin/python AK_data_changes/ar_analysis.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, '..', 'jkp_data.parquet')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'files', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET    = 'ret_exc_lead1m'
TRAIN_END = pd.Timestamp('2015-12-31')
NLAGS     = 24
MIN_OBS   = 60
N_STOCKS  = 200   # individual-stock sample size
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Load & split
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
df['eom'] = pd.to_datetime(df['eom'])
df = df.dropna(subset=[TARGET])
train = df[df['eom'] <= TRAIN_END].copy()

# (a) Equal-weighted market return
mkt = train.groupby('eom')[TARGET].mean().sort_index()
print(f"Market return series : T = {len(mkt)} months")

# (b) Individual stock sample
stock_series = {}
for sid, grp in train.groupby('id'):
    s = grp.set_index('eom')[TARGET].sort_index().dropna()
    if len(s) >= MIN_OBS:
        stock_series[sid] = s

sample_ids = np.random.choice(list(stock_series.keys()),
                               size=min(N_STOCKS, len(stock_series)), replace=False)
print(f"Stock sample         : {len(sample_ids)} stocks  (≥{MIN_OBS} months each)")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def plot_acf_pacf(series, title, filepath, nlags=NLAGS):
    acf_vals  = acf( series, nlags=nlags, fft=True)
    pacf_vals = pacf(series, nlags=nlags)
    conf      = 1.96 / np.sqrt(len(series))
    lags      = np.arange(nlags + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, vals, label in zip(axes,
                                [acf_vals, pacf_vals],
                                ['ACF', 'PACF']):
        ax.bar(lags[1:], vals[1:], width=0.6, color='#4393c3')
        ax.axhline( conf, color='red', lw=1.2, linestyle='--', label=f'±1.96/√T')
        ax.axhline(-conf, color='red', lw=1.2, linestyle='--')
        ax.axhline(0,     color='black', lw=0.7)
        ax.set_xlabel('Lag (months)'); ax.set_ylabel(label)
        ax.set_title(f'{label}: {title}'); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150); plt.close()


def run_diagnostics(series, label):
    """ADF + Ljung-Box at several lag horizons."""
    adf_stat, adf_p, *_ = adfuller(series, autolag='AIC')
    lb = acorr_ljungbox(series, lags=[6, 12, 18, 24], return_df=True)
    print(f"\n  {label}")
    print(f"    ADF stat={adf_stat:.3f}  p={adf_p:.4f}  "
          f"→ {'stationary' if adf_p < 0.05 else 'NON-stationary'}")
    print(f"    Ljung-Box:")
    for _, row in lb.iterrows():
        lag = int(row['lb_stat'] / row['lb_pvalue'] * 0 + row.name) if 'lb_stat' in row else '?'
        lag = row.name
        sig = '* reject white noise' if row['lb_pvalue'] < 0.05 else '  fail to reject'
        print(f"      lag={lag:>2}  Q={row['lb_stat']:>7.3f}  p={row['lb_pvalue']:.4f}  {sig}")
    return lb


def fit_arima_safe(series, order):
    try:
        res = ARIMA(series, order=order).fit(method_kwargs={'warn_convergence': False})
        return res
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1  Visual ACF / PACF
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 68)
print("STEP 1  Visual ACF / PACF")
print("═" * 68)

plot_acf_pacf(mkt.values, 'Equal-weighted market return',
              os.path.join(OUT_DIR, 'S1a_acf_pacf_market.png'))
print("  Saved: S1a_acf_pacf_market.png")

# Median stock ACF (average across sample)
stock_acfs = []
for sid in sample_ids:
    s = stock_series[sid].values
    a = acf(s, nlags=NLAGS, fft=True)[1:]
    stock_acfs.append(a)

mean_stock_acf  = np.nanmean(stock_acfs, axis=0)
se_stock_acf    = np.nanstd(stock_acfs,  axis=0) / np.sqrt(len(stock_acfs))
conf_mkt        = 1.96 / np.sqrt(len(mkt))

fig, ax = plt.subplots(figsize=(11, 4))
lags_x = np.arange(1, NLAGS + 1)
ax.bar(lags_x, mean_stock_acf, width=0.6, color='#4393c3', label='Mean ACF across stocks')
ax.fill_between(lags_x,
                mean_stock_acf - 1.96 * se_stock_acf,
                mean_stock_acf + 1.96 * se_stock_acf,
                alpha=0.25, color='#4393c3', label='±1.96 SE across stocks')
avg_conf = 1.96 / np.sqrt(MIN_OBS)
ax.axhline( avg_conf, color='red',   lw=1.2, linestyle='--', label=f'±1.96/√T (min T={MIN_OBS})')
ax.axhline(-avg_conf, color='red',   lw=1.2, linestyle='--')
ax.axhline(0,         color='black', lw=0.7)
ax.set_xlabel('Lag (months)'); ax.set_ylabel('ACF')
ax.set_title(f'S1b: Mean ACF across {len(sample_ids)} individual stocks')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'S1b_acf_stocks_mean.png'), dpi=150); plt.close()
print("  Saved: S1b_acf_stocks_mean.png")

print(f"\n  Market ACF at lag 1 : {acf(mkt.values, nlags=1, fft=True)[1]:+.4f}")
print(f"  Market ACF at lag 2 : {acf(mkt.values, nlags=2, fft=True)[2]:+.4f}")
print(f"  Mean stock ACF lag 1: {mean_stock_acf[0]:+.4f}  (SE={se_stock_acf[0]:.4f})")
print(f"  Mean stock ACF lag 2: {mean_stock_acf[1]:+.4f}  (SE={se_stock_acf[1]:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2  Stationarity + raw autocorrelation tests
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 68)
print("STEP 2  Stationarity (ADF) + Ljung-Box on raw returns")
print("═" * 68)

lb_mkt = run_diagnostics(mkt.values, "Market return")

# Fraction of individual stocks where Ljung-Box rejects at lag 12
lb_stock_pvals_lag12 = []
for sid in sample_ids:
    s = stock_series[sid].values
    lb = acorr_ljungbox(s, lags=[12], return_df=True)
    lb_stock_pvals_lag12.append(lb['lb_pvalue'].iloc[0])

frac_reject = np.mean(np.array(lb_stock_pvals_lag12) < 0.05)
print(f"\n  Individual stocks — Ljung-Box lag 12 (sample of {len(sample_ids)})")
print(f"    Fraction rejecting white noise: {frac_reject:.1%}")
print(f"    Expected under H0 (size=5%)  : 5.0%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3  AR(1): fit → check residuals
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 68)
print("STEP 3  AR(1) — fit and residual diagnostics")
print("═" * 68)

# --- 3a. Market return ---
print("\n  3a. Market return")
res_ar1_mkt = fit_arima_safe(mkt.values, (1, 0, 0))
if res_ar1_mkt is not None:
    phi = res_ar1_mkt.arparams[0]
    print(f"    AR(1) coefficient φ = {phi:+.4f}  "
          f"(95% CI: [{phi - 1.96*res_ar1_mkt.bse[0]:+.4f}, "
          f"{phi + 1.96*res_ar1_mkt.bse[0]:+.4f}])")
    print(f"    AIC = {res_ar1_mkt.aic:.2f}   BIC = {res_ar1_mkt.bic:.2f}")
    run_diagnostics(res_ar1_mkt.resid, "AR(1) residuals — market")
    plot_acf_pacf(res_ar1_mkt.resid,
                  'AR(1) residuals — market',
                  os.path.join(OUT_DIR, 'S3a_ar1_resid_market.png'))
    print("    Saved: S3a_ar1_resid_market.png")

# --- 3b. Individual stocks: distribution of AR(1) φ and residual LB p-values ---
print("\n  3b. Individual stocks")
ar1_phis, ar1_aics, resid_lb_pvals = [], [], []

for sid in sample_ids:
    s   = stock_series[sid].values
    res = fit_arima_safe(s, (1, 0, 0))
    if res is None:
        continue
    ar1_phis.append(res.arparams[0])
    ar1_aics.append(res.aic)
    lb = acorr_ljungbox(res.resid, lags=[12], return_df=True)
    resid_lb_pvals.append(lb['lb_pvalue'].iloc[0])

ar1_phis       = np.array(ar1_phis)
resid_lb_pvals = np.array(resid_lb_pvals)
frac_resid_sig = np.mean(resid_lb_pvals < 0.05)

print(f"    Stocks fitted       : {len(ar1_phis)}")
print(f"    Mean φ              : {ar1_phis.mean():+.4f}  (std={ar1_phis.std():.4f})")
print(f"    Fraction φ > 0      : {(ar1_phis > 0).mean():.1%}")
print(f"    AR(1) residuals: fraction still autocorrelated (LB lag 12, p<0.05): "
      f"{frac_resid_sig:.1%}")

if frac_resid_sig > 0.10:
    print(f"\n    → AR(1) residuals still show structure in {frac_resid_sig:.0%} of stocks.")
    print(f"      Proceeding to ARMA grid (Step 4).")
    proceed_to_arma = True
else:
    print(f"\n    → AR(1) residuals are clean for most stocks.")
    print(f"      ARMA grid may not be needed, but running for completeness.")
    proceed_to_arma = True   # run it regardless — let the data speak

# Plot φ distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(ar1_phis, bins=50, color='#4393c3', edgecolor='white', lw=0.3)
axes[0].axvline(0, color='red', lw=1.5, linestyle='--', label='φ=0')
axes[0].axvline(ar1_phis.mean(), color='black', lw=1.5, label=f'mean={ar1_phis.mean():+.3f}')
axes[0].set_xlabel('AR(1) φ'); axes[0].set_title('S3b: AR(1) coefficient distribution')
axes[0].legend()

axes[1].hist(resid_lb_pvals, bins=30, color='#d6604d', edgecolor='white', lw=0.3)
axes[1].axvline(0.05, color='black', lw=1.5, linestyle='--', label='p=0.05')
axes[1].set_xlabel('Ljung-Box p-value (lag 12, AR(1) residuals)')
axes[1].set_title('S3c: Residual autocorrelation after AR(1)')
axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'S3_ar1_stocks.png'), dpi=150); plt.close()
print("    Saved: S3_ar1_stocks.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4  ARMA(p,q) grid — AIC/BIC + residual tests
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 68)
print("STEP 4  ARMA(p,q) grid  (p,q ∈ {0,1,2})")
print("═" * 68)

orders = [(p, 0, q) for p in range(3) for q in range(3)]
order_labels = [f'ARMA({p},{q})' for p, _, q in orders]

# --- 4a. Market return: show full grid ---
print("\n  4a. Market return — full AIC/BIC table")
mkt_results = []
for order, lbl in zip(orders, order_labels):
    res = fit_arima_safe(mkt.values, order)
    if res is None:
        mkt_results.append({'model': lbl, 'aic': np.inf, 'bic': np.inf, 'lb_p': np.nan})
        continue
    lb = acorr_ljungbox(res.resid, lags=[12], return_df=True)
    mkt_results.append({
        'model': lbl,
        'aic'  : res.aic,
        'bic'  : res.bic,
        'lb_p' : lb['lb_pvalue'].iloc[0]
    })

mkt_grid = pd.DataFrame(mkt_results).sort_values('aic')
best_aic_mkt = mkt_grid.iloc[0]['model']
best_bic_mkt = mkt_grid.sort_values('bic').iloc[0]['model']

print(f"\n  {'Model':<14} {'AIC':>10} {'BIC':>10} {'LB p(lag12)':>14}  Clean?")
print("  " + "-" * 58)
for _, row in mkt_grid.iterrows():
    flag = '← best AIC' if row['model'] == best_aic_mkt else (
           '← best BIC' if row['model'] == best_bic_mkt else '')
    clean = 'yes' if row['lb_p'] > 0.05 else 'residuals still autocorr'
    print(f"  {row['model']:<14} {row['aic']:>10.2f} {row['bic']:>10.2f} "
          f"{row['lb_p']:>14.4f}  {clean}  {flag}")

# Fit best-AIC model and show final residuals
best_order_mkt = orders[order_labels.index(best_aic_mkt)]
res_best_mkt   = fit_arima_safe(mkt.values, best_order_mkt)
if res_best_mkt is not None:
    print(f"\n  Best AIC model ({best_aic_mkt}) parameters:")
    print(f"    {dict(zip(res_best_mkt.param_names, res_best_mkt.params))}")
    plot_acf_pacf(res_best_mkt.resid,
                  f'{best_aic_mkt} residuals — market',
                  os.path.join(OUT_DIR, f'S4a_best_arma_resid_market.png'))
    print(f"    Saved: S4a_best_arma_resid_market.png")

# --- 4b. Individual stocks: which model wins most often ---
print(f"\n  4b. Individual stocks — which ARMA(p,q) wins AIC/BIC most often")
print(f"      (sample of {len(sample_ids)} stocks)")

aic_wins = {lbl: 0 for lbl in order_labels}
bic_wins = {lbl: 0 for lbl in order_labels}
best_resid_clean = 0

for sid in sample_ids:
    s = stock_series[sid].values
    best_aic_val, best_bic_val = np.inf, np.inf
    best_aic_lbl = best_bic_lbl = 'ARMA(0,0)'
    best_resid_lb = np.nan

    for order, lbl in zip(orders, order_labels):
        res = fit_arima_safe(s, order)
        if res is None:
            continue
        if res.aic < best_aic_val:
            best_aic_val, best_aic_lbl = res.aic, lbl
            lb = acorr_ljungbox(res.resid, lags=[12], return_df=True)
            best_resid_lb = lb['lb_pvalue'].iloc[0]
        if res.bic < best_bic_val:
            best_bic_val, best_bic_lbl = res.bic, lbl

    aic_wins[best_aic_lbl] += 1
    bic_wins[best_bic_lbl] += 1
    if not np.isnan(best_resid_lb) and best_resid_lb > 0.05:
        best_resid_clean += 1

total = len(sample_ids)
print(f"\n  {'Model':<14} {'Best AIC':>10} {'Best BIC':>10}")
print("  " + "-" * 36)
for lbl in order_labels:
    print(f"  {lbl:<14} {aic_wins[lbl]/total:>9.1%}  {bic_wins[lbl]/total:>9.1%}")

print(f"\n  Best-AIC model has clean residuals (LB lag 12 p>0.05): "
      f"{best_resid_clean/total:.1%} of stocks")

# Heatmap of AIC wins
aic_grid_arr = np.zeros((3, 3))
for p in range(3):
    for q in range(3):
        aic_grid_arr[p, q] = aic_wins[f'ARMA({p},{q})'] / total * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
im = axes[0].imshow(aic_grid_arr, cmap='Blues', aspect='auto')
axes[0].set_xticks([0,1,2]); axes[0].set_yticks([0,1,2])
axes[0].set_xticklabels(['q=0','q=1','q=2'])
axes[0].set_yticklabels(['p=0','p=1','p=2'])
axes[0].set_title('S4b: % of stocks — best AIC\nARMA(p,q)')
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, f'{aic_grid_arr[i,j]:.1f}%', ha='center', va='center', fontsize=10)
plt.colorbar(im, ax=axes[0])

bic_grid_arr = np.zeros((3, 3))
for p in range(3):
    for q in range(3):
        bic_grid_arr[p, q] = bic_wins[f'ARMA({p},{q})'] / total * 100

im2 = axes[1].imshow(bic_grid_arr, cmap='Oranges', aspect='auto')
axes[1].set_xticks([0,1,2]); axes[1].set_yticks([0,1,2])
axes[1].set_xticklabels(['q=0','q=1','q=2'])
axes[1].set_yticklabels(['p=0','p=1','p=2'])
axes[1].set_title('S4b: % of stocks — best BIC\nARMA(p,q)')
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f'{bic_grid_arr[i,j]:.1f}%', ha='center', va='center', fontsize=10)
plt.colorbar(im2, ax=axes[1])
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'S4b_arma_grid_wins.png'), dpi=150); plt.close()
print("\n  Saved: S4b_arma_grid_wins.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5  Summary — what the data actually says
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 68)
print("STEP 5  Summary")
print("═" * 68)

print(f"""
  Level A — Equal-weighted market return
  ───────────────────────────────────────
  ACF/PACF     : lag-1 ACF = {acf(mkt.values, nlags=1, fft=True)[1]:+.4f}  — visible or not?
  Ljung-Box    : see Step 2 output above
  Best AIC     : {best_aic_mkt}
  Best BIC     : {best_bic_mkt}
  AR(1) φ      : {res_ar1_mkt.arparams[0]:+.4f}

  Level B — Individual stocks ({len(sample_ids)} sampled)
  ──────────────────────────────────────────────────
  Mean AR(1) φ             : {ar1_phis.mean():+.4f}  (std {ar1_phis.std():.4f})
  LB reject after AR(1)    : {frac_resid_sig:.1%} of stocks
  Best-AIC model           : ARMA({max(aic_wins, key=aic_wins.get)})  wins {max(aic_wins.values())/total:.0%} of stocks
  Best-BIC model           : ARMA({max(bic_wins, key=bic_wins.get)})  wins {max(bic_wins.values())/total:.0%} of stocks
  Best model residuals clean: {best_resid_clean/total:.1%} of stocks
""")

print("  Interpretation guide")
print("  ─────────────────────")
print("  If AR(0) dominates BIC AND best-model residuals are clean")
print("  → returns are effectively white noise; AR/ARMA adds no signal.")
print()
print("  If a non-trivial ARMA wins frequently AND residuals become clean")
print("  → there IS exploitable autocorrelation; AR features could help.")
print()
print("  If residuals are never fully clean regardless of order")
print("  → conditional heteroskedasticity (GARCH), not AR structure.")
