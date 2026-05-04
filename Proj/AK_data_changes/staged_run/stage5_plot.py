# =============================================================================
# Stage 5 — Final comparison table + all plots
#
# Run from Proj/:
#   ../venv/bin/python AK_data_changes/staged_run/stage5_plot.py
#
# Requires: stage1, stage2, stage3 (optional), stage4 (optional) checkpoints
# Saves plots to AK_data_changes/files/results_staged/
# =============================================================================

import matplotlib
matplotlib.use('Agg')

import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from utils import TARGET, portfolio_weights, oos_r2

CKPT     = Path('AK_data_changes/files/checkpoints')
PLOT_DIR = Path('AK_data_changes/files/results_staged')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load all available checkpoints ───────────────────────────────────────────
print("Loading checkpoints...")

# Stage 2 (required)
results    = pd.read_parquet(CKPT / 'results.parquet')
ridge_pf   = pd.read_parquet(CKPT / 'ridge_pf.parquet')
ols_pf     = pd.read_parquet(CKPT / 'ols_pf.parquet')
s2_metrics = json.load(open(CKPT / 'stage2_metrics.json'))
ridge_coef = np.load(CKPT / 'ridge_coef.npy')
gpca_names = json.load(open(CKPT / 'gpca_output_names.json'))

# Stage 3 (optional)
ptk_pf = None
s3_metrics = {}
if (CKPT / 'ptk_returns.parquet').exists():
    ptk_pf = pd.read_parquet(CKPT / 'ptk_returns.parquet')
    if len(ptk_pf) == 0:
        ptk_pf = None
    if (CKPT / 'stage3_metrics.json').exists():
        s3_metrics = json.load(open(CKPT / 'stage3_metrics.json'))

# Stage 4 (optional)
nn_pf = None
s4_metrics = {}
if (CKPT / 'nn_pf.parquet').exists():
    nn_pf = pd.read_parquet(CKPT / 'nn_pf.parquet')
    if (CKPT / 'stage4_metrics.json').exists():
        s4_metrics = json.load(open(CKPT / 'stage4_metrics.json'))

# ── Plot 1: Ridge vs Market ───────────────────────────────────────────────────
ridge_pf.index = pd.to_datetime(ridge_pf.index)
cum = (1 + ridge_pf).cumprod() - 1

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(cum.index, cum['Market'] * 100,
        color='#888888', linestyle='--', lw=1.4,
        label=f"Market   (SR = {ridge_pf['Market'].mean() * 12 / (ridge_pf['Market'].std() * np.sqrt(12)):.2f})")
ax.plot(cum.index, cum['Ridge'] * 100,
        color='#d6604d', lw=2.0,
        label=f"Ridge    (SR = {s2_metrics['ridge_sharpe']:.2f})")
ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.4)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylabel('Cumulative excess return (%)')
ax.set_title('Expanding-window Ridge vs. Market — test period (Jan 2019 – Dec 2024)\n'
             'Ridge: rank-based zero-cost long–short  |  Market: equal-weighted long-only')
ax.legend(framealpha=0.9)
ax.grid(axis='y', lw=0.4, alpha=0.5)
fig.tight_layout()
plt.savefig(PLOT_DIR / 'ridge_vs_market.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved ridge_vs_market.png")

# ── Plot 2: Ridge feature importance ─────────────────────────────────────────
coefs = pd.Series(ridge_coef, index=gpca_names)
top_features = pd.concat([coefs.nlargest(10), coefs.nsmallest(10)]).sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#d6604d' if c < 0 else '#4393c3' for c in top_features.values]
top_features.plot(kind='barh', color=colors, ax=ax)
ax.set_title('Top 10 Positive and Negative Predictors (Final Ridge Model, trained through 2023)')
ax.set_xlabel('Coefficient value (standardised features)')
ax.grid(axis='x', lw=0.5, alpha=0.5)
fig.tight_layout()
plt.savefig(PLOT_DIR / 'ridge_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved ridge_feature_importance.png")

# ── Print comparison table ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FINAL COMPARISON: ALL MODELS")
print("=" * 80)

comparison_data = []

comparison_data.append({
    'Model': 'Historical Avg', 'Type': 'Baseline',
    'OOS R²': f"{s2_metrics['hist_r2_test']:+.4f}",
    'Sharpe': '0.00', 'IC Mean': 'N/A'
})

comparison_data.append({
    'Model': 'OLS', 'Type': 'Baseline',
    'OOS R²': f"{s2_metrics['ols_r2_test']:+.4f}",
    'Sharpe': f"{s2_metrics['ols_sharpe']:.2f}", 'IC Mean': 'N/A'
})

comparison_data.append({
    'Model': 'Ridge', 'Type': 'Linear (L2)',
    'OOS R²': f"{s2_metrics['ridge_r2']:+.4f}",
    'Sharpe': f"{s2_metrics['ridge_sharpe']:.2f}",
    'IC Mean': f"{s2_metrics['ridge_ic_mean']:+.4f}"
})

if s4_metrics:
    comparison_data.append({
        'Model': 'GKX NN3 (2020)', 'Type': 'Non-linear (MSE)',
        'OOS R²': f"{s4_metrics['r2_nn_paper']:+.4%}",
        'Sharpe': f"{s4_metrics['nn_sharpe']:.2f}",
        'IC Mean': f"{s4_metrics['nn_ic_mean']:+.4f}"
    })

if s3_metrics.get('ptk_sharpe') is not None:
    comparison_data.append({
        'Model': 'PTK-SDF (2024)', 'Type': 'Gradient-based',
        'OOS R²': 'N/A',
        'Sharpe': f"{s3_metrics['ptk_sharpe']:.2f}", 'IC Mean': 'N/A'
    })

comparison_df = pd.DataFrame(comparison_data)
print(f"\n{'Model':<22} {'Type':<22} {'OOS R²':<12} {'Sharpe':<10} {'IC Mean':<10}")
print("=" * 80)
for _, row in comparison_df.iterrows():
    print(f"{row['Model']:<22} {row['Type']:<22} {row['OOS R²']:<12} {row['Sharpe']:<10} {row['IC Mean']:<10}")
print("=" * 80)

# ── Plot 3: All models cumulative returns ─────────────────────────────────────
all_returns = pd.DataFrame()

ridge_pf.index = pd.to_datetime(ridge_pf.index)
ols_pf.index   = pd.to_datetime(ols_pf.index)
all_returns['Ridge']  = ridge_pf['Ridge']
all_returns['OLS']    = ols_pf['OLS']

if nn_pf is not None and len(nn_pf) > 0:
    nn_pf.index = pd.to_datetime(nn_pf.index)
    all_returns['GKX_NN3'] = nn_pf['NN_GKX']

if ptk_pf is not None and len(ptk_pf) > 0:
    ptk_pf.index = pd.to_datetime(ptk_pf.index)
    all_returns['PTK_SDF'] = ptk_pf['PTK_SDF']

all_returns['Market'] = ridge_pf['Market']

cum_returns = (1 + all_returns).cumprod() - 1

fig, ax = plt.subplots(figsize=(14, 6))

colors_map = {
    'OLS'    : '#a6bddb',
    'Ridge'  : '#4393c3',
    'GKX_NN3': '#238b45',
    'PTK_SDF': '#8c510a',
    'Market' : '#666666',
}

for col in cum_returns.columns:
    linestyle = '--' if col == 'Market' else '-'
    lw = 1.5 if col == 'Market' else 2.0
    ax.plot(cum_returns.index, cum_returns[col] * 100,
            color=colors_map.get(col, '#333333'),
            linestyle=linestyle, lw=lw, label=col)

ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.4)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylabel('Cumulative Excess Return (%)')
ax.set_title('Model Comparison: Out-of-Sample Performance (2019–2024)')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax.grid(axis='y', lw=0.4, alpha=0.5)
fig.tight_layout()
plt.savefig(PLOT_DIR / 'all_models_cumulative.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved all_models_cumulative.png")

print(f"\nAll plots saved to {PLOT_DIR}")
print("Stage 5 complete.")
