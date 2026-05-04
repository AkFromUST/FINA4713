"""
ic_screening.py  —  Information Coefficient feature screening
--------------------------------------------------------------
For each of the 185 features, computes monthly cross-sectional
Spearman rank correlation with next-month excess return:

    IC_j,t = Spearmanr( feature_j across stocks in month t,
                         ret_exc_lead1m across those stocks )

Summary stats per feature:
    mean_IC   — average predictive power over training months
    std_IC    — consistency
    IC_tstat  = mean_IC / std_IC * sqrt(T)
    ICIR      = mean_IC / std_IC   (information ratio)

Filter: keep features where |IC_tstat| >= THRESHOLD on training set.
Cross-check on validation set to guard against look-ahead.

Outputs (all in AK_data_changes/files/results/):
    ic_stats.csv            — full table for all 185 features
    selected_features.json  — kept feature names, ready for GKX NN3
    ic_tstat_bar.png        — ranked bar chart, coloured by keep/drop
    ic_train_val_scatter.png — train vs val IC t-stat (stability check)
    ic_threshold_table.png  — how many features survive at 1.0 / 1.5 / 2.0

Run from Proj/:  ../venv/bin/python AK_data_changes/ic_screening.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, '..', 'jkp_data.parquet')
OUT_DIR     = os.path.join(SCRIPT_DIR, 'files', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET      = 'ret_exc_lead1m'
TRAIN_END   = '2015-12-31'
VAL_END     = '2018-12-31'
THRESHOLD   = 1.5      # primary keep/drop cutoff on |IC t-stat|
MIN_STOCKS  = 50       # minimum stocks per month for IC to be meaningful
MISSING_CAP = 0.50     # feature filter applied before IC (same as ak_pipeline)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
df['eom'] = pd.to_datetime(df['eom'])
df['log_me'] = np.log1p(df['me'].clip(lower=0))
df = df.dropna(subset=[TARGET])

META     = ['id', 'eom', 'excntry', TARGET, 'me']
ALL_FEAT = [c for c in df.columns if c not in META]   # includes log_me

# Cast object columns to numeric
for c in ALL_FEAT:
    if df[c].dtype == object:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Temporal splits
train = df[df['eom'] <= TRAIN_END].copy()
val   = df[(df['eom'] > TRAIN_END) & (df['eom'] <= VAL_END)].copy()

# Apply <50% missing filter (same as ak_pipeline)
missing_pct = train[ALL_FEAT].isna().mean()
FEATURES    = missing_pct[missing_pct < MISSING_CAP].index.tolist()

print(f"Features after <50% filter : {len(FEATURES)}")
print(f"Train months : {train['eom'].nunique()}  "
      f"({train['eom'].min().date()} – {train['eom'].max().date()})")
print(f"Val months   : {val['eom'].nunique()}  "
      f"({val['eom'].min().date()} – {val['eom'].max().date()})")


# ── Core: compute monthly IC matrix ──────────────────────────────────────────
def compute_ic_matrix(df_split, features, target=TARGET, min_stocks=MIN_STOCKS):
    """
    Returns DataFrame of shape (T_months × n_features) with Spearman IC values.
    NaN for months where a feature has too few non-missing observations.
    IC is computed on raw (unpreprocessed) features — Spearman rank correlation
    is robust to outliers and scale, so no preprocessing needed here.
    """
    months = sorted(df_split['eom'].unique())
    rows   = []

    for month in months:
        grp = df_split[df_split['eom'] == month][[target] + features].dropna(subset=[target])
        n   = len(grp)
        if n < min_stocks:
            rows.append([np.nan] * len(features))
            continue

        y     = grp[target].values
        row   = []
        for feat in features:
            x     = grp[feat].values.astype(float)
            valid = ~np.isnan(x)
            if valid.sum() < min_stocks:
                row.append(np.nan)
            else:
                r, _ = spearmanr(x[valid], y[valid])
                row.append(r)
        rows.append(row)

    return pd.DataFrame(rows, index=months, columns=features)


# ── Compute IC matrices ───────────────────────────────────────────────────────
print("\nComputing monthly IC on training set...")
ic_train = compute_ic_matrix(train, FEATURES)
print(f"  IC matrix shape: {ic_train.shape}  "
      f"(months × features)")

print("Computing monthly IC on validation set...")
ic_val = compute_ic_matrix(val, FEATURES)
print(f"  IC matrix shape: {ic_val.shape}")


# ── Summarise ─────────────────────────────────────────────────────────────────
def summarise_ic(ic_matrix, suffix=''):
    T       = ic_matrix.notna().sum()           # effective months per feature
    mean_ic = ic_matrix.mean()
    std_ic  = ic_matrix.std()
    tstat   = mean_ic / std_ic * np.sqrt(T)
    icir    = mean_ic / std_ic

    return pd.DataFrame({
        f'mean_ic{suffix}' : mean_ic,
        f'std_ic{suffix}'  : std_ic,
        f'tstat{suffix}'   : tstat,
        f'icir{suffix}'    : icir,
        f'T{suffix}'       : T,
    })

stats_train = summarise_ic(ic_train, '_train')
stats_val   = summarise_ic(ic_val,   '_val')
stats       = pd.concat([stats_train, stats_val], axis=1)
stats.index.name = 'feature'
stats = stats.sort_values('tstat_train', key=abs, ascending=False)


# ── Apply filter ──────────────────────────────────────────────────────────────
stats['keep'] = stats['tstat_train'].abs() >= THRESHOLD
n_keep = stats['keep'].sum()
n_drop = (~stats['keep']).sum()

print(f"\n{'─'*55}")
print(f"IC t-stat threshold : |t| ≥ {THRESHOLD}")
print(f"  Keep : {n_keep} features")
print(f"  Drop : {n_drop} features")
print(f"{'─'*55}")


# ── Print top / bottom features ───────────────────────────────────────────────
print(f"\nTop 20 features by |IC t-stat| (training):")
print(f"  {'Feature':<28} {'Mean IC':>9} {'Std IC':>9} {'t-stat':>9} "
      f"{'val t-stat':>11}  Keep?")
print("  " + "─" * 75)
for feat, row in stats.head(20).iterrows():
    flag = '✓' if row['keep'] else '✗'
    print(f"  {feat:<28} {row['mean_ic_train']:>+9.4f} {row['std_ic_train']:>9.4f} "
          f"{row['tstat_train']:>+9.2f} {row['tstat_val']:>+11.2f}  {flag}")

print(f"\nBottom 10 features (weakest signal):")
print(f"  {'Feature':<28} {'Mean IC':>9} {'t-stat':>9}  Keep?")
print("  " + "─" * 55)
for feat, row in stats.tail(10).iterrows():
    flag = '✓' if row['keep'] else '✗'
    print(f"  {feat:<28} {row['mean_ic_train']:>+9.4f} {row['tstat_train']:>+9.2f}  {flag}")


# ── Threshold sensitivity ─────────────────────────────────────────────────────
print(f"\nFeatures kept at different thresholds:")
print(f"  {'Threshold':>10} {'Kept':>8} {'Dropped':>10}")
print("  " + "─" * 30)
for thr in [1.0, 1.5, 2.0, 2.5, 3.0]:
    k = (stats['tstat_train'].abs() >= thr).sum()
    print(f"  {thr:>10.1f} {k:>8} {len(stats)-k:>10}")


# ── Train vs val stability check ──────────────────────────────────────────────
kept_stats = stats[stats['keep']]
both_pos   = ((kept_stats['tstat_train'] > 0) == (kept_stats['tstat_val'] > 0)).mean()
print(f"\nAmong the {n_keep} kept features:")
print(f"  Same sign in train & val  : {both_pos:.1%}")
print(f"  Mean |val t-stat|         : {kept_stats['tstat_val'].abs().mean():.2f}")


# ── Save outputs ──────────────────────────────────────────────────────────────
stats.to_csv(os.path.join(OUT_DIR, 'ic_stats.csv'))

selected = stats[stats['keep']].index.tolist()
with open(os.path.join(OUT_DIR, 'selected_features.json'), 'w') as f:
    json.dump(selected, f, indent=2)
print(f"\nSaved: ic_stats.csv  ({len(stats)} features)")
print(f"Saved: selected_features.json  ({len(selected)} features)")


# ── Figure 1: ranked IC t-stat bar chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
x      = np.arange(len(stats))
colors = ['#2166ac' if t else '#d6604d' for t in stats['keep']]

ax.bar(x, stats['tstat_train'].values, color=colors, width=1.0, linewidth=0)
ax.axhline( THRESHOLD, color='black', lw=1.2, linestyle='--',
            label=f'+{THRESHOLD} threshold')
ax.axhline(-THRESHOLD, color='black', lw=1.2, linestyle='--')
ax.axhline(0, color='black', lw=0.6)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2166ac', label=f'Keep ({n_keep})'),
                   Patch(facecolor='#d6604d', label=f'Drop ({n_drop})')]
ax.legend(handles=legend_elements, loc='upper right')
ax.set_xlabel('Feature (ranked by |IC t-stat|)')
ax.set_ylabel('IC t-statistic (training, 2005–2015)')
ax.set_title(f'IC Screening — all {len(stats)} features ranked by |IC t-stat|  '
             f'(threshold = ±{THRESHOLD})')
ax.set_xlim(-1, len(stats))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'ic_tstat_bar.png'), dpi=150)
plt.close()
print("Saved: ic_tstat_bar.png")


# ── Figure 2: train vs val IC t-stat scatter ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
kept = stats[stats['keep']]
drop = stats[~stats['keep']]

ax.scatter(drop['tstat_train'], drop['tstat_val'],
           s=12, alpha=0.5, color='#d6604d', label=f'Dropped ({n_drop})', zorder=2)
ax.scatter(kept['tstat_train'], kept['tstat_val'],
           s=18, alpha=0.8, color='#2166ac', label=f'Kept ({n_keep})', zorder=3)

lim = max(abs(stats['tstat_train']).max(), abs(stats['tstat_val']).max()) * 1.05
ax.axhline(0, color='black', lw=0.7); ax.axvline(0, color='black', lw=0.7)
ax.axvline( THRESHOLD, color='black', lw=1, linestyle='--', alpha=0.6)
ax.axvline(-THRESHOLD, color='black', lw=1, linestyle='--', alpha=0.6)
ax.plot([-lim, lim], [-lim, lim], color='grey', lw=0.8, linestyle=':', alpha=0.6,
        label='y = x')
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('IC t-stat — training (2005–2015)')
ax.set_ylabel('IC t-stat — validation (2016–2018)')
ax.set_title('Stability: train vs val IC t-stat per feature\n'
             '(dots near y=x → signal is consistent across periods)')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'ic_train_val_scatter.png'), dpi=150)
plt.close()
print("Saved: ic_train_val_scatter.png")


# ── Figure 3: threshold sensitivity table ────────────────────────────────────
thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
kept_counts = [(stats['tstat_train'].abs() >= t).sum() for t in thresholds]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(range(len(thresholds)), kept_counts,
              color='#4393c3', width=0.5, edgecolor='white')
for bar, k in zip(bars, kept_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(k), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.axhline(kept_counts[1], color='red', lw=1.5, linestyle='--',
           label=f'Our choice (t≥{THRESHOLD}) → {kept_counts[1]} features')
ax.set_xticks(range(len(thresholds)))
ax.set_xticklabels([f'|t| ≥ {t}' for t in thresholds])
ax.set_ylabel('Features kept')
ax.set_title('IC Threshold Sensitivity — how many features survive?')
ax.set_ylim(0, len(stats) * 1.1)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'ic_threshold_sensitivity.png'), dpi=150)
plt.close()
print("Saved: ic_threshold_sensitivity.png")


# ── Final summary ─────────────────────────────────────────────────────────────
print(f"""
{'═'*55}
IC SCREENING COMPLETE
{'═'*55}
  Started with          : {len(FEATURES)} features
  Kept (|t| ≥ {THRESHOLD})       : {n_keep} features
  Dropped               : {n_drop} features

  Signal stability (kept features):
    Same sign train/val : {both_pos:.1%}
    Mean |val t-stat|   : {kept_stats['tstat_val'].abs().mean():.2f}

  Selected features saved to:
    {OUT_DIR}/selected_features.json
{'═'*55}
""")
