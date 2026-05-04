# =============================================================================
# Stage 2 — Ridge regression (alpha tuning + expanding window) + OLS
#
# Run from Proj/:
#   ../venv/bin/python AK_data_changes/staged_run/stage2_ridge.py
#
# Requires: stage1 checkpoints
# Saves: results.parquet, ridge_pf.parquet, ols_pf.parquet,
#        best_alpha.json, ridge_coef.npy, stage2_metrics.json
# =============================================================================

import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LinearRegression

warnings.filterwarnings('ignore')

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw): return it

sys.path.insert(0, str(Path(__file__).parent))
from utils import (TARGET, TRAIN_END, VAL_END, FEATURE_GROUPS, GroupedPCA,
                   fit_preprocessor, apply_preprocessor, apply_signed_log1p,
                   load_and_prep_df, portfolio_weights, oos_r2)

CKPT = Path('AK_data_changes/files/checkpoints')

# ── Load stage-1 checkpoints ──────────────────────────────────────────────────
print("Loading stage-1 checkpoints...")
IC_FEATURES  = json.load(open(CKPT / 'IC_FEATURES.json'))
LOG_FEATURES = json.load(open(CKPT / 'LOG_FEATURES.json'))
gpca         = pickle.load(open(CKPT / 'gpca.pkl', 'rb'))

X_tr  = np.load(CKPT / 'X_tr.npy')
X_val = np.load(CKPT / 'X_val.npy')
X_te  = np.load(CKPT / 'X_te.npy')
y_tr  = np.load(CKPT / 'y_tr.npy')
y_val = np.load(CKPT / 'y_val.npy')
y_te  = np.load(CKPT / 'y_te.npy')
hist_avg     = float(np.load(CKPT / 'hist_avg.npy')[0])
train_meta   = pd.read_parquet(CKPT / 'train_meta.parquet')
test_meta    = pd.read_parquet(CKPT / 'test_meta.parquet')

y_null_val = np.full(len(y_val), hist_avg)
y_null_te  = np.full(len(y_te),  hist_avg)

to_num_ic = lambda df_: df_[IC_FEATURES].apply(pd.to_numeric, errors='coerce')

# ── 1. Ridge alpha tuning on validation set ───────────────────────────────────
alpha_grid = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

def compute_portfolio_metrics(df_eval, pred_col):
    monthly = []
    for _, grp in df_eval.groupby('eom'):
        w = portfolio_weights(grp[pred_col].values)
        monthly.append((w * grp[TARGET].values).sum())
    pf = pd.Series(monthly)
    ann_ret = pf.mean() * 12
    ann_vol = pf.std() * np.sqrt(12)
    return ann_ret, ann_vol, (ann_ret / ann_vol if ann_vol > 0 else 0.0)

print(f"\n{'Alpha':>10} | {'Val Sharpe':>10} | {'Val Ann. Ret':>12}")
print("-" * 38)

print("Reloading parquet for expanding window and val metadata...")
df      = load_and_prep_df('../jkp_data.parquet', LOG_FEATURES)
val_df  = df[(df['eom'] > TRAIN_END) & (df['eom'] <= VAL_END)]
test_df = df[df['eom'] > VAL_END]

val_metrics = {}
val_eval = val_df[['eom', TARGET]].copy().reset_index(drop=True)
for alpha in alpha_grid:
    m = Ridge(alpha=alpha).fit(X_tr, y_tr)
    val_eval['ridge_pred'] = m.predict(X_val)
    ann_ret, _, sharpe = compute_portfolio_metrics(val_eval, 'ridge_pred')
    val_metrics[alpha] = sharpe
    print(f"{alpha:>10.0f} | {sharpe:>10.4f} | {ann_ret:>12.2%}")

best_alpha = max(val_metrics, key=val_metrics.get)
print(f"\n-> Best α = {best_alpha:.0f}  (val Sharpe = {val_metrics[best_alpha]:.4f})")

# ── 2. Ridge expanding window ─────────────────────────────────────────────────
records = []

for year in tqdm(range(2019, 2025), desc="Walk-Forward Years"):
    train_cut  = pd.Timestamp(f'{year}-01-01')
    test_start = pd.Timestamp(f'{year}-01-01')
    test_end   = pd.Timestamp(f'{year + 1}-01-01')

    df_tr = df[df['eom'] < train_cut]
    df_te = df[(df['eom'] >= test_start) & (df['eom'] < test_end)].dropna(subset=[TARGET])
    if df_te.empty:
        continue

    lo, hi, imp_r, sc_r = fit_preprocessor(to_num_ic(df_tr))
    Xtr_ic = apply_preprocessor(to_num_ic(df_tr), lo, hi, imp_r, sc_r)
    Xte_ic = apply_preprocessor(to_num_ic(df_te), lo, hi, imp_r, sc_r)

    Xtr = gpca.transform(pd.DataFrame(Xtr_ic, columns=IC_FEATURES))
    Xte = gpca.transform(pd.DataFrame(Xte_ic, columns=IC_FEATURES))
    ytr = df_tr[TARGET].values

    ridge = Ridge(alpha=best_alpha).fit(Xtr, ytr)

    tmp = df_te[['eom', TARGET]].copy().reset_index(drop=True)
    tmp['ridge_pred'] = ridge.predict(Xte)
    tmp['hist_avg']   = ytr.mean()
    records.append(tmp)

    print(f"  {year}: train {len(df_tr):>7,} obs  →  test {len(df_te):>6,} obs")

results = pd.concat(records, ignore_index=True)
print(f"\nTotal test obs: {len(results):,}")

# ── 3. Ridge portfolio construction ──────────────────────────────────────────
monthly = []
for eom_date, grp in results.groupby('eom'):
    ret_xs = grp[TARGET].values
    monthly.append({
        'date'   : eom_date,
        'Ridge'  : (portfolio_weights(grp['ridge_pred'].values) * ret_xs).sum(),
        'HistAvg': 0.0,
        'Market' : ret_xs.mean(),
    })

ridge_pf = pd.DataFrame(monthly).set_index('date').sort_index()
ridge_pf.index = pd.to_datetime(ridge_pf.index)

# Ridge metrics
y_true = results[TARGET].values
y_null = results['hist_avg'].values
ridge_r2 = oos_r2(y_true, results['ridge_pred'].values, y_null)
ic_ridge  = results.groupby('eom').apply(lambda g: spearmanr(g['ridge_pred'], g[TARGET])[0])
ridge_ic_mean   = float(ic_ridge.mean())
ridge_ic_tstat  = float(ic_ridge.mean() / ic_ridge.std() * np.sqrt(len(ic_ridge)))
ridge_sharpe    = float(ridge_pf['Ridge'].mean() * 12 / (ridge_pf['Ridge'].std() * np.sqrt(12)))

# ── 4. OLS ────────────────────────────────────────────────────────────────────
print("\nTraining OLS...")
ols_model = LinearRegression().fit(X_tr, y_tr)
ols_pred_test = ols_model.predict(X_te)

stock_hist_avg = train_meta.groupby('id')[TARGET].mean()
test_hist_avg  = test_meta['id'].map(stock_hist_avg).fillna(hist_avg).values
hist_r2_test   = oos_r2(y_te, test_hist_avg, test_hist_avg)
ols_r2_test    = oos_r2(y_te, ols_pred_test, test_hist_avg)

test_df_reset = test_df[['eom', TARGET]].copy().reset_index(drop=True)
test_df_reset['ols_pred'] = ols_pred_test

ols_portfolio = []
for eom_date, grp in test_df_reset.groupby('eom'):
    if len(grp) > 0:
        ret_xs = grp[TARGET].values
        ols_portfolio.append({
            'date': eom_date,
            'OLS': (portfolio_weights(grp['ols_pred'].values) * ret_xs).sum(),
        })

ols_pf = pd.DataFrame(ols_portfolio).set_index('date').sort_index()
ols_pf.index = pd.to_datetime(ols_pf.index)
ols_sharpe = float(ols_pf['OLS'].mean() * 12 / (ols_pf['OLS'].std() * np.sqrt(12)))

print(f"  OLS Test R²: {ols_r2_test:+.4f}  |  Sharpe: {ols_sharpe:.2f}")
print(f"  Ridge Test R²: {ridge_r2:+.4f}  |  Sharpe: {ridge_sharpe:.2f}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
print("Saving stage-2 outputs...")
results.to_parquet(CKPT / 'results.parquet', index=False)
ridge_pf.to_parquet(CKPT / 'ridge_pf.parquet')
ols_pf.to_parquet(CKPT / 'ols_pf.parquet')
np.save(CKPT / 'ridge_coef.npy', ridge.coef_)
json.dump({'best_alpha': best_alpha}, open(CKPT / 'best_alpha.json', 'w'))
json.dump({
    'hist_r2_test' : hist_r2_test,
    'ols_r2_test'  : ols_r2_test,
    'ols_sharpe'   : ols_sharpe,
    'ridge_r2'     : ridge_r2,
    'ridge_ic_mean': ridge_ic_mean,
    'ridge_ic_tstat': ridge_ic_tstat,
    'ridge_sharpe' : ridge_sharpe,
}, open(CKPT / 'stage2_metrics.json', 'w'))

print("Stage 2 complete.")
