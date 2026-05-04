# =============================================================================
# Stage 1 — Data loading, feature engineering, IC screening, Grouped PCA
#
# Run from Proj/:
#   ../venv/bin/python AK_data_changes/staged_run/stage1_preprocess.py
#
# Saves to AK_data_changes/files/checkpoints/:
#   FEATURES.json, LOG_FEATURES.json, IC_FEATURES.json
#   gpca.pkl, preprocessor.pkl
#   X_tr.npy, X_val.npy, X_te.npy
#   y_tr.npy, y_val.npy, y_te.npy
#   hist_avg.npy
#   test_meta.parquet, train_meta.parquet
# =============================================================================

import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from utils import (TARGET, TRAIN_END, VAL_END, META, FEATURE_GROUPS,
                   GroupedPCA, fit_preprocessor, apply_preprocessor,
                   apply_signed_log1p)

CKPT = Path('AK_data_changes/files/checkpoints')
CKPT.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading parquet...")
df = pd.read_parquet('../../jkp_data.parquet')
print(f"  Shape: {df.shape}")

df['log_me'] = np.log1p(df['me'].clip(lower=0))
FEATURES = list(dict.fromkeys([c for c in df.columns if c not in META] + ['log_me']))
df = df.dropna(subset=[TARGET])

train = df[df['eom'] <= TRAIN_END]
val   = df[(df['eom'] > TRAIN_END) & (df['eom'] <= VAL_END)]
test  = df[df['eom'] > VAL_END]
print(f"  Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")

# ── 2. Selective log-transform ────────────────────────────────────────────────
SKEW_THRESHOLD = 1.0
train_skew = train[FEATURES].apply(pd.to_numeric, errors='coerce').skew()
LOG_FEATURES = train_skew[train_skew.abs() > SKEW_THRESHOLD].index.tolist()
print(f"  Log-transform: {len(LOG_FEATURES)}/{len(FEATURES)} features (|skew| > {SKEW_THRESHOLD})")

df    = apply_signed_log1p(df, LOG_FEATURES)
train = df[df['eom'] <= TRAIN_END]
val   = df[(df['eom'] > TRAIN_END) & (df['eom'] <= VAL_END)]
test  = df[df['eom'] > VAL_END]

# ── 3. IC Screening ───────────────────────────────────────────────────────────
IC_THRESHOLD = 1.5

def compute_ic_matrix(df_split, feats):
    df_num = df_split[feats + [TARGET]].apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
    df_num['eom'] = df_split['eom'].values
    rows = []
    for _, grp in df_num.groupby('eom'):
        y = grp[TARGET].values
        row = []
        for f in feats:
            x = grp[f].values
            both_valid = ~np.isnan(x) & ~np.isnan(y)
            if both_valid.sum() >= 50:
                xv = x[both_valid]; yv = y[both_valid]
                xr = xv.argsort().argsort().astype(float)
                yr = yv.argsort().argsort().astype(float)
                xr -= xr.mean(); yr -= yr.mean()
                denom = np.sqrt((xr**2).sum() * (yr**2).sum())
                row.append(float(np.dot(xr, yr) / denom) if denom > 0 else 0.0)
            else:
                row.append(np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=feats)

print("Computing monthly Spearman IC on training data...")
ic_mat   = compute_ic_matrix(train, FEATURES)
mean_ic  = ic_mat.mean()
std_ic   = ic_mat.std()
T_ic     = ic_mat.notna().sum()
ic_tstat = mean_ic / std_ic * np.sqrt(T_ic)

IC_FEATURES = ic_tstat[ic_tstat.abs() >= IC_THRESHOLD].index.tolist()
print(f"  IC Screening: {len(FEATURES)} → {len(IC_FEATURES)} kept  (|t| ≥ {IC_THRESHOLD})")

del ic_mat  # free ~100 MB

# ── 4. Preprocessing (fit on training only) ───────────────────────────────────
to_num_ic = lambda df_: df_[IC_FEATURES].apply(pd.to_numeric, errors='coerce')

print("Fitting preprocessor and GroupedPCA...")
low_ic, high_ic, imp_ic, sc_ic = fit_preprocessor(to_num_ic(train))

X_tr_ic  = apply_preprocessor(to_num_ic(train), low_ic, high_ic, imp_ic, sc_ic)
X_val_ic = apply_preprocessor(to_num_ic(val),   low_ic, high_ic, imp_ic, sc_ic)
X_te_ic  = apply_preprocessor(to_num_ic(test),  low_ic, high_ic, imp_ic, sc_ic)

X_tr_ic_df  = pd.DataFrame(X_tr_ic,  columns=IC_FEATURES)
X_val_ic_df = pd.DataFrame(X_val_ic, columns=IC_FEATURES)
X_te_ic_df  = pd.DataFrame(X_te_ic,  columns=IC_FEATURES)

gpca      = GroupedPCA(FEATURE_GROUPS, var_threshold=0.90)
X_tr_pca  = gpca.fit_transform(X_tr_ic_df)
X_val_pca = gpca.transform(X_val_ic_df)
X_te_pca  = gpca.transform(X_te_ic_df)

y_tr = train[TARGET].values
y_val = val[TARGET].values
y_te  = test[TARGET].values
hist_avg = y_tr.mean()

print(f"  PCA output shape: {X_tr_pca.shape[1]} components")

# ── 5. Save checkpoints ───────────────────────────────────────────────────────
print("Saving checkpoints...")

json.dump(FEATURES,     open(CKPT / 'FEATURES.json',     'w'))
json.dump(LOG_FEATURES, open(CKPT / 'LOG_FEATURES.json', 'w'))
json.dump(IC_FEATURES,  open(CKPT / 'IC_FEATURES.json',  'w'))
json.dump(gpca.output_names, open(CKPT / 'gpca_output_names.json', 'w'))

pickle.dump(gpca,                           open(CKPT / 'gpca.pkl', 'wb'))
pickle.dump((low_ic, high_ic, imp_ic, sc_ic), open(CKPT / 'preprocessor.pkl', 'wb'))

np.save(CKPT / 'X_tr.npy',  X_tr_pca)
np.save(CKPT / 'X_val.npy', X_val_pca)
np.save(CKPT / 'X_te.npy',  X_te_pca)
np.save(CKPT / 'y_tr.npy',  y_tr)
np.save(CKPT / 'y_val.npy', y_val)
np.save(CKPT / 'y_te.npy',  y_te)
np.save(CKPT / 'hist_avg.npy', np.array([hist_avg]))

train[['id', 'eom', TARGET]].to_parquet(CKPT / 'train_meta.parquet', index=False)
test[['id', 'eom', TARGET]].to_parquet(CKPT / 'test_meta.parquet', index=False)

print("Stage 1 complete.")
print(f"  Features: {len(FEATURES)} raw → {len(IC_FEATURES)} IC → {X_tr_pca.shape[1]} PCA")
print(f"  Checkpoints saved to {CKPT}")
