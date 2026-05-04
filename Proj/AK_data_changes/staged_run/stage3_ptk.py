# =============================================================================
# Stage 3 — PTK-SDF expanding window
#
# Run from Proj/:
#   ../venv/bin/python AK_data_changes/staged_run/stage3_ptk.py
#
# Requires: stage1 checkpoints
# Saves: ptk_returns.parquet, stage3_metrics.json
#
# This stage is isolated so that the heavy df_all_ptk copy does not coexist
# in memory with the Ridge or GKX arrays.
# =============================================================================

import sys
import json
import warnings
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

warnings.filterwarnings('ignore')

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw): return it

sys.path.insert(0, str(Path(__file__).parent))
from utils import TARGET, TRAIN_END, VAL_END, load_and_prep_df

CKPT = Path('AK_data_changes/files/checkpoints')

# ── Device (MPS → CUDA → CPU, no override) ───────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print(f'Device: {DEVICE}')

# ── PTK config ────────────────────────────────────────────────────────────────
class Config:
    HIDDEN_DIM           = 32
    DNN_EPOCHS           = 15
    DNN_LEARNING_RATE    = 1e-3
    RIDGE_PENALTY        = 1e-3
    ROLLING_WINDOW       = 12
    MIN_STOCKS_PER_MONTH = 20
    MAX_TRAIN_SAMPLES    = 100000
    PTK_TRAIN_SPLIT      = 0.7

ptk_cfg = Config()

# ── PTK model classes ─────────────────────────────────────────────────────────
class PTKNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(-1)


def msrr_loss(predictions, returns):
    portfolio_return = (predictions * returns).sum()
    return ((1 - portfolio_return) ** 2) / len(returns)


def train_dnn(X_train, R_train, input_dim, cfg):
    model = PTKNetwork(input_dim, cfg.HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.DNN_LEARNING_RATE)
    X_t = torch.FloatTensor(X_train).to(DEVICE)
    R_t = torch.FloatTensor(R_train).to(DEVICE)
    for epoch in range(cfg.DNN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = msrr_loss(model(X_t), R_t)
        loss.backward()
        optimizer.step()
    return model.cpu()


def compute_ptk_gradients(model, X, R):
    model.eval()
    X_t = torch.FloatTensor(X).requires_grad_(True)
    R_t = torch.FloatTensor(R)
    predictions = model(X_t)
    portfolio_return = (predictions * R_t).sum()
    gradients = torch.autograd.grad(
        outputs=portfolio_return, inputs=model.parameters(),
        create_graph=False, retain_graph=False
    )
    scaling = 1.0 / np.sqrt(len(R))
    grad_vector = torch.cat([g.flatten() for g in gradients])
    return (scaling * grad_vector).detach().numpy()


class PTK_Pricer:
    def __init__(self, ridge_penalty):
        self.ridge_penalty = ridge_penalty
        self.weights = None
        self.scaler = StandardScaler()

    def fit(self, F):
        T, P = F.shape
        F_scaled = self.scaler.fit_transform(F)
        F_bar = F_scaled.mean(axis=0)
        covariance = (F_scaled.T @ F_scaled) / T
        zI = self.ridge_penalty * np.eye(P)
        try:
            inv_matrix = np.linalg.inv(zI + covariance)
            self.weights = inv_matrix @ F_bar
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(zI + covariance) @ F_bar
        return self

    def predict(self, F_new):
        F_new_scaled = self.scaler.transform(F_new.reshape(1, -1))
        return (F_new_scaled @ self.weights)[0]


def preprocess_data_paper_style(df, features, verbose=True):
    df_processed = df.copy()
    missing_pct = df_processed[features].isna().mean()
    features = missing_pct[missing_pct < 0.30].index.tolist()
    if verbose:
        print(f"  Features kept (<30% missing): {len(features)}")

    def _rank_std(s):
        r = s.rank(method='average')
        n = r.notna().sum()
        return (r / (n - 1) - 0.5) if n > 1 else r * 0.0

    df_processed[features] = df_processed.groupby('eom')[features].transform(_rank_std)
    df_processed[features] = df_processed.groupby('eom')[features].transform(
        lambda s: s.fillna(s.median())
    )
    return df_processed, features


def get_monthly_data(df_subset, features):
    X_list, R_list, month_list = [], [], []
    for month, grp in df_subset.groupby('eom'):
        X_month = grp[features].values
        R_month = grp[TARGET].values
        if len(X_month) >= ptk_cfg.MIN_STOCKS_PER_MONTH:
            X_list.append(X_month)
            R_list.append(R_month)
            month_list.append(month)
    return X_list, R_list, month_list


# ── Load stage-1 outputs ──────────────────────────────────────────────────────
IC_FEATURES  = json.load(open(CKPT / 'IC_FEATURES.json'))
LOG_FEATURES = json.load(open(CKPT / 'LOG_FEATURES.json'))

print("Loading and preparing data for PTK...")
df   = load_and_prep_df('../jkp_data.parquet', LOG_FEATURES)
test = df[df['eom'] > VAL_END]
test_months = sorted(test['eom'].unique())
print(f"  Test months: {len(test_months)}")

print("Applying PTK preprocessing (rank-standardize to [-0.5, 0.5])...")
df_all_ptk, ptk_features = preprocess_data_paper_style(df, IC_FEATURES, verbose=True)
print(f"  PTK Features kept: {len(ptk_features)}")

# Free the original df — df_all_ptk is all we need now
del df
import gc; gc.collect()

# ── PTK expanding window ──────────────────────────────────────────────────────
ptk_sdf_returns_list = []
ptk_stock_predictions = []

for test_month in tqdm(test_months, desc="Expanding Window PTK"):
    train_data = df_all_ptk[df_all_ptk['eom'] < test_month]

    if len(train_data) < ptk_cfg.MIN_STOCKS_PER_MONTH * 12:
        continue

    X_monthly, R_monthly, month_list = get_monthly_data(train_data, ptk_features)

    if len(X_monthly) < ptk_cfg.ROLLING_WINDOW:
        continue

    window = min(ptk_cfg.ROLLING_WINDOW, len(X_monthly))
    X_train_stack = [X_monthly[i] for i in range(-window, 0)]
    R_train_stack = [R_monthly[i] for i in range(-window, 0)]
    X_train_full  = np.vstack(X_train_stack)
    R_train_full  = np.concatenate(R_train_stack)

    if len(X_train_full) > ptk_cfg.MAX_TRAIN_SAMPLES:
        idx = np.random.choice(len(X_train_full), ptk_cfg.MAX_TRAIN_SAMPLES, replace=False)
        X_train_full = X_train_full[idx]
        R_train_full = R_train_full[idx]

    scaler_local = StandardScaler()
    X_train_scaled = scaler_local.fit_transform(X_train_full)

    try:
        model = train_dnn(X_train_scaled, R_train_full, len(ptk_features), ptk_cfg)
    except Exception:
        continue

    ptk_factors = []
    ptk_factor_months = []
    for i in range(len(X_monthly)):
        X_month = X_monthly[i]
        R_month = R_monthly[i]
        if len(X_month) < ptk_cfg.MIN_STOCKS_PER_MONTH:
            continue
        X_scaled = scaler_local.transform(X_month)
        try:
            factors = compute_ptk_gradients(model, X_scaled, R_month)
            ptk_factors.append(factors)
            ptk_factor_months.append(month_list[i])
        except Exception:
            continue

    if len(ptk_factors) < 12:
        continue

    F_matrix = np.array(ptk_factors)
    n_train  = int(len(F_matrix) * ptk_cfg.PTK_TRAIN_SPLIT)
    if n_train < 6:
        continue

    F_train          = F_matrix[:n_train]
    F_test           = F_matrix[n_train:]
    test_months_ptk  = ptk_factor_months[n_train:]

    try:
        pricer = PTK_Pricer(ridge_penalty=ptk_cfg.RIDGE_PENALTY)
        pricer.fit(F_train)
    except Exception:
        continue

    sdf_return = None
    for i, month in enumerate(test_months_ptk):
        if month == test_month:
            sdf_return = pricer.predict(F_test[i])
            break

    if sdf_return is not None:
        ptk_sdf_returns_list.append({'date': test_month, 'PTK_SDF': sdf_return})

    test_month_data = df_all_ptk[df_all_ptk['eom'] == test_month]
    if len(test_month_data) >= ptk_cfg.MIN_STOCKS_PER_MONTH:
        X_test_month  = test_month_data[ptk_features].values
        R_test_month  = test_month_data[TARGET].values
        X_test_scaled = scaler_local.transform(X_test_month)
        try:
            test_factor = compute_ptk_gradients(model, X_test_scaled, R_test_month)
            sdf_return  = pricer.predict(test_factor)
            ptk_sdf_returns_list.append({'date': test_month, 'PTK_SDF': sdf_return})
        except Exception:
            pass

# ── Compute PTK metrics and save ──────────────────────────────────────────────
if len(ptk_sdf_returns_list) > 0:
    sdf_returns = np.array([item['PTK_SDF'] for item in ptk_sdf_returns_list])
    sdf_dates   = [item['date'] for item in ptk_sdf_returns_list]

    ptk_pf = pd.DataFrame({'PTK_SDF': sdf_returns},
                          index=pd.to_datetime(sdf_dates)).sort_index()

    ptk_sharpe  = float(np.sqrt(12) * sdf_returns.mean() / (sdf_returns.std() + 1e-8))
    ptk_ann_ret = float(sdf_returns.mean() * 12)
    ptk_ann_vol = float(sdf_returns.std() * np.sqrt(12))

    print(f"\nPTK-SDF  Ann.Ret: {ptk_ann_ret:.2%}  Vol: {ptk_ann_vol:.2%}  Sharpe: {ptk_sharpe:.2f}")

    ptk_pf.to_parquet(CKPT / 'ptk_returns.parquet')
    json.dump({'ptk_sharpe': ptk_sharpe, 'ptk_ann_ret': ptk_ann_ret,
               'ptk_ann_vol': ptk_ann_vol, 'ptk_n_months': len(ptk_pf)},
              open(CKPT / 'stage3_metrics.json', 'w'))
    print("Stage 3 complete.")
else:
    print("\nNo PTK-SDF predictions generated — empty checkpoint written.")
    pd.DataFrame(columns=['PTK_SDF']).to_parquet(CKPT / 'ptk_returns.parquet')
    json.dump({'ptk_sharpe': None}, open(CKPT / 'stage3_metrics.json', 'w'))
