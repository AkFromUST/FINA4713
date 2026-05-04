# =============================================================================
# Stage 4 — GKX NN3 (Gu, Kelly & Xiu 2020)
#
# Run from Proj/:
#   ../venv/bin/python AK_data_changes/staged_run/stage4_gkx.py
#
# Requires: stage1 checkpoints
# Saves: nn_pf.parquet, nn_ic.parquet, stage4_metrics.json
# =============================================================================

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from utils import TARGET, portfolio_weights, oos_r2

CKPT = Path('AK_data_changes/files/checkpoints')
MIN_STOCKS_PER_MONTH = 20

# ── Device (MPS → CUDA → CPU, no override) ───────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print(f'Device: {DEVICE}')
if DEVICE.type == 'mps':
    print('  Apple Silicon GPU (MPS) — Metal acceleration enabled')
elif DEVICE.type == 'cuda':
    print(f'  NVIDIA GPU: {torch.cuda.get_device_name(0)}')

# ── Load stage-1 checkpoints ──────────────────────────────────────────────────
print("Loading checkpoints...")
X_tr  = np.load(CKPT / 'X_tr.npy')
X_val = np.load(CKPT / 'X_val.npy')
X_te  = np.load(CKPT / 'X_te.npy')
y_tr  = np.load(CKPT / 'y_tr.npy')
y_val = np.load(CKPT / 'y_val.npy')
y_te  = np.load(CKPT / 'y_te.npy')
test_meta = pd.read_parquet(CKPT / 'test_meta.parquet')
hist_avg  = float(np.load(CKPT / 'hist_avg.npy')[0])

print(f"  X_train: {X_tr.shape}  X_val: {X_val.shape}  X_test: {X_te.shape}")

# ── GKX NN3 training function ─────────────────────────────────────────────────
def train_gkx_nn3(X_train, y_train, X_val_arr, y_val_arr, input_dim,
                  batch_size=1024, learning_rate=0.001,
                  lambda_l1=1e-5, patience=5, max_epochs=100):

    class GKX_NN3(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.layer1    = nn.Linear(d, 32)
            self.layer2    = nn.Linear(32, 16)
            self.layer3    = nn.Linear(16, 8)
            self.output    = nn.Linear(8, 1)
            self.activation = nn.ReLU()
        def forward(self, x):
            x = self.activation(self.layer1(x))
            x = self.activation(self.layer2(x))
            x = self.activation(self.layer3(x))
            return self.output(x).squeeze(-1)

    model     = GKX_NN3(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0,
                         pin_memory=(DEVICE.type == 'cuda'))

    X_val_t = torch.FloatTensor(X_val_arr).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val_arr).to(DEVICE)

    best_val_loss   = float('inf')
    best_state_dict = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            mse_loss = criterion(model(X_batch), y_batch)
            l1_loss  = sum(p.abs().sum() for p in model.parameters())
            (mse_loss + lambda_l1 * l1_loss).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state_dict  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model.cpu()

# ── Scale target ──────────────────────────────────────────────────────────────
scaler_y = StandardScaler()
scaler_y.fit(y_tr.reshape(-1, 1))
y_tr_scaled  = scaler_y.transform(y_tr.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

# ── Train 5-seed ensemble ─────────────────────────────────────────────────────
print("\nTraining GKX NN3 ensemble (5 seeds)...")
nn_predictions_list = []

for seed in [42, 123, 456, 789, 101112]:
    print(f"  seed={seed}...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = train_gkx_nn3(
        X_tr, y_tr_scaled, X_val, y_val_scaled,
        input_dim=X_tr.shape[1],
        batch_size=1024, learning_rate=0.001,
        lambda_l1=1e-5, patience=5, max_epochs=100
    )

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_te)).numpy()
    nn_predictions_list.append(pred_scaled)

nn_pred_scaled = np.mean(nn_predictions_list, axis=0)
nn_pred = scaler_y.inverse_transform(nn_pred_scaled.reshape(-1, 1)).flatten()

# ── Evaluation ────────────────────────────────────────────────────────────────
y_true_nn = y_te
ss_res    = np.sum((y_true_nn - nn_pred) ** 2)
r2_nn_paper = float(1 - ss_res / np.sum(y_true_nn ** 2))
r2_nn_mean  = float(1 - ss_res / np.sum((y_true_nn - hist_avg) ** 2))

test_nn = test_meta[['eom', TARGET]].copy().reset_index(drop=True)
test_nn['pred_nn_gkx'] = nn_pred

nn_ic = test_nn.groupby('eom').apply(
    lambda g: spearmanr(g['pred_nn_gkx'], g[TARGET])[0]
)
nn_ic_mean  = float(nn_ic.mean())
nn_ic_tstat = float(nn_ic.mean() / nn_ic.std() * np.sqrt(len(nn_ic))) if nn_ic.std() > 0 else 0.0

# ── Portfolio construction ────────────────────────────────────────────────────
nn_portfolio = []
for eom_date, grp in test_nn.groupby('eom'):
    if len(grp) >= MIN_STOCKS_PER_MONTH:
        ret_xs = grp[TARGET].values
        preds  = grp['pred_nn_gkx'].values
        nn_portfolio.append({
            'date'  : eom_date,
            'NN_GKX': (portfolio_weights(preds) * ret_xs).sum(),
        })

nn_pf = pd.DataFrame(nn_portfolio).set_index('date').sort_index()
nn_pf.index = pd.to_datetime(nn_pf.index)

nn_sharpe  = float(nn_pf['NN_GKX'].mean() * 12 / (nn_pf['NN_GKX'].std() * np.sqrt(12)))
nn_ann_ret = float(nn_pf['NN_GKX'].mean() * 12)
nn_ann_vol = float(nn_pf['NN_GKX'].std() * np.sqrt(12))

print(f"\nGKX NN3  OOS R²(paper): {r2_nn_paper:+.4%}")
print(f"         Mean IC: {nn_ic_mean:+.4f}  t-stat: {nn_ic_tstat:+.2f}")
print(f"         Ann.Ret: {nn_ann_ret:.2%}  Vol: {nn_ann_vol:.2%}  Sharpe: {nn_sharpe:.2f}")

# ── Save ─────────────────────────────────────────────────────────────────────
print("Saving stage-4 outputs...")
nn_pf.to_parquet(CKPT / 'nn_pf.parquet')
pd.Series(nn_ic, name='nn_ic').to_frame().to_parquet(CKPT / 'nn_ic.parquet')
json.dump({
    'r2_nn_paper': r2_nn_paper,
    'r2_nn_mean' : r2_nn_mean,
    'nn_ic_mean' : nn_ic_mean,
    'nn_ic_tstat': nn_ic_tstat,
    'nn_sharpe'  : nn_sharpe,
    'nn_ann_ret' : nn_ann_ret,
    'nn_ann_vol' : nn_ann_vol,
}, open(CKPT / 'stage4_metrics.json', 'w'))

print("Stage 4 complete.")
