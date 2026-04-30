"""
nn_model.py
FINA 4713 – Final Group Project
Neural Network models following Gu, Kelly & Xiu (2020)

GKX ARCHITECTURE SPEC
----------------------
Five feed-forward networks, geometric pyramid hidden-layer sizes:
  NN1 : [32]
  NN2 : [32, 16]
  NN3 : [32, 16, 8]   ← best performer in the paper
  NN4 : [32, 16, 8, 4]
  NN5 : [32, 16, 8, 4, 2]
Activation  : ReLU at every hidden unit
Batch norm  : applied after each hidden layer (Ioffe & Szegedy 2015)
Output      : linear (no activation)

GKX TRAINING SPEC
-----------------
Objective   : MSE (l2 loss) + L1 penalty on all weight matrices
Optimiser   : Adam (Kingma & Ba 2014) with default lr=0.001
Regularisers: L1 weight penalty, batch normalisation, early stopping
Ensemble    : 10 random seeds; final prediction = mean over seeds
Batch size  : 10,000
Target      : raw excess return ret_exc_lead1m  (NOT standardised)
              OOS-R² benchmarked against zero (GKX eq. 19)

OUTPUTS (→ NN/files/)
---------------------
  models/   nn1.pt … nn5.pt          (best-seed model state dicts)
  results/  nn_results.csv           (val + test OOS-R² per architecture)
            nn_test_predictions.parquet
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── resolve imports from parent Proj/ directory ───────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from models import clean_xy, infer_feature_cols, oos_r2, META_COLS, TARGET

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_DATA_DIR    = os.path.join(_PROJ, "files", "data")
_OUT_DIR     = os.path.join(_HERE, "files")
_MODELS_DIR  = os.path.join(_OUT_DIR, "models")
_RESULTS_DIR = os.path.join(_OUT_DIR, "results")

# GKX architectures (hidden layer sizes, geometric pyramid rule)
GKX_ARCHITECTURES = {
    "NN1": [32],
    "NN2": [32, 16],
    "NN3": [32, 16, 8],
    "NN4": [32, 16, 8, 4],
    "NN5": [32, 16, 8, 4, 2],
}

N_SEEDS    = 10        # ensemble size
BATCH_SIZE = 10_000
LR         = 1e-3      # Adam default
MAX_EPOCHS = 100
PATIENCE   = 5         # early stopping patience (on val MSE)

# L1 lambda grid — tuned on val OOS-R² for each architecture
L1_GRID = [1e-5, 1e-4, 1e-3]


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────
class GKXNet(nn.Module):
    """
    Feed-forward network following GKX (2020).
    Hidden layers: Linear → BatchNorm → ReLU
    Output layer : Linear (no activation)
    """
    def __init__(self, input_dim: int, layer_sizes: list):
        super().__init__()
        layers = []
        prev = input_dim
        for h in layer_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def l1_penalty(self) -> torch.Tensor:
        return sum(p.abs().sum() for name, p in self.named_parameters()
                   if "weight" in name)


# ─────────────────────────────────────────────────────────────────────────────
# Single-seed training
# ─────────────────────────────────────────────────────────────────────────────
def _train_one(
    model:     GKXNet,
    X_train:   np.ndarray,
    y_train:   np.ndarray,
    X_val:     np.ndarray,
    y_val:     np.ndarray,
    l1_lambda: float,
    device:    torch.device,
    verbose:   bool = False,
) -> tuple[GKXNet, float]:
    """
    Train one GKXNet instance with Adam + L1 + early stopping.
    Returns (best_model, best_val_mse).
    """
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xvl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    yvl = torch.tensor(y_val,   dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    mse_fn = nn.MSELoss()

    best_val_mse   = float("inf")
    best_state     = {k: v.clone() for k, v in model.state_dict().items()}
    patience_count = 0

    model.to(device)

    for epoch in range(MAX_EPOCHS):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = mse_fn(pred, yb) + l1_lambda * model.l1_penalty()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xvl)
            val_mse  = mse_fn(val_pred, yvl).item()

        if verbose:
            print(f"    epoch {epoch+1:3d}  val_mse={val_mse:.6e}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val_mse


# ─────────────────────────────────────────────────────────────────────────────
# L1-lambda tuning
# ─────────────────────────────────────────────────────────────────────────────
def _tune_l1(
    input_dim:   int,
    layer_sizes: list,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    device:      torch.device,
    verbose:     bool = True,
) -> float:
    """
    Grid-search L1 lambda on val MSE using a single seed (seed=0).
    Returns best lambda.
    """
    best_lambda = L1_GRID[0]
    best_mse    = float("inf")

    for lam in L1_GRID:
        torch.manual_seed(0)
        model = GKXNet(input_dim, layer_sizes).to(device)
        _, val_mse = _train_one(model, X_train, y_train, X_val, y_val, lam, device)
        if verbose:
            print(f"    λ={lam:.0e}  val_mse={val_mse:.6e}")
        if val_mse < best_mse:
            best_mse    = val_mse
            best_lambda = lam

    return best_lambda


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble training
# ─────────────────────────────────────────────────────────────────────────────
def _train_ensemble(
    input_dim:   int,
    layer_sizes: list,
    l1_lambda:   float,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    X_test:      np.ndarray,
    device:      torch.device,
    n_seeds:     int = N_SEEDS,
    verbose:     bool = True,
) -> tuple[np.ndarray, np.ndarray, GKXNet]:
    """
    Train N_SEEDS models, return (mean_val_preds, mean_test_preds, best_seed_model).
    The 'best_seed_model' is the one with lowest val MSE — saved as the .pt artefact.
    """
    Xvl_t  = torch.tensor(X_val,  dtype=torch.float32).to(device)
    Xte_t  = torch.tensor(X_test, dtype=torch.float32).to(device)

    val_preds  = []
    test_preds = []
    best_mse   = float("inf")
    best_model = None

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        model = GKXNet(input_dim, layer_sizes).to(device)
        model, val_mse = _train_one(model, X_train, y_train, X_val, y_val,
                                    l1_lambda, device)
        model.eval()
        with torch.no_grad():
            vp = model(Xvl_t).cpu().numpy()
            tp = model(Xte_t).cpu().numpy()
        val_preds.append(vp)
        test_preds.append(tp)

        if val_mse < best_mse:
            best_mse   = val_mse
            best_model = model

        if verbose:
            print(f"    seed {seed+1:2d}/{n_seeds}  val_mse={val_mse:.6e}")

    return (np.mean(val_preds,  axis=0),
            np.mean(test_preds, axis=0),
            best_model)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_nn(
    train_df:    pd.DataFrame | None = None,
    val_df:      pd.DataFrame | None = None,
    test_df:     pd.DataFrame | None = None,
    feature_cols: list | None        = None,
    architectures: dict              = GKX_ARCHITECTURES,
    n_seeds:     int                 = N_SEEDS,
    data_dir:    str                 = _DATA_DIR,
    out_dir:     str                 = _OUT_DIR,
    verbose:     bool                = True,
) -> dict:
    """
    Full GKX neural-network pipeline.

    Parameters
    ----------
    train_df, val_df, test_df : pre-processed DataFrames; loaded from data_dir if None.
    feature_cols : predictor columns; inferred from train_df if None.
    architectures : dict mapping name → hidden-layer-size list (default: all 5 GKX nets).
    n_seeds : ensemble size (GKX default = 10).
    data_dir, out_dir : path overrides.
    verbose : print progress.

    Returns
    -------
    dict with keys:
        results      — DataFrame: architecture × {val_oos_r2, test_oos_r2, best_l1}
        predictions  — DataFrame: test-set predictions for all architectures
        models       — dict: name → best-seed GKXNet
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"\n  Device : {device}")

    models_dir  = os.path.join(out_dir, "models")
    results_dir = os.path.join(out_dir, "results")
    for d in (models_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    # ── [1] Load data ─────────────────────────────────────────────────────────
    def _load(df, name):
        if df is None:
            path = os.path.join(data_dir, f"{name}_processed.parquet")
            if verbose:
                print(f"  Loading {path}")
            return pd.read_parquet(path)
        return df

    if verbose:
        print("\n[1] Data")
    train_df = _load(train_df, "train")
    val_df   = _load(val_df,   "val")
    test_df  = _load(test_df,  "test")

    if feature_cols is None:
        feature_cols = infer_feature_cols(train_df)
    if verbose:
        print(f"  Features : {len(feature_cols)}")

    # ── [2] Clean arrays (drop NaN-target rows) ───────────────────────────────
    X_train, y_train, train_clean = clean_xy(train_df, feature_cols)
    X_val,   y_val,   val_clean   = clean_xy(val_df,   feature_cols)
    X_test,  y_test,  test_clean  = clean_xy(test_df,  feature_cols)

    if verbose:
        print(f"  train {len(y_train):,} | val {len(y_val):,} | test {len(y_test):,}")

    input_dim = X_train.shape[1]

    # ── [3] Train each architecture ───────────────────────────────────────────
    result_rows = []
    pred_df     = test_clean[["id", "eom", "excntry", TARGET]].copy()
    pred_df     = pred_df.rename(columns={TARGET: "y_true"})
    if "me_raw" in test_clean.columns:
        pred_df["me_raw"] = test_clean["me_raw"].values

    trained_models = {}

    for name, layer_sizes in architectures.items():
        t0 = time.time()
        if verbose:
            print(f"\n[NN] {name}  layers={layer_sizes}")

        # L1-lambda tuning
        if verbose:
            print(f"  Tuning L1 lambda (seed=0, {len(L1_GRID)} candidates):")
        best_l1 = _tune_l1(input_dim, layer_sizes,
                            X_train, y_train, X_val, y_val,
                            device, verbose)
        if verbose:
            print(f"  → best λ={best_l1:.0e}")

        # Ensemble
        if verbose:
            print(f"  Training ensemble ({n_seeds} seeds, λ={best_l1:.0e}):")
        val_pred, test_pred, best_model = _train_ensemble(
            input_dim, layer_sizes, best_l1,
            X_train, y_train, X_val, y_val, X_test,
            device, n_seeds, verbose,
        )

        # Metrics — OOS-R² vs zero (GKX eq. 19)
        val_r2  = oos_r2(y_val,  val_pred)
        test_r2 = oos_r2(y_test, test_pred)
        elapsed = time.time() - t0

        if verbose:
            print(f"  Val OOS-R²={val_r2:+.6f}  |  Test OOS-R²={test_r2:+.6f}"
                  f"  ({elapsed:.0f}s)")

        result_rows.append({
            "model":       name,
            "val_oos_r2":  val_r2,
            "test_oos_r2": test_r2,
            "best_l1":     best_l1,
            "layer_sizes": str(layer_sizes),
            "n_seeds":     n_seeds,
        })

        pred_col = f"pred_{name.lower()}"
        pred_df[pred_col] = test_pred

        trained_models[name] = best_model

        # Save best-seed model state dict
        torch.save(best_model.state_dict(),
                   os.path.join(models_dir, f"{name.lower()}.pt"))

    # ── [4] Summary ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(result_rows)

    if verbose:
        sep = "─" * 58
        print(f"\n  {sep}")
        print(f"  {'Model':<8} {'Val OOS-R²':>12} {'Test OOS-R²':>13}")
        print(f"  {sep}")
        for row in result_rows:
            print(f"  {row['model']:<8} {row['val_oos_r2']:>+12.6f} "
                  f"{row['test_oos_r2']:>+13.6f}")
        print(f"  {sep}")

    # ── [5] Save ──────────────────────────────────────────────────────────────
    results_df.to_csv(os.path.join(results_dir, "nn_results.csv"), index=False)
    pred_df.to_parquet(os.path.join(results_dir, "nn_test_predictions.parquet"),
                       index=False)

    if verbose:
        print(f"\n  Saved → {models_dir}/  {{nn1..nn5}}.pt")
        print(f"  Saved → {results_dir}/  nn_results.csv, nn_test_predictions.parquet")

    return {
        "results":     results_df,
        "predictions": pred_df,
        "models":      trained_models,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  FINA 4713 — Neural Networks (GKX 2020)")
    print("  NN1–NN5 · ReLU · BatchNorm · L1 · Ensemble")
    print("=" * 58)
    run_nn()
    print("\n  Done.\n")
