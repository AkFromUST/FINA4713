"""
nn_hunloss.py
FINA 4713 – Final Group Project
Neural Networks with custom directional loss — ported from
NN_pricing_OLD_torch_changed_setting.ipynb (Vanilla_Project)

WHAT THIS FILE IS
-----------------
Direct adaptation of the second KRX notebook onto our JKP dataset.
Architecture, loss function, and optimizer are identical to the notebook;
only the data pipeline is replaced with our preprocessed parquets.

ARCHITECTURES  (Net1–Net6, input_dim inferred at runtime)
----------------------------------------------------------
Net1 : input → 16 → 1
Net2 : input → 16 → 8 → 1
Net3 : input → 16 → 8 → 4 → 1
Net4 : input → 16 → 8 → 4 → 2 → 1
Net5 : input → 32 → 16 → 8 → 1
Net6 : two-branch multi-path network:
         Branch A : input → 64 → 32 → 16 → 8 → [4] → 2 → 1
         Branch B : input → 16 → 8 → [4] → 2 → 1
         Merge    : concat(A@4, B@4) → 8 → 4 → 2 → 1
         Final    : concat(A_out, B_out, merge_out) → Linear(3,1)
All hidden units use ReLU; output is linear.

LOSS FUNCTION  (loss_hun — directional / asymmetric)
-----------------------------------------------------
Penalises two types of errors:
  bull_bear : predicted positive  when actual is negative  (wrong direction, upside)
  bull      : predicted anything  when actual is positive  (missing/damaging upside)
Does NOT penalise predicting negative when actual is negative
(i.e. correctly avoiding losers is rewarded by silence).
Financially motivated: prioritises not being on the wrong side of winners/losers.

OPTIMIZER  : RAdam (Rectified Adam), lr=0.01
EARLY STOP : patience=20 on val loss
BATCH SIZE : 512 (small, matching notebook's 120 scaled to our larger dataset)

OUTPUTS (→ NN/files/hunloss/)
------------------------------
  models/   net1.pt … net6.pt
  results/  hunloss_results.csv
            hunloss_test_predictions.parquet
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── parent Proj/ on path ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from models import clean_xy, infer_feature_cols, oos_r2, META_COLS, TARGET

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
_DATA_DIR    = os.path.join(_PROJ, "files", "data")
_OUT_DIR     = os.path.join(_HERE, "files", "hunloss")

LR         = 0.01     # matches notebook
MAX_EPOCHS = 700      # matches notebook (ite=700)
PATIENCE   = 20       # matches notebook
BATCH_SIZE = 512


# ─────────────────────────────────────────────────────────────────────────────
# Architectures  (identical to notebook, input_dim injected at runtime)
# ─────────────────────────────────────────────────────────────────────────────
class Net1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.name = "Net1"
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Net2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.name = "Net2"
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Net3(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.name = "Net3"
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Net4(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.name = "Net4"
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 2)
        self.fc5 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class Net5(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.name = "Net5"
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Net6(nn.Module):
    """
    Multi-branch architecture from changed_setting notebook.
    Internal layer sizes are fixed (same as notebook); only
    the input projection changes with input_dim.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.name = "Net6"

        # Branch A: input → 64 → 32 → 16 → 8 → 4 → 2 → 1
        self.a1  = nn.Linear(input_dim, 64)
        self.a2  = nn.Linear(64, 32)
        self.a3  = nn.Linear(32, 16)
        self.a4  = nn.Linear(16, 8)
        self.a5  = nn.Linear(8, 4)    # ← intermediate used in merge
        self.a6  = nn.Linear(4, 2)
        self.a7  = nn.Linear(2, 1)    # Branch A scalar output

        # Branch B: input → 16 → 8 → 4 → 2 → 1
        self.b1  = nn.Linear(input_dim, 16)
        self.b2  = nn.Linear(16, 8)
        self.b3  = nn.Linear(8, 4)    # ← intermediate used in merge
        self.b4  = nn.Linear(4, 2)
        self.b5  = nn.Linear(2, 1)    # Branch B scalar output

        # Merge: concat(A@4, B@4) = 8 → 4 → 2 → 1
        self.m1  = nn.Linear(8, 4)
        self.m2  = nn.Linear(4, 2)
        self.m3  = nn.Linear(2, 1)    # Merge scalar output

        # Final: concat(A_out, B_out, merge_out) = 3 → 1
        self.res = nn.Linear(3, 1)

    def forward(self, x):
        # Branch A
        a = F.relu(self.a1(x))
        a = F.relu(self.a2(a))
        a = F.relu(self.a3(a))
        a = F.relu(self.a4(a))
        a_mid = F.relu(self.a5(a))   # 4-dim intermediate
        a_out = F.relu(self.a6(a_mid))
        a_out = self.a7(a_out)       # scalar

        # Branch B
        b = F.relu(self.b1(x))
        b = F.relu(self.b2(b))
        b_mid = F.relu(self.b3(b))   # 4-dim intermediate
        b_out = F.relu(self.b4(b_mid))
        b_out = self.b5(b_out)       # scalar

        # Merge
        m = torch.cat((a_mid, b_mid), dim=1)  # 8-dim
        m = F.relu(self.m1(m))
        m = F.relu(self.m2(m))
        m_out = self.m3(m)            # scalar

        # Final residual combination
        return self.res(torch.cat((a_out, b_out, m_out), dim=1))


def build_nets(input_dim: int) -> list:
    return [Net1(input_dim), Net2(input_dim), Net3(input_dim),
            Net4(input_dim), Net5(input_dim), Net6(input_dim)]


# ─────────────────────────────────────────────────────────────────────────────
# Loss function  (loss_hun — directional / asymmetric)
# ─────────────────────────────────────────────────────────────────────────────
def loss_hun(output: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
    Asymmetric directional loss from the notebook.
    Penalises:
      - Predicting up when market goes down  (bull_bear)
      - Any prediction error when market goes up  (bull)
    Silent when market goes down and prediction is also down/neutral.
    """
    diff      = output - real
    bull_bear = (real < 0).float() * (output > 0).float() * diff
    bull      = (real > 0).float() * diff
    return ((bull_bear + bull) ** 2).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop  (matches notebook's torch_train)
# ─────────────────────────────────────────────────────────────────────────────
def _train_one(
    model:   nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    device:  torch.device,
    verbose: bool = False,
) -> tuple[nn.Module, float]:
    """
    Train with RAdam + loss_hun + early stopping (patience=20).
    Returns (best_model, best_val_loss).
    """
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xvl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    yvl = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1).to(device)

    loader  = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=False)
    opt     = torch.optim.RAdam(model.parameters(), lr=LR)

    best_val_loss  = float("inf")
    best_state     = {k: v.clone() for k, v in model.state_dict().items()}
    patience_count = 0

    model.to(device)

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_losses = []
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            out  = model(Xb)
            loss = loss_hun(out, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_out  = model(Xvl)
            val_loss = loss_hun(val_out, yvl).item()

        if verbose and epoch % 50 == 0:
            print(f"    epoch {epoch:4d}  train={np.mean(train_losses):.4e}"
                  f"  val={val_loss:.4e}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count > PATIENCE:
                if verbose:
                    print(f"    early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_hunloss(
    train_df:     pd.DataFrame | None = None,
    val_df:       pd.DataFrame | None = None,
    test_df:      pd.DataFrame | None = None,
    feature_cols: list | None         = None,
    data_dir:     str                 = _DATA_DIR,
    out_dir:      str                 = _OUT_DIR,
    verbose:      bool                = True,
) -> dict:
    """
    Full pipeline for Net1–Net6 with loss_hun + RAdam on our JKP data.

    Returns
    -------
    dict with keys:
        results      — DataFrame: model × {val_oos_r2, test_oos_r2, val_hunloss}
        predictions  — DataFrame: test-set predictions for all 6 nets
        models       — dict: name → trained nn.Module
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

    # ── [2] Arrays ───────────────────────────────────────────────────────────
    X_train, y_train, train_clean = clean_xy(train_df, feature_cols)
    X_val,   y_val,   val_clean   = clean_xy(val_df,   feature_cols)
    X_test,  y_test,  test_clean  = clean_xy(test_df,  feature_cols)

    if verbose:
        print(f"  train {len(y_train):,} | val {len(y_val):,} | test {len(y_test):,}")

    input_dim = X_train.shape[1]
    nets      = build_nets(input_dim)

    # ── [3] Train each net ────────────────────────────────────────────────────
    result_rows    = []
    pred_df        = test_clean[["id", "eom", "excntry", TARGET]].copy()
    pred_df        = pred_df.rename(columns={TARGET: "y_true"})
    if "me_raw" in test_clean.columns:
        pred_df["me_raw"] = test_clean["me_raw"].values

    Xte_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    trained_models = {}

    for net in nets:
        t0 = time.time()
        if verbose:
            print(f"\n[NN] {net.name}")

        model, best_val_loss = _train_one(
            net, X_train, y_train, X_val, y_val, device, verbose
        )

        model.eval()
        with torch.no_grad():
            Xvl_t    = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_pred  = model(Xvl_t).squeeze(1).cpu().numpy()
            test_pred = model(Xte_t).squeeze(1).cpu().numpy()

        val_r2  = oos_r2(y_val,  val_pred)
        test_r2 = oos_r2(y_test, test_pred)
        elapsed = time.time() - t0

        if verbose:
            print(f"  Val OOS-R²={val_r2:+.6f}  |  Test OOS-R²={test_r2:+.6f}"
                  f"  |  val_hunloss={best_val_loss:.4e}  ({elapsed:.0f}s)")

        result_rows.append({
            "model":        net.name,
            "val_oos_r2":   val_r2,
            "test_oos_r2":  test_r2,
            "val_hunloss":  best_val_loss,
        })

        pred_df[f"pred_{net.name.lower()}"] = test_pred
        trained_models[net.name] = model

        torch.save(model.state_dict(),
                   os.path.join(models_dir, f"{net.name.lower()}.pt"))

    # ── [4] Summary ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(result_rows)

    if verbose:
        sep = "─" * 62
        print(f"\n  {sep}")
        print(f"  {'Model':<6} {'Val OOS-R²':>12} {'Test OOS-R²':>13} {'Val HunLoss':>13}")
        print(f"  {sep}")
        for row in result_rows:
            print(f"  {row['model']:<6} {row['val_oos_r2']:>+12.6f} "
                  f"{row['test_oos_r2']:>+13.6f} {row['val_hunloss']:>13.4e}")
        print(f"  {sep}")

    # ── [5] Save ──────────────────────────────────────────────────────────────
    results_df.to_csv(os.path.join(results_dir, "hunloss_results.csv"), index=False)
    pred_df.to_parquet(os.path.join(results_dir, "hunloss_test_predictions.parquet"),
                       index=False)

    if verbose:
        print(f"\n  Saved → {models_dir}/  {{net1..net6}}.pt")
        print(f"  Saved → {results_dir}/  hunloss_results.csv, hunloss_test_predictions.parquet")

    return {
        "results":     results_df,
        "predictions": pred_df,
        "models":      trained_models,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  FINA 4713 — Neural Networks with Directional Loss (HunLoss)")
    print("  Net1–Net6 · RAdam · loss_hun · JKP data")
    print("=" * 62)
    run_hunloss()
    print("\n  Done.\n")
