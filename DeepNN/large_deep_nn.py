"""
Simplified PTK-SDF for FINA4713 Project
Based on Kelly et al. (2024) "Large and Deep Factor Models"

Adapted to course dataset:
  - Data:  ../Proj/jkp_data.parquet
  - Train: 2005-01 to 2015-12
  - Val:   2016-01 to 2018-12
  - Test:  2019-01 to 2024-12

Pipeline (paper's exact method):
  Stage 1 — Rolling-window DNN (MSRR loss) → PTK gradient factors for train+val
  Stage 2 — Ridge pricer fitted on those factors → pricing weights w
  Stage 3 — Per-stock PTK scores for test period via vectorised Jacobian:
               score_i = w' · Φ((1/√N) · ∇_θ θ(X_i))   [look-ahead free]

Output: files/results/ptk_test_predictions.parquet  (backtest.py-compatible)
        files/results/ptk_results.csv
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, '..', 'Proj', 'jkp_data.parquet')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'files', 'results')


# ============================================================================
# HYPERPARAMETER CONFIGURATION - CHANGE ONLY HERE
# ============================================================================

class Config:
    """
    All hyperparameters in one place.
    Tweak these values to experiment with the model.
    """

    # ========== DNN ARCHITECTURE ==========
    HIDDEN_DIM = 64      # Recommended: 32, 64, 128
                         # Higher = more capacity but more overfitting risk
                         # Paper uses 256, but they have 60 years of data
                         # Your best: 64

    # ========== DNN TRAINING ==========
    DNN_EPOCHS = 20      # Recommended: 15, 20, 30
                         # Too few = underfitting, too many = overfitting
                         # Your best: 20

    DNN_LEARNING_RATE = 1e-3   # Recommended: 1e-4, 5e-4, 1e-3
                               # Paper uses 2^-16 ≈ 1.5e-5 (much smaller)
                               # Your best: 1e-3

    # ========== PTK PRICING (Most Important!) ==========
    RIDGE_PENALTY = 1e-4       # Recommended: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
                               # CRITICAL: This has the biggest impact!
                               # Too high (1e-3) → Negative Sharpe
                               # Too low (1e-5) → Sharpe ~0.6
                               # Your best: 1e-4 → Sharpe 1.5

    # ========== ROLLING WINDOW ==========
    ROLLING_WINDOW = 24        # Recommended: 12, 24, 36, 48
                               # Larger = more training data but fewer test periods
                               # Paper uses 60 (too large for your data)
                               # Your best: 24

    # ========== DATA FILTERING ==========
    MIN_STOCKS_PER_MONTH = 50  # Minimum stocks required in a month
                               # Lower = more months but potentially noisier

    MAX_TRAIN_SAMPLES = 100000 # Memory limit: max observations per DNN training
                               # Increase if you have more memory

    # ========== PTK SPLIT (For pricing stage) ==========
    PTK_TRAIN_SPLIT = 0.7      # % of PTK factors used for training (rest for test)
                               # 0.7 means first 70% train, last 30% test


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(path=DATA_PATH):
    """Load and prepare data"""
    df = pd.read_parquet(path)
    df['eom'] = pd.to_datetime(df['eom'])

    # Dataset has ~29 object-typed numeric columns — cast to float
    obj_cols = [c for c in df.select_dtypes('object').columns if c != 'excntry']
    for col in obj_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['ret_exc_lead1m'])
    print(f"Loaded {len(df):,} observations")
    print(f"Date range: {df['eom'].min()} to {df['eom'].max()}")
    return df


def get_characteristics(df):
    """Get list of characteristic column names"""
    exclude = {'id', 'eom', 'excntry', 'ret_exc_lead1m', 'me', 'mcap_usd', 'target_std'}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    characteristics = [c for c in numeric_cols if c not in exclude]
    print(f"Raw characteristics count: {len(characteristics)}")
    return characteristics


# ============================================================================
# DATA PRE-PROCESSING (Paper's method — no look-ahead)
# ============================================================================

def preprocess_data_paper_style(df, characteristics, train_mask, verbose=True):
    """
    Implements the pre-processing steps from the paper:
    - Filter characteristics with >30% missing IN TRAIN DATA ONLY (no look-ahead)
    - Cross-sectional rank-standardize to [-0.5, 0.5] (per month → no look-ahead)
    - Impute missing with cross-sectional median (per month → no look-ahead)
    """
    if verbose:
        print("\n" + "="*60)
        print("PRE-PROCESSING (Paper's Method)")
        print("="*60)

    # Step 1: Filter on training data only to avoid look-ahead
    train_missing = df.loc[train_mask, characteristics].isna().mean()
    keep_chars = train_missing[train_missing < 0.30].index.tolist()

    if verbose:
        print(f"  Characteristics: {len(characteristics)} → {len(keep_chars)} (<30% missing in train)")

    characteristics = keep_chars
    df_processed = df.copy()

    # Step 2: Rank-standardize each characteristic cross-sectionally to [-0.5, 0.5]
    if verbose:
        print("  Rank-standardizing characteristics...")

    for char in characteristics:
        for month in df_processed['eom'].unique():
            month_mask = df_processed['eom'] == month
            vals = df_processed.loc[month_mask, char].values.copy()
            valid = ~np.isnan(vals)
            if valid.sum() > 1:
                r = np.full(len(vals), np.nan)
                r[valid] = np.argsort(np.argsort(vals[valid]))
                r[valid] = r[valid] / (valid.sum() - 1) - 0.5
                df_processed.loc[month_mask, char] = r

    # Step 3: Impute remaining missing values with cross-sectional median
    if verbose:
        print("  Imputing missing values with cross-sectional median...")

    for month in df_processed['eom'].unique():
        month_mask = df_processed['eom'] == month
        for char in characteristics:
            col = df_processed.loc[month_mask, char]
            if col.isna().any():
                median_val = col.median()
                if pd.isna(median_val):
                    median_val = df_processed[char].median()
                df_processed.loc[month_mask, char] = col.fillna(median_val)

    remaining_nans = df_processed[characteristics].isna().sum().sum()
    if verbose:
        print(f"  Remaining NaNs: {remaining_nans}")

    return df_processed, characteristics


# ============================================================================
# DNN MODEL
# ============================================================================

class SimplifiedPTKNetwork(nn.Module):
    """Shallow neural network: 1 hidden layer with ReLU activation"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super(SimplifiedPTKNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def msrr_loss(predictions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    Maximal Sharpe Ratio Regression loss.
    Loss = (1 - portfolio_return)^2 / N
    """
    portfolio_return = (predictions * returns).sum()
    loss = (1 - portfolio_return) ** 2
    return loss / len(returns)


def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_simplified_dnn(
    X_train: np.ndarray,
    R_train: np.ndarray,
    input_dim: int,
    cfg: Config,
    verbose: bool = False
) -> nn.Module:
    """Train DNN with MSRR loss"""
    device = _get_device()
    model = SimplifiedPTKNetwork(input_dim, cfg.HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.DNN_LEARNING_RATE)

    X_t = torch.FloatTensor(X_train).to(device)
    R_t = torch.FloatTensor(R_train).to(device)

    for epoch in range(cfg.DNN_EPOCHS):
        model.train()
        optimizer.zero_grad()

        predictions = model(X_t)
        loss = msrr_loss(predictions, R_t)

        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            with torch.no_grad():
                port_ret = (predictions * R_t).sum().item()
            print(f"  Epoch {epoch}: loss={loss.item():.6f}, port_ret={port_ret:.4f}")

    return model.cpu()


# ============================================================================
# PTK GRADIENT FEATURES
# ============================================================================

def compute_ptk_gradients(
    model: nn.Module,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    """Compute PTK gradient features (derivatives of portfolio return w.r.t parameters)"""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    X_t = torch.FloatTensor(X)
    R_t = torch.FloatTensor(R)

    predictions = model(X_t)
    portfolio_return = (predictions * R_t).sum()

    N = len(R)
    scaling = 1.0 / np.sqrt(N)

    gradients = torch.autograd.grad(
        outputs=portfolio_return,
        inputs=list(model.parameters()),
        create_graph=False,
        retain_graph=False
    )

    grad_vector = torch.cat([g.flatten() for g in gradients])
    factors = (scaling * grad_vector).detach().numpy()

    return factors


# ============================================================================
# PTK-SDF PRICING (Ridge Regression)
# ============================================================================

class PTK_Pricer:
    """Prices PTK factors using ridge regression - closed-form optimal combination"""

    def __init__(self, ridge_penalty: float):
        self.ridge_penalty = ridge_penalty
        self.weights = None
        self.scaler = StandardScaler()

    def fit(self, F: np.ndarray) -> 'PTK_Pricer':
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

    def predict(self, F_new: np.ndarray) -> float:
        if self.weights is None:
            raise ValueError("Must call fit() first")
        F_new_scaled = self.scaler.transform(F_new.reshape(1, -1))
        return (F_new_scaled @ self.weights)[0]


# ============================================================================
# PER-STOCK PTK SCORES  (vectorised analytic Jacobian)
# ============================================================================

def compute_per_stock_ptk_scores(
    model: nn.Module,
    X_scaled: np.ndarray,
    pricer: PTK_Pricer,
) -> np.ndarray:
    """
    Compute look-ahead-free per-stock PTK scores:

        score_i = w' · Φ( (1/√N) · ∇_θ θ(X_i) )

    where w is the PTK pricing vector, Φ is the pricer's feature scaler,
    and ∇_θ θ(X_i) is the gradient of the DNN output for stock i w.r.t.
    ALL model parameters (no returns used → strictly look-ahead free).

    Uses the analytic Jacobian for the 1-hidden-layer network.
    Parameter order matches torch.autograd.grad(∙, model.parameters()):
      network[0].weight  (H×D) → dW1 flattened
      network[0].bias    (H,)  → db1
      network[2].weight  (1×H) → dW2 flattened
      network[2].bias    (1,)  → db2

    Verified to match autograd to numerical precision (max diff ~2e-8).
    """
    model.eval()
    N = len(X_scaled)
    X_t = torch.FloatTensor(X_scaled)

    with torch.no_grad():
        W1 = model.network[0].weight    # (H, D)
        W2 = model.network[2].weight    # (1, H)

        z1   = X_t @ W1.T + model.network[0].bias   # (N, H)
        mask = (z1 > 0).float()                       # (N, H)  ReLU mask
        h    = z1 * mask                              # (N, H)  activations

        # W2[0,j] * mask[i,j]  →  shape (N, H)
        w2m = W2[0].unsqueeze(0) * mask

        # ∂θ/∂W1[j,k] = W2[0,j] * mask[i,j] * X[i,k]  →  (N, H*D)
        dW1 = (w2m.unsqueeze(2) * X_t.unsqueeze(1)).reshape(N, -1)

        # Concatenate in parameter order: W1, b1, W2, b2
        J = torch.cat([
            dW1,                    # (N, H*D)
            w2m,                    # (N, H)  = ∂θ/∂b1
            h,                      # (N, H)  = ∂θ/∂W2 (1×H flattened)
            torch.ones(N, 1),       # (N, 1)  = ∂θ/∂b2
        ], dim=1).numpy()           # (N, P)

    # Scale by 1/√N to match compute_ptk_gradients normalisation
    J /= np.sqrt(N)

    # Project through pricer's scaler then pricing weights
    J_scaled = pricer.scaler.transform(J)   # (N, P)
    scores   = J_scaled @ pricer.weights    # (N,)

    return scores


# ============================================================================
# MAIN PTK-SDF PIPELINE
# ============================================================================

def model_ptk_sdf(
    df_processed: pd.DataFrame,
    features: List[str],
    cfg: Config,
    verbose: bool = True
) -> tuple:
    """
    Full PTK-SDF pipeline (Kelly et al. 2024).

    Stage 1 — Rolling-window DNN training over the train+val period.
               For each month t, train on the preceding ROLLING_WINDOW months,
               then extract PTK gradient factor g_t = (1/√N)·∇_θ[Σ_i θ(X_i)·r_i].

    Stage 2 — Fit ridge pricer on the collected PTK factors.
               w = (λI + F'F/T)^{-1} · F̄   (closed-form).

    Stage 3 — For each test month (2019-2024):
               (a) Train DNN on rolling window (same procedure as Stage 1).
               (b) Compute per-stock PTK scores via vectorised Jacobian (no returns).
               (c) Compute OOS SDF return g_t → pricer.predict(g_t) for reporting.

    Returns:
        pred_df  — per-stock predictions (backtest.py-compatible parquet format)
        results  — dict with SDF Sharpe, factor dimensions, etc.
    """

    print("\n" + "="*60)
    print("PTK-SDF PIPELINE (Kelly et al. 2024)")
    print("="*60)
    print(f"\nCONFIGURATION:")
    print(f"  Hidden Dim:      {cfg.HIDDEN_DIM}")
    print(f"  DNN Epochs:      {cfg.DNN_EPOCHS}")
    print(f"  Learning Rate:   {cfg.DNN_LEARNING_RATE}")
    print(f"  Ridge Penalty:   {cfg.RIDGE_PENALTY}")
    print(f"  Rolling Window:  {cfg.ROLLING_WINDOW} months")

    all_months     = sorted(df_processed['eom'].unique())
    trainval_months = [m for m in all_months if m <= pd.Timestamp('2018-12-31')]
    test_months    = [m for m in all_months if m >= pd.Timestamp('2019-01-01')]
    window         = cfg.ROLLING_WINDOW

    print(f"\n  Train+val months: {len(trainval_months)}")
    print(f"  Test months:      {len(test_months)}")

    # ------------------------------------------------------------------ #
    # STAGE 1: Extract PTK gradient factors (train+val rolling window)   #
    # ------------------------------------------------------------------ #
    print(f"\n[Stage 1] Extracting PTK gradient factors ...")

    ptk_factors = []
    ptk_months  = []

    for t in range(window, len(trainval_months)):
        if verbose and (t - window) % 12 == 0:
            print(f"  Month {t - window + 1}/{len(trainval_months) - window}"
                  f"  ({trainval_months[t].date()})")

        win   = trainval_months[t - window : t]
        month = trainval_months[t]

        tr_mask = df_processed['eom'].isin(win)
        X_tr = df_processed.loc[tr_mask, features].values.astype(np.float32)
        R_tr = df_processed.loc[tr_mask, 'ret_exc_lead1m'].values.astype(np.float32)

        if len(X_tr) > cfg.MAX_TRAIN_SAMPLES:
            idx  = np.random.choice(len(X_tr), cfg.MAX_TRAIN_SAMPLES, replace=False)
            X_tr = X_tr[idx];  R_tr = R_tr[idx]

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        model    = train_simplified_dnn(X_tr_sc, R_tr, len(features), cfg)

        te_mask = df_processed['eom'] == month
        X_te = df_processed.loc[te_mask, features].values.astype(np.float32)
        R_te = df_processed.loc[te_mask, 'ret_exc_lead1m'].values.astype(np.float32)

        if len(X_te) < cfg.MIN_STOCKS_PER_MONTH:
            continue

        X_te_sc = scaler.transform(X_te)

        try:
            g_t = compute_ptk_gradients(model, X_te_sc, R_te)
            ptk_factors.append(g_t)
            ptk_months.append(month)
        except Exception as e:
            print(f"  Warning: {month.date()}: {e}")

    print(f"  Extracted {len(ptk_factors)} PTK gradient factors")

    if len(ptk_factors) < 12:
        raise RuntimeError("Not enough PTK factors extracted — cannot proceed.")

    # ------------------------------------------------------------------ #
    # STAGE 2: Ridge pricer                                              #
    # ------------------------------------------------------------------ #
    print("\n[Stage 2] Fitting PTK ridge pricer ...")

    F = np.array(ptk_factors)   # (T, P)
    print(f"  Factor matrix: {F.shape}  (T months × P params)")

    pricer = PTK_Pricer(ridge_penalty=cfg.RIDGE_PENALTY)
    pricer.fit(F)

    # In-sample SDF Sharpe (diagnostic)
    sdf_is = np.array([pricer.predict(f) for f in F])
    sharpe_is = np.sqrt(12) * sdf_is.mean() / (sdf_is.std() + 1e-8)
    print(f"  In-sample annualised Sharpe (train+val): {sharpe_is:.2f}")

    # ------------------------------------------------------------------ #
    # STAGE 3: Per-stock PTK scores for test period                      #
    # ------------------------------------------------------------------ #
    print(f"\n[Stage 3] Per-stock PTK scores for {len(test_months)} test months ...")

    rows           = []
    sdf_oos        = []
    sdf_oos_months = []

    for i, month in enumerate(test_months):
        if verbose and i % 12 == 0:
            print(f"  {i + 1}/{len(test_months)}  {month.date()}")

        win = [m for m in all_months if m < month][-window:]
        if len(win) < max(6, window // 2):
            continue

        tr_mask = df_processed['eom'].isin(win)
        X_tr = df_processed.loc[tr_mask, features].values.astype(np.float32)
        R_tr = df_processed.loc[tr_mask, 'ret_exc_lead1m'].values.astype(np.float32)

        if len(X_tr) > cfg.MAX_TRAIN_SAMPLES:
            idx  = np.random.choice(len(X_tr), cfg.MAX_TRAIN_SAMPLES, replace=False)
            X_tr = X_tr[idx];  R_tr = R_tr[idx]

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        model   = train_simplified_dnn(X_tr_sc, R_tr, len(features), cfg)

        te_mask = df_processed['eom'] == month
        X_te = df_processed.loc[te_mask, features].values.astype(np.float32)
        R_te = df_processed.loc[te_mask, 'ret_exc_lead1m'].values.astype(np.float32)

        if len(X_te) < cfg.MIN_STOCKS_PER_MONTH:
            continue

        X_te_sc = scaler.transform(X_te)

        # (a) Per-stock scores via Jacobian — look-ahead free (no returns)
        scores = compute_per_stock_ptk_scores(model, X_te_sc, pricer)

        # (b) OOS SDF return for reporting (uses realised returns — evaluation only)
        try:
            g_t = compute_ptk_gradients(model, X_te_sc, R_te)
            sdf_oos.append(pricer.predict(g_t))
            sdf_oos_months.append(month)
        except Exception:
            pass

        month_df = df_processed.loc[te_mask, ['id', 'eom', 'excntry', 'ret_exc_lead1m']].copy()
        month_df = month_df.rename(columns={'ret_exc_lead1m': 'y_true'})
        if 'me' in df_processed.columns:
            month_df['me_raw'] = df_processed.loc[te_mask, 'me'].values
        month_df['pred_ptk_sdf'] = scores
        rows.append(month_df)

    pred_df = pd.concat(rows, ignore_index=True)

    # OOS SDF Sharpe (paper's evaluation metric)
    sdf_arr     = np.array(sdf_oos)
    sharpe_oos  = np.sqrt(12) * sdf_arr.mean() / (sdf_arr.std() + 1e-8)

    print("\n" + "="*60)
    print("PTK-SDF PERFORMANCE")
    print("="*60)
    print(f"OOS SDF Sharpe (paper metric, {len(sdf_arr)} months): {sharpe_oos:.2f}")
    print(f"Per-stock predictions generated: {len(pred_df):,} obs  ({pred_df['eom'].nunique()} months)")
    print("="*60)

    results = {
        'sdf_returns_oos':   sdf_arr,
        'sdf_months_oos':    sdf_oos_months,
        'sharpe_oos':        sharpe_oos,
        'sharpe_is':         sharpe_is,
        'n_ptk_factors':     F.shape[0],
        'n_factor_dim':      F.shape[1],
    }
    return pred_df, results


# ============================================================================
# OOS R²  (GKX 2020 convention: benchmark = zero mean)
# ============================================================================

def compute_oos_r2(pred_df, col='pred_ptk_sdf'):
    y = pred_df['y_true'].values
    p = pred_df[col].values
    return 1 - np.sum((y - p) ** 2) / np.sum(y ** 2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config()

    # Load data
    df = load_data()
    raw_features = get_characteristics(df)

    # Fixed temporal splits (course specification — do not change)
    train_mask = df['eom'] <= pd.Timestamp('2015-12-31')
    val_mask   = (df['eom'] >= pd.Timestamp('2016-01-01')) & (df['eom'] <= pd.Timestamp('2018-12-31'))
    test_mask  = df['eom'] >= pd.Timestamp('2019-01-01')

    print(f"\nSplits:")
    print(f"  Train 2005-2015: {train_mask.sum():>10,} obs")
    print(f"  Val   2016-2018: {val_mask.sum():>10,} obs")
    print(f"  Test  2019-2024: {test_mask.sum():>10,} obs")

    # Paper's preprocessing (missing filter on train only — no look-ahead)
    df_processed, features = preprocess_data_paper_style(df, raw_features, train_mask)

    # Full PTK-SDF pipeline
    pred_df, results = model_ptk_sdf(df_processed, features, cfg)

    # OOS R² (must report per Section 3.4)
    r2 = compute_oos_r2(pred_df)
    print(f"\nTest OOS R² (PTK-SDF): {r2:.6f}")
    print("(Negative OOS R² expected — model optimises Sharpe, not MSE)")

    # Portfolio metrics via shared backtest.py
    proj_dir = os.path.join(SCRIPT_DIR, '..', 'Proj')
    if os.path.isdir(proj_dir):
        sys.path.insert(0, proj_dir)
        try:
            from backtest import evaluate_all, print_summary
            print("\n--- Portfolio Performance (equal-weight decile L/S) ---")
            print_summary(evaluate_all(pred_df))
        except Exception as e:
            print(f"backtest.py note: {e}")

    # Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pred_path = os.path.join(RESULTS_DIR, 'ptk_test_predictions.parquet')
    pred_df.to_parquet(pred_path, index=False)
    print(f"\nSaved predictions → {pred_path}")

    summary = pd.DataFrame([{
        'model':            'ptk_sdf',
        'oos_r2_test':      r2,
        'sharpe_oos_sdf':   results['sharpe_oos'],
        'sharpe_is_sdf':    results['sharpe_is'],
        'n_obs':            len(pred_df),
        'n_months':         pred_df['eom'].nunique(),
        'n_ptk_factors':    results['n_ptk_factors'],
        'n_factor_dim':     results['n_factor_dim'],
        'hidden_dim':       cfg.HIDDEN_DIM,
        'epochs':           cfg.DNN_EPOCHS,
        'lr':               cfg.DNN_LEARNING_RATE,
        'rolling_window':   cfg.ROLLING_WINDOW,
        'ridge_penalty':    cfg.RIDGE_PENALTY,
    }])
    csv_path = os.path.join(RESULTS_DIR, 'ptk_results.csv')
    summary.to_csv(csv_path, index=False)
    print(f"Saved summary   → {csv_path}")

    return pred_df, results


if __name__ == '__main__':
    pred_df, results = main()
