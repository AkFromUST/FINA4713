"""
Simplified PTK-SDF for FINA4713 Project
Based on Kelly et al. (2024) "Large and Deep Factor Models"

Adapted to course dataset:
  - Data:  ../Proj/jkp_data.parquet
  - Train: 2005-01 to 2015-12
  - Val:   2016-01 to 2018-12
  - Test:  2019-01 to 2024-12

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
    MAX_TRAIN_SAMPLES = 100000 # Memory limit: max observations per DNN training

    # ========== PTK SPLIT (For pricing stage) ==========
    PTK_TRAIN_SPLIT = 0.7      # % of PTK factors used for training (rest for test)


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
    print(f"Loaded {len(df):,} observations  ({df['eom'].min().date()} – {df['eom'].max().date()})")
    return df


def get_characteristics(df):
    """Get list of characteristic column names"""
    exclude = {'id', 'eom', 'excntry', 'ret_exc_lead1m', 'me', 'mcap_usd', 'target_std'}
    chars = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    print(f"Raw characteristics count: {len(chars)}")
    return chars


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
# ROLLING-WINDOW PER-STOCK PREDICTION  (primary output for backtest.py)
# ============================================================================

def generate_test_predictions(df_processed, features, cfg, verbose=True):
    """
    For each test month (2019-01 to 2024-12):
      1. Train DNN on the preceding ROLLING_WINDOW months.
      2. Run DNN forward pass on the test month's stocks → per-stock signal.

    The MSRR loss trains the DNN to find portfolio weights that maximise Sharpe,
    so the DNN output θ(X_i,t) is directly the stock-level signal.

    Returns a DataFrame compatible with backtest.py:
      id, eom, excntry, y_true, me_raw, pred_ptk_sdf
    """
    all_months  = sorted(df_processed['eom'].unique())
    test_months = [m for m in all_months if m >= pd.Timestamp('2019-01-01')]

    print(f"\n[Rolling prediction] {len(test_months)} test months  (window = {cfg.ROLLING_WINDOW} months)")

    rows = []
    for i, month in enumerate(test_months):
        if verbose and i % 12 == 0:
            print(f"  {i+1}/{len(test_months)}  {month.date()}")

        win = [m for m in all_months if m < month][-cfg.ROLLING_WINDOW:]
        if len(win) < max(6, cfg.ROLLING_WINDOW // 2):
            continue

        tr_mask = df_processed['eom'].isin(win)
        X_tr    = df_processed.loc[tr_mask, features].values.astype(np.float32)
        R_tr    = df_processed.loc[tr_mask, 'ret_exc_lead1m'].values.astype(np.float32)

        if len(X_tr) > cfg.MAX_TRAIN_SAMPLES:
            idx  = np.random.choice(len(X_tr), cfg.MAX_TRAIN_SAMPLES, replace=False)
            X_tr = X_tr[idx];  R_tr = R_tr[idx]

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        model    = train_simplified_dnn(X_tr_sc, R_tr, len(features), cfg)

        te_mask = df_processed['eom'] == month
        X_te    = df_processed.loc[te_mask, features].values.astype(np.float32)
        if len(X_te) < cfg.MIN_STOCKS_PER_MONTH:
            continue

        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(scaler.transform(X_te))).numpy()

        month_df = df_processed.loc[te_mask, ['id', 'eom', 'excntry', 'ret_exc_lead1m']].copy()
        month_df = month_df.rename(columns={'ret_exc_lead1m': 'y_true'})

        # me is excluded from preprocessing so its original value is preserved
        if 'me' in df_processed.columns:
            month_df['me_raw'] = df_processed.loc[te_mask, 'me'].values

        month_df['pred_ptk_sdf'] = preds
        rows.append(month_df)

    pred_df = pd.concat(rows, ignore_index=True)
    print(f"Generated {len(pred_df):,} stock-month predictions  ({pred_df['eom'].nunique()} months)")
    return pred_df


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

    # Paper's preprocessing (missing filter fitted on train only — no look-ahead)
    df_processed, features = preprocess_data_paper_style(df, raw_features, train_mask)

    # Rolling-window per-stock predictions for the test period
    pred_df = generate_test_predictions(df_processed, features, cfg)

    # OOS R² (must report per Section 3.4)
    r2 = compute_oos_r2(pred_df)
    print(f"\nTest OOS R² (PTK-SDF): {r2:.6f}")

    # Portfolio metrics via shared backtest.py
    proj_dir = os.path.join(SCRIPT_DIR, '..', 'Proj')
    if os.path.isdir(proj_dir):
        sys.path.insert(0, proj_dir)
        try:
            from backtest import evaluate_all, print_summary
            print("\n--- Portfolio Performance (equal-weight decile L/S) ---")
            print_summary(evaluate_all(pred_df))
        except Exception as e:
            print(f"backtest.py unavailable: {e}")

    # Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pred_path = os.path.join(RESULTS_DIR, 'ptk_test_predictions.parquet')
    pred_df.to_parquet(pred_path, index=False)
    print(f"\nSaved predictions → {pred_path}")

    summary = pd.DataFrame([{
        'model':          'ptk_sdf',
        'oos_r2_test':    r2,
        'n_obs':          len(pred_df),
        'n_months':       pred_df['eom'].nunique(),
        'hidden_dim':     cfg.HIDDEN_DIM,
        'epochs':         cfg.DNN_EPOCHS,
        'lr':             cfg.DNN_LEARNING_RATE,
        'rolling_window': cfg.ROLLING_WINDOW,
        'ridge_penalty':  cfg.RIDGE_PENALTY,
    }])
    csv_path = os.path.join(RESULTS_DIR, 'ptk_results.csv')
    summary.to_csv(csv_path, index=False)
    print(f"Saved summary   → {csv_path}")

    return pred_df


if __name__ == '__main__':
    pred_df = main()
