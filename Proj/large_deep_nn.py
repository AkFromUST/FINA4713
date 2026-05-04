"""
Simplified PTK-SDF for Limited Data (2005-2024)
Based on Kelly et al. (2024) "Large and Deep Factor Models"

Key simplifications for your data:
- Shallow network (1 hidden layer) instead of deep wide networks
- 24-month rolling window instead of 60 months
- Proper imputation (paper's method) instead of dropping NaN
- MSRR loss (the core innovation)
- PTK gradient features + ridge pricing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')
import os

import random

def set_all_seeds(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    Use the SAME seed value for every run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # Make PyTorch deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For Python's hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All random seeds set to: {seed}")

# Call this BEFORE any other imports or code
set_all_seeds(42)

# ============================================================================
# HYPERPARAMETER CONFIGURATION - CHANGE ONLY HERE
# ============================================================================

class Config:
    """
    All hyperparameters in one place.
    Tweak these values to experiment with the model.
    """
    
    # ========== DNN ARCHITECTURE ==========
    HIDDEN_DIM = 32      # Recommended: 32, 64, 128
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
    RIDGE_PENALTY = 1e-5     # Recommended: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
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

def load_data(path='jkp_data.parquet'):
    """Load and prepare data"""
    df = pd.read_parquet(path)
    df['eom'] = pd.to_datetime(df['eom'])
    df = df.dropna(subset=['ret_exc_lead1m'])
    print(f"Loaded {len(df):,} observations")
    print(f"Date range: {df['eom'].min()} to {df['eom'].max()}")
    return df

def get_characteristics(df):
    """Get list of characteristic column names"""
    exclude = ['id', 'eom', 'excntry', 'ret_exc_lead1m', 'me', 'mcap_usd', 'target_std']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    characteristics = [c for c in numeric_cols if c not in exclude]
    print(f"Raw characteristics count: {len(characteristics)}")
    return characteristics


# ============================================================================
# DATA PRE-PROCESSING (Paper's method - impute, don't drop)
# ============================================================================

def preprocess_data_paper_style(df, characteristics, cfg: Config, verbose=True):
    """
    Implements the pre-processing steps from the paper:
    - Filter characteristics with >30% missing
    - Rank-standardize to [-0.5, 0.5]
    - Impute missing with cross-sectional median
    """
    
    if verbose:
        print("\n" + "="*60)
        print("PRE-PROCESSING (Paper's Method)")
        print("="*60)
    
    df_processed = df.copy()
    
    # Step 1: Filter characteristics with <30% missing
    missing_pct = df_processed[characteristics].isna().mean()
    keep_chars = missing_pct[missing_pct < 0.30].index.tolist()
    
    if verbose:
        print(f"  Characteristics: {len(characteristics)} → {len(keep_chars)} (<30% missing)")
    
    characteristics = keep_chars
    
    # Step 2: Rank-standardize each characteristic cross-sectionally to [-0.5, 0.5]
    if verbose:
        print("  Rank-standardizing characteristics...")
    
    for char in characteristics:
        for month in df_processed['eom'].unique():
            month_mask = df_processed['eom'] == month
            values = df_processed.loc[month_mask, char].values
            
            if len(values) > 0:
                rank = np.argsort(np.argsort(values))
                if len(rank) > 1:
                    normalized = (rank / (len(rank) - 1) - 0.5)
                else:
                    normalized = np.zeros_like(rank)
                df_processed.loc[month_mask, char] = normalized
    
    # Step 3: Impute remaining missing values with cross-sectional median
    if verbose:
        print("  Imputing missing values with cross-sectional median...")
    
    for month in df_processed['eom'].unique():
        month_mask = df_processed['eom'] == month
        for char in characteristics:
            median_val = df_processed.loc[month_mask, char].median()
            if pd.isna(median_val):
                median_val = df_processed[char].median()
            df_processed.loc[month_mask, char] = df_processed.loc[month_mask, char].fillna(median_val)
    
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


def train_simplified_dnn(
    X_train: np.ndarray,
    R_train: np.ndarray,
    input_dim: int,
    cfg: Config,
    verbose: bool = False
) -> nn.Module:
    """Train DNN with MSRR loss"""
    
    model = SimplifiedPTKNetwork(input_dim, cfg.HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=cfg.DNN_LEARNING_RATE)
    
    X_t = torch.FloatTensor(X_train)
    R_t = torch.FloatTensor(R_train)
    
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
    
    return model


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
    
    X_t = torch.FloatTensor(X).requires_grad_(True)
    R_t = torch.FloatTensor(R)
    
    predictions = model(X_t)
    portfolio_return = (predictions * R_t).sum()
    
    N = len(R)
    scaling = 1.0 / np.sqrt(N)
    
    gradients = torch.autograd.grad(
        outputs=portfolio_return,
        inputs=model.parameters(),
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
# MAIN PTK-SDF PIPELINE
# ============================================================================

def model_ptk_sdf(
    df: pd.DataFrame,
    train_mask,
    val_mask,
    test_mask,
    features: List[str],
    cfg: Config,
    verbose: bool = True
) -> tuple:
    """
    Simplified PTK-SDF implementation.
    
    IMPORTANT: The Sharpe ratio reported is computed on the TEST set of the PTK pricing stage.
    This is truly out-of-sample - these months were NEVER used for training the PTK pricer.
    """
    
    print("\n" + "="*60)
    print("SIMPLIFIED PTK-SDF (Limited Data Version)")
    print("="*60)
    print(f"\nCONFIGURATION:")
    print(f"  Hidden Dim: {cfg.HIDDEN_DIM}")
    print(f"  DNN Epochs: {cfg.DNN_EPOCHS}")
    print(f"  DNN Learning Rate: {cfg.DNN_LEARNING_RATE}")
    print(f"  Ridge Penalty: {cfg.RIDGE_PENALTY}")
    print(f"  Rolling Window: {cfg.ROLLING_WINDOW}")
    
    # Preprocess data
    df_processed, features = preprocess_data_paper_style(df, features, cfg, verbose)
    
    # Get dates
    dates = df_processed['eom']
    
    # Get unique months in each split
    train_months = sorted(dates[train_mask].unique())
    val_months = sorted(dates[val_mask].unique())
    test_months = sorted(dates[test_mask].unique())
    
    print(f"\nMonths after preprocessing:")
    print(f"  Train months: {len(train_months)}")
    print(f"  Val months: {len(val_months)}")
    print(f"  Test months: {len(test_months)}")
    
    # Function to get data for a list of months
    def get_month_data(month_list):
        X_list = []
        R_list = []
        valid_months = []
        
        for month in month_list:
            month_mask = dates == month
            X_month = df_processed.loc[month_mask, features].values
            R_month = df_processed.loc[month_mask, 'ret_exc_lead1m'].values
            
            if len(X_month) >= cfg.MIN_STOCKS_PER_MONTH:
                X_list.append(X_month)
                R_list.append(R_month)
                valid_months.append(month)
        
        return X_list, R_list, valid_months
    
    # Get data for each split
    X_train_list, R_train_list, train_months_valid = get_month_data(train_months)
    X_val_list, R_val_list, val_months_valid = get_month_data(val_months)
    X_test_list, R_test_list, test_months_valid = get_month_data(test_months)
    
    print(f"\nMonths with >= {cfg.MIN_STOCKS_PER_MONTH} stocks:")
    print(f"  Train: {len(X_train_list)}")
    print(f"  Val: {len(X_val_list)}")
    print(f"  Test: {len(X_test_list)}")
    
    # Combine train+val for PTK extraction
    X_all_list = X_train_list + X_val_list
    R_all_list = R_train_list + R_val_list
    all_months = train_months_valid + val_months_valid
    
    window = min(cfg.ROLLING_WINDOW, len(X_all_list) - 1)
    print(f"\nRolling window size: {window} months")
    print(f"Total months for PTK extraction: {len(X_all_list)}")
    
    # Step 1: Rolling window estimation - extract PTK factors
    print("\n[Step 1] Training DNNs and extracting PTK factors...")
    
    all_ptk_factors = []
    factor_months = []
    
    for t in range(window, len(X_all_list)):
        if (t - window) % 12 == 0:
            print(f"  Processing month {t}/{len(X_all_list)}")
        
        # Training data: previous window months
        X_train_stack = []
        R_train_stack = []
        for i in range(t - window, t):
            X_train_stack.append(X_all_list[i])
            R_train_stack.append(R_all_list[i])
        
        X_train_full = np.vstack(X_train_stack)
        R_train_full = np.concatenate(R_train_stack)
        
        # Sample for memory
        if len(X_train_full) > cfg.MAX_TRAIN_SAMPLES:
            idx = np.random.choice(len(X_train_full), cfg.MAX_TRAIN_SAMPLES, replace=False)
            X_train_full = X_train_full[idx]
            R_train_full = R_train_full[idx]
        
        # Standardize features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_full)
        
        # Train DNN
        model = train_simplified_dnn(
            X_train_scaled, R_train_full,
            input_dim=len(features),
            cfg=cfg,
            verbose=False
        )
        
        # Extract PTK factors for the next month
        X_test_month = X_all_list[t]
        R_test_month = R_all_list[t]
        X_test_scaled = scaler_X.transform(X_test_month)
        
        try:
            factors = compute_ptk_gradients(model, X_test_scaled, R_test_month)
            all_ptk_factors.append(factors)
            factor_months.append(all_months[t])
        except Exception as e:
            print(f"    Warning: Failed for month {all_months[t]}: {e}")
    
    print(f"\nExtracted {len(all_ptk_factors)} PTK factor vectors")
    
    if len(all_ptk_factors) < 12:
        print("ERROR: Not enough PTK factors extracted")
        df_processed['pred_ptk'] = 0
        return df_processed, {'sdf_returns': np.array([0]), 'sharpe': 0, 'type': 'ERROR'}
    
    # Step 2: PTK pricing
    print("\n[Step 2] PTK Pricing with Ridge Regression...")
    
    F_matrix = np.array(all_ptk_factors)
    print(f"  Factor matrix shape: {F_matrix.shape}")
    
    # Split: first 70% train PTK pricer, last 30% test (true out-of-sample)
    split = int(len(F_matrix) * cfg.PTK_TRAIN_SPLIT)
    
    F_train = F_matrix[:split]
    F_test = F_matrix[split:]
    test_months_ptk = factor_months[split:]
    
    print(f"  PTK training samples (used to estimate weights): {len(F_train)}")
    print(f"  PTK test samples (true out-of-sample): {len(F_test)}")
    
    # Fit PTK pricer on training factors
    pricer = PTK_Pricer(ridge_penalty=cfg.RIDGE_PENALTY)
    pricer.fit(F_train)
    
    # Generate SDF returns on test factors (TRULY OUT-OF-SAMPLE)
    sdf_returns = np.array([pricer.predict(f) for f in F_test])
    
    # Store in dataframe
    df_processed['pred_ptk'] = np.nan
    for month, ret in zip(test_months_ptk, sdf_returns):
        df_processed.loc[df_processed['eom'] == month, 'pred_ptk'] = ret
    
    # EVALUATION: Sharpe ratio on TEST set only
    sharpe_annualized = np.sqrt(12) * sdf_returns.mean() / (sdf_returns.std() + 1e-8)
    
    print("\n" + "="*60)
    print("PTK-SDF PERFORMANCE (Test Set Only)")
    print("="*60)
    print(f"These results are from {len(sdf_returns)} months of TRUE OUT-OF-SAMPLE data.")
    print(f"These months were NEVER used to train the PTK pricer.")
    print("-" * 60)
    print(f"Mean monthly SDF return: {sdf_returns.mean():.6f}")
    print(f"Std monthly SDF return: {sdf_returns.std():.6f}")
    print(f"Annualized Sharpe Ratio: {sharpe_annualized:.2f}")
    print(f"Total return: {(1 + sdf_returns).prod() - 1:.2%}")
    print("="*60)
    
    if sharpe_annualized < 1.0:
        print("\nNOTE: Lower Sharpe than paper (3.8-3.9) due to:")
        print("  - Shorter time period (2005-2024 vs 1963-2024)")
        print("  - Simpler network vs paper's deep wide network")
        print("  - Smaller rolling window vs paper's 60 months")
    elif sharpe_annualized > 2.0:
        print("\nWARNING: Sharpe > 2.0 - check for look-ahead bias or overfitting")
    
    return df_processed, {
        'sdf_returns': sdf_returns,
        'sharpe': sharpe_annualized,
        'dates': test_months_ptk,
        'n_factors': F_matrix.shape[1],
        'n_test_months': len(sdf_returns)
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main_with_ptk():
    """Main function to run simplified PTK-SDF"""
    
    # Load configuration
    cfg = Config()
    
    # Load data
    df = load_data('jkp_data.parquet')
    raw_features = get_characteristics(df)
    
    # Check date range
    first_date = df['eom'].min()
    last_date = df['eom'].max()
    print(f"\nData spans {first_date.date()} to {last_date.date()}")
    
    # Split dates (60/20/20 split for your 20-year data)
    unique_dates = sorted(df['eom'].unique())
    n_months = len(unique_dates)
    
    train_end_idx = int(n_months * 0.6)
    val_end_idx = int(n_months * 0.8)
    
    train_end_date = unique_dates[train_end_idx]
    val_end_date = unique_dates[val_end_idx]
    
    train_mask = df['eom'] <= train_end_date
    val_mask = (df['eom'] > train_end_date) & (df['eom'] <= val_end_date)
    test_mask = df['eom'] > val_end_date
    
    print(f"\nSplit configuration:")
    print(f"  Train end: {train_end_date.date()}")
    print(f"  Val end: {val_end_date.date()}")
    
    # Run PTK-SDF
    df_processed, results = model_ptk_sdf(
        df, train_mask, val_mask, test_mask, raw_features,
        cfg=cfg,
        verbose=True
    )
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Test Period: {results['dates'][0].date()} to {results['dates'][-1].date()}")
    print(f"Number of test months: {results['n_test_months']}")
    print(f"PTK-SDF Annualized Sharpe Ratio: {results['sharpe']:.2f}")
    print(f"Number of PTK factors: {results['n_factors']:,}")
    print("="*60)
    print("\nIMPORTANT: The Sharpe ratio above is computed on TRUE OUT-OF-SAMPLE data.")
    print("These months were never used to train the PTK pricer.")
    
    return df_processed, results


if __name__ == "__main__":
    df, results = main_with_ptk()