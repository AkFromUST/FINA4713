"""
Simplified PTK-SDF for Limited Data (2005-2024)
Based on Kelly et al. (2024) "Large and Deep Factor Models"

Key simplifications for your data:
- Shallow network (1 hidden layer, 32 nodes) instead of deep wide networks
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
from typing import Tuple, List, Optional, Dict
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


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

def preprocess_data_paper_style(df, characteristics, verbose=True):
    """
    Implements the pre-processing steps from the paper:
    - Impute missing with cross-sectional median
    - Rank-standardize to [-0.5, 0.5]
    - Filter characteristics with >30% missing
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
                # Get ranks (handling NaN by pushing to end)
                rank = np.argsort(np.argsort(values))
                # Normalize to [-0.5, 0.5]
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
    
    # Verify no NaNs remain
    remaining_nans = df_processed[characteristics].isna().sum().sum()
    if verbose:
        print(f"  Remaining NaNs: {remaining_nans}")
    
    return df_processed, characteristics


# ============================================================================
# SIMPLIFIED DNN (Shallow, avoids overfitting)
# ============================================================================

class SimplifiedPTKNetwork(nn.Module):
    """
    Shallow neural network designed for limited data.
    - 1 hidden layer, 32 neurons (much smaller than paper's 256)
    - ReLU activation
    - Output: portfolio weight for each stock
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
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
    This is the key innovation from the paper.
    
    Loss = (1 - portfolio_return)^2 / N
    
    Minimizing this maximizes the Sharpe ratio of the portfolio.
    """
    portfolio_return = (predictions * returns).sum()
    loss = (1 - portfolio_return) ** 2
    return loss / len(returns)


def train_simplified_dnn(
    X_train: np.ndarray,
    R_train: np.ndarray,
    input_dim: int,
    hidden_dim: int = 32,
    epochs: int = 30,
    lr: float = 1e-5,
    verbose: bool = False
) -> nn.Module:
    """Train simplified DNN with MSRR loss"""
    
    model = SimplifiedPTKNetwork(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_t = torch.FloatTensor(X_train)
    R_t = torch.FloatTensor(R_train)
    
    for epoch in range(epochs):
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
    """
    Compute PTK gradient features.
    
    These are the derivatives of the portfolio return with respect to
    each model parameter. They become the "factors" for pricing.
    """
    model.eval()
    
    X_t = torch.FloatTensor(X).requires_grad_(True)
    R_t = torch.FloatTensor(R)
    
    predictions = model(X_t)
    portfolio_return = (predictions * R_t).sum()
    
    # Scale by 1/sqrt(N) as in paper
    N = len(R)
    scaling = 1.0 / np.sqrt(N)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=portfolio_return,
        inputs=model.parameters(),
        create_graph=False,
        retain_graph=False
    )
    
    # Concatenate all gradients into a single vector
    grad_vector = torch.cat([g.flatten() for g in gradients])
    factors = (scaling * grad_vector).detach().numpy()
    
    return factors


# ============================================================================
# PTK-SDF PRICING (Ridge Regression)
# ============================================================================

class PTK_Pricer:
    """
    Prices PTK factors using ridge regression.
    This is the closed-form optimal combination of factors.
    """
    
    def __init__(self, ridge_penalty: float = 1e-5):
        self.ridge_penalty = ridge_penalty
        self.weights = None
        self.scaler = StandardScaler()
        
    def fit(self, F: np.ndarray) -> 'PTK_Pricer':
        """Estimate optimal weights for PTK factors"""
        T, P = F.shape
        
        # Standardize factors
        F_scaled = self.scaler.fit_transform(F)
        
        # Sample mean
        F_bar = F_scaled.mean(axis=0)
        
        # Sample covariance
        covariance = (F_scaled.T @ F_scaled) / T
        
        # Ridge solution: (zI + Σ)^(-1) μ
        zI = self.ridge_penalty * np.eye(P)
        
        try:
            inv_matrix = np.linalg.inv(zI + covariance)
            self.weights = inv_matrix @ F_bar
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            self.weights = np.linalg.pinv(zI + covariance) @ F_bar
        
        return self
    
    def predict(self, F_new: np.ndarray) -> float:
        """Compute SDF return for new factors"""
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
    rolling_window: int = 24,  # Smaller window for limited data
    hidden_dim: int = 64,
    ridge_penalty: float = 1e-5,
    verbose: bool = True
):
    """
    Simplified PTK-SDF implementation for limited data.
    
    Key simplifications:
    - Shallow network (1 hidden layer, 32 nodes)
    - 24-month rolling window (vs paper's 60)
    - Proper imputation (no dropping)
    """
    
    print("\n" + "="*60)
    print("SIMPLIFIED PTK-SDF (Limited Data Version)")
    print("="*60)
    
    # First, preprocess data properly (impute, don't drop)
    df_processed, features = preprocess_data_paper_style(df, features, verbose)
    
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
            
            # Require at least 50 stocks per month
            if len(X_month) >= 50:
                X_list.append(X_month)
                R_list.append(R_month)
                valid_months.append(month)
        
        return X_list, R_list, valid_months
    
    # Get data for each split
    X_train_list, R_train_list, train_months_valid = get_month_data(train_months)
    X_val_list, R_val_list, val_months_valid = get_month_data(val_months)
    X_test_list, R_test_list, test_months_valid = get_month_data(test_months)
    
    print(f"\nMonths with >=50 stocks:")
    print(f"  Train: {len(X_train_list)}")
    print(f"  Val: {len(X_val_list)}")
    print(f"  Test: {len(X_test_list)}")
    
    # Combine train+val for training
    X_all_list = X_train_list + X_val_list
    R_all_list = R_train_list + R_val_list
    all_months = train_months_valid + val_months_valid
    
    if len(X_all_list) < rolling_window + 1:
        print(f"\nWARNING: Need at least {rolling_window + 1} months, have {len(X_all_list)}")
        print(f"Reducing rolling window to {len(X_all_list) - 1}")
        rolling_window = max(12, len(X_all_list) - 1)
    
    print(f"\nRolling window size: {rolling_window} months")
    print(f"Total months for PTK extraction: {len(X_all_list)}")
    
    # Step 1: Rolling window estimation
    print("\n[Step 1] Training DNNs and extracting PTK factors...")
    
    all_ptk_factors = []
    factor_months = []
    
    for t in range(rolling_window, len(X_all_list)):
        if (t - rolling_window) % 12 == 0:
            print(f"  Processing month {t}/{len(X_all_list)}")
        
        # Training data: previous rolling_window months
        X_train_stack = []
        R_train_stack = []
        for i in range(t - rolling_window, t):
            X_train_stack.append(X_all_list[i])
            R_train_stack.append(R_all_list[i])
        
        X_train_full = np.vstack(X_train_stack)
        R_train_full = np.concatenate(R_train_stack)
        
        # Sample for memory (max 100k observations)
        if len(X_train_full) > 100000:
            idx = np.random.choice(len(X_train_full), 100000, replace=False)
            X_train_full = X_train_full[idx]
            R_train_full = R_train_full[idx]
        
        # Standardize features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_full)
        
        # Train DNN
        model = train_simplified_dnn(
            X_train_scaled, R_train_full,
            input_dim=len(features),
            hidden_dim=hidden_dim,
            epochs=20,
            lr=1e-3,
            verbose=False
        )
        
        # Extract PTK factors for the next month (test)
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
        return df_processed, {'sdf_returns': np.array([0]), 'sharpe': 0}
    
    # Step 2: PTK pricing
    print("\n[Step 2] PTK Pricing with Ridge Regression...")
    
    F_matrix = np.array(all_ptk_factors)
    print(f"  Factor matrix shape: {F_matrix.shape}")
    
    # Use last 70% for training PTK, first 30% for testing
    # (Reversed because time order - later months for testing)
    split = int(len(F_matrix) * 0.7)
    
    F_train = F_matrix[:split]
    F_test = F_matrix[split:]
    test_months_ptk = factor_months[split:]
    
    print(f"  PTK training samples: {len(F_train)}")
    print(f"  PTK test samples: {len(F_test)}")
    
    # Fit PTK pricer
    pricer = PTK_Pricer(ridge_penalty=ridge_penalty)
    pricer.fit(F_train)
    
    # Generate SDF returns
    sdf_returns = np.array([pricer.predict(f) for f in F_test])
    
    # Store in dataframe
    df_processed['pred_ptk'] = np.nan
    for month, ret in zip(test_months_ptk, sdf_returns):
        df_processed.loc[df_processed['eom'] == month, 'pred_ptk'] = ret
    
    # Step 3: Evaluation
    sharpe_annualized = np.sqrt(12) * sdf_returns.mean() / (sdf_returns.std() + 1e-8)
    
    print("\n" + "="*60)
    print("PTK-SDF PERFORMANCE")
    print("="*60)
    print(f"Mean monthly SDF return: {sdf_returns.mean():.6f}")
    print(f"Std monthly SDF return: {sdf_returns.std():.6f}")
    print(f"Annualized Sharpe Ratio: {sharpe_annualized:.2f}")
    print(f"Total return: {(1 + sdf_returns).prod() - 1:.2%}")
    print("="*60)
    
    if sharpe_annualized < 1.0:
        print("\nNOTE: Lower Sharpe than paper (3.8-3.9) due to:")
        print("  - Shorter time period (2005-2024 vs 1963-2024)")
        print("  - Simpler network (32 vs 256 hidden units)")
        print("  - Smaller rolling window (24 vs 60 months)")
    
    return df_processed, {
        'sdf_returns': sdf_returns,
        'sharpe': sharpe_annualized,
        'dates': test_months_ptk,
        'n_factors': F_matrix.shape[1]
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main_with_ptk():
    """Main function to run simplified PTK-SDF"""
    
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
    
    train_end_idx = int(n_months * 0.6)   # 60% for training
    val_end_idx = int(n_months * 0.8)     # 20% for validation
    
    train_end_date = unique_dates[train_end_idx]
    val_end_date = unique_dates[val_end_idx]
    
    train_mask = df['eom'] <= train_end_date
    val_mask = (df['eom'] > train_end_date) & (df['eom'] <= val_end_date)
    test_mask = df['eom'] > val_end_date
    
    print(f"\nSplit configuration:")
    print(f"  Train end: {train_end_date.date()} ({train_mask.sum():,} obs)")
    print(f"  Val end: {val_end_date.date()} ({val_mask.sum():,} obs)")
    print(f"  Test: {df[test_mask].shape[0]:,} obs")
    
    # Run PTK-SDF with simplified settings
    df_processed, results = model_ptk_sdf(
        df, train_mask, val_mask, test_mask, raw_features,
        rolling_window=24,      # Smaller than paper's 60
        hidden_dim=64,          # Much smaller than paper's 256
        ridge_penalty=1e-4,
        verbose=True
    )
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"PTK-SDF Annualized Sharpe Ratio: {results['sharpe']:.2f}")
    print(f"Number of PTK factors: {results['n_factors']:,}")
    print("="*60)
    
    return df_processed, results


if __name__ == "__main__":
    df, results = main_with_ptk()