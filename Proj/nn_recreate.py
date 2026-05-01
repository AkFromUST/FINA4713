"""
Replication of Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine Learning"
Neural Network Model NN3 (their best performing architecture)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict, List, Optional


# ============================================================================
# DATA LOADING FUNCTIONS
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
    exclude = ['id', 'eom', 'excntry', 'ret_exc_lead1m', 'me', 'target_std']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    characteristics = [c for c in numeric_cols if c not in exclude]
    print(f"Using {len(characteristics)} characteristics")
    return characteristics


# ============================================================================
# ARCHITECTURE - Exactly as described in paper
# ============================================================================

class GKX_NN3(nn.Module):
    """
    Neural network with exactly 3 hidden layers: 32, 16, 8 neurons.
    """
    
    def __init__(self, input_dim: int):
        super(GKX_NN3, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.output(x)
        return x.squeeze(-1)


# ============================================================================
# REGULARIZATION AND TRAINING
# ============================================================================

def train_with_early_stopping(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 1024,
    learning_rate: float = 0.00001,
    lambda_l1: float = 1e-5,
    patience: int = 5,
    max_epochs: int = 100,
    verbose: bool = False
) -> Tuple[nn.Module, Dict]:
    """
    Train with early stopping and L1 regularization.
    """
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # Create DataLoader
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state_dict = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss_sum = 0
        
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            mse_loss = criterion(predictions, y_batch)
            
            # L1 regularization
            l1_loss = sum(p.abs().sum() for p in model.parameters())
            total_loss = mse_loss + lambda_l1 * l1_loss
            
            total_loss.backward()
            optimizer.step()
            train_loss_sum += total_loss.item()
        
        avg_train_loss = train_loss_sum / len(loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = criterion(val_predictions, y_val_t).item()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early stopping check - FIXED: check for nan/inf
        if np.isnan(val_loss) or np.isinf(val_loss):
            if verbose:
                print(f"Warning: NaN/Inf validation loss at epoch {epoch}")
            # Keep previous best model if exists, otherwise continue
            if best_state_dict is None:
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping due to NaN/Inf at epoch {epoch}")
                break
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model - FIXED: ensure we always have a state dict
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    else:
        # If no improvement ever, use the final model
        if verbose:
            print("Warning: No improvement in validation loss, using final model")
    
    return model, history


# ============================================================================
# ENSEMBLE - Multiple random seeds
# ============================================================================

class GKX_Ensemble:
    """
    Ensemble averaging across multiple random seeds.
    """
    
    def __init__(self, input_dim: int, n_ensemble: int = 5, seed_list: Optional[List[int]] = None):
        self.input_dim = input_dim
        self.n_ensemble = n_ensemble
        
        if seed_list is None:
            self.seed_list = [42, 123, 456, 789, 101112]
        else:
            self.seed_list = seed_list[:n_ensemble]
            
        self.models: List[GKX_NN3] = []
        self.histories: List[Dict] = []
        
    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        lambda_l1: float = 1e-5,
        patience: int = 5,
        max_epochs: int = 100,
        verbose: bool = True
    ) -> None:
        """Train ensemble of models with different random seeds."""
        
        self.models = []
        self.histories = []
        
        for i, seed in enumerate(self.seed_list):
            if verbose:
                print(f"Training ensemble model {i+1}/{len(self.seed_list)} (seed={seed})")
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = GKX_NN3(self.input_dim)
            
            trained_model, history = train_with_early_stopping(
                model, X_train, y_train, X_val, y_val,
                batch_size=batch_size,
                learning_rate=learning_rate,
                lambda_l1=lambda_l1,
                patience=patience,
                max_epochs=max_epochs,
                verbose=verbose
            )
            
            self.models.append(trained_model)
            self.histories.append(history)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction: average of all models' forecasts."""
        X_t = torch.FloatTensor(X)
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_t).numpy()
            predictions.append(pred)
        
        return np.mean(predictions, axis=0)

def model_neuralnet(df, train_mask, val_mask, test_mask, features):
    """
    Train neural network on training data, validate on validation.
    Implements missing data handling as per paper: replace with cross-sectional medians.
    """
    
    print("\n" + "="*60)
    print("Training GKX (2020) NN3 Neural Network")
    print("="*60)
    
    # Extract data
    X_train = df.loc[train_mask, features].copy()
    y_train = df.loc[train_mask, 'ret_exc_lead1m'].copy()
    
    X_val = df.loc[val_mask, features].copy()
    y_val = df.loc[val_mask, 'ret_exc_lead1m'].copy()
    
    X_test = df.loc[test_mask, features].copy()
    y_test = df.loc[test_mask, 'ret_exc_lead1m'].copy()
    
    print(f"Raw training samples: {len(X_train):,}")
    print(f"Raw validation samples: {len(X_val):,}")
    print(f"Raw test samples: {len(X_test):,}")
    
    # ================================================================
    # MISSING DATA HANDLING - As per paper (footnote 30, page 2248)
    # "missing characteristics are replaced with the cross-sectional 
    # median at each month for each stock, respectively"
    # ================================================================
    
    def impute_with_cross_sectional_median(X, date_series):
        """
        Impute missing values with cross-sectional median per month.
        This matches the paper's description.
        """
        X_imputed = X.copy()
        
        # Get unique dates (months)
        unique_dates = date_series.unique()
        
        for date in unique_dates:
            # Get rows for this date
            date_mask = date_series == date
            X_date = X_imputed[date_mask]
            
            if len(X_date) > 0:
                # Compute cross-sectional median for each feature
                date_medians = X_date.median()
                
                # Fill NaN with the column's median for that date
                for col in X_imputed.columns:
                    col_median = date_medians[col]
                    if pd.isna(col_median):
                        # If entire column is NaN for this date, use overall median
                        col_median = X_imputed[col].median()
                    X_imputed.loc[date_mask, col] = X_imputed.loc[date_mask, col].fillna(col_median)
        
        return X_imputed
    
    # Get date series for each split
    train_dates = df.loc[train_mask, 'eom']
    val_dates = df.loc[val_mask, 'eom']
    test_dates = df.loc[test_mask, 'eom']
    
    # Impute missing values
    print("\nImputing missing values with cross-sectional medians...")
    X_train = impute_with_cross_sectional_median(X_train, train_dates)
    X_val = impute_with_cross_sectional_median(X_val, val_dates)
    X_test = impute_with_cross_sectional_median(X_test, test_dates)
    
    # Also impute any remaining NaNs with overall median
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_val.median())
    X_test = X_test.fillna(X_test.median())
    
    print(f"After imputation - Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # Convert to numpy and drop any remaining NaN rows (should be very few)
    X_train = X_train.values
    y_train = y_train.values
    X_val = X_val.values
    y_val = y_val.values
    X_test = X_test.values
    y_test = y_test.values
    
    # Remove rows where y is NaN (should be none after imputation)
    train_nan_mask = ~np.isnan(y_train)
    val_nan_mask = ~np.isnan(y_val)
    test_nan_mask = ~np.isnan(y_test)
    
    X_train = X_train[train_nan_mask]
    y_train = y_train[train_nan_mask]
    X_val = X_val[val_nan_mask]
    y_val = y_val[val_nan_mask]
    X_test = X_test[test_nan_mask]
    y_test = y_test[test_nan_mask]
    
    print(f"After final cleaning - Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # Check for constant columns in training data
    std_train = X_train.std(axis=0)
    constant_cols = np.where((std_train == 0) | np.isnan(std_train))[0]
    if len(constant_cols) > 0:
        print(f"Removing {len(constant_cols)} constant columns")
        keep_cols = [i for i in range(X_train.shape[1]) if i not in constant_cols]
        X_train = X_train[:, keep_cols]
        X_val = X_val[:, keep_cols]
        X_test = X_test[:, keep_cols]
    
    print(f"Features after removal: {X_train.shape[1]}")
    
    # Check if we have enough data
    if len(X_train) < 1000:
        print("WARNING: Very few training samples after cleaning. Consider checking data quality.")
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    
    # Check for NaN in scaled data
    print(f"NaN in X_train_scaled: {np.isnan(X_train_scaled).sum()}")
    print(f"NaN in y_train_scaled: {np.isnan(y_train_scaled).sum()}")
    
    # Train ensemble
    ensemble = GKX_Ensemble(
        input_dim=X_train_scaled.shape[1],
        n_ensemble=5
    )
    
    ensemble.train_all(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        batch_size=1024,
        learning_rate=0.001,
        lambda_l1=1e-5,
        patience=5,
        max_epochs=100,
        verbose=True
    )
    
    # Generate predictions for test set
    y_pred_test_scaled = ensemble.predict(X_test_scaled)
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
    
    # Store predictions in dataframe
    df['pred_nn'] = np.nan
    test_indices = df.loc[test_mask].index
    df.loc[test_indices, 'pred_nn'] = y_pred_test
    
    # Calculate OOS R2 using PAPER'S FORMULA (zero benchmark)
    ss_res = np.sum((y_test - y_pred_test) ** 2)
    ss_tot = np.sum(y_test ** 2)
    r2_oos_paper = 1 - ss_res / ss_tot
    
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE PERFORMANCE")
    print("="*60)
    print(f"Paper's R² (zero benchmark): {r2_oos_paper:.4%}")
    print(f"MSE: {mean_squared_error(y_test, y_pred_test):.8f}")
    
    # Also compute historical mean benchmark for comparison
    ss_tot_mean = np.sum((y_test - y_test.mean()) ** 2)
    r2_oos_mean = 1 - ss_res / ss_tot_mean
    print(f"R² (historical mean benchmark): {r2_oos_mean:.4%}")
    print("="*60)
    print("\nNOTE: Paper reports 0.40% monthly R² (zero benchmark) for NN3")
    
    return df, ensemble


def main_with_nn():
    """
    Main function to run neural network.
    """
    
    # Load data
    df = load_data('jkp_data.parquet')
    features = get_characteristics(df)
    
    # Check date range
    first_date = df['eom'].min()
    last_date = df['eom'].max()
    print(f"\nData spans {first_date.date()} to {last_date.date()}")
    
    # Calculate split points based on available data
    total_years = (last_date - first_date).days / 365.25
    
    if total_years >= 18:
        # Use paper-like split if enough data
        train_end = first_date + pd.DateOffset(years=10)
        val_end = train_end + pd.DateOffset(years=5)
    else:
        # For smaller datasets, use proportional split
        train_end = first_date + pd.DateOffset(years=int(total_years * 0.5))
        val_end = train_end + pd.DateOffset(years=int(total_years * 0.25))
    
    train_mask = df['eom'] <= train_end
    val_mask = (df['eom'] > train_end) & (df['eom'] <= val_end)
    test_mask = df['eom'] > val_end
    
    print(f"\nTrain end: {train_end.date()}")
    print(f"Val end: {val_end.date()}")
    print(f"Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")
    
    # Train neural network
    df, nn_model = model_neuralnet(df, train_mask, val_mask, test_mask, features)
    
    # Evaluate
    print("\n" + "="*60)
    
    # Get test predictions (non-nan)
    test_pred_mask = df['pred_nn'].notna() & (df['eom'] > val_end)
    if test_pred_mask.sum() > 0:
        y_true = df.loc[test_pred_mask, 'ret_exc_lead1m'].values
        y_pred = df.loc[test_pred_mask, 'pred_nn'].values
        
        # Paper's benchmark (zero)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum(y_true ** 2)
        r2_paper = 1 - ss_res / ss_tot
        
        # Historical mean benchmark (for comparison)
        ss_tot_mean = np.sum((y_true - y_true.mean()) ** 2)
        r2_mean = 1 - ss_res / ss_tot_mean
        
        print(f"\n{'Model':<20} {'R² (zero bench)':<18} {'R² (mean bench)':<18} {'MSE':<12}")
        print("-" * 68)
        print(f"{'Neural Net (NN3)':<20} {r2_paper:16.4%} {r2_mean:16.4%} {mean_squared_error(y_true, y_pred):12.8f}")
    else:
        print("WARNING: No valid test predictions available")
    
    print("="*60)
    
    return df, {'r2_paper': r2_paper if 'r2_paper' in locals() else None}


if __name__ == "__main__":
    df, results = main_with_nn()
