import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class SimpleStockNN(nn.Module):
    """
    3-layer feedforward neural network
    Input: stock characteristics -> Hidden layers -> Output: predicted return
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_nn(model, X_train, y_train, X_val, y_val, verbose=True):
    """
    Train neural network with early stopping
    """
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    # DataLoader for batching
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(50):
        model.train()
        train_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss/len(loader):.6f}, val_loss={val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
    
    model.load_state_dict(best_model_state)
    return model

# ============================================================================
# MODEL (iv): NEURAL NETWORK
# ============================================================================

def model_neuralnet(df, train_mask, val_mask, test_mask, features):
    """
    Train neural network on training data, tune on validation
    """
    print("\nTraining Neural Network...")
    
    # Get data
    X_train = df[train_mask][features].values
    y_train = df[train_mask]['target_std'].values
    X_val = df[val_mask][features].values
    y_val = df[val_mask]['target_std'].values
    X_test = df[test_mask][features].values
    
    # Hyperparameter tuning on validation set
    hidden_dims = [32, 64, 128]
    dropouts = [0.2, 0.3, 0.5]
    
    best_val_loss = float('inf')
    best_model = None
    best_params = None
    
    for hidden_dim in hidden_dims:
        for dropout in dropouts:
            model = SimpleStockNN(input_dim=len(features), hidden_dim=hidden_dim, dropout=dropout)
            model = train_nn(model, X_train, y_train, X_val, y_val, verbose=False)
            
            # Evaluate on validation
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32)
                val_pred = model(X_val_t).numpy()
                val_loss = np.mean((val_pred - y_val) ** 2)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_params = (hidden_dim, dropout)
    
    print(f"Best: hidden_dim={best_params[0]}, dropout={best_params[1]}, val_loss={best_val_loss:.6f}")
    
    # Predict on all splits
    model.eval()
    with torch.no_grad():
        X_all = df[features].values
        X_all_t = torch.tensor(X_all, dtype=torch.float32)
        df['pred_nn'] = best_model(X_all_t).numpy()
    
    return df, best_model

# ============================================================================
# ADD TO MAIN FUNCTION
# ============================================================================

def main_with_nn():
    # Load data
    df = load_data('jkp_data.parquet')
    features = get_characteristics(df)
    
    # Split dates
    train_mask = df['eom'] <= '2015-12-31'
    val_mask = (df['eom'] > '2015-12-31') & (df['eom'] <= '2018-12-31')
    test_mask = df['eom'] > '2018-12-31'
    
    print(f"\nTrain: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")
    
    # Preprocess
    df, _ = preprocess(df, features, train_mask)
    
    # All models
    df = model_historical_average(df, train_mask)
    df = model_ols(df, train_mask, features)
    df = model_elasticnet(df, train_mask, val_mask, features)
    df, nn_model = model_neuralnet(df, train_mask, val_mask, test_mask, features)
    
    # Evaluate all four models
    print("\n" + "="*60)
    y_true = df[test_mask]['ret_exc_lead1m']
    y_baseline = y_true.mean()
    
    results = []
    for name, pred_col in [('Historical Avg', 'pred_hist'),
                            ('OLS', 'pred_ols'),
                            ('Elastic Net', 'pred_enet'),
                            ('Neural Net', 'pred_nn')]:
        y_pred = df[test_mask][pred_col].values
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_baseline) ** 2)
        r2 = 1 - ss_res / ss_tot
        mse = mean_squared_error(y_true, y_pred)
        results.append({'Model': name, 'Test R²': r2, 'Test MSE': mse})
    
    print(pd.DataFrame(results).to_string(index=False))
    print("="*60)
    
    # Portfolio from best model
    best_model_name = results[np.argmax([r['Test R²'] for r in results])]['Model']
    pred_col = {'Historical Avg': 'pred_hist', 'OLS': 'pred_ols', 
                'Elastic Net': 'pred_enet', 'Neural Net': 'pred_nn'}[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    build_portfolio(df, test_mask, pred_col)
    
    return df, results

if __name__ == "__main__":
    df, results = main_with_nn()