import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(path='jkp.data.parquet'):
    df = pd.read_parquet(path)
    df['eom'] = pd.to_datetime(df['eom'])
    df = df.dropna(subset=['ret_exc_lead1m'])
    print(f"Loaded {len(df):,} observations")
    print(f"Date range: {df['eom'].min()} to {df['eom'].max()}")
    return df

def get_characteristics(df):
    exclude = ['id', 'eom', 'excntry', 'ret_exc_lead1m', 'me']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    characteristics = [c for c in numeric_cols if c not in exclude]
    print(f"Using {len(characteristics)} characteristics")
    return characteristics

# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess(df, features, train_mask):
    """Preprocess with proper index handling"""
    df = df.copy()
    df = df.reset_index(drop=True)
    
    train_indices = train_mask.values if hasattr(train_mask, 'values') else train_mask
    
    # 1. Winsorize
    for f in features:
        train_vals = df.loc[train_indices, f].dropna()
        if len(train_vals) > 0:
            lower, upper = train_vals.quantile([0.01, 0.99])
            df[f] = df[f].clip(lower, upper)
    
    # 2. Log transform skewed variables
    for f in features:
        train_vals = df.loc[train_indices, f].dropna()
        if len(train_vals) > 0 and abs(train_vals.skew()) > 1.0:
            min_val = df[f].min()
            if min_val <= 0:
                df[f] = df[f] - min_val + 0.001
            df[f] = np.log1p(df[f])
    
    # 3. Impute missing with monthly median
    for f in features:
        monthly_medians = df.loc[train_indices].groupby('eom')[f].median()
        overall_median = df.loc[train_indices, f].median()
        
        df[f] = df.groupby('eom')[f].transform(
            lambda x: x.fillna(monthly_medians.get(x.name, overall_median))
        )
        df[f] = df[f].fillna(overall_median)
    
    # 4. Standardize features
    scaler = StandardScaler()
    scaler.fit(df.loc[train_indices, features])
    df[features] = scaler.transform(df[features])
    
    # 5. Standardize target within each month
    df['target_std'] = df.groupby('eom')['ret_exc_lead1m'].transform(
        lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)
    )
    
    return df, scaler

# ============================================================================
# MODEL (i): HISTORICAL AVERAGE
# ============================================================================

def historical_average(df, train_mask):
    train_indices = train_mask.values if hasattr(train_mask, 'values') else train_mask
    stock_means = df.loc[train_indices].groupby('id')['ret_exc_lead1m'].mean()
    overall_mean = df.loc[train_indices, 'ret_exc_lead1m'].mean()
    df['pred_hist'] = df['id'].map(stock_means).fillna(overall_mean)
    return df

# ============================================================================
# MODEL (ii): OLS
# ============================================================================

def ols_model(df, train_mask, features):
    train_indices = train_mask.values if hasattr(train_mask, 'values') else train_mask
    ols = LinearRegression()
    ols.fit(df.loc[train_indices, features], df.loc[train_indices, 'target_std'])
    df['pred_ols'] = ols.predict(df[features].values)
    return df

# ============================================================================
# MODEL (iii): GENERIC ELASTIC NET
# ============================================================================

def generic_elasticnet(df, train_mask, val_mask, features):
    train_indices = train_mask.values if hasattr(train_mask, 'values') else train_mask
    val_indices = val_mask.values if hasattr(val_mask, 'values') else val_mask
    
    X_train = df.loc[train_indices, features].values
    y_train = df.loc[train_indices, 'target_std'].values
    X_val = df.loc[val_indices, features].values
    y_val = df.loc[val_indices, 'target_std'].values
    
    alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    l1_ratios = [0.3, 0.5, 0.7, 0.9]
    
    best_r2 = -np.inf
    best_model = None
    best_params = None
    
    for alpha in alphas:
        for l1 in l1_ratios:
            model = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10000, random_state=42)
            model.fit(X_train, y_train)
            r2 = r2_score(y_val, model.predict(X_val))
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_params = (alpha, l1)
    
    print(f"Best: alpha={best_params[0]}, l1_ratio={best_params[1]}, val R²={best_r2:.6f}")
    print(f"Selected {np.sum(np.abs(best_model.coef_) > 1e-6)} / {len(features)} features")
    
    df['pred_enet'] = best_model.predict(df[features].values)
    return df

# ============================================================================
# MODEL (iv): ROLLING WINDOW ELASTIC NET (PIE-LASSO style)
# ============================================================================

def rolling_window_elasticnet(df, test_mask, features):
    print("\n" + "="*60)
    print("ROLLING WINDOW ELASTIC NET (PIE-LASSO style)")
    print("="*60)
    
    test_indices = test_mask.values if hasattr(test_mask, 'values') else test_mask
    test_dates = sorted(df.loc[test_indices, 'eom'].unique())
    
    all_predictions = []
    selected_counts = []
    
    for i, month in enumerate(test_dates):
        # Training: last 60 months before this month
        train_cutoff = month - pd.DateOffset(months=1)
        train_start = train_cutoff - pd.DateOffset(months=60)
        
        train_data = df[(df['eom'] >= train_start) & (df['eom'] <= train_cutoff)]
        test_data = df[df['eom'] == month]
        
        if len(train_data) < 500 or len(test_data) < 50:
            preds = np.zeros(len(test_data))
        else:
            scaler_roll = StandardScaler()
            X_train_roll = scaler_roll.fit_transform(train_data[features])
            X_test_roll = scaler_roll.transform(test_data[features])
            y_train_roll = train_data['target_std'].values
            
            model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42)
            model.fit(X_train_roll, y_train_roll)
            preds = model.predict(X_test_roll)
            selected_counts.append(np.sum(np.abs(model.coef_) > 1e-6))
        
        all_predictions.append(pd.DataFrame({
            'id': test_data['id'].values,
            'eom': month,
            'pred_enet_rolling': preds
        }))
        
        if (i + 1) % 12 == 0:
            avg_selected = np.mean(selected_counts) if selected_counts else 0
            print(f"  Month {i+1}/{len(test_dates)}: avg selected {avg_selected:.1f} features")
    
    predictions = pd.concat(all_predictions, ignore_index=True)
    df = df.merge(predictions, on=['id', 'eom'], how='left')
    df['pred_enet_rolling'] = df['pred_enet_rolling'].fillna(0)
    
    return df

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_models(df, test_mask):
    test_indices = test_mask.values if hasattr(test_mask, 'values') else test_mask
    
    y_true = df.loc[test_indices, 'ret_exc_lead1m'].values
    y_baseline = y_true.mean()
    
    pred_cols = ['pred_hist', 'pred_ols', 'pred_enet', 'pred_enet_rolling']
    
    results = []
    for col in pred_cols:
        if col in df.columns:
            y_pred = df.loc[test_indices, col].values
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_baseline) ** 2)
            r2 = 1 - ss_res / ss_tot
            mse = mean_squared_error(y_true, y_pred)
            
            name = col.replace('pred_', '').replace('_', ' ').title()
            results.append({'Model': name, 'Test R²': r2, 'Test MSE': mse})
    
    return pd.DataFrame(results)

# ============================================================================
# PORTFOLIO CONSTRUCTION
# ============================================================================

def build_portfolio(df, test_mask, pred_col, name="Portfolio"):
    test_indices = test_mask.values if hasattr(test_mask, 'values') else test_mask
    test = df.loc[test_indices].copy()
    
    all_weights = []
    
    for date in test['eom'].unique():
        month = test[test['eom'] == date].copy()
        month = month.sort_values(pred_col, ascending=False)
        
        n = len(month)
        decile_size = n // 10
        
        if decile_size > 0:
            month['weight'] = 0.0
            month.iloc[:decile_size, month.columns.get_loc('weight')] = 1.0 / decile_size
            month.iloc[-decile_size:, month.columns.get_loc('weight')] = -1.0 / decile_size
            
            # L1 normalization (Transformer paper style)
            month['weight'] = month['weight'] / (month['weight'].abs().sum() + 1e-10)
        else:
            month['weight'] = 0.0
        
        all_weights.append(month[['id', 'eom', 'weight']])
    
    weights = pd.concat(all_weights, ignore_index=True)
    
    # Calculate returns
    portfolio = weights.merge(test[['id', 'eom', 'ret_exc_lead1m']], on=['id', 'eom'])
    monthly_returns = portfolio.groupby('eom').apply(
        lambda x: (x['weight'] * x['ret_exc_lead1m']).sum()
    )
    
    ann_return = monthly_returns.mean() * 12
    ann_vol = monthly_returns.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Turnover
    weights_wide = weights.pivot(index='eom', columns='id', values='weight').fillna(0)
    turnover = weights_wide.diff().abs().sum(axis=1).mean() / 2
    
    print(f"\n{name}:")
    print(f"  Annualized Return: {ann_return:.2%}")
    print(f"  Annualized Volatility: {ann_vol:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Monthly Turnover: {turnover:.2%}")
    
    return monthly_returns, sharpe

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("ENHANCED ELASTIC NET PIPELINE")
    print("Inspired by: PIE-LASSO and Kelly Transformer (2025)")
    print("="*70)
    
    # Load
    df = load_data('jkp_data.parquet')
    features = get_characteristics(df)
    
    # Split dates
    train_mask = df['eom'] <= '2015-12-31'
    val_mask = (df['eom'] > '2015-12-31') & (df['eom'] <= '2018-12-31')
    test_mask = df['eom'] > '2018-12-31'
    
    print(f"\nTrain: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")
    
    # Preprocess
    df, _ = preprocess(df, features, train_mask)
    
    # Models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    df = historical_average(df, train_mask)
    print("✓ Historical Average")
    
    df = ols_model(df, train_mask, features)
    print("✓ OLS")
    
    df = generic_elasticnet(df, train_mask, val_mask, features)
    print("✓ Generic Elastic Net")
    
    df = rolling_window_elasticnet(df, test_mask, features)
    print("✓ Rolling Window Elastic Net (PIE-LASSO)")
    
    # Evaluate
    print("\n" + "="*70)
    print("TEST SET PERFORMANCE (Jan 2019 - Dec 2024)")
    print("="*70)
    results = evaluate_models(df, test_mask)
    print(results.to_string(index=False))
    
    # Portfolio comparison
    print("\n" + "="*70)
    print("PORTFOLIO PERFORMANCE COMPARISON")
    print("="*70)
    
    build_portfolio(df, test_mask, 'pred_hist', 'Historical Average')
    build_portfolio(df, test_mask, 'pred_ols', 'OLS')
    build_portfolio(df, test_mask, 'pred_enet', 'Generic Elastic Net')
    build_portfolio(df, test_mask, 'pred_enet_rolling', 'Rolling Window (PIE-LASSO)')
    
    return df, results

if __name__ == "__main__":
    df, results = main()