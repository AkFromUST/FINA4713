"""
models.py
FINA 4713 – Final Group Project
Section 3.3 – Model Estimation

Three models are implemented:
  (i)  HistoricalAverageBenchmark  — per-stock mean return from training data
  (ii) OLS                         — linear regression, no regularisation
  (iii) Ridge                      — L2-penalised OLS, α tuned on validation set

PUBLIC API
----------
Each model can be used independently:

    from models import HistoricalAverageBenchmark, fit_ols, tune_ridge, fit_ridge, oos_r2

Or run the full pipeline through the orchestrator:

    results = run_models()                          # loads defaults from files/data/
    results = run_models(train_df, val_df, test_df) # pass DataFrames directly
    results = run_models(feature_cols=[...])        # custom predictor set

run_models() returns a dict:
    {
        "models":      {"hist_avg": ..., "ols": ..., "ridge": ...},
        "results":     DataFrame  — OOS-R² for val + test per model,
        "predictions": DataFrame  — id, eom, y_true, pred_* for test set,
        "alpha_search": DataFrame — val OOS-R² for every alpha tried,
        "best_alpha":   float,
    }
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
_DATA_DIR    = "files/data"
_RESULTS_DIR = "files/results"
_MODELS_DIR  = "files/models"

META_COLS = ["id", "eom", "excntry"]
TARGET    = "ret_exc_lead1m"
_NON_FEAT = set(META_COLS) | {TARGET, "me_raw"}

# Alpha search grid for Ridge.
# With N ≈ 1.6 M training obs, the OLS estimator is already low-variance,
# so meaningful L2 regularisation only activates when α >> N.
# We sweep 15 log-spaced values from 1e2 to 1e9.
DEFAULT_ALPHAS = np.logspace(2, 9, 15)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    OOS R² = 1 − SS_res / SS_tot  (SS_tot uses evaluation-set mean).
    Negative means the model is beaten by predicting the sample mean.
    Small or negative values are normal for monthly stock returns.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def infer_feature_cols(df: pd.DataFrame) -> list:
    """Return all columns that are not metadata, target, or me_raw."""
    return [c for c in df.columns if c not in _NON_FEAT]


def clean_xy(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Drop rows where the target is NaN (stocks at end-of-sample with no
    forward return), then return (X array, y array, cleaned DataFrame).
    """
    df_c = df.dropna(subset=[TARGET]).reset_index(drop=True)
    X = df_c[feature_cols].values.astype(np.float64)
    y = df_c[TARGET].values.astype(np.float64)
    return X, y, df_c


# ─────────────────────────────────────────────────────────────────────────────
# Model (i): Historical-average benchmark
# ─────────────────────────────────────────────────────────────────────────────
class HistoricalAverageBenchmark:
    """
    Predict each stock's future return as its mean excess return in the
    training period.

    Why this benchmark?
    Every ML model is evaluated relative to this baseline.  It captures
    persistent cross-sectional differences in expected returns (e.g., value
    stocks that historically earned high returns are predicted high) but
    ignores all time-varying signals.  Beating it is a minimum bar for any
    feature-based model.

    Fallback: stocks absent from training (new listings) receive the global
    training mean, which is ≈ 0 for excess returns — a neutral prior.
    """

    def fit(self, df_train: pd.DataFrame) -> "HistoricalAverageBenchmark":
        valid = df_train.dropna(subset=[TARGET])
        self.stock_means_  = valid.groupby("id")[TARGET].mean().to_dict()
        self.global_mean_  = float(valid[TARGET].mean())
        self.n_stocks_fit_ = len(self.stock_means_)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["id"].map(self.stock_means_).fillna(self.global_mean_).values


# ─────────────────────────────────────────────────────────────────────────────
# Model (ii): OLS baseline
# ─────────────────────────────────────────────────────────────────────────────
def fit_ols(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Ordinary least squares on all features (no regularisation).

    Why OLS?
    Transparent baseline: coefficients are directly interpretable as marginal
    return predictions per unit of each standardised characteristic.
    With 185 correlated predictors OLS tends to overfit, which motivates
    Ridge as the next step.
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model (iii): Ridge regression
# ─────────────────────────────────────────────────────────────────────────────
def tune_ridge(
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    X_val:    np.ndarray,
    y_val:    np.ndarray,
    alphas:   np.ndarray = DEFAULT_ALPHAS,
    verbose:  bool = True,
) -> pd.DataFrame:
    """
    Grid-search Ridge alpha on the VALIDATION set; return results DataFrame.

    Why Ridge?
    - L2 shrinkage distributes weight across correlated factors rather than
      picking one arbitrarily (unlike Lasso), which is appropriate when the
      signal is diffuse across many characteristics (GKX 2020).
    - Closed-form solution — no convergence issues, scales to N = 1.6 M.
    - Unlike OLS, Ridge remains numerically stable under multicollinearity.

    Why this alpha range (1e2 – 1e9)?
    The Ridge objective is ||Xw − y||² + α||w||².  With N ≈ 1.6 M the data
    gradient is O(N), so regularisation only materially shrinks coefficients
    when α is O(N) or larger.  The upper end lets us observe when
    regularisation becomes too aggressive (predictions collapse toward zero).

    Returns DataFrame with columns [alpha, val_r2] for every alpha tried.
    """
    rows = []
    if verbose:
        print(f"  Searching {len(alphas)} alphas: "
              f"{alphas[0]:.1e} → {alphas[-1]:.1e}")
    for i, alpha in enumerate(alphas):
        m   = Ridge(alpha=alpha, fit_intercept=True, solver="auto")
        m.fit(X_train, y_train)
        r2  = oos_r2(y_val, m.predict(X_val))
        rows.append({"alpha": float(alpha), "val_r2": r2})
        if verbose:
            print(f"    [{i+1:2d}/{len(alphas)}]  α={alpha:.2e}  val OOS-R²={r2:+.6f}")
    return pd.DataFrame(rows)


def fit_ridge(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    best_alpha: float,
) -> Ridge:
    """Fit Ridge with a fixed alpha (call after tune_ridge selects best_alpha)."""
    model = Ridge(alpha=best_alpha, fit_intercept=True, solver="auto")
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_models(
    train_df:     pd.DataFrame | None = None,
    val_df:       pd.DataFrame | None = None,
    test_df:      pd.DataFrame | None = None,
    feature_cols: list | None         = None,
    alphas:       np.ndarray          = DEFAULT_ALPHAS,
    save:         bool                = True,
    data_dir:     str                 = _DATA_DIR,
    results_dir:  str                 = _RESULTS_DIR,
    models_dir:   str                 = _MODELS_DIR,
    verbose:      bool                = True,
) -> dict:
    """
    Full model-estimation pipeline.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame, optional
        Pre-processed splits.  If None, loaded from data_dir/*.parquet.
    feature_cols : list, optional
        Predictor column names.  If None, inferred from train_df
        (all columns except id, eom, excntry, ret_exc_lead1m, me_raw).
    alphas : array-like
        Ridge alpha search grid.
    save : bool
        Whether to write artefacts to disk.
    data_dir, results_dir, models_dir : str
        Override default output paths.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        models      — {"hist_avg", "ols", "ridge"}
        results     — DataFrame: model × {val_oos_r2, test_oos_r2}
        predictions — DataFrame: test-set predictions for all models
        alpha_search — DataFrame: alpha → val_r2 from Ridge search
        best_alpha  — float
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    def _load(df, name):
        if df is None:
            path = f"{data_dir}/{name}_processed.parquet"
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

    # ── Clean arrays (drop NaN-target rows) ───────────────────────────────────
    X_train, y_train, train_clean = clean_xy(train_df, feature_cols)
    X_val,   y_val,   val_clean   = clean_xy(val_df,   feature_cols)
    X_test,  y_test,  test_clean  = clean_xy(test_df,  feature_cols)
    if verbose:
        print(f"  After dropping NaN-target rows — "
              f"train {len(y_train):,} | val {len(y_val):,} | test {len(y_test):,}")

    # ── (i) Historical-average benchmark ──────────────────────────────────────
    if verbose:
        print("\n[2] Historical-average benchmark")
    hist = HistoricalAverageBenchmark().fit(train_df)
    if verbose:
        print(f"  Stocks fitted : {hist.n_stocks_fit_:,}  |  "
              f"global fallback : {hist.global_mean_:+.6f}")
    hist_val_pred  = hist.predict(val_clean)
    hist_test_pred = hist.predict(test_clean)
    hist_val_r2    = oos_r2(y_val,  hist_val_pred)
    hist_test_r2   = oos_r2(y_test, hist_test_pred)
    if verbose:
        print(f"  Val OOS-R²={hist_val_r2:+.6f}  |  Test OOS-R²={hist_test_r2:+.6f}")

    # ── (ii) OLS ──────────────────────────────────────────────────────────────
    if verbose:
        print("\n[3] OLS baseline")
    ols = fit_ols(X_train, y_train)
    ols_val_pred  = ols.predict(X_val)
    ols_test_pred = ols.predict(X_test)
    ols_val_r2    = oos_r2(y_val,  ols_val_pred)
    ols_test_r2   = oos_r2(y_test, ols_test_pred)
    if verbose:
        print(f"  |coef| max={np.abs(ols.coef_).max():.4f}  "
              f"mean={np.abs(ols.coef_).mean():.6f}")
        print(f"  Val OOS-R²={ols_val_r2:+.6f}  |  Test OOS-R²={ols_test_r2:+.6f}")

    # ── (iii) Ridge — alpha search ────────────────────────────────────────────
    if verbose:
        print("\n[4] Ridge — alpha search on validation set")
    alpha_df   = tune_ridge(X_train, y_train, X_val, y_val, alphas, verbose)
    best_row   = alpha_df.loc[alpha_df["val_r2"].idxmax()]
    best_alpha = float(best_row["alpha"])
    if verbose:
        print(f"\n  Best α={best_alpha:.4e}  "
              f"(val OOS-R²={best_row['val_r2']:+.6f})")
        print(f"\n[5] Refit Ridge (α={best_alpha:.4e}) on training data")

    ridge = fit_ridge(X_train, y_train, best_alpha)
    ridge_val_pred  = ridge.predict(X_val)
    ridge_test_pred = ridge.predict(X_test)
    ridge_val_r2    = oos_r2(y_val,  ridge_val_pred)
    ridge_test_r2   = oos_r2(y_test, ridge_test_pred)
    if verbose:
        print(f"  |coef| max={np.abs(ridge.coef_).max():.6f}  "
              f"mean={np.abs(ridge.coef_).mean():.6f}")
        print(f"  Val OOS-R²={ridge_val_r2:+.6f}  |  Test OOS-R²={ridge_test_r2:+.6f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    model_rows = [
        ("Historical average",          hist_val_r2,  hist_test_r2,  None),
        ("OLS",                          ols_val_r2,   ols_test_r2,   None),
        (f"Ridge (α={best_alpha:.2e})",  ridge_val_r2, ridge_test_r2, best_alpha),
    ]
    results_df = pd.DataFrame([
        {"model": n, "val_oos_r2": v, "test_oos_r2": t, "best_alpha": a}
        for n, v, t, a in model_rows
    ])
    if verbose:
        print(f"\n  {'─'*52}")
        print(f"  {'Model':<30} {'Val OOS-R²':>10} {'Test OOS-R²':>11}")
        print(f"  {'─'*52}")
        for n, v, t, _ in model_rows:
            print(f"  {n:<30} {v:>+10.6f} {t:>+11.6f}")
        print(f"  {'─'*52}")

    # ── Test predictions DataFrame ────────────────────────────────────────────
    pred_df = test_clean[["id", "eom", "excntry", TARGET]].copy()
    pred_df = pred_df.rename(columns={TARGET: "y_true"})
    pred_df["pred_hist_avg"] = hist_test_pred
    pred_df["pred_ols"]      = ols_test_pred
    pred_df["pred_ridge"]    = ridge_test_pred
    if "me_raw" in test_clean.columns:
        pred_df["me_raw"]    = test_clean["me_raw"].values

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        os.makedirs(models_dir,  exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        joblib.dump(hist,  f"{models_dir}/hist_avg.pkl")
        joblib.dump(ols,   f"{models_dir}/ols.pkl")
        joblib.dump(ridge, f"{models_dir}/ridge.pkl")

        alpha_df.to_csv(  f"{results_dir}/ridge_alpha_search.csv",  index=False)
        pred_df.to_parquet(f"{results_dir}/test_predictions.parquet", index=False)
        results_df.to_csv( f"{results_dir}/model_results.csv",        index=False)

        if verbose:
            print(f"\n  Saved → {models_dir}/  {{hist_avg, ols, ridge}}.pkl")
            print(f"  Saved → {results_dir}/  "
                  "model_results.csv, ridge_alpha_search.csv, test_predictions.parquet")

    return {
        "models":       {"hist_avg": hist, "ols": ols, "ridge": ridge},
        "results":      results_df,
        "predictions":  pred_df,
        "alpha_search": alpha_df,
        "best_alpha":   best_alpha,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  FINA 4713 — Section 3.3 : Model Estimation")
    print("=" * 58)
    run_models()
    print("\n  Done.\n")
