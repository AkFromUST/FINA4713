"""
choosing_predictors.py
FINA 4713 – Final Group Project
Feature Selection via Greedy Backward Elimination

Algorithm
---------
1. Start with all preprocessed features.
2. Tune Ridge alpha once on the full feature set (fixes alpha for the search).
3. Greedy loop:
     a. Try removing each feature in the current set one at a time.
     b. For each candidate set, fit OLS and Ridge (fixed alpha).
     c. Compute adj-OOS-R² on the validation set for Ridge (selection criterion).
     d. Accept the removal that yields the highest Ridge adj-OOS-R².
     e. Stop when no single removal improves the metric.
4. Refit the full pipeline (run_models) on the selected feature set — this call
   picks up every model in models.py automatically, so adding new models there
   requires no changes here.
5. Save processed data slices and all artefacts to files/choosing_predictors/.

Adjusted OOS-R²
---------------
adj-OOS-R² = 1 - (1 - OOS-R²) * (N_val - 1) / (N_val - k - 1)
where k = number of predictors, N_val = validation-set size after dropping
NaN-target rows.

Selection criterion: Ridge adj-OOS-R².
  Ridge was the only model with positive test OOS-R² in the baseline run,
  consistent with GKX (2020).  OLS and Ridge metrics are both logged at every
  step for inspection.

Scalability note
----------------
The greedy search uses only OLS + Ridge for speed (both have closed-form /
fast solvers).  The final evaluation calls run_models(), which automatically
includes every model present in models.py at runtime.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression

from models import (
    oos_r2,
    infer_feature_cols,
    clean_xy,
    tune_ridge,
    run_models,
    META_COLS,
    TARGET,
)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
_DATA_DIR = "files/data"
_OUT_DIR  = "files/choosing_predictors"

_ME_RAW = "me_raw"


# ─────────────────────────────────────────────────────────────────────────────
# Metric
# ─────────────────────────────────────────────────────────────────────────────
def adj_oos_r2(r2: float, n_val: int, k: int) -> float:
    """
    Adjusted OOS-R² penalises for predictor count.
    Returns -inf when the formula is undefined (n_val - k - 1 <= 0).
    """
    denom = n_val - k - 1
    if denom <= 0:
        return float("-inf")
    return 1.0 - (1.0 - r2) * (n_val - 1) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Single-trial fit: OLS + Ridge on a candidate feature subset
# ─────────────────────────────────────────────────────────────────────────────
def _trial(
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    ridge_alpha: float,
    n_val:       int,
    k:           int,
) -> dict:
    """
    Fit OLS and Ridge (fixed alpha) on X_train; evaluate on X_val.
    Returns a dict of raw and adjusted OOS-R² for both models.
    """
    # OLS
    ols = LinearRegression(fit_intercept=True)
    ols.fit(X_train, y_train)
    ols_r2_raw = oos_r2(y_val, ols.predict(X_val))

    # Ridge (fixed alpha — no grid search during greedy steps)
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True, solver="auto")
    ridge.fit(X_train, y_train)
    ridge_r2_raw = oos_r2(y_val, ridge.predict(X_val))

    return {
        "ols_oos_r2":     ols_r2_raw,
        "ols_adj_r2":     adj_oos_r2(ols_r2_raw,   n_val, k),
        "ridge_oos_r2":   ridge_r2_raw,
        "ridge_adj_r2":   adj_oos_r2(ridge_r2_raw, n_val, k),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Greedy backward elimination
# ─────────────────────────────────────────────────────────────────────────────
def greedy_backward_elimination(
    train_df:      pd.DataFrame,
    val_df:        pd.DataFrame,
    feature_cols:  list,
    initial_alpha: float,
    verbose:       bool = True,
) -> tuple[list, pd.DataFrame]:
    """
    Greedy backward elimination driven by Ridge adj-OOS-R² on the validation set.

    Parameters
    ----------
    train_df, val_df : pre-processed DataFrames (NaN-target rows are dropped internally)
    feature_cols     : starting feature list (ordered)
    initial_alpha    : Ridge alpha fixed for all search trials
    verbose          : print progress

    Returns
    -------
    best_features : list of selected feature column names
    log_df        : DataFrame with one row per greedy step
    """
    # Pre-extract arrays once (column slicing is cheap on numpy)
    X_train_all, y_train, _ = clean_xy(train_df, feature_cols)
    X_val_all,   y_val,   _ = clean_xy(val_df,   feature_cols)
    n_val = len(y_val)

    # Map feature name → column index in the pre-extracted arrays
    feat_to_idx = {c: i for i, c in enumerate(feature_cols)}

    def _arrays(feats: list):
        idx = [feat_to_idx[c] for c in feats]
        return X_train_all[:, idx], X_val_all[:, idx]

    current_features = list(feature_cols)
    log_rows = []

    # ── Step 0: baseline metrics on full feature set ──────────────────────────
    Xtr, Xvl = _arrays(current_features)
    current_metrics = _trial(
        Xtr, y_train, Xvl, y_val,
        initial_alpha, n_val, len(current_features),
    )
    log_rows.append({
        "step":           0,
        "n_features":     len(current_features),
        "removed":        None,
        "ridge_adj_r2":   current_metrics["ridge_adj_r2"],
        "ridge_oos_r2":   current_metrics["ridge_oos_r2"],
        "ols_adj_r2":     current_metrics["ols_adj_r2"],
        "ols_oos_r2":     current_metrics["ols_oos_r2"],
        "improved":       None,   # baseline — no comparison yet
        "elapsed_s":      0.0,
    })

    if verbose:
        print(f"\n  Step 0 (baseline): k={len(current_features)}  "
              f"Ridge adj-OOS-R²={current_metrics['ridge_adj_r2']:+.6f}")

    step = 0
    while len(current_features) > 1:
        step += 1
        n = len(current_features)
        t0 = time.time()

        if verbose:
            print(f"  Step {step:3d}: trying {n} removals (k={n}) ...", end="", flush=True)

        best_removal = None
        best_metrics = None
        best_ridge_adj = current_metrics["ridge_adj_r2"]

        for feat in current_features:
            candidate = [f for f in current_features if f != feat]
            Xtr, Xvl = _arrays(candidate)
            m = _trial(Xtr, y_train, Xvl, y_val, initial_alpha, n_val, len(candidate))
            if m["ridge_adj_r2"] > best_ridge_adj:
                best_ridge_adj = m["ridge_adj_r2"]
                best_removal   = feat
                best_metrics   = m

        elapsed = time.time() - t0

        if best_removal is None:
            # No removal improves the metric — stop
            log_rows.append({
                "step":         step,
                "n_features":   n,
                "removed":      None,
                "ridge_adj_r2": current_metrics["ridge_adj_r2"],
                "ridge_oos_r2": current_metrics["ridge_oos_r2"],
                "ols_adj_r2":   current_metrics["ols_adj_r2"],
                "ols_oos_r2":   current_metrics["ols_oos_r2"],
                "improved":     False,
                "elapsed_s":    elapsed,
            })
            if verbose:
                print(f" done ({elapsed:.1f}s)  no improvement — stopping at k={n}")
            break

        # Accept removal
        current_features.remove(best_removal)
        current_metrics = best_metrics
        log_rows.append({
            "step":         step,
            "n_features":   len(current_features),
            "removed":      best_removal,
            "ridge_adj_r2": best_metrics["ridge_adj_r2"],
            "ridge_oos_r2": best_metrics["ridge_oos_r2"],
            "ols_adj_r2":   best_metrics["ols_adj_r2"],
            "ols_oos_r2":   best_metrics["ols_oos_r2"],
            "improved":     True,
            "elapsed_s":    elapsed,
        })
        if verbose:
            print(f" done ({elapsed:.1f}s)  removed='{best_removal}'  "
                  f"Ridge adj-OOS-R²={best_metrics['ridge_adj_r2']:+.6f}  "
                  f"→ k={len(current_features)}")

    return current_features, pd.DataFrame(log_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_choosing_predictors(
    data_dir: str = _DATA_DIR,
    out_dir:  str = _OUT_DIR,
    verbose:  bool = True,
) -> dict:
    """
    Full feature-selection pipeline.

    Reads preprocessed data from data_dir, runs greedy backward elimination,
    writes all artefacts to out_dir, and returns a summary dict.

    Returns
    -------
    dict with keys:
        best_features     — list of selected feature names
        best_ridge_adj_r2 — Ridge adj-OOS-R² at the selected k
        selection_log     — DataFrame: one row per greedy step
        final_results     — return value of run_models() on selected features
    """
    for sub in ("", "/data", "/models", "/results"):
        os.makedirs(out_dir + sub, exist_ok=True)

    # ── [1] Load data ──────────────────────────────────────────────────────────
    if verbose:
        print("\n[1] Loading preprocessed data")
    train_df = pd.read_parquet(f"{data_dir}/train_processed.parquet")
    val_df   = pd.read_parquet(f"{data_dir}/val_processed.parquet")
    test_df  = pd.read_parquet(f"{data_dir}/test_processed.parquet")

    feature_cols = infer_feature_cols(train_df)
    if verbose:
        print(f"  Starting features : {len(feature_cols)}")
        print(f"  Train / Val / Test : "
              f"{len(train_df):,} / {len(val_df):,} / {len(test_df):,} rows")

    # ── [2] Tune Ridge alpha once on the full feature set ─────────────────────
    if verbose:
        print("\n[2] Tuning initial Ridge alpha (full feature set, fixed for search)")
    X_train_full, y_train_full, _ = clean_xy(train_df, feature_cols)
    X_val_full,   y_val_full,   _ = clean_xy(val_df,   feature_cols)

    alpha_df     = tune_ridge(X_train_full, y_train_full,
                               X_val_full,  y_val_full, verbose=verbose)
    best_row     = alpha_df.loc[alpha_df["val_r2"].idxmax()]
    initial_alpha = float(best_row["alpha"])
    if verbose:
        print(f"\n  Fixed search alpha : {initial_alpha:.4e}  "
              f"(val OOS-R²={best_row['val_r2']:+.6f})")

    # ── [3] Greedy backward elimination ───────────────────────────────────────
    if verbose:
        print("\n[3] Greedy backward elimination")
        print(f"  Selection criterion : Ridge adj-OOS-R² on validation set")
        print(f"  Stopping rule       : first step with no improvement\n")

    best_features, log_df = greedy_backward_elimination(
        train_df, val_df, feature_cols, initial_alpha, verbose
    )

    n_selected = len(best_features)
    # Best metrics = last row where improved=True (or step 0 if nothing improved)
    improved_rows = log_df[log_df["improved"] == True]
    if len(improved_rows):
        best_ridge_adj_r2 = float(improved_rows.iloc[-1]["ridge_adj_r2"])
    else:
        best_ridge_adj_r2 = float(log_df.iloc[0]["ridge_adj_r2"])

    if verbose:
        print(f"\n  Selected {n_selected} features  "
              f"(removed {len(feature_cols) - n_selected})")
        print(f"  Best Ridge adj-OOS-R² (val) : {best_ridge_adj_r2:+.6f}")

    # ── [4] Save selection log and feature lists ───────────────────────────────
    log_df.to_csv(f"{out_dir}/selection_log.csv", index=False)

    with open(f"{out_dir}/best_features.json", "w") as fh:
        json.dump(best_features, fh, indent=2)

    with open(f"{out_dir}/best_features.txt", "w") as fh:
        fh.write(f"# {n_selected} selected features\n")
        for feat in best_features:
            fh.write(feat + "\n")

    if verbose:
        print(f"\n  Saved → {out_dir}/selection_log.csv")
        print(f"  Saved → {out_dir}/best_features.json / .txt")

    # ── [5] Save processed data slices (selected features only) ───────────────
    meta_keep = [c for c in (META_COLS + [TARGET, _ME_RAW]) if c in train_df.columns]
    keep_cols = meta_keep + best_features

    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = f"{out_dir}/data/{split}_processed.parquet"
        df[keep_cols].to_parquet(path, index=False)

    if verbose:
        print(f"  Saved → {out_dir}/data/  (train / val / test with {n_selected} features)")

    # ── [6] Final model run on selected features ───────────────────────────────
    # run_models() is called directly — any new model added to models.py is
    # automatically included here without modifying this file.
    if verbose:
        print(f"\n[4] Final model evaluation on selected {n_selected} features")

    final_results = run_models(
        train_df     = train_df,
        val_df       = val_df,
        test_df      = test_df,
        feature_cols = best_features,
        save         = True,
        data_dir     = f"{out_dir}/data",
        results_dir  = f"{out_dir}/results",
        models_dir   = f"{out_dir}/models",
        verbose      = verbose,
    )

    # ── [7] Summary printout ──────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*58}")
        print(f"  FEATURE SELECTION COMPLETE")
        print(f"  Features selected : {n_selected} / {len(feature_cols)}")
        print(f"  Ridge adj-OOS-R²  : {best_ridge_adj_r2:+.6f}")
        print(f"  Artefacts         : {out_dir}/")
        print(f"    selection_log.csv   — step-by-step greedy log")
        print(f"    best_features.json  — selected feature list")
        print(f"    data/               — processed splits (selected features)")
        print(f"    models/             — fitted model objects (.pkl)")
        print(f"    results/            — OOS-R² table, predictions")
        print(f"{'='*58}")

    return {
        "best_features":      best_features,
        "best_ridge_adj_r2":  best_ridge_adj_r2,
        "selection_log":      log_df,
        "final_results":      final_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  FINA 4713 — Feature Selection")
    print("  Greedy Backward Elimination · Ridge adj-OOS-R²")
    print("=" * 58)
    run_choosing_predictors()
    print("\n  Done.\n")
