"""
baseline_claude.py
FINA 4713 – Final Group Project
Sections 3.1 (Data Preparation) + 3.2 (Train / Val / Test Split)

PREDICTOR SET
-------------
All JKP characteristics are retained after one data-quality filter:
drop any feature with > 50 % missing values in the TRAINING window.
This keeps broad coverage across value, momentum, quality, profitability,
investment, accruals, risk, and liquidity factors, while avoiding columns
so sparse that imputed values would dominate the signal.
The 50 % threshold is computed on training data only — no look-ahead.

PREPROCESSING PIPELINE  (all parameters fit on TRAIN only)
----------------------------------------------------------
  a) Signed-log transform  — features with |skewness| > 1.0.
       Strictly-positive: log1p(x)
       Mixed-sign:        sign(x) * log1p(|x|)
     Applied first so winsorisation operates on the compressed distribution.
  b) Winsorisation  — clip at training [1st, 99th] percentiles.
  c) Median imputation — fill NaN with training median.
  d) Z-score standardise — training mean / std → features ≈ N(0, 1).

OUTPUTS  (→ files/)
-------------------
  data/train_processed.parquet
  data/val_processed.parquet
  data/test_processed.parquet
  preprocessing/params.pkl
  preprocessing/feature_summary.csv
  preprocessing/summary.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH       = "jkp_data.parquet"
DATA_OUT_DIR    = "files/data"
PREPROC_OUT_DIR = "files/preprocessing"

SKEW_THRESH = 1.0
WINSOR_LO   = 0.01
WINSOR_HI   = 0.99
MISS_THRESH = 0.50
MIN_OBS     = 50

TRAIN_START = "2005-01-01";  TRAIN_END = "2015-12-31"
VAL_START   = "2016-01-01";  VAL_END   = "2018-12-31"
TEST_START  = "2019-01-01";  TEST_END  = "2024-12-31"

META_COLS = ["id", "eom", "excntry"]
TARGET    = "ret_exc_lead1m"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load JKP parquet; cast Arrow-string numeric columns to float64."""
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    obj_cols = [c for c in df.columns if df[c].dtype == "object" and c not in META_COLS]
    for c in obj_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} cols  |  "
          f"{df['eom'].min().date()} → {df['eom'].max().date()}")
    print(f"  Countries : {sorted(df['excntry'].unique())}")
    print(f"  Object cols cast to float : {len(obj_cols)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Temporal split
# ─────────────────────────────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame) -> tuple:
    """Strict date-based split — no shuffling of any kind."""
    train = df[(df["eom"] >= TRAIN_START) & (df["eom"] <= TRAIN_END)].copy()
    val   = df[(df["eom"] >= VAL_START)   & (df["eom"] <= VAL_END)].copy()
    test  = df[(df["eom"] >= TEST_START)  & (df["eom"] <= TEST_END)].copy()
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"  {name:5s}: {split['eom'].min().date()} → {split['eom'].max().date()}"
              f"  |  {len(split):>9,} obs  |  {split['eom'].nunique()} months")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature curation
# ─────────────────────────────────────────────────────────────────────────────
def curate_features(df_train: pd.DataFrame, all_feature_cols: list) -> tuple:
    """Drop features with > MISS_THRESH missing in training data only."""
    miss    = df_train[all_feature_cols].isna().mean().sort_values(ascending=False)
    kept    = miss[miss <= MISS_THRESH].index.tolist()
    dropped = miss[miss >  MISS_THRESH].index.tolist()
    print(f"  {len(all_feature_cols)} total  →  {len(dropped)} dropped "
          f"(>{MISS_THRESH:.0%} missing)  →  {len(kept)} retained")
    return kept, {"total": len(all_feature_cols), "dropped": dropped,
                  "kept": kept, "miss_rate_train": miss.to_dict()}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fit preprocessing  (TRAINING DATA ONLY)
# ─────────────────────────────────────────────────────────────────────────────
def fit_preprocessing(df_train: pd.DataFrame, feature_cols: list) -> dict:
    """Fit and return all preprocessing parameters from training data only."""
    params: dict = {}
    X = df_train[feature_cols].copy()

    # a) Skewness detection
    skewed, positive_only = [], []
    for col in feature_cols:
        vals = X[col].dropna()
        if len(vals) < MIN_OBS:
            continue
        if abs(float(stats.skew(vals))) > SKEW_THRESH:
            skewed.append(col)
            if (vals > 0).all():
                positive_only.append(col)
    params["skewed"]        = skewed
    params["positive_only"] = positive_only
    X = _apply_log(X, skewed, positive_only)

    # b) Winsorisation bounds
    params["winsor"] = {col: (float(X[col].quantile(WINSOR_LO)),
                              float(X[col].quantile(WINSOR_HI)))
                        for col in feature_cols}
    X = _apply_winsor(X, params["winsor"], feature_cols)

    # c) Imputation medians
    params["medians"] = {col: float(X[col].median()) for col in feature_cols}
    X = X.fillna(params["medians"])

    # d) Standardisation
    params["means"] = {col: float(X[col].mean()) for col in feature_cols}
    params["stds"]  = {col: max(float(X[col].std()), 1e-8) for col in feature_cols}

    return params


# ─────────────────────────────────────────────────────────────────────────────
# 5. Apply preprocessing  (transform only — no re-fitting)
# ─────────────────────────────────────────────────────────────────────────────
def apply_preprocessing(df: pd.DataFrame, feature_cols: list, params: dict) -> pd.DataFrame:
    """Apply the fitted pipeline to any split."""
    X = df[feature_cols].copy()
    X = _apply_log(X, params["skewed"], params["positive_only"])
    X = _apply_winsor(X, params["winsor"], feature_cols)
    X = X.fillna(params["medians"])
    for col in feature_cols:
        X[col] = (X[col] - params["means"][col]) / params["stds"][col]

    meta = df[META_COLS + [TARGET]].copy().reset_index(drop=True)
    if "me" in df.columns:
        meta["me_raw"] = df["me"].values   # raw market cap for portfolio weighting
    return pd.concat([meta, X.reset_index(drop=True)], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _apply_log(X, skewed, positive_only):
    for col in skewed:
        if col not in X.columns:
            continue
        if col in positive_only:
            X[col] = np.log1p(X[col])
        else:
            X[col] = np.sign(X[col]) * np.log1p(np.abs(X[col]))
    return X


def _apply_winsor(X, winsor, feature_cols):
    for col in feature_cols:
        lo, hi = winsor[col]
        X[col] = X[col].clip(lo, hi)
    return X


def sanity_check(train_proc: pd.DataFrame, feature_cols: list) -> None:
    means = train_proc[feature_cols].mean()
    stds  = train_proc[feature_cols].std()
    print(f"  |mean| > 0.05          : {(means.abs() > 0.05).sum()}  (expected ~0)")
    print(f"  std outside [0.8, 1.2] : {((stds < 0.8) | (stds > 1.2)).sum()}  (expected ~0)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────────────────────
def save_artefacts(train_proc, val_proc, test_proc,
                   params, feature_cols, feature_summary) -> None:
    os.makedirs(DATA_OUT_DIR,    exist_ok=True)
    os.makedirs(PREPROC_OUT_DIR, exist_ok=True)

    # Processed splits
    train_proc.to_parquet(f"{DATA_OUT_DIR}/train_processed.parquet", index=False)
    val_proc.to_parquet(  f"{DATA_OUT_DIR}/val_processed.parquet",   index=False)
    test_proc.to_parquet( f"{DATA_OUT_DIR}/test_processed.parquet",  index=False)
    print(f"  data/  ← train/val/test_processed.parquet")

    # Preprocessing params
    joblib.dump(params, f"{PREPROC_OUT_DIR}/params.pkl")

    # Feature summary CSV
    miss_s  = pd.Series(feature_summary["miss_rate_train"], name="miss_rate_train")
    feat_df = miss_s.to_frame()
    feat_df["kept"]     = feat_df.index.isin(feature_cols)
    feat_df["log_type"] = feat_df.index.map(
        lambda c: ("log1p" if c in params["positive_only"] else "signed_log")
        if c in params["skewed"] else "none")
    feat_df.sort_values("miss_rate_train", ascending=False).to_csv(
        f"{PREPROC_OUT_DIR}/feature_summary.csv")

    # Text summary
    lines = [
        "=" * 58, "FINA 4713 — Preprocessing Summary", "=" * 58, "",
        f"  Train : 2005-01 – 2015-12  |  {len(train_proc):,} obs",
        f"  Val   : 2016-01 – 2018-12  |  {len(val_proc):,} obs",
        f"  Test  : 2019-01 – 2024-12  |  {len(test_proc):,} obs", "",
        f"  Characteristics considered : {feature_summary['total']}",
        f"  Dropped (>50% missing)     : {len(feature_summary['dropped'])}",
        f"  Retained                   : {len(feature_cols)}", "",
        "  Pipeline (fit on TRAIN only):",
        f"    a) Signed-log  : {len(params['skewed'])} features  "
        f"({len(params['positive_only'])} log1p, "
        f"{len(params['skewed']) - len(params['positive_only'])} signed-log)",
        f"    b) Winsorise   : [{WINSOR_LO:.0%}, {WINSOR_HI:.0%}]",
        f"    c) Impute      : median",
        f"    d) Standardise : z-score",
        "", "  Dropped features:",
    ] + [f"    {c}" for c in feature_summary["dropped"]]
    with open(f"{PREPROC_OUT_DIR}/summary.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  preprocessing/  ← params.pkl, feature_summary.csv, summary.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    SEP = "=" * 58
    print(f"\n{SEP}")
    print("  FINA 4713 — Sections 3.1 + 3.2")
    print(SEP)

    print("\n[1] Load")
    df = load_data()

    print("\n[2] Temporal split")
    train_raw, val_raw, test_raw = temporal_split(df)

    all_feature_cols = [c for c in df.columns if c not in META_COLS + [TARGET]]
    print("\n[3] Feature curation")
    feature_cols, feature_summary = curate_features(train_raw, all_feature_cols)

    print("\n[4] Fit preprocessing on training data")
    params = fit_preprocessing(train_raw, feature_cols)
    print(f"  Log-transformed : {len(params['skewed'])} "
          f"({len(params['positive_only'])} log1p + "
          f"{len(params['skewed']) - len(params['positive_only'])} signed-log)")

    print("\n[5] Apply to all splits")
    train_proc = apply_preprocessing(train_raw, feature_cols, params)
    val_proc   = apply_preprocessing(val_raw,   feature_cols, params)
    test_proc  = apply_preprocessing(test_raw,  feature_cols, params)
    print(f"  Train {train_proc.shape} | Val {val_proc.shape} | Test {test_proc.shape}")

    print("\n[6] Sanity check")
    sanity_check(train_proc, feature_cols)

    print("\n[7] Save")
    save_artefacts(train_proc, val_proc, test_proc, params, feature_cols, feature_summary)

    print(f"\n  Done.  files/ → {os.path.abspath('files')}\n")
    return train_proc, val_proc, test_proc, feature_cols, params


if __name__ == "__main__":
    main()
