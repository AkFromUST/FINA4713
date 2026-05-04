# Vanilla Implementation — What's in `../Vanilla_Project/`

## What the directory is

A collection of ML code inspired by **Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine Learning"** (GKX). The paper is the landmark work applying a full ML zoo — OLS, Ridge, Lasso, Elastic Net, Random Forest, GBRT, and Neural Nets (NN1–NN5) — to predict monthly stock returns from firm characteristics. The punchline: regularised and nonlinear models outperform the historical average and OLS; NNs and tree-based models have the best test OOS-R² and generate economically large Sharpe ratios.

The directory contains three layers of code:

| File / dir | What it is |
|---|---|
| `baseline_claude.py` | Identical to `Proj/baseline_claude.py` — our JKP preprocessing pipeline |
| `models.py` | Identical to `Proj/models.py` — Hist avg, OLS, Ridge |
| `elastic_net.py` | Extended ElasticNet + portfolio construction (described below) |
| `nn.py` | PyTorch 3-layer feedforward NN (described below) |
| `pricing_data.py` | KRX (Korean) data loader — **NOT our data**, ignore |
| `NN_pricing_OLD_torch.ipynb` | Old notebook on KRX data — 5 NN architectures, rolling training, CUDA; **not compatible with our data** |
| `NN_pricing_OLD_torch_changed_setting.ipynb` | Variant of above — same story |
| `Asset-Pricing-via-ML/` | Empty |
| `Empirical-Asset-Pricing-via-Machine-Learning-Evidence-from-the-German-Stock-Market/` | Empty |
| `gu_kelly_xiu_2020_...pdf` | The GKX 2020 paper |

---

## `elastic_net.py` — what it does

Four models + evaluation + portfolio, all operating on our same date splits (train ≤ 2015-12, val 2016–2018, test 2019–2024).

### Models

**1. Generic ElasticNet (global fit)**
- Grid-searches 5 alphas × 4 l1_ratios on the validation set.
- Picks the best combo, refits on train, predicts everywhere.
- Column produced: `pred_enet`

**2. Rolling-window ElasticNet**
- For each test month, trains on the preceding 60 months.
- Re-standardises features within each rolling window.
- Fixed hyperparams: alpha=0.001, l1_ratio=0.5.
- Column produced: `pred_enet_rolling`

**3. Seed-averaged ElasticNet**
- Trains 5 identical models (different random seeds) on the full train set.
- Averages raw predictions (`pred_enet_seedavg`) and L1-normalised predictions (`pred_enet_seedavg_norm`).
- Conceptually analogous to "ensemble" predictions.

**4. Historical average + OLS** — same benchmarks as in `Proj/models.py`.

### Evaluation

`evaluate_models()` computes OOS-R² and MSE for all prediction columns on the test set.

### Portfolio construction (already coded)

`build_portfolio(df, test_mask, pred_col)`:
- Each month: sort stocks by predicted return, long top decile, short bottom decile.
- Equal-weighted within each leg; L1-normalised so gross exposure = 1.
- Computes and prints:
  - Annualised return = `mean(monthly_ret) × 12`
  - Annualised vol = `std(monthly_ret) × √12`
  - Sharpe ratio = ann_return / ann_vol

**This is the portfolio code we can directly reuse/extend for CAGR and drawdown.**

---

## `nn.py` — what it does

A self-contained PyTorch module.

**Architecture: `SimpleStockNN`**
- Input → Linear(hidden) → ReLU → Dropout → Linear(hidden/2) → ReLU → Dropout → Linear(1)
- Default: hidden=64, dropout=0.3

**Training: `train_nn()`**
- Adam optimiser, lr=0.001, weight_decay=1e-5
- Batch size 1024, max 50 epochs
- Early stopping: patience=5 on validation MSE loss

**Orchestrator: `model_neuralnet()`**
- Grid-searches 3 hidden dims × 3 dropout values (9 combos) on validation loss
- Keeps best model, predicts on all splits
- Column produced: `pred_nn`

**Important note**: `nn.py` references functions (`load_data`, `preprocess`, `build_portfolio`) from `elastic_net.py` — it can't run standalone without that context. We will need to wire it into our own pipeline.

---

## What's useful vs what to ignore

| Keep | Ignore |
|---|---|
| `elastic_net.py` — ElasticNet models and `build_portfolio()` | `pricing_data.py` — KRX-specific data loader |
| `nn.py` — NN architecture and training loop | Both notebooks — KRX data, incompatible |
| Portfolio/Sharpe logic | The two empty subdirs |

---

## What we need to do to plug this into our pipeline

The goal: run GKX-style models on **our** JKP preprocessed data (`files/data/*.parquet`) and compute financial KPIs.

### Step 1 — Wire ElasticNet to our data
`elastic_net.py` reads raw `jkp_data.parquet` and preprocesses inline. We already have clean preprocessed splits. We need to either:
- (a) Pass our processed parquets directly and skip the preprocessing inside `elastic_net.py`, or
- (b) Refactor `elastic_net.py` to accept pre-processed DataFrames (cleaner).

### Step 2 — Wire the NN to our pipeline
`nn.py`'s `model_neuralnet()` expects DataFrames with boolean masks (the old style). We'll refactor it to accept our `(X_train, y_train, X_val, y_val, X_test)` numpy arrays directly — simpler and consistent with `models.py`.

### Step 3 — Financial KPIs
Extend the `build_portfolio()` logic to compute:
- **Sharpe ratio** (already there)
- **CAGR** = `(1 + total_return)^(12/n_months) - 1`
- **Max drawdown**
- **Calmar ratio** = CAGR / |max_drawdown|
- **Monthly hit rate** (% months positive)
- Cumulative return chart (optional)

### Step 4 — Run and compare
Produce a results table comparing all models on:
- OOS-R² (val + test)
- Portfolio Sharpe, CAGR, max drawdown

---

## Summary

The Vanilla_Project gives us two models we don't yet have: **ElasticNet** (with rolling-window and seed-averaging variants) and a **feedforward NN**. Both are functional but built around the old raw-data pipeline — we need to rewire them to consume our preprocessed splits. The portfolio/KPI infrastructure is already sketched in `elastic_net.py`'s `build_portfolio()` and just needs extending.
