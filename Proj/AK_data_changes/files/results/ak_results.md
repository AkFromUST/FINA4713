# AK Pipeline — GKX NN3 Results
*Run completed: 2026-05-04 21:58*

---

## Feature Engineering Summary

| Stage | Features |
|---|---|
| Raw columns | 202 |
| After <50% missing filter | 185 |
| After selective log-transform (\|skew\| > 1.0) | 169 log-transformed, 16 passed through |
| After IC screening (\|t\| ≥ 1.5) | 140 kept, 45 dropped |
| After Grouped PCA (90% var/group) | 82 components |

### Selective Log-Transform
- Method: `sign(x) × log1p(|x|)` — preserves sign, handles negatives, NaN-safe
- Threshold: |skewness| > 1.0 (computed on training data only)
- **169 / 185 features log-transformed**
- Note: Spearman IC is invariant to monotone transforms — IC screening results are unaffected; benefit is in preprocessing quality and NN gradient stability

### IC Screening
- Training period: 2005–2015 (132 months)
- Threshold: |IC t-stat| ≥ 1.5
- **Kept 140 / dropped 45** features
- Sign consistency train/val: **92.1%**
- Mean |val t-stat| (kept features): 2.46

### Grouped PCA (13 JKP Themes)

| Group | Raw | Surviving | Components | Var Expl |
|---|---|---|---|---|
| momentum | 22 | 21 | 7 | 90.7% |
| seasonality | 6 | 5 | 5 | 100.0% |
| value | 17 | 12 | 9 | 91.0% |
| profitability | 16 | 15 | 7 | 90.8% |
| profit_growth | 15 | 13 | 9 | 90.0% |
| investment | 18 | 4 | 4 | 100.0% |
| accruals | 14 | 7 | 6 | 97.6% |
| debt_issuance | 14 | 12 | 6 | 90.7% |
| leverage | 6 | 4 | 2 | 90.0% |
| low_risk | 23 | 20 | 8 | 91.5% |
| quality | 16 | 14 | 10 | 91.6% |
| size_liquidity | 13 | 10 | 6 | 94.3% |
| other | 4 | 3 | 3 | 100.0% |

| **TOTAL** | **184** | **140** | **82** | — |

---

## GKX NN3 Results (OOS: 2019–2024, 71 months)

| Config | Features | OOS R² | Sharpe | Ann. Return | Ann. Vol | IC Mean | IC t-stat |
|---|---|---|---|---|---|---|---|
| Baseline (no IC/PCA) | 185 | +0.0001 | **2.03** | 9.90% | 4.88% | +0.0764 | +6.38 |
| IC + Grouped PCA | 82 | +0.0002 | **2.15** | 10.31% | 4.80% | +0.0764 | +6.87 |

> **Skeleton GKX NN3 benchmark** (from jkp_project_skeleton_code.ipynb): Sharpe = 2.05, IC t-stat = 6.57

### Verdict
IC+PCA wins on Sharpe: **2.15** vs 2.03

---

## Output Files

| File | Description |
|---|---|
| `feature_skewness.csv` | Skewness of all 185 features (training data) |
| `log_features.json` | 169 features that were log-transformed |
| `skewness_distribution.png` | Histogram of feature skewness with threshold |
| `ic_stats.csv` | IC t-stats for all 185 features |
| `selected_features.json` | 140 IC-selected feature names |
| `ic_tstat_bar.png` | Feature ranking bar chart |
| `pca_group_summary.png` | PCA components per JKP theme |
| `nn3_cumulative.png` | Cumulative return chart (OOS) |
| `ak_results_nn3.csv` | Numeric results table |
| `ak_results.md` | This report |
