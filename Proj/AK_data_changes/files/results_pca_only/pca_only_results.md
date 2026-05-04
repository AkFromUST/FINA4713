# AK Pipeline (PCA-only) — GKX NN3 Results
*Run completed: 2026-05-04 22:33*

---

## Setup
This run uses Grouped PCA **without IC pre-screening**. All 185 features (after
<50% missing filter) are fed directly into the PCA grouped by JKP theme. Compare against
`ak_pipeline.py` (IC+PCA) to isolate the contribution of the IC screening step.

---

## Feature Engineering Summary

| Stage | Features |
|---|---|
| Raw columns | 202 |
| After <50% missing filter | 185 |
| After selective log-transform (\|skew\| > 1.0) | 169 log-transformed, 16 passed through |
| After Grouped PCA, no IC filter (90% var/group) | 106 components |

### Selective Log-Transform
- Method: `sign(x) × log1p(|x|)` — preserves sign, handles negatives, NaN-safe
- Threshold: |skewness| > 1.0 (computed on training data only)
- **169 / 185 features log-transformed**

### Grouped PCA — No IC Pre-Filter

| Group | Raw feats | In data | Components | Var expl |
|---|---|---|---|---|
| momentum | 22 | 22 | 7 | 90.5% |
| short_term_reversal | 1 | 1 | 1 | 100.0% |
| seasonality | 6 | 6 | 6 | 100.0% |
| value | 17 | 17 | 12 | 91.7% |
| profitability | 16 | 16 | 8 | 91.5% |
| profit_growth | 15 | 15 | 11 | 91.4% |
| investment | 18 | 18 | 12 | 90.1% |
| accruals | 14 | 14 | 8 | 92.3% |
| debt_issuance | 14 | 14 | 7 | 93.8% |
| leverage | 6 | 6 | 3 | 92.8% |
| low_risk | 23 | 23 | 10 | 91.4% |
| quality | 16 | 16 | 11 | 90.0% |
| size_liquidity | 13 | 13 | 6 | 91.2% |
| other | 4 | 4 | 4 | 100.0% |

| **TOTAL** | **185** | **185** | **106** | — |

Ungrouped features (not in any JKP theme, excluded from PCA): 0
- none

---

## GKX NN3 Results (OOS: 2019–2024, 71 months)

| Config | Features | OOS R² | Sharpe | Ann. Return | Ann. Vol | IC Mean | IC t-stat |
|---|---|---|---|---|---|---|---|
| Baseline (no PCA) | 185 | +0.0001 | **2.03** | 9.90% | 4.88% | +0.0764 | +6.38 |
| PCA-only (no IC) | 106 | +0.0001 | **2.00** | 9.04% | 4.53% | +0.0635 | +6.21 |

> **Skeleton GKX NN3 benchmark**: Sharpe = 2.05, IC t-stat = 6.57
> **ak_pipeline.py IC+PCA benchmark**: Sharpe = (see ak_results.md)

### Verdict
Baseline wins on Sharpe: **2.03** vs 2.00

---

## Output Files

| File | Description |
|---|---|
| `feature_skewness.csv` | Skewness of all 185 features (training data) |
| `log_features.json` | 169 features that were log-transformed |
| `skewness_distribution.png` | Histogram of feature skewness with threshold |
| `pca_group_summary.png` | PCA components per JKP theme |
| `nn3_cumulative.png` | Cumulative return chart (OOS) |
| `pca_only_results_nn3.csv` | Numeric results table |
| `pca_only_results.md` | This report |
