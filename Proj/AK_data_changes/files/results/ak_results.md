# AK Pipeline — GKX NN3 Results
*Run completed: 2026-05-04 21:03*

---

## Feature Engineering Summary

| Stage | Features |
|---|---|
| Raw columns | 202 |
| After <50% missing filter | 185 |
| After IC screening (\|t\| ≥ 1.5) | 42 |
| After Grouped PCA (90% var/group) | 33 components |

### IC Screening
- Training period: 2005–2015 (132 months)
- Threshold: |IC t-stat| ≥ 1.5
- **Kept 42 / dropped 9** features
- Sign consistency train/val: **100.0%**
- Mean |val t-stat| (kept features): 3.33

### Grouped PCA (13 JKP Themes)

| Group | Raw | Surviving | Components | Var Expl |
|---|---|---|---|---|
| momentum | 22 | 4 | 3 | 90.4% |
| seasonality | 6 | 1 | 1 | 100.0% |
| value | 17 | 7 | 6 | 94.8% |
| profitability | 16 | 6 | 4 | 92.2% |
| profit_growth | 15 | 1 | 1 | 100.0% |
| investment | 18 | 2 | 2 | 100.0% |
| debt_issuance | 14 | 3 | 2 | 98.4% |
| low_risk | 23 | 7 | 3 | 95.7% |
| quality | 16 | 5 | 5 | 100.0% |
| size_liquidity | 13 | 4 | 4 | 100.0% |
| other | 4 | 2 | 2 | 100.0% |

| **TOTAL** | **164** | **42** | **33** | — |

---

## GKX NN3 Results (OOS: 2019–2024, 71 months)

| Config | Features | OOS R² | Sharpe | Ann. Return | Ann. Vol | IC Mean | IC t-stat |
|---|---|---|---|---|---|---|---|
| Baseline (no IC/PCA) | 185 | -0.0000 | **1.79** | 8.36% | 4.68% | +0.0617 | +5.72 |
| IC + Grouped PCA | 33 | +0.0001 | **1.88** | 8.33% | 4.43% | +0.0694 | +6.99 |

> **Skeleton GKX NN3 benchmark** (from jkp_project_skeleton_code.ipynb): Sharpe = 2.05, IC t-stat = 6.57

### Verdict
IC+PCA wins on Sharpe: **1.88** vs 1.79

---

## Output Files

| File | Description |
|---|---|
| `ic_stats.csv` | IC t-stats for all 185 features |
| `selected_features.json` | 42 IC-selected feature names |
| `ic_tstat_bar.png` | Feature ranking bar chart |
| `pca_group_summary.png` | PCA components per JKP theme |
| `nn3_cumulative.png` | Cumulative return chart (OOS) |
| `ak_results_nn3.csv` | Numeric results table |
| `ak_results.md` | This report |
