# AK Pipeline — GKX NN3 Results
*Run completed: 2026-05-04 20:21*

---

## Feature Engineering Summary

| Stage | Features |
|---|---|
| Raw columns | 202 |
| After <50% missing filter | 185 |
| After IC screening (\|t\| ≥ 1.5) | 140 |
| After Grouped PCA (90% var/group) | 87 components |

### IC Screening
- Training period: 2005–2015 (132 months)
- Threshold: |IC t-stat| ≥ 1.5
- **Kept 140 / dropped 45** features
- Sign consistency train/val: **92.1%**
- Mean |val t-stat| (kept features): 2.46

### Grouped PCA (13 JKP Themes)

| Group | Raw | Surviving | Components | Var Expl |
|---|---|---|---|---|
| momentum | 22 | 21 | 8 | 91.1% |
| seasonality | 6 | 5 | 5 | 100.0% |
| value | 17 | 12 | 10 | 93.8% |
| profitability | 16 | 15 | 8 | 92.3% |
| profit_growth | 15 | 13 | 10 | 93.3% |
| investment | 18 | 4 | 4 | 100.0% |
| accruals | 14 | 7 | 6 | 97.5% |
| debt_issuance | 14 | 12 | 6 | 90.4% |
| leverage | 6 | 4 | 3 | 97.6% |
| low_risk | 23 | 20 | 8 | 91.4% |
| quality | 16 | 14 | 10 | 90.9% |
| size_liquidity | 13 | 10 | 6 | 93.1% |
| other | 4 | 3 | 3 | 100.0% |

| **TOTAL** | **184** | **140** | **87** | — |

---

## GKX NN3 Results (OOS: 2019–2024, 71 months)

| Config | Features | OOS R² | Sharpe | Ann. Return | Ann. Vol | IC Mean | IC t-stat |
|---|---|---|---|---|---|---|---|
| Baseline (no IC/PCA) | 185 | +0.0002 | **2.33** | 10.19% | 4.36% | +0.0664 | +6.22 |
| IC + Grouped PCA | 87 | +0.0002 | **2.37** | 10.81% | 4.56% | +0.0780 | +7.56 |

> **Skeleton GKX NN3 benchmark** (from jkp_project_skeleton_code.ipynb): Sharpe = 2.05, IC t-stat = 6.57

### Verdict
IC+PCA wins on Sharpe: **2.37** vs 2.33

---

## Output Files

| File | Description |
|---|---|
| `ic_stats.csv` | IC t-stats for all 185 features |
| `selected_features.json` | 140 IC-selected feature names |
| `ic_tstat_bar.png` | Feature ranking bar chart |
| `pca_group_summary.png` | PCA components per JKP theme |
| `nn3_cumulative.png` | Cumulative return chart (OOS) |
| `ak_results_nn3.csv` | Numeric results table |
| `ak_results.md` | This report |
