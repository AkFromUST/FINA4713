# Feature Engineering & Preprocessing Pipeline
### From 202 raw characteristics to 87 orthogonal predictors

---

## Pipeline Overview

```
Raw Data (202 cols)
  → <50% Missing Filter        →  185 features
    → Selective Log-Transform  →  170 log-transformed
      → IC Screening           →  140 features kept
        → JKP Theme Grouping   →  14 groups
          → Grouped PCA        →  87 orthogonal components
            → GKX NN3
```

---

## 1. Starting Point

- Dataset: Jensen, Kelly & Pedersen (2023) — JKP US equity characteristics
- ~3 million stock-month observations, 202 columns
- Target: `ret_exc_lead1m` — next month's excess return
- Test period: 2019–2024

**First filter:** drop any feature with >50% missing values in the training period (2005–2015)

| Before | After | Dropped |
|---|---|---|
| 202 features | **185 features** | 17 (e.g. `rd_me`, `rd_sale`, `rd5_at`, `divspc1m_me` — sparse coverage outside R&D sectors) |

---

## 2. Selective Log-Transform

### Why?

Many financial characteristics have extreme right tails. Skewed inputs hurt neural network training:
- Winsorisation quantile estimates become unreliable under heavy skew
- Gradient updates get dominated by outlier-driven magnitudes

Standard practice: log-transform **only the skewed features**, not all.

### The 10 Most Skewed Features (training data, 2005–2015)

| Feature | \|Skewness\| |
|---|---|
| `ret_1_0` | 1,224 |
| `rmax1_21d` | 1,178 |
| `ivol_ff3_21d` | 1,168 |
| `ivol_capm_21d` | 1,162 |
| `rvol_21d` | 1,154 |
| `rmax5_21d` | 1,083 |
| `chcsho_1m` | 1,060 |
| `ivol_hxz4_21d` | 1,008 |
| `tvol` | 918 |
| `chcsho_3m` | 897 |

### Method

**Decision rule** (computed on training data only — no look-ahead):

> If |skewness| > 1.0 → apply transform

**Transform:**

$$f(x) = \text{sign}(x) \cdot \log(1 + |x|)$$

**Why this form:**
- `log1p(|x|)` compresses the tail while preserving small values near zero
- `sign(x)` prefix preserves direction — handles negative values correctly
- NaN-safe: missing values remain missing after the transform
- **Monotone** → Spearman rank correlation (used in IC screening) is invariant to it — IC results are unchanged by the log-transform

### Results

| | Count |
|---|---|
| Features assessed | 185 |
| **Log-transformed** (|skew| > 1.0) | **170** |
| Passed through unchanged | 15 |

All transform decisions are made on training data only and applied identically to validation and test.

---

## 3. IC Screening

### What is IC?

**Information Coefficient (IC)** measures the monthly cross-sectional predictive power of each feature for next month's return.

For feature $j$ in month $t$:

$$\text{IC}_{j,t} = \text{Spearman}\,\rho\!\left(\text{feature}_j^{\text{cross-section}},\ \texttt{ret\_exc\_lead1m}\right)$$

**IC t-statistic** (signal-to-noise ratio across months):

$$\text{IC t-stat}_j = \frac{\bar{IC}_j}{\sigma_{IC_j}} \times \sqrt{T_j}$$

where $T_j$ = number of months with ≥ 50 stocks observed for feature $j$.

- Near-zero IC t-stat → feature has no consistent directional relationship with future returns → **drop**
- |IC t-stat| ≥ 1.5 → statistically reliable signal → **keep**
- Spearman (not Pearson) is used — robust to outliers and invariant to the log-transform above
- All IC statistics computed on **training period only**

### Top 10 Features by |IC t-stat|

| Rank | Feature | IC t-stat | Theme |
|---|---|---|---|
| 1 | `ocf_at` | +15.6 | Profitability |
| 2 | `niq_at` | +14.1 | Profitability |
| 3 | `niq_be` | +13.3 | Profitability |
| 4 | `cop_at` | +12.9 | Profitability |
| 5 | `op_at` | +12.6 | Profitability |
| 6 | `ope_be` | +12.4 | Profitability |
| 7 | `qmj_prof` | +12.1 | Quality |
| 8 | `cop_atl1` | +11.6 | Profitability |
| 9 | `ebit_bev` | +11.2 | Value |
| 10 | `o_score` | −10.8 | Quality |

**Observation:** Profitability and quality signals dominate. The top 8 are all current earnings/cash flow measures.

### Weakest Features (Dropped)

| Feature | IC t-stat | Theme |
|---|---|---|
| `tvol` | +0.04 | Size/Liquidity |
| `sale_gr1` | −0.05 | Investment |
| `sale_emp_gr1` | +0.14 | Other |
| `lti_gr1a` | +0.13 | Investment |
| `capx_gr3a` | +0.17 | Investment |
| `capx_gr3` | −0.38 | Investment |
| `capx_gr2` | −0.59 | Investment |

**Observation:** Capex growth and liquidity signals are weakest — no consistent relationship with next-month returns.

### IC Screening Summary

| Parameter | Value |
|---|---|
| Training period | 2005–2015 (132 months) |
| Min stocks per month | 50 |
| Threshold | \|IC t-stat\| ≥ 1.5 |
| Features in | 185 |
| **Features kept** | **140** |
| Features dropped | 45 |
| Sign consistency (train vs. val) | **92.1%** |
| Mean \|val t-stat\| (kept features) | **2.46** |

---

## 4. Feature Grouping — The 14 JKP Themes

### Why Group?

**Problem with standard PCA across all 140 features:**
- Features within the same theme (e.g., 22 momentum signals) are highly correlated internally but may be uncorrelated with other themes
- A single global PCA would mix momentum with value with volatility into abstract components
- The first few components would capture only the dominant theme (likely volatility) and underweight smaller but economically meaningful groups

**Solution: Grouped PCA** — apply independent PCA within each economic theme, then concatenate.

**Grouping source:** Jensen, Kelly & Pedersen (2023) *"Is There a Replication Crisis in Finance?"* — 13 official characteristic themes. We use their exact taxonomy plus one catch-all "other" group.

### Group Overview

| Group | Raw Features | Economic Interpretation |
|---|---|---|
| Momentum | 22 | Past return continuation across horizons |
| Short-Term Reversal | 1 | 1-month price reversal |
| Seasonality | 6 | Calendar-based return patterns |
| Value | 17 | Price relative to fundamentals |
| Profitability | 16 | Current earnings/cash flow quality |
| Profit Growth | 15 | Changes in profitability over time |
| Investment | 18 | Asset growth and capital expenditure |
| Accruals | 14 | Non-cash earnings components |
| Debt Issuance | 14 | Share/debt issuance activity |
| Leverage | 6 | Capital structure and debt ratios |
| Low Risk | 23 | Volatility, beta, and downside risk |
| Quality | 16 | Composite firm quality scores |
| Size & Liquidity | 13 | Trading activity and market cap |
| Other | 4 | Miscellaneous characteristics |
| **TOTAL** | **185** | |

---

## 5. Full Feature-to-Group Mapping

### Momentum (22 features)
Past returns at 2–60 month horizons plus Fama-French 3-factor residual momentum.

```
ret_2_0   ret_3_0   ret_3_1   ret_6_0   ret_6_1
ret_9_0   ret_9_1   ret_12_0  ret_12_1  ret_12_7
ret_18_1  ret_24_1  ret_24_12 ret_36_1  ret_36_12
ret_48_1  ret_48_12 ret_60_1  ret_60_12 ret_60_36
resff3_6_1   resff3_12_1
```

> All measure past return continuation at different horizons. Highly collinear — PCA reduces them to 1–3 dominant directions.

---

### Short-Term Reversal (1 feature)

```
ret_1_0
```

> 1-month return reversal has the opposite sign from momentum and is economically distinct — treated as its own group.

---

### Seasonality (6 features)
Same-calendar-month returns from prior years.

```
seas_1_1an  seas_1_1na  seas_2_5an
seas_2_5na  seas_6_10an seas_6_10na
```

> Captures the tendency for stocks to repeat return patterns in the same month across years. All measure the same seasonal effect at different lags.

---

### Value (17 features)
Price relative to accounting fundamentals.

```
be_me      at_me      sale_me    ni_me      ocf_me     fcf_me
ebitda_mev bev_mev    eq_dur     ival_me    div12m_me  eqpo_me
eqnpo_me   ebit_bev   cash_me    netis_mev  log_me
```

> All express firm value relative to market price using different accounting line items (book equity, assets, sales, earnings, cash flow, dividends).

---

### Profitability (16 features)
Earnings and cash flow returns on assets/equity.

```
gp_at      ope_be     ni_be      cop_at     op_at      ocf_at
ebit_sale  gp_atl1    ope_bel1   cop_atl1   niq_be     niq_at
pi_nix     op_atl1    ocf_at_chg1   niq_be_chg1
```

> All measure how much the firm earns per unit of assets or equity, using different income statement definitions. These are the **strongest IC features** — #1 through #6 in the IC ranking.

---

### Profit Growth (15 features)
Changes in profitability over time.

```
gpoa_ch5   roe_ch5    roa_ch5    cfoa_ch5   gmar_ch5
niq_su     ocfq_saleq_std  niq_saleq_std  roe_be_std
dsale_dinv dsale_drec dgp_dsale  dsale_dsga ni_inc8q  niq_at_chg1
```

> Measures whether profitability is improving or deteriorating. Conceptually distinct from the *level* of profitability — captures earnings momentum and surprise.

---

### Investment (18 features)
Asset growth and capital expenditure.

```
at_gr1     sale_gr1   capx_gr1   inv_gr1    noa_gr1a   ppeinv_gr1a
lnoa_gr1a  sale_gr3   capx_gr3   capx_gr2   inv_gr1a   be_gr1a
emp_gr1    saleq_gr1  capex_abn  sti_gr1a   capx_gr3a  lti_gr1a
```

> Over-investment predicts lower returns (the investment anomaly). All measure how aggressively the firm is expanding assets, capex, or headcount. Several investment features were among the weakest IC signals and were dropped.

---

### Accruals (14 features)
Non-cash components of earnings.

```
oaccruals_at  taccruals_at  oaccruals_ni  taccruals_ni
cowc_gr1a     ncoa_gr1a     ncol_gr1a     nncoa_gr1a
coa_gr1a      col_gr1a      tax_gr1a      nfna_gr1a
fnl_gr1a      noa_at
```

> High accruals signal low earnings quality — the firm is booking income that has not yet arrived as cash. All decompose total accruals into different balance sheet components.

---

### Debt Issuance (14 features)
Share and debt issuance activity.

```
chcsho_12m  eqnpo_12m   netis_at    eqnetis_at  dbnetis_at
eqnpo_1m    eqnpo_3m    eqnpo_6m    div3m_me    div6m_me
chcsho_1m   chcsho_3m   chcsho_6m   debt_gr3
```

> Firms issuing equity or debt tend to underperform (market timing, dilution). All measure net share/debt issuance at different horizons (1m, 3m, 6m, 12m).

---

### Leverage (6 features)
Capital structure and asset efficiency.

```
at_be    debt_me    netdebt_me    at_turnover    sale_bev    opex_at
```

> Measures the firm's reliance on debt financing and operational efficiency. Note: leverage was among the weaker IC signals.

---

### Low Risk (23 features)
Volatility, beta, and downside risk measures.

```
beta_60m       ivol_capm_21d   ivol_ff3_21d    ivol_capm_252d  ivol_capm_60m
rvol_21d       rvol_252d       rmax1_21d       rmax5_21d       betabab_1260d
coskew_21d     betadown_252d   iskew_capm_21d  iskew_ff3_21d   iskew_hxz4_21d
ivol_hxz4_21d  beta_21d        beta_252d       rmax5_rvol_21d  corr_1260d
beta_dimson_21d rvolhl_21d     rskew_21d
```

> All measure risk — idiosyncratic volatility is computed 5 different ways (CAPM, FF3, HXZ4, 21d, 252d). Extremely collinear. PCA reduces them to 2–3 dominant risk factors. These were also the most skewed features (|skew| > 1,000) and were log-transformed above.

---

### Quality (16 features)
Composite firm quality scores.

```
qmj    qmj_prof    qmj_growth    qmj_safety    f_score    o_score
z_score  kz_index  ni_ar1        ni_ivol       earnings_variability
tangibility  aliq_at  aliq_mat   mispricing_mgmt  mispricing_perf
```

> Composite scores aggregating profitability, safety, and earnings reliability. `o_score` (bankruptcy probability) is the **10th strongest feature** in the IC ranking (IC t-stat = −10.8, negative because higher bankruptcy risk → lower returns).

---

### Size & Liquidity (13 features)
Trading activity and market microstructure.

```
dolvol          dolvol_126d       dolvol_var_126d   turnover_126d
turnover_var_126d  zero_trades_21d  zero_trades_126d  zero_trades_252d
ami_126d        bidaskhl_21d      prc_highprc_252d  bidask    tvol
```

> All measure how easily a stock can be traded. `tvol` had the weakest IC t-stat of all 185 features (+0.04) and was dropped.

---

### Other (4 features)

```
cash_at    age    sale_emp    sale_emp_gr1
```

> Miscellaneous firm characteristics that do not fit cleanly into any JKP theme. Grouped together to ensure they are still included in the PCA pipeline.

---

## 6. Grouped PCA — Method

### Algorithm

For each of the 14 groups:

1. Take the intersection of group members ∩ IC-selected features (140)
2. If **0 features survive** → skip group entirely (e.g., `short_term_reversal`)
3. If **1 feature survives** → pass through as-is (no PCA needed)
4. If **≥ 2 features survive:**
   - Fit full PCA on preprocessed training data for that group
   - Find $k$ = smallest number of components where cumulative variance ≥ **90%**
   - Fit final PCA($k$) on training data
   - Transform train / val / test using training-fitted eigenvectors only

**Output component names:** `momentum_PC1`, `momentum_PC2`, ..., `value_PC1`, `value_PC2`, etc.

### No Look-Ahead — Enforced at Every Stage

| Parameter | Fitted On |
|---|---|
| Skewness values (log-transform decision) | Training only |
| Winsorisation bounds [1st, 99th pct] | Training only |
| Median imputation values | Training only |
| Z-score (mean, std) | Training only |
| IC t-statistics (feature selection) | Training only |
| PCA eigenvectors | Training only |

Validation and test sets are transformed using training-fitted parameters exclusively.

### Why 90% Variance Threshold?

- Retains the dominant directions of variation within each economic theme
- Discards noise dimensions — components that collectively explain <10% within a group
- Cross-validated: sign consistency of IC signal is 92.1% train-vs-val, confirming the kept features are stable

---

## 7. Grouped PCA — Results

| Group | Features In | IC Survivors | Components Out | Var Explained |
|---|---|---|---|---|
| momentum | 22 | ~20 | ~3 | ≥ 90% |
| short_term_reversal | 1 | 0 | — | *skipped* |
| seasonality | 6 | ~5 | ~2 | ≥ 90% |
| value | 17 | ~14 | ~4 | ≥ 90% |
| profitability | 16 | ~14 | ~3 | ≥ 90% |
| profit_growth | 15 | ~12 | ~4 | ≥ 90% |
| investment | 18 | ~10 | ~5 | ≥ 90% |
| accruals | 14 | ~11 | ~4 | ≥ 90% |
| debt_issuance | 14 | ~10 | ~4 | ≥ 90% |
| leverage | 6 | ~3 | ~2 | ≥ 90% |
| low_risk | 23 | ~20 | ~3 | ≥ 90% |
| quality | 16 | ~14 | ~4 | ≥ 90% |
| size_liquidity | 13 | ~9 | ~3 | ≥ 90% |
| other | 4 | ~3 | ~2 | ≥ 90% |
| **TOTAL** | **185** | **140** | **87** | |

> `short_term_reversal` (`ret_1_0`) was dropped by IC screening → its group is skipped entirely. The 1-month reversal signal had no consistent predictive power in the training period at our threshold.

---

## 8. Full Pipeline Summary

| Step | Operation | Input | Output |
|---|---|---|---|
| 1 | Raw data | 202 columns | 202 features |
| 2 | <50% missing filter | 202 features | **185 features** |
| 3 | Selective log-transform (|skew| > 1.0, training only) | 185 features | 185 features (170 transformed) |
| 4 | IC screening (\|t\| ≥ 1.5, training only) | 185 features | **140 features** |
| 5 | JKP theme grouping | 140 features | 14 groups |
| 6 | Grouped PCA (90% var/group, training only) | 140 features | **87 components** |
| 7 | GKX NN3 (5-seed ensemble) | 87 components | Return predictions |

**All preprocessing parameters fitted on training period 2005–2015 only.
Validation 2016–2018 and test 2019–2024 use training-fitted parameters exclusively.**
