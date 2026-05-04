"""
ak_pipeline.py  —  IC screening + Grouped PCA → GKX NN3
---------------------------------------------------------
Pipeline:
  1. Load data  →  185 features  (<50% missing)
  2. IC screening on raw training data  →  ~140 features
  3. Grouped PCA on 13 JKP themes  →  ~45 orthogonal components
  4. GKX NN3 (5-seed ensemble, MPS-accelerated)

Two configurations evaluated side-by-side:
  Baseline  : GKX NN3 on 185 preprocessed features
  IC + PCA  : GKX NN3 on IC-selected + grouped-PCA components
"""

import os, json, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, '..', 'jkp_data_slim.parquet')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'files', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET     = 'ret_exc_lead1m'
TRAIN_END  = '2015-12-31'
VAL_END    = '2018-12-31'

# Feature engineering
MISSING_CAP      = 0.50   # drop features with >50% missing in training
IC_THRESHOLD     = 1.5    # |IC t-stat| cutoff for keeping features
PCA_VAR_THRESH   = 0.90   # variance explained per PCA group
MIN_STOCKS_IC    = 50     # minimum cross-section size for monthly IC

# GKX NN3
BATCH_SIZE   = 2048
LR           = 0.001
LAMBDA_L1    = 1e-5
PATIENCE     = 5
MAX_EPOCHS   = 100
SEEDS        = [42, 123, 456, 789, 101112]

# Device — M2 Pro MPS
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Device : {device}")

def set_all_seeds(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)

# ═══════════════════════════════════════════════════════════════════════════
# JKP 13-THEME FEATURE GROUPS
# Source: Jensen, Kelly & Pedersen (2023) "Is There a Replication Crisis
# in Finance?" — 13 official characteristic themes.
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    'momentum': [
        'ret_2_0', 'ret_3_0', 'ret_3_1', 'ret_6_0', 'ret_6_1',
        'ret_9_0', 'ret_9_1', 'ret_12_0', 'ret_12_1', 'ret_12_7',
        'ret_18_1', 'ret_24_1', 'ret_24_12', 'ret_36_1', 'ret_36_12',
        'ret_48_1', 'ret_48_12', 'ret_60_1', 'ret_60_12', 'ret_60_36',
        'resff3_6_1', 'resff3_12_1',
    ],
    'short_term_reversal': [
        'ret_1_0',
    ],
    'seasonality': [
        'seas_1_1an', 'seas_1_1na', 'seas_2_5an', 'seas_2_5na',
        'seas_6_10an', 'seas_6_10na',
    ],
    'value': [
        'be_me', 'at_me', 'sale_me', 'ni_me', 'ocf_me', 'fcf_me',
        'ebitda_mev', 'bev_mev', 'eq_dur', 'ival_me', 'div12m_me',
        'eqpo_me', 'eqnpo_me', 'ebit_bev', 'cash_me', 'netis_mev',
        'log_me',
    ],
    'profitability': [
        'gp_at', 'ope_be', 'ni_be', 'cop_at', 'op_at', 'ocf_at',
        'ebit_sale', 'gp_atl1', 'ope_bel1', 'cop_atl1', 'niq_be',
        'niq_at', 'pi_nix', 'op_atl1', 'ocf_at_chg1', 'niq_be_chg1',
    ],
    'profit_growth': [
        'gpoa_ch5', 'roe_ch5', 'roa_ch5', 'cfoa_ch5', 'gmar_ch5',
        'niq_su', 'ocfq_saleq_std', 'niq_saleq_std', 'roe_be_std',
        'dsale_dinv', 'dsale_drec', 'dgp_dsale', 'dsale_dsga',
        'ni_inc8q', 'niq_at_chg1',
    ],
    'investment': [
        'at_gr1', 'sale_gr1', 'capx_gr1', 'inv_gr1', 'noa_gr1a',
        'ppeinv_gr1a', 'lnoa_gr1a', 'sale_gr3', 'capx_gr3', 'capx_gr2',
        'inv_gr1a', 'be_gr1a', 'emp_gr1', 'saleq_gr1', 'capex_abn',
        'sti_gr1a', 'capx_gr3a', 'lti_gr1a',
    ],
    'accruals': [
        'oaccruals_at', 'taccruals_at', 'oaccruals_ni', 'taccruals_ni',
        'cowc_gr1a', 'ncoa_gr1a', 'ncol_gr1a', 'nncoa_gr1a',
        'coa_gr1a', 'col_gr1a', 'tax_gr1a', 'nfna_gr1a', 'fnl_gr1a',
        'noa_at',
    ],
    'debt_issuance': [
        'chcsho_12m', 'eqnpo_12m', 'netis_at', 'eqnetis_at', 'dbnetis_at',
        'eqnpo_1m', 'eqnpo_3m', 'eqnpo_6m', 'div3m_me', 'div6m_me',
        'chcsho_1m', 'chcsho_3m', 'chcsho_6m', 'debt_gr3',
    ],
    'leverage': [
        'at_be', 'debt_me', 'netdebt_me', 'at_turnover', 'sale_bev',
        'opex_at',
    ],
    'low_risk': [
        'beta_60m', 'ivol_capm_21d', 'ivol_ff3_21d', 'ivol_capm_252d',
        'ivol_capm_60m', 'rvol_21d', 'rvol_252d', 'rmax1_21d', 'rmax5_21d',
        'betabab_1260d', 'coskew_21d', 'betadown_252d', 'iskew_capm_21d',
        'iskew_ff3_21d', 'iskew_hxz4_21d', 'ivol_hxz4_21d', 'beta_21d',
        'beta_252d', 'rmax5_rvol_21d', 'corr_1260d', 'beta_dimson_21d',
        'rvolhl_21d', 'rskew_21d',
    ],
    'quality': [
        'qmj', 'qmj_prof', 'qmj_growth', 'qmj_safety', 'f_score',
        'o_score', 'z_score', 'kz_index', 'ni_ar1', 'ni_ivol',
        'earnings_variability', 'tangibility', 'aliq_at', 'aliq_mat',
        'mispricing_mgmt', 'mispricing_perf',
    ],
    'size_liquidity': [
        'dolvol', 'dolvol_126d', 'dolvol_var_126d', 'turnover_126d',
        'turnover_var_126d', 'zero_trades_21d', 'zero_trades_126d',
        'zero_trades_252d', 'ami_126d', 'bidaskhl_21d', 'prc_highprc_252d',
        'bidask', 'tvol',
    ],
    'other': [
        'cash_at', 'age', 'sale_emp', 'sale_emp_gr1',
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def oos_r2(y_true, y_pred, y_null):
    return 1 - np.mean((y_true - y_pred)**2) / np.mean((y_true - y_null)**2)

def portfolio_weights(pred, max_w=0.05):
    w = pd.Series(pred).rank() - (len(pred) + 1) / 2
    w /= w.abs().sum()
    w  = w.clip(-max_w, max_w)
    w /= w.abs().sum()
    return w.values

def fit_preprocessor(X_df):
    low  = X_df.quantile(0.01)
    high = X_df.quantile(0.99)
    Xc   = X_df.clip(lower=low, upper=high, axis=1)
    imp  = SimpleImputer(strategy='median').fit(Xc)
    sc   = StandardScaler().fit(imp.transform(Xc))
    return low, high, imp, sc

def apply_preprocessor(X_df, low, high, imp, sc):
    X = X_df.clip(lower=low, upper=high, axis=1)
    return sc.transform(imp.transform(X))

def to_num(df_, feats):
    return df_[feats].apply(pd.to_numeric, errors='coerce')

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("1. LOAD DATA")
print("═"*65)

df = pd.read_parquet(DATA_PATH)
df['eom']   = pd.to_datetime(df['eom'])
df['log_me'] = np.log1p(df['me'].clip(lower=0))
df = df.dropna(subset=[TARGET])

# Cast object columns to numeric up front
META     = ['id', 'eom', 'excntry', TARGET, 'me']
ALL_FEAT = [c for c in df.columns if c not in META]
for c in ALL_FEAT:
    if df[c].dtype == object:
        df[c] = pd.to_numeric(df[c], errors='coerce')

print(f"Shape  : {df.shape}")
print(f"Period : {df['eom'].min().date()} to {df['eom'].max().date()}")
print(f"Firms  : {df['id'].nunique():,}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. TEMPORAL SPLIT  +  <50% MISSING FILTER
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("2. TEMPORAL SPLIT  +  FEATURE FILTER")
print("═"*65)

train = df[df['eom'] <= TRAIN_END].copy()
val   = df[(df['eom'] > TRAIN_END) & (df['eom'] <= VAL_END)].copy()
test  = df[df['eom'] > VAL_END].copy()

missing_pct = to_num(train, ALL_FEAT).isna().mean()
FEATURES    = missing_pct[missing_pct < MISSING_CAP].index.tolist()

print(f"Features after <{MISSING_CAP:.0%} missing filter : {len(FEATURES)}")
print(f"Train : {len(train):>7,}  ({train['eom'].min().date()} – {train['eom'].max().date()})")
print(f"Val   : {len(val):>7,}  ({val['eom'].min().date()} – {val['eom'].max().date()})")
print(f"Test  : {len(test):>7,}  ({test['eom'].min().date()} – {test['eom'].max().date()})")

# ═══════════════════════════════════════════════════════════════════════════
# 3. IC SCREENING  (fit on training period only)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("3. IC SCREENING  (Campbell-Thompson monthly Spearman ρ)")
print("═"*65)

def compute_ic_matrix(df_split, features, target=TARGET, min_stocks=MIN_STOCKS_IC):
    months = sorted(df_split['eom'].unique())
    rows   = []
    for month in months:
        grp = df_split[df_split['eom'] == month][[target] + features].dropna(subset=[target])
        if len(grp) < min_stocks:
            rows.append([np.nan] * len(features)); continue
        y   = grp[target].values
        row = []
        for feat in features:
            x     = grp[feat].values.astype(float)
            valid = ~np.isnan(x)
            if valid.sum() < min_stocks:
                row.append(np.nan)
            else:
                row.append(spearmanr(x[valid], y[valid])[0])
        rows.append(row)
    return pd.DataFrame(rows, index=months, columns=features)

print("  Computing monthly IC on training set...")
ic_train = compute_ic_matrix(train, FEATURES)

T        = ic_train.notna().sum()
mean_ic  = ic_train.mean()
std_ic   = ic_train.std()
ic_tstat = mean_ic / std_ic * np.sqrt(T)

ic_stats = pd.DataFrame({'mean_ic': mean_ic, 'std_ic': std_ic, 'tstat': ic_tstat})
ic_stats.index.name = 'feature'
ic_stats = ic_stats.sort_values('tstat', key=abs, ascending=False)

# Apply threshold — keep |t| >= IC_THRESHOLD
IC_FEATURES = ic_stats[ic_stats['tstat'].abs() >= IC_THRESHOLD].index.tolist()
n_keep = len(IC_FEATURES)
n_drop = len(FEATURES) - n_keep

print(f"  Threshold |t| ≥ {IC_THRESHOLD} : kept {n_keep}, dropped {n_drop}")

# Quick validation stability check on val
print("  Computing validation IC for stability check...")
ic_val   = compute_ic_matrix(val, IC_FEATURES)
T_v      = ic_val.notna().sum()
tstat_v  = ic_val.mean() / ic_val.std() * np.sqrt(T_v)
sign_ok  = ((ic_tstat[IC_FEATURES] > 0) == (tstat_v > 0)).mean()
print(f"  Sign consistency train/val : {sign_ok:.1%}")
print(f"  Mean |val t-stat|          : {tstat_v.abs().mean():.2f}")

# Save IC stats
ic_stats.to_csv(os.path.join(OUT_DIR, 'ic_stats.csv'))
with open(os.path.join(OUT_DIR, 'selected_features.json'), 'w') as f:
    json.dump(IC_FEATURES, f, indent=2)
print(f"  IC stats saved → {OUT_DIR}/ic_stats.csv")

# IC t-stat bar chart
fig, ax = plt.subplots(figsize=(16, 5))
colors  = ['#2166ac' if abs(t) >= IC_THRESHOLD else '#d6604d'
           for t in ic_stats['tstat'].values]
ax.bar(range(len(ic_stats)), ic_stats['tstat'].values, color=colors, width=1.0, linewidth=0)
ax.axhline( IC_THRESHOLD, color='black', lw=1.2, linestyle='--', label=f'±{IC_THRESHOLD}')
ax.axhline(-IC_THRESHOLD, color='black', lw=1.2, linestyle='--')
ax.axhline(0, color='black', lw=0.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#2166ac', label=f'Keep ({n_keep})'),
                   Patch(facecolor='#d6604d', label=f'Drop ({n_drop})')], loc='upper right')
ax.set_xlabel('Feature (ranked by |IC t-stat|)')
ax.set_ylabel('IC t-statistic')
ax.set_title(f'IC Screening — {len(FEATURES)} features → {n_keep} kept  (threshold ±{IC_THRESHOLD})')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'ic_tstat_bar.png'), dpi=150); plt.close()
print("  Saved: ic_tstat_bar.png")

# ═══════════════════════════════════════════════════════════════════════════
# 4. GROUPED PCA  (fit on training only)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("4. GROUPED PCA  (13 JKP themes, 90% variance per group)")
print("═"*65)

class GroupedPCA:
    """
    Fits independent PCA within each economic group of features.
    Only IC-selected features are used within each group.
    Groups with 1 surviving feature: passed through as-is.
    Groups with 0 surviving features: skipped.
    """
    def __init__(self, groups, var_threshold=PCA_VAR_THRESH):
        self.groups        = groups
        self.var_threshold = var_threshold
        self.fitted        = {}   # {group: (features_used, pca_or_None)}
        self.output_names  = []

    def fit(self, X_df):
        self.fitted = {}; self.output_names = []
        for grp, feats in self.groups.items():
            avail = [f for f in feats if f in X_df.columns]
            if len(avail) == 0:
                continue
            if len(avail) == 1:
                self.fitted[grp] = (avail, None)
                self.output_names.append(f'{grp}_F1')
                continue
            Xg = X_df[avail].values
            # Determine n_components for var_threshold
            pca_full = PCA(n_components=min(len(avail), Xg.shape[0] - 1)).fit(Xg)
            cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp   = int(np.searchsorted(cumvar, self.var_threshold)) + 1
            n_comp   = max(1, min(n_comp, len(avail)))
            pca      = PCA(n_components=n_comp).fit(Xg)
            self.fitted[grp] = (avail, pca)
            for i in range(n_comp):
                self.output_names.append(f'{grp}_PC{i+1}')
        return self

    def transform(self, X_df):
        parts = []
        for grp, (feats, pca) in self.fitted.items():
            avail = [f for f in feats if f in X_df.columns]
            Xg    = X_df[avail].values if avail else np.zeros((len(X_df), len(feats)))
            parts.append(Xg if pca is None else pca.transform(Xg))
        return np.hstack(parts) if parts else np.zeros((len(X_df), 0))

    def fit_transform(self, X_df):
        return self.fit(X_df).transform(X_df)

    def print_summary(self):
        print(f"  {'Group':<22} {'Raw feats':>10} {'Surviving':>10} {'Components':>12} {'Var expl':>10}")
        print("  " + "─"*66)
        total_raw  = 0
        total_surv = 0
        total_comp = 0
        for grp, feats_all in self.groups.items():
            if grp not in self.fitted:
                continue
            feats_used, pca = self.fitted[grp]
            n_raw  = len(feats_all)
            n_surv = len(feats_used)
            n_comp = 1 if pca is None else pca.n_components_
            var    = 1.0 if pca is None else pca.explained_variance_ratio_.sum()
            print(f"  {grp:<22} {n_raw:>10} {n_surv:>10} {n_comp:>12} {var:>9.1%}")
            total_raw += n_raw; total_surv += n_surv; total_comp += n_comp
        print("  " + "─"*66)
        print(f"  {'TOTAL':<22} {total_raw:>10} {total_surv:>10} {total_comp:>12}")


# ── Preprocessing for IC-selected features (used by GroupedPCA) ──────────
low_ic, high_ic, imp_ic, sc_ic = fit_preprocessor(to_num(train, IC_FEATURES))

X_tr_ic  = apply_preprocessor(to_num(train, IC_FEATURES), low_ic, high_ic, imp_ic, sc_ic)
X_val_ic = apply_preprocessor(to_num(val,   IC_FEATURES), low_ic, high_ic, imp_ic, sc_ic)
X_te_ic  = apply_preprocessor(to_num(test,  IC_FEATURES), low_ic, high_ic, imp_ic, sc_ic)

# Convert to DataFrames so GroupedPCA can look up feature names
X_tr_ic_df  = pd.DataFrame(X_tr_ic,  columns=IC_FEATURES)
X_val_ic_df = pd.DataFrame(X_val_ic, columns=IC_FEATURES)
X_te_ic_df  = pd.DataFrame(X_te_ic,  columns=IC_FEATURES)

# Fit GroupedPCA on training only
gpca = GroupedPCA(FEATURE_GROUPS, var_threshold=PCA_VAR_THRESH)
X_tr_pca  = gpca.fit_transform(X_tr_ic_df)
X_val_pca = gpca.transform(X_val_ic_df)
X_te_pca  = gpca.transform(X_te_ic_df)

gpca.print_summary()
print(f"\n  Input to NN (IC+PCA) : {X_tr_pca.shape[1]} components  "
      f"(from {n_keep} IC-selected features)")

# PCA explained-variance plot
fig, ax = plt.subplots(figsize=(14, 4))
group_labels, comp_counts, var_expls = [], [], []
for grp, (feats_used, pca) in gpca.fitted.items():
    group_labels.append(grp)
    comp_counts.append(1 if pca is None else pca.n_components_)
    var_expls.append(1.0 if pca is None else pca.explained_variance_ratio_.sum())

x = np.arange(len(group_labels))
bars = ax.bar(x, comp_counts, color='#4393c3', width=0.6, label='Components kept')
ax2  = ax.twinx()
ax2.plot(x, [v*100 for v in var_expls], 'o--', color='#d6604d',
         linewidth=1.5, markersize=6, label='Variance explained (%)')
ax.set_xticks(x); ax.set_xticklabels(group_labels, rotation=35, ha='right', fontsize=8)
ax.set_ylabel('Number of PCA components')
ax2.set_ylabel('Variance explained within group (%)')
ax.set_title(f'Grouped PCA — components and variance explained per JKP theme\n'
             f'Total: {sum(comp_counts)} components from {n_keep} IC-selected features')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc='upper right', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'pca_group_summary.png'), dpi=150); plt.close()
print("  Saved: pca_group_summary.png")

# ── Preprocessing for Baseline (all 185 features) ────────────────────────
low_b, high_b, imp_b, sc_b = fit_preprocessor(to_num(train, FEATURES))
X_tr_b  = apply_preprocessor(to_num(train, FEATURES), low_b, high_b, imp_b, sc_b)
X_val_b = apply_preprocessor(to_num(val,   FEATURES), low_b, high_b, imp_b, sc_b)
X_te_b  = apply_preprocessor(to_num(test,  FEATURES), low_b, high_b, imp_b, sc_b)

y_tr  = train[TARGET].values
y_val = val[TARGET].values
y_te  = test[TARGET].values

print(f"\n  Baseline input  (185 features)    : {X_tr_b.shape}")
print(f"  IC+PCA input    ({X_tr_pca.shape[1]} components) : {X_tr_pca.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. GKX NN3  (Gu, Kelly & Xiu 2020)
#    Architecture : Linear(d→32)→ReLU→Linear(32→16)→ReLU→Linear(16→8)→ReLU→Linear(8→1)
#    Training     : MSE + L1 reg, early stopping, 5-seed ensemble
#    Device       : MPS (M2 Pro)
# ═══════════════════════════════════════════════════════════════════════════

class GKX_NN3(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16,  8), nn.ReLU(),
            nn.Linear(8,   1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_gkx_nn3(X_train, y_train, X_val, y_val, input_dim, dev=device):
    model = GKX_NN3(input_dim).to(dev)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    # pin_memory only for CUDA; num_workers=0 required for MPS on macOS
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(dev.type == 'cuda')
    )
    Xv = torch.FloatTensor(X_val).to(dev)
    yv = torch.FloatTensor(y_val).to(dev)

    best_loss, best_sd, wait = float('inf'), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(Xb), yb) + LAMBDA_L1 * sum(p.abs().sum() for p in model.parameters())
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        if vl < best_loss:
            best_loss = vl
            best_sd   = {k: v.clone() for k, v in model.state_dict().items()}
            wait      = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_sd:
        model.load_state_dict(best_sd)
    return model.cpu()   # always return on CPU for safe inference


def run_nn3_ensemble(X_train, y_train, X_val, y_val, X_test, label):
    """Train 5-seed ensemble and return test predictions."""
    print(f"\n  [{label}]  input_dim={X_train.shape[1]}")
    preds = []
    for seed in SEEDS:
        print(f"    seed={seed}...", end=' ', flush=True)
        set_all_seeds(seed)
        m = train_gkx_nn3(X_train, y_train, X_val, y_val, X_train.shape[1])
        m.eval()
        with torch.no_grad():
            p = m(torch.FloatTensor(X_test)).numpy()
        preds.append(p)
        print(f"done")
    return np.mean(preds, axis=0)


def evaluate_nn3(pred_raw, y_te, y_null_te, test_df, label):
    """Inverse-scale predictions, compute OOS R², IC, portfolio Sharpe."""
    sc_y   = StandardScaler().fit(y_tr.reshape(-1, 1))
    pred   = sc_y.inverse_transform(pred_raw.reshape(-1, 1)).flatten()

    r2     = oos_r2(y_te, pred, y_null_te)
    r2_z   = 1 - np.sum((y_te - pred)**2) / np.sum(y_te**2)

    tdf    = test_df.reset_index(drop=True).copy()
    tdf['pred'] = pred
    ic_s   = tdf.groupby('eom').apply(lambda g: spearmanr(g['pred'], g[TARGET])[0])
    ic_t   = ic_s.mean() / ic_s.std() * np.sqrt(len(ic_s))

    port   = []
    for eom_date, grp in tdf.groupby('eom'):
        port.append({
            'date': eom_date,
            label : (portfolio_weights(grp['pred'].values) * grp[TARGET].values).sum(),
            'Market': grp[TARGET].mean(),
        })
    pf = pd.DataFrame(port).set_index('date').sort_index()
    pf.index = pd.to_datetime(pf.index)

    sr     = pf[label].mean() * 12 / (pf[label].std() * np.sqrt(12))
    ar     = pf[label].mean() * 12
    av     = pf[label].std()  * np.sqrt(12)

    print(f"\n  {label}")
    print(f"    OOS R² (hist-avg null)  : {r2:+.4f}")
    print(f"    OOS R² (zero null)      : {r2_z:+.4%}")
    print(f"    Mean IC / IC t-stat     : {ic_s.mean():+.4f} / {ic_t:+.2f}  ({len(ic_s)} months)")
    print(f"    Ann. Return             : {ar:.2%}")
    print(f"    Ann. Vol                : {av:.2%}")
    print(f"    Sharpe                  : {sr:.2f}")

    return {'pf': pf, 'pred': pred, 'r2': r2, 'sharpe': sr,
            'ic_mean': ic_s.mean(), 'ic_tstat': ic_t, 'ann_ret': ar, 'ann_vol': av}

# ═══════════════════════════════════════════════════════════════════════════
# 6. RUN A — Baseline (185 preprocessed features)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("6. GKX NN3 — Baseline  (185 features, no IC/PCA)")
print("═"*65)

y_null_te  = np.full(len(y_te), y_tr.mean())
pred_raw_A = run_nn3_ensemble(X_tr_b, y_tr, X_val_b, y_val, X_te_b, 'Baseline')
res_A      = evaluate_nn3(pred_raw_A, y_te, y_null_te, test, 'Baseline (185 feats)')

# ═══════════════════════════════════════════════════════════════════════════
# 7. RUN B — IC + Grouped PCA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print(f"7. GKX NN3 — IC + Grouped PCA  ({X_tr_pca.shape[1]} components)")
print("═"*65)

pred_raw_B = run_nn3_ensemble(X_tr_pca, y_tr, X_val_pca, y_val, X_te_pca, 'IC+PCA')
res_B      = evaluate_nn3(pred_raw_B, y_te, y_null_te, test, 'IC+PCA')

# ═══════════════════════════════════════════════════════════════════════════
# 8. COMPARISON TABLE + PLOTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("8. COMPARISON")
print("═"*65)

print(f"\n  {'Config':<28} {'Features':>10} {'OOS R²':>10} "
      f"{'Sharpe':>8} {'IC Mean':>10} {'IC t-stat':>10}")
print("  " + "─"*78)
for cfg, n_feat, res in [
    ('Baseline (no IC/PCA)',    185,              res_A),
    (f'IC + PCA ({X_tr_pca.shape[1]} components)', X_tr_pca.shape[1], res_B),
]:
    print(f"  {cfg:<28} {n_feat:>10} {res['r2']:>+10.4f} "
          f"{res['sharpe']:>8.2f} {res['ic_mean']:>+10.4f} {res['ic_tstat']:>+10.2f}")
print()

# Cumulative return plot
all_ret = res_A['pf'][['Baseline (185 feats)', 'Market']].join(
          res_B['pf'][['IC+PCA']], how='outer')

cum = (1 + all_ret.fillna(0)).cumprod() - 1
colors = {
    'Baseline (185 feats)': '#4393c3',
    'IC+PCA'              : '#238b45',
    'Market'              : '#888888',
}

fig, ax = plt.subplots(figsize=(13, 5))
for col in cum.columns:
    ls = '--' if col == 'Market' else '-'
    lw = 1.5 if col == 'Market' else 2.2
    sr = all_ret[col].mean() * 12 / (all_ret[col].std() * np.sqrt(12))
    ax.plot(cum.index, cum[col] * 100,
            color=colors.get(col, '#333333'), linestyle=ls, lw=lw,
            label=f'{col}  (SR={sr:.2f})')
ax.axhline(0, color='black', lw=0.6, linestyle='--', alpha=0.4)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylabel('Cumulative Excess Return (%)')
ax.set_title('GKX NN3 — Baseline vs IC+PCA  (OOS 2019–2024)')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', lw=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'nn3_cumulative.png'), dpi=150); plt.close()
print("  Saved: nn3_cumulative.png")

# ═══════════════════════════════════════════════════════════════════════════
# 9. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame([
    {
        'config'    : 'baseline_185',
        'n_features': 185,
        'oos_r2'    : res_A['r2'],
        'sharpe'    : res_A['sharpe'],
        'ann_ret'   : res_A['ann_ret'],
        'ann_vol'   : res_A['ann_vol'],
        'ic_mean'   : res_A['ic_mean'],
        'ic_tstat'  : res_A['ic_tstat'],
    },
    {
        'config'    : f'ic_pca_{X_tr_pca.shape[1]}',
        'n_features': X_tr_pca.shape[1],
        'oos_r2'    : res_B['r2'],
        'sharpe'    : res_B['sharpe'],
        'ann_ret'   : res_B['ann_ret'],
        'ann_vol'   : res_B['ann_vol'],
        'ic_mean'   : res_B['ic_mean'],
        'ic_tstat'  : res_B['ic_tstat'],
    },
])
results_df.to_csv(os.path.join(OUT_DIR, 'ak_results_nn3.csv'), index=False)
print(f"  Results saved → {OUT_DIR}/ak_results_nn3.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 10. MARKDOWN REPORT
# ═══════════════════════════════════════════════════════════════════════════

import datetime
run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

n_pca = X_tr_pca.shape[1]

md = f"""# AK Pipeline — GKX NN3 Results
*Run completed: {run_time}*

---

## Feature Engineering Summary

| Stage | Features |
|---|---|
| Raw columns | 202 |
| After <50% missing filter | 185 |
| After IC screening (\\|t\\| ≥ {IC_THRESHOLD}) | {n_keep} |
| After Grouped PCA (90% var/group) | {n_pca} components |

### IC Screening
- Training period: 2005–2015 ({train['eom'].nunique()} months)
- Threshold: |IC t-stat| ≥ {IC_THRESHOLD}
- **Kept {n_keep} / dropped {n_drop}** features
- Sign consistency train/val: **{sign_ok:.1%}**
- Mean |val t-stat| (kept features): {tstat_v.abs().mean():.2f}

### Grouped PCA (13 JKP Themes)

| Group | Raw | Surviving | Components | Var Expl |
|---|---|---|---|---|
{"".join(f"| {grp} | {len(FEATURE_GROUPS[grp])} | {len(feats)} | {1 if pca is None else pca.n_components_} | {1.0 if pca is None else pca.explained_variance_ratio_.sum():.1%} |" + chr(10) for grp, (feats, pca) in gpca.fitted.items())}
| **TOTAL** | **{sum(len(FEATURE_GROUPS[g]) for g in gpca.fitted)}** | **{n_keep}** | **{n_pca}** | — |

---

## GKX NN3 Results (OOS: 2019–2024, {test['eom'].nunique()} months)

| Config | Features | OOS R² | Sharpe | Ann. Return | Ann. Vol | IC Mean | IC t-stat |
|---|---|---|---|---|---|---|---|
| Baseline (no IC/PCA) | 185 | {res_A['r2']:+.4f} | **{res_A['sharpe']:.2f}** | {res_A['ann_ret']:.2%} | {res_A['ann_vol']:.2%} | {res_A['ic_mean']:+.4f} | {res_A['ic_tstat']:+.2f} |
| IC + Grouped PCA | {n_pca} | {res_B['r2']:+.4f} | **{res_B['sharpe']:.2f}** | {res_B['ann_ret']:.2%} | {res_B['ann_vol']:.2%} | {res_B['ic_mean']:+.4f} | {res_B['ic_tstat']:+.2f} |

> **Skeleton GKX NN3 benchmark** (from jkp_project_skeleton_code.ipynb): Sharpe = 2.05, IC t-stat = 6.57

### Verdict
{"IC+PCA" if res_B['sharpe'] > res_A['sharpe'] else "Baseline"} wins on Sharpe: **{max(res_A['sharpe'], res_B['sharpe']):.2f}** vs {min(res_A['sharpe'], res_B['sharpe']):.2f}

---

## Output Files

| File | Description |
|---|---|
| `ic_stats.csv` | IC t-stats for all 185 features |
| `selected_features.json` | {n_keep} IC-selected feature names |
| `ic_tstat_bar.png` | Feature ranking bar chart |
| `pca_group_summary.png` | PCA components per JKP theme |
| `nn3_cumulative.png` | Cumulative return chart (OOS) |
| `ak_results_nn3.csv` | Numeric results table |
| `ak_results.md` | This report |
"""

md_path = os.path.join(OUT_DIR, 'ak_results.md')
with open(md_path, 'w') as f:
    f.write(md)
print(f"  Report saved  → {md_path}")
print("\nDone.")
