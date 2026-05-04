# =============================================================================
# utils.py — shared constants, classes, and functions for all stage scripts
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List

TARGET    = 'ret_exc_lead1m'
TRAIN_END = '2015-12-31'
VAL_END   = '2018-12-31'
META      = ['id', 'eom', 'excntry', TARGET, 'me']
MIN_STOCKS_PER_MONTH = 20


def fit_preprocessor(X_df):
    low  = X_df.quantile(0.01)
    high = X_df.quantile(0.99)
    X_clipped = X_df.clip(lower=low, upper=high, axis=1)
    imp = SimpleImputer(strategy='median').fit(X_clipped)
    sc  = StandardScaler().fit(imp.transform(X_clipped))
    return low, high, imp, sc


def apply_preprocessor(X_df, low, high, imp, sc):
    X = X_df.clip(lower=low, upper=high, axis=1)
    X = imp.transform(X)
    return sc.transform(X)


def apply_signed_log1p(df_in, log_feats):
    """sign(x)*log1p(|x|) — monotone, so Spearman IC is invariant."""
    df_out = df_in.copy()
    present = [f for f in log_feats if f in df_out.columns]
    if present:
        X = df_out[present].apply(pd.to_numeric, errors='coerce')
        df_out[present] = np.sign(X) * np.log1p(np.abs(X))
    return df_out


def portfolio_weights(pred, max_w=0.05):
    n  = len(pred)
    w  = pd.Series(pred).rank() - (n + 1) / 2
    w /= w.abs().sum()
    w  = w.clip(-max_w, max_w)
    w /= w.abs().sum()
    return w.values


def oos_r2(y_true, y_pred, y_null):
    return 1 - np.mean((y_true - y_pred)**2) / np.mean((y_true - y_null)**2)


def load_and_prep_df(parquet_path, log_features):
    """Load parquet, add log_me, drop NaN target, apply log transform."""
    df = pd.read_parquet(parquet_path)
    df['log_me'] = np.log1p(df['me'].clip(lower=0))
    df = df.dropna(subset=[TARGET])
    df = apply_signed_log1p(df, log_features)
    return df


FEATURE_GROUPS = {
    'momentum': [
        'ret_2_0', 'ret_3_0', 'ret_3_1', 'ret_6_0', 'ret_6_1',
        'ret_9_0', 'ret_9_1', 'ret_12_0', 'ret_12_1', 'ret_12_7',
        'ret_18_1', 'ret_24_1', 'ret_24_12', 'ret_36_1', 'ret_36_12',
        'ret_48_1', 'ret_48_12', 'ret_60_1', 'ret_60_12', 'ret_60_36',
        'resff3_6_1', 'resff3_12_1',
    ],
    'short_term_reversal': ['ret_1_0'],
    'seasonality': [
        'seas_1_1an', 'seas_1_1na', 'seas_2_5an', 'seas_2_5na',
        'seas_6_10an', 'seas_6_10na',
    ],
    'value': [
        'be_me', 'at_me', 'sale_me', 'ni_me', 'ocf_me', 'fcf_me',
        'ebitda_mev', 'bev_mev', 'eq_dur', 'ival_me', 'div12m_me',
        'eqpo_me', 'eqnpo_me', 'ebit_bev', 'cash_me', 'netis_mev', 'log_me',
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
        'coa_gr1a', 'col_gr1a', 'tax_gr1a', 'nfna_gr1a', 'fnl_gr1a', 'noa_at',
    ],
    'debt_issuance': [
        'chcsho_12m', 'eqnpo_12m', 'netis_at', 'eqnetis_at', 'dbnetis_at',
        'eqnpo_1m', 'eqnpo_3m', 'eqnpo_6m', 'div3m_me', 'div6m_me',
        'chcsho_1m', 'chcsho_3m', 'chcsho_6m', 'debt_gr3',
    ],
    'leverage': ['at_be', 'debt_me', 'netdebt_me', 'at_turnover', 'sale_bev', 'opex_at'],
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
    'other': ['cash_at', 'age', 'sale_emp', 'sale_emp_gr1'],
}


class GroupedPCA:
    """Fits independent PCA within each economic group (90% variance per group)."""
    def __init__(self, groups, var_threshold=0.90):
        self.groups = groups
        self.var_threshold = var_threshold
        self.fitted = {}
        self.output_names = []

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
            pca_full = PCA(n_components=min(len(avail), Xg.shape[0]-1)).fit(Xg)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp = int(np.searchsorted(cumvar, self.var_threshold)) + 1
            n_comp = max(1, min(n_comp, len(avail)))
            pca = PCA(n_components=n_comp).fit(Xg)
            self.fitted[grp] = (avail, pca)
            for i in range(n_comp):
                self.output_names.append(f'{grp}_PC{i+1}')
        return self

    def transform(self, X_df):
        parts = []
        for grp, (feats, pca) in self.fitted.items():
            avail = [f for f in feats if f in X_df.columns]
            Xg = X_df[avail].values if avail else np.zeros((len(X_df), len(feats)))
            parts.append(Xg if pca is None else pca.transform(Xg))
        return np.hstack(parts) if parts else np.zeros((len(X_df), 0))

    def fit_transform(self, X_df):
        return self.fit(X_df).transform(X_df)
