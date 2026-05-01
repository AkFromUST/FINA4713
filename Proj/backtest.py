"""
backtest.py
FINA 4713 — Generalised long-short portfolio backtesting

PUBLIC API
----------
run_backtest(pred_df, pred_col, ...)  →  dict
evaluate_all(pred_df, ...)            →  pd.DataFrame  (one row per model)
print_summary(summary_df)            →  None

INPUT FORMAT
------------
pred_df must have:
    eom       datetime — end-of-month rebalancing date
    y_true    float    — realised excess return (ret_exc_lead1m; rf already removed)
    pred_<X>  float    — model signal; higher = more bullish

Optional:
    me_raw    float    — market cap (needed for value-weighted portfolios)
    excntry   str      — ISO country code (needed for country-neutral mode)

PORTFOLIO CONSTRUCTION
-----------------------
Each month:
  1. Rank stocks by pred_col (globally, or within-country if country_neutral=True).
  2. Long  = top    `quantile` fraction by rank.
  3. Short = bottom `quantile` fraction by rank.
  4. L/S return = weighted_mean(long y_true) − weighted_mean(short y_true).
     Weighting: 'equal' (default) or 'value' (market-cap; needs me_raw).

METRICS  (all annualised where applicable, months × 12)
---------------------------------------------------------
  cagr         Compound annual growth rate of the L/S strategy
  ann_ret      Arithmetic annualised mean monthly return
  ann_vol      Annualised return volatility (monthly std × √12)
  sharpe       ann_ret / ann_vol  (rf = 0; y_true is already excess)
  sortino      ann_ret / downside_vol  (downside = negative months only)
  max_drawdown Peak-to-trough on cumulative L/S wealth (negative number)
  calmar       CAGR / |max_drawdown|
  hit_rate     Fraction of months with positive L/S return
  n_months     Calendar months in the backtest
"""

import sys
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_mean(returns: pd.Series, weights: pd.Series | None) -> float:
    if weights is None:
        return float(returns.mean())
    w = weights.clip(lower=0)
    total = w.sum()
    if total == 0:
        return float(returns.mean())
    return float((returns * w).sum() / total)


def _monthly_ls(
    month_df:        pd.DataFrame,
    pred_col:        str,
    quantile:        float,
    weighting:       str,
    country_neutral: bool,
) -> tuple[float, float, float]:
    """Return (ls_return, long_return, short_return) for one rebalancing period."""
    df = month_df.copy()

    if country_neutral and "excntry" in df.columns:
        df["_pct"] = (
            df.groupby("excntry")[pred_col]
            .rank(pct=True, method="average")
        )
    else:
        df["_pct"] = df[pred_col].rank(pct=True, method="average")

    long_mask  = df["_pct"] >= (1.0 - quantile)
    short_mask = df["_pct"] <= quantile

    long_df  = df[long_mask]
    short_df = df[short_mask]

    if len(long_df) == 0 or len(short_df) == 0:
        return np.nan, np.nan, np.nan

    w_col = "me_raw" if (weighting == "value" and "me_raw" in df.columns) else None

    long_ret  = _weighted_mean(long_df["y_true"],  long_df[w_col]  if w_col else None)
    short_ret = _weighted_mean(short_df["y_true"], short_df[w_col] if w_col else None)

    return long_ret - short_ret, long_ret, short_ret


def _compute_metrics(ls: pd.Series) -> dict:
    """Compute annualised metrics from a monthly L/S return series."""
    r = ls.dropna()
    n = len(r)

    base = {"n_months": n}
    nans = dict.fromkeys(
        ["cagr", "ann_ret", "ann_vol", "sharpe", "sortino",
         "max_drawdown", "calmar", "hit_rate"], np.nan
    )
    if n < 2:
        return {**nans, **base}

    wealth  = (1 + r).cumprod()
    cagr    = float(wealth.iloc[-1] ** (12 / n) - 1)
    ann_ret = float(r.mean() * 12)
    ann_vol = float(r.std(ddof=1) * np.sqrt(12))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    neg      = r[r < 0]
    down_vol = float(neg.std(ddof=1) * np.sqrt(12)) if len(neg) > 1 else np.nan
    sortino  = ann_ret / down_vol if (pd.notna(down_vol) and down_vol > 0) else np.nan

    peak   = wealth.cummax()
    max_dd = float(((wealth - peak) / peak).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "cagr":         cagr,
        "ann_ret":      ann_ret,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "max_drawdown": max_dd,
        "calmar":       calmar,
        "hit_rate":     float((r > 0).mean()),
        "n_months":     n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    pred_df:         pd.DataFrame,
    pred_col:        str,
    quantile:        float = 0.1,
    weighting:       str   = "equal",
    country_neutral: bool  = False,
) -> dict:
    """
    Monthly-rebalanced long-short backtest for a single prediction column.

    Parameters
    ----------
    pred_df         DataFrame — must have eom, y_true, pred_col
                    (optional: me_raw for value-weighting, excntry for country-neutral)
    pred_col        column name of the model signal
    quantile        top/bottom fraction for L/S legs; default 0.1 = decile
    weighting       "equal" (default) or "value" (market-cap, needs me_raw)
    country_neutral if True, stocks are ranked within each country

    Returns
    -------
    dict:
        metrics          — performance dict (cagr, sharpe, …)
        monthly_returns  — pd.Series (eom index) of L/S monthly returns
        long_returns     — pd.Series of long-leg monthly returns
        short_returns    — pd.Series of short-leg monthly returns
        cumulative       — pd.Series of cumulative L/S wealth (base = 1)
    """
    missing = {"eom", "y_true", pred_col} - set(pred_df.columns)
    if missing:
        raise ValueError(f"pred_df is missing required columns: {missing}")
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1); got {quantile}")

    rows = []
    for eom, grp in pred_df.groupby("eom"):
        valid = grp.dropna(subset=[pred_col, "y_true"])
        if len(valid) < 10:
            continue
        ls, lng, sht = _monthly_ls(valid, pred_col, quantile, weighting, country_neutral)
        rows.append({"eom": eom, "ls": ls, "long": lng, "short": sht})

    if not rows:
        raise ValueError(f"No valid rebalancing months found for '{pred_col}'")

    ret_df = pd.DataFrame(rows).set_index("eom").sort_index()
    ls_series = ret_df["ls"]

    return {
        "metrics":         _compute_metrics(ls_series),
        "monthly_returns": ls_series,
        "long_returns":    ret_df["long"],
        "short_returns":   ret_df["short"],
        "cumulative":      (1 + ls_series.dropna()).cumprod(),
    }


def evaluate_all(
    pred_df:         pd.DataFrame,
    pred_cols:       list | None = None,
    quantile:        float = 0.1,
    weighting:       str   = "equal",
    country_neutral: bool  = False,
) -> pd.DataFrame:
    """
    Run run_backtest for every pred_* column (or a specified list).

    pred_cols defaults to all columns whose name starts with 'pred_'.

    Returns a summary DataFrame with one row per model and columns:
    [cagr, ann_ret, ann_vol, sharpe, sortino, max_drawdown, calmar, hit_rate, n_months].
    """
    if pred_cols is None:
        pred_cols = [c for c in pred_df.columns if c.startswith("pred_")]
    if not pred_cols:
        raise ValueError("No pred_* columns found in pred_df and none specified.")

    _COL_ORDER = ["cagr", "ann_ret", "ann_vol", "sharpe", "sortino",
                  "max_drawdown", "calmar", "hit_rate", "n_months"]

    rows = []
    for col in pred_cols:
        try:
            result  = run_backtest(pred_df, col, quantile, weighting, country_neutral)
            metrics = result["metrics"]
        except Exception as exc:
            metrics = dict.fromkeys(_COL_ORDER, np.nan)
            metrics["error"] = str(exc)
        metrics["model"] = col.removeprefix("pred_")
        rows.append(metrics)

    summary = pd.DataFrame(rows).set_index("model")
    present = [c for c in _COL_ORDER if c in summary.columns]
    return summary[present]


def print_summary(df: pd.DataFrame) -> None:
    """Pretty-print the DataFrame returned by evaluate_all."""

    def pct(v):
        return f"{v * 100:+.1f}%" if pd.notna(v) else "    n/a"

    def x2(v):
        return f"{v:+.2f}x" if pd.notna(v) else "   n/a"

    def cnt(v):
        return str(int(v)) if pd.notna(v) else "?"

    hdr = (
        f"{'Model':<12} {'CAGR':>8} {'AnnRet':>8} {'AnnVol':>8} "
        f"{'Sharpe':>8} {'Sortino':>8} {'MaxDD':>9} {'Calmar':>8} "
        f"{'HitRate':>8} {'N':>5}"
    )
    sep = "─" * len(hdr)
    print(f"\n  {sep}")
    print(f"  {hdr}")
    print(f"  {sep}")
    for model, row in df.iterrows():
        print(
            f"  {str(model):<12}"
            f" {pct(row.get('cagr')):>9}"
            f" {pct(row.get('ann_ret')):>9}"
            f" {pct(row.get('ann_vol')):>9}"
            f" {x2(row.get('sharpe')):>9}"
            f" {x2(row.get('sortino')):>9}"
            f" {pct(row.get('max_drawdown')):>10}"
            f" {x2(row.get('calmar')):>9}"
            f" {pct(row.get('hit_rate')):>9}"
            f" {cnt(row.get('n_months')):>5}"
        )
    print(f"  {sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — python backtest.py [path/to/predictions.parquet] [options]
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    _DEFAULT = "NN/files/hunloss/results/hunloss_test_predictions.parquet"

    parser = argparse.ArgumentParser(
        description="Run long-short backtest on a predictions parquet."
    )
    parser.add_argument(
        "parquet", nargs="?", default=_DEFAULT,
        help=f"Path to predictions parquet (default: {_DEFAULT})"
    )
    parser.add_argument(
        "--quantile", type=float, default=0.1,
        help="Top/bottom fraction for L/S legs (default: 0.1 = decile)"
    )
    parser.add_argument(
        "--value-weighted", action="store_true",
        help="Weight by market cap (me_raw) instead of equal-weight"
    )
    parser.add_argument(
        "--country-neutral", action="store_true",
        help="Rank within each country before forming global legs"
    )
    args = parser.parse_args()

    print(f"\n{'=' * 62}")
    print("  FINA 4713 — Long-Short Portfolio Backtest")
    print(f"{'=' * 62}")
    print(f"  Source   : {args.parquet}")
    print(f"  Quantile : {args.quantile:.0%} ({args.quantile:.0%} long / {args.quantile:.0%} short)")
    print(f"  Weighting: {'value (me_raw)' if args.value_weighted else 'equal'}")
    print(f"  Neutral  : {'country-neutral' if args.country_neutral else 'global ranking'}")

    pred_df = pd.read_parquet(args.parquet)
    print(f"  Loaded   : {len(pred_df):,} rows  |  "
          f"{pred_df['eom'].min().date()} → {pred_df['eom'].max().date()}")

    summary = evaluate_all(
        pred_df,
        quantile        = args.quantile,
        weighting       = "value" if args.value_weighted else "equal",
        country_neutral = args.country_neutral,
    )

    print_summary(summary)
