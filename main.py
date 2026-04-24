"""
Project: Portfolio Choice and Performance Evaluation
Author: Abhishek Ujjawal
Date: 2026-04-24
Description:
    Clean Python implementation of a FIN41360 Portfolio and Risk Management
    project. The script analyses Kenneth French industry and factor data,
    constructs efficient frontiers, applies shrinkage estimators, evaluates
    out-of-sample robustness, and optionally benchmarks mutual funds using
    Yahoo Finance data.
"""

# ============================================================
# Imports and Configuration
# ============================================================

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
START_DATE = "1980-01-01"
END_DATE = "2025-12-31"
SPLIT_DATE = "2002-12-31"

plt.rcParams.update(
    {
        "figure.figsize": (12, 8),
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


# ============================================================
# Data Loading
# ============================================================

def resolve_data_file(filename: str) -> Path:
    """Find a data file in ./data first, then in the project root."""
    data_path = DATA_DIR / filename
    root_path = PROJECT_ROOT / filename
    if data_path.exists():
        return data_path
    if root_path.exists():
        return root_path
    raise FileNotFoundError(f"Could not find {filename} in {DATA_DIR} or {PROJECT_ROOT}")


def load_and_clean_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(resolve_data_file(filename))
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df.loc[START_DATE:END_DATE]


def load_project_data():
    industry_df = load_and_clean_data("industry_portfolios_monthly.csv")
    ff3_df = load_and_clean_data("ff3_factors_monthly.csv")
    ff5_df = load_and_clean_data("ff5_factors_monthly.csv")
    rf = ff3_df["RF"]
    return industry_df, ff3_df, ff5_df, rf


# ============================================================
# Portfolio Statistics and Optimisation
# ============================================================

def portfolio_stats(weights, mu, cov, rf=0.0):
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights.T @ cov @ weights))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def find_gmv_portfolio(mu, cov, bounds=None):
    n_assets = len(mu)
    bounds = bounds or [(-1.0, 2.0)] * n_assets
    init = np.ones(n_assets) / n_assets
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        lambda w: w.T @ cov @ w,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"GMV optimisation failed: {result.message}")
    return result.x


def find_tangency_portfolio(mu, cov, rf, bounds=None):
    n_assets = len(mu)
    bounds = bounds or [(-1.0, 2.0)] * n_assets
    init = np.ones(n_assets) / n_assets
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def negative_sharpe(w):
        ret, vol, _ = portfolio_stats(w, mu, cov, rf)
        return -((ret - rf) / vol) if vol > 0 else 1e6

    result = minimize(
        negative_sharpe,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Tangency optimisation failed: {result.message}")
    return result.x


def efficient_frontier(mu, cov, n_points=80, bounds=None):
    n_assets = len(mu)
    bounds = bounds or [(-1.0, 2.0)] * n_assets
    init = np.ones(n_assets) / n_assets
    sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    w_gmv = find_gmv_portfolio(mu, cov, bounds)
    gmv_return = w_gmv @ mu
    max_return = float(np.max(mu) * 1.2)
    targets = np.linspace(gmv_return, max_return, n_points)

    frontier_returns, frontier_vols, frontier_weights = [], [], []
    for target in targets:
        constraints = [
            sum_constraint,
            {"type": "eq", "fun": lambda w, target=target: w @ mu - target},
        ]
        result = minimize(
            lambda w: w.T @ cov @ w,
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if result.success:
            frontier_returns.append(target)
            frontier_vols.append(np.sqrt(result.fun))
            frontier_weights.append(result.x)

    return np.array(frontier_returns), np.array(frontier_vols), frontier_weights


# ============================================================
# Shrinkage Estimators
# ============================================================

def bayes_stein_shrinkage(returns: pd.DataFrame):
    observations, n_assets = returns.shape
    mu = returns.mean().values
    cov = returns.cov().values
    inv_cov = np.linalg.pinv(cov)
    ones = np.ones(n_assets)

    w_gmv = inv_cov @ ones / (ones.T @ inv_cov @ ones)
    mu_gmv = w_gmv @ mu
    diff = mu - mu_gmv * ones
    q_stat = diff.T @ inv_cov @ diff
    phi = (n_assets + 2) / ((n_assets + 2) + observations * q_stat)
    mu_bs = (1 - phi) * mu + phi * mu_gmv * ones

    return mu_bs, phi


def ledoit_wolf_shrinkage(returns: pd.DataFrame):
    x = returns.values
    observations, n_assets = x.shape
    centered = x - x.mean(axis=0)

    sample_cov = centered.T @ centered / observations
    target_scale = np.trace(sample_cov) / n_assets
    target = target_scale * np.eye(n_assets)

    denominator = np.sum((sample_cov - target) ** 2)
    beta = 0.0
    for row in centered:
        beta += np.sum((np.outer(row, row) - sample_cov) ** 2)
    beta /= observations ** 2

    delta = min(1.0, max(0.0, beta / denominator)) if denominator > 0 else 0.0
    shrunk_cov = delta * target + (1 - delta) * sample_cov
    shrunk_cov *= observations / (observations - 1)
    return shrunk_cov, delta


# ============================================================
# Plotting Helpers
# ============================================================

def plot_frontiers(frontiers, title, filename):
    plt.figure()
    for label, returns, vols in frontiers:
        plt.plot(vols, returns, linewidth=2, label=label)

    plt.xlabel("Monthly volatility")
    plt.ylabel("Monthly return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / filename)
    plt.close()


# ============================================================
# Efficient Frontiers and Shrinkage Analysis
# ============================================================

def run_efficient_frontier_analysis(industry_df, rf):
    rf_mean = rf.mean()
    mu_sample = industry_df.mean().values
    cov_sample = industry_df.cov().values

    ret_sample, vol_sample, _ = efficient_frontier(mu_sample, cov_sample)

    mu_bs, phi = bayes_stein_shrinkage(industry_df)
    ret_bs, vol_bs, _ = efficient_frontier(mu_bs, cov_sample)

    cov_lw, delta = ledoit_wolf_shrinkage(industry_df)
    ret_bslw, vol_bslw, _ = efficient_frontier(mu_bs, cov_lw)

    w_gmv = find_gmv_portfolio(mu_sample, cov_sample)
    w_tan = find_tangency_portfolio(mu_sample, cov_sample, rf_mean)

    print("\nTASK 2: Mean-Variance Efficient Frontiers")
    print(f"Bayes-Stein shrinkage intensity phi: {phi:.4f}")
    print(f"Ledoit-Wolf shrinkage intensity delta: {delta:.4f}")
    print(f"Sample GMV return, vol, Sharpe: {portfolio_stats(w_gmv, mu_sample, cov_sample, rf_mean)}")
    print(f"Sample tangency return, vol, Sharpe: {portfolio_stats(w_tan, mu_sample, cov_sample, rf_mean)}")

    plot_frontiers(
        [
            ("Sample estimates", ret_sample, vol_sample),
            ("Bayes-Stein means", ret_bs, vol_bs),
            ("Bayes-Stein + Ledoit-Wolf", ret_bslw, vol_bslw),
        ],
        "Efficient Frontiers: 30 Industry Portfolios",
        "figure_1_efficient_frontiers.png",
    )

    return {
        "mu_sample": mu_sample,
        "cov_sample": cov_sample,
        "mu_bs": mu_bs,
        "cov_lw": cov_lw,
        "w_tangency": w_tan,
    }


# ============================================================
# Factor Portfolio Analysis
# ============================================================

def run_factor_frontier_analysis(industry_df, ff5_df, rf):
    industry_excess = industry_df.sub(rf, axis=0)
    ff3_cols = ["Mkt-RF", "SMB", "HML"]
    ff5_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    universes = {
        "30 industries excess": industry_excess,
        "FF3 factors": ff5_df[ff3_cols],
        "FF5 factors": ff5_df[ff5_cols],
    }

    frontiers = []
    print("\nTASK 5: Factor Opportunity Sets")
    for label, data in universes.items():
        mu = data.mean().values
        cov = data.cov().values
        rets, vols, _ = efficient_frontier(mu, cov)
        w_tan = find_tangency_portfolio(mu, cov, rf=0.0)
        _, _, sr = portfolio_stats(w_tan, mu, cov, rf=0.0)
        print(f"{label}: tangency Sharpe = {sr:.4f}")
        frontiers.append((label, rets, vols))

    plot_frontiers(
        frontiers,
        "Industry and Fama-French Factor Frontiers",
        "figure_2_factor_frontiers.png",
    )


# ============================================================
# Out-of-Sample Performance Evaluation
# ============================================================

def frontier_return_at_vol(frontier_returns, frontier_vols, target_vol):
    gmv_idx = int(np.argmin(frontier_vols))
    upper_vols = frontier_vols[gmv_idx:]
    upper_returns = frontier_returns[gmv_idx:]
    if target_vol < upper_vols.min() or target_vol > upper_vols.max():
        return np.nan
    return float(np.interp(target_vol, upper_vols, upper_returns))


def jobson_korkie_test(r1, r2):
    r1 = pd.Series(r1).dropna()
    r2 = pd.Series(r2).dropna()
    aligned = pd.concat([r1, r2], axis=1).dropna()
    x = aligned.iloc[:, 0]
    y = aligned.iloc[:, 1]

    sr_x = x.mean() / x.std(ddof=1)
    sr_y = y.mean() / y.std(ddof=1)
    se = np.sqrt((1 + 0.5 * sr_x ** 2 + 1 + 0.5 * sr_y ** 2) / len(aligned))
    z_stat = (sr_x - sr_y) / se if se > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value


def run_oos_analysis(industry_df, rf):
    train = industry_df.loc[:SPLIT_DATE]
    test = industry_df.loc["2003-01-01":]
    rf_train_mean = rf.loc[:SPLIT_DATE].mean()
    rf_test = rf.loc[test.index]

    mu_train = train.mean().values
    cov_train = train.cov().values

    w_gmv = find_gmv_portfolio(mu_train, cov_train)
    w_tan = find_tangency_portfolio(mu_train, cov_train, rf_train_mean)
    w_mid = 0.5 * w_gmv + 0.5 * w_tan
    w_mid = w_mid / w_mid.sum()

    mu_test = test.mean().values
    cov_test = test.cov().values
    frontier_rets, frontier_vols, _ = efficient_frontier(mu_test, cov_test)

    oos_returns = {
        "GMV": pd.Series(test.values @ w_gmv, index=test.index),
        "Tangency": pd.Series(test.values @ w_tan, index=test.index),
        "Mid-risk": pd.Series(test.values @ w_mid, index=test.index),
    }

    print("\nTASK 7: Out-of-Sample Performance")
    for name, series in oos_returns.items():
        excess = series - rf_test
        ret = series.mean()
        vol = series.std(ddof=1)
        sharpe = excess.mean() / excess.std(ddof=1)
        frontier_ret = frontier_return_at_vol(frontier_rets, frontier_vols, vol)
        print(
            f"{name}: return={ret:.4f}, vol={vol:.4f}, "
            f"Sharpe={sharpe:.4f}, frontier return at same vol={frontier_ret:.4f}"
        )

    z_stat, p_value = jobson_korkie_test(oos_returns["GMV"], oos_returns["Tangency"])
    print(f"Jobson-Korkie GMV vs Tangency: z={z_stat:.4f}, p={p_value:.4f}")


# ============================================================
# Advanced Optimisation Techniques
# ============================================================

def resampled_frontier(returns, n_points=30, n_sims=50):
    mu = returns.mean().values
    cov = returns.cov().values
    base_returns, _, _ = efficient_frontier(mu, cov, n_points=n_points)

    resampled_returns = []
    resampled_vols = []
    for _ in base_returns:
        weights = []
        for _sim in range(n_sims):
            sample = returns.sample(len(returns), replace=True)
            mu_b = sample.mean().values
            cov_b = sample.cov().values
            try:
                w = find_tangency_portfolio(mu_b, cov_b, rf=0.0)
                weights.append(w)
            except RuntimeError:
                continue

        if weights:
            w_avg = np.mean(weights, axis=0)
            w_avg = w_avg / w_avg.sum()
            ret, vol, _ = portfolio_stats(w_avg, mu, cov)
            resampled_returns.append(ret)
            resampled_vols.append(vol)

    return np.array(resampled_returns), np.array(resampled_vols)


def run_advanced_optimisation(industry_df):
    train = industry_df.loc[:SPLIT_DATE]
    mu = train.mean().values
    cov = train.cov().values

    uncon_rets, uncon_vols, _ = efficient_frontier(mu, cov)
    constrained_rets, constrained_vols, _ = efficient_frontier(
        mu,
        cov,
        bounds=[(0.0, 0.10)] * len(mu),
    )
    resampled_rets, resampled_vols = resampled_frontier(train)

    print("\nTASK 8: Advanced Optimisation")
    print("Compared unconstrained, long-only 10% cap, and resampled frontiers.")

    plot_frontiers(
        [
            ("Unconstrained", uncon_rets, uncon_vols),
            ("Long-only 10% cap", constrained_rets, constrained_vols),
            ("Resampled", resampled_rets, resampled_vols),
        ],
        "Advanced Portfolio Optimisation",
        "figure_3_advanced_optimisation.png",
    )


# ============================================================
# Mutual Fund Performance Assessment
# ============================================================

def download_monthly_returns(tickers, start, end):
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is not installed.")

    prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(prices.columns, pd.MultiIndex):
        close = prices["Close"]
    else:
        close = prices[["Close"]].rename(columns={"Close": tickers[0]})

    return close.resample("ME").last().pct_change().dropna(how="all")


def run_regression(y, x):
    y = np.asarray(y)
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    x_design = np.column_stack([np.ones(len(x)), x])
    beta_hat, _, _, _ = np.linalg.lstsq(x_design, y, rcond=None)
    residuals = y - x_design @ beta_hat
    mse = np.sum(residuals ** 2) / (len(y) - x_design.shape[1])
    var_cov = mse * np.linalg.inv(x_design.T @ x_design)

    alpha = beta_hat[0]
    t_alpha = alpha / np.sqrt(var_cov[0, 0])
    tracking_error = np.sqrt(mse)
    information_ratio = alpha / tracking_error if tracking_error > 0 else np.nan
    return alpha, beta_hat[1:], t_alpha, information_ratio


def run_mutual_fund_assessment(ff3_df):
    if not YFINANCE_AVAILABLE:
        print("\nTASK 9 skipped: yfinance is not installed.")
        return

    funds = ["VFINX", "FMAGX", "AGTHX"]
    returns = download_monthly_returns(funds, "1976-01-01", "2025-12-31")

    ff = ff3_df[["Mkt-RF", "RF"]].copy()
    ff.index = pd.date_range("1980-01-31", periods=len(ff), freq="ME")

    print("\nTASK 9: Mutual Fund CAPM Assessment")
    for fund in funds:
        if fund not in returns.columns:
            continue

        aligned = pd.concat([returns[fund], ff], axis=1).dropna()
        excess_fund = aligned[fund] - aligned["RF"]
        alpha, beta, t_alpha, ir = run_regression(excess_fund, aligned[["Mkt-RF"]])
        print(
            f"{fund}: alpha={alpha:.4f}, beta={beta[0]:.4f}, "
            f"t(alpha)={t_alpha:.2f}, IR={ir:.4f}"
        )


# ============================================================
# Main Execution
# ============================================================

def main():
    industry_df, ff3_df, ff5_df, rf = load_project_data()

    print("Portfolio Choice and Performance Evaluation")
    print(f"Sample: {industry_df.index.min().date()} to {industry_df.index.max().date()}")
    print(f"Monthly observations: {len(industry_df)}")
    print(f"Average monthly risk-free rate: {rf.mean():.4%}")

    run_efficient_frontier_analysis(industry_df, rf)
    run_factor_frontier_analysis(industry_df, ff5_df, rf)
    run_oos_analysis(industry_df, rf)
    run_advanced_optimisation(industry_df)
    run_mutual_fund_assessment(ff3_df)


if __name__ == "__main__":
    main()
