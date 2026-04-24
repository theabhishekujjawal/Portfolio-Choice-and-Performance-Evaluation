# Project Brief

## Project Goal and Problem Statement

This project evaluates portfolio construction and performance using Modern Portfolio Theory, shrinkage estimators, factor models, practical ETF proxies, out-of-sample testing, and mutual fund benchmarking. The central problem is whether mean-variance optimized portfolios that look efficient in-sample remain robust out-of-sample when expected returns and covariances are estimated with noise.

## Dataset Used

- Source: Kenneth French Data Library
- Sample period: January 1980 to December 2025
- Frequency: Monthly
- Size: 552 monthly observations
- Files:
  - `industry_portfolios_monthly.csv`: 30 industry portfolio returns
  - `ff3_factors_monthly.csv`: Mkt-RF, SMB, HML, RF
  - `ff5_factors_monthly.csv`: Mkt-RF, SMB, HML, RMW, CMA, RF
  - `risk_free_rate_monthly.csv`: monthly risk-free rate
- Optional external data: Yahoo Finance prices for selected individual stocks, ETF proxies, and mutual funds.

## Methods and Mathematical Basis

- Mean-variance optimization: minimizes \(\mathbf{w}^{\top}\boldsymbol{\Sigma}\mathbf{w}\) subject to a target return and full-investment constraint.
- Global Minimum Variance portfolio: \(\mathbf{w}_{GMV} = \boldsymbol{\Sigma}^{-1}\mathbf{1}/(\mathbf{1}^{\top}\boldsymbol{\Sigma}^{-1}\mathbf{1})\).
- Tangency portfolio: maximizes \(SR = (E[R_p]-R_f)/\sigma_p\).
- Capital Market Line: combines the risk-free asset and tangency portfolio.
- Bayes-Stein shrinkage: shrinks sample expected returns toward the GMV mean.
- Ledoit-Wolf shrinkage: shrinks the sample covariance matrix toward a structured target.
- Fama-French factor models: compare industry assets with market, size, value, profitability, and investment factor portfolios.
- Out-of-sample testing: estimates weights on 1980-2002 data and evaluates them on 2003-2025 data.
- Jobson-Korkie test: compares Sharpe ratios statistically.
- Mutual fund regressions: estimate Jensen's alpha, beta exposures, t-statistics, and Information Ratios.

## Key Results and Conclusions

- Sample mean-variance frontiers are optimistic and vulnerable to estimation error.
- Bayes-Stein shrinkage materially reduces expected-return optimism.
- Ledoit-Wolf covariance shrinkage is modest because the sample contains many observations relative to the number of assets.
- Industry portfolios dominate individual stocks because idiosyncratic risk is diversified away.
- FF5 factors offer a richer opportunity set than FF3 factors.
- GMV portfolios are more robust out-of-sample than tangency portfolios.
- Constrained and resampled portfolios sacrifice in-sample performance for more realistic allocations.
- Mutual fund performance must be judged relative to market, factor, and tangency portfolio benchmarks.

## Libraries and Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `yfinance`
- Python standard library: `pathlib`, `warnings`
