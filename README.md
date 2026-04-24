# Portfolio Choice and Performance Evaluation
> Mean-variance portfolio optimization with shrinkage, factor models, and out-of-sample performance testing.

## Background & Motivation

Portfolio construction is one of the central problems in quantitative finance. Modern Portfolio Theory provides an elegant framework for choosing asset weights by balancing expected return against risk, but real-world implementation is difficult because expected returns and covariances must be estimated from noisy historical data. Small estimation errors can create extreme portfolio weights and exaggerated in-sample Sharpe ratios.

This project studies that problem in a realistic investment management setting. Using Kenneth French industry and factor data, it compares classical mean-variance optimization with shrinkage estimators, factor-based opportunity sets, practical ETF proxies, constrained optimization, and out-of-sample performance testing. The goal is to understand which portfolio methods are theoretically attractive and which remain robust when tested on unseen data.

## Theory & Methodology

### Mean-Variance Optimization

In plain English, mean-variance optimization finds the portfolio with the lowest risk for a chosen expected return.

\[
E[R_p] = \mathbf{w}^{\top}\boldsymbol{\mu}
\]

where \(E[R_p]\) is expected portfolio return, \(\mathbf{w}\) is the vector of asset weights, and \(\boldsymbol{\mu}\) is the vector of expected asset returns.

\[
\sigma_p^2 = \mathbf{w}^{\top}\boldsymbol{\Sigma}\mathbf{w}
\]

where \(\sigma_p^2\) is portfolio variance and \(\boldsymbol{\Sigma}\) is the covariance matrix of asset returns.

The efficient frontier is obtained by solving:

\[
\min_{\mathbf{w}} \mathbf{w}^{\top}\boldsymbol{\Sigma}\mathbf{w}
\]

subject to:

\[
\mathbf{w}^{\top}\boldsymbol{\mu} = \mu_p,\quad \mathbf{w}^{\top}\mathbf{1}=1
\]

where \(\mu_p\) is the target portfolio return and \(\mathbf{1}\) is a vector of ones.

### Global Minimum Variance Portfolio

The Global Minimum Variance portfolio is the lowest-risk portfolio available from the asset universe. It does not require expected return estimates, making it more stable than return-driven portfolios.

\[
\mathbf{w}_{GMV} =
\frac{\boldsymbol{\Sigma}^{-1}\mathbf{1}}
{\mathbf{1}^{\top}\boldsymbol{\Sigma}^{-1}\mathbf{1}}
\]

where \(\mathbf{w}_{GMV}\) is the GMV weight vector.

### Tangency Portfolio and Sharpe Ratio

The tangency portfolio maximizes compensation per unit of risk.

\[
SR_p = \frac{E[R_p] - R_f}{\sigma_p}
\]

where \(SR_p\) is the Sharpe ratio, \(R_f\) is the risk-free rate, and \(\sigma_p\) is portfolio volatility.

### Capital Market Line

The Capital Market Line combines the risk-free asset with the tangency portfolio.

\[
E[R_c] = R_f + \frac{E[R_T] - R_f}{\sigma_T}\sigma_c
\]

where \(E[R_T]\) and \(\sigma_T\) are the tangency portfolio return and volatility, and \(\sigma_c\) is the chosen complete portfolio volatility.

### Bayes-Stein Mean Shrinkage

Bayes-Stein shrinkage reduces noisy expected return estimates by pulling sample means toward a common target, here the GMV portfolio mean.

\[
\hat{\boldsymbol{\mu}}_{BS}
=
(1-\phi)\hat{\boldsymbol{\mu}}
+
\phi\mu_{GMV}\mathbf{1}
\]

where \(\hat{\boldsymbol{\mu}}_{BS}\) is the shrinkage mean vector, \(\hat{\boldsymbol{\mu}}\) is the sample mean vector, \(\phi\) is the shrinkage intensity, and \(\mu_{GMV}\) is the GMV expected return.

### Ledoit-Wolf Covariance Shrinkage

Ledoit-Wolf shrinkage stabilizes the covariance matrix by blending the sample covariance with a structured target.

\[
\hat{\boldsymbol{\Sigma}}_{LW}
=
\delta\mathbf{F} + (1-\delta)\mathbf{S}
\]

where \(\hat{\boldsymbol{\Sigma}}_{LW}\) is the shrunk covariance matrix, \(\delta\) is the shrinkage intensity, \(\mathbf{F}\) is the target matrix, and \(\mathbf{S}\) is the sample covariance matrix.

### Factor Models

The CAPM explains excess return using market exposure:

\[
R_{p,t}-R_{f,t}
=
\alpha_p + \beta_p(R_{M,t}-R_{f,t})+\epsilon_{p,t}
\]

where \(\alpha_p\) is abnormal return, \(\beta_p\) is market beta, and \(\epsilon_{p,t}\) is residual return.

The Fama-French 3-factor model adds size and value:

\[
R_{p,t}-R_{f,t}
=
\alpha_p+\beta_M MKT_t+\beta_S SMB_t+\beta_H HML_t+\epsilon_t
\]

The Fama-French 5-factor model adds profitability and investment:

\[
R_{p,t}-R_{f,t}
=
\alpha_p+\beta_M MKT_t+\beta_S SMB_t+\beta_H HML_t+\beta_R RMW_t+\beta_C CMA_t+\epsilon_t
\]

where \(MKT\) is market excess return, \(SMB\) is size, \(HML\) is value, \(RMW\) is profitability, and \(CMA\) is investment.

## Dataset

- Source: Kenneth French Data Library
- Size: 552 monthly observations
- Time period: January 1980 to December 2025
- Features:
  - 30 industry portfolio returns
  - Fama-French 3 factors: Mkt-RF, SMB, HML
  - Fama-French 5 factors: Mkt-RF, SMB, HML, RMW, CMA
  - Risk-free rate
- Additional optional data: Yahoo Finance prices for stocks, ETF proxies, and mutual funds
- Preprocessing:
  - Parsed dates into a monthly time-series index
  - Filtered all datasets to the same sample period
  - Converted industry returns to excess returns where factor comparisons required it
  - Aligned monthly observations before regression and performance testing

## Project Structure

```text
portfolio-choice-performance/
├── main.py
├── notebook.ipynb
├── README.md
├── story.md
├── requirements.txt
└── data/ (if applicable)
```

## How to Run

```bash
git clone https://github.com/YOUR-USERNAME/portfolio-choice-performance.git
cd portfolio-choice-performance
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Key Results & Findings

- Sample mean-variance optimization produced the most optimistic in-sample frontier.
- Bayes-Stein shrinkage reduced expected-return estimation error and produced more conservative frontiers.
- Ledoit-Wolf covariance shrinkage was modest because the sample contains 552 monthly observations for 30 industry portfolios.
- Industry portfolios dominated individual stocks because diversification reduces idiosyncratic risk.
- Fama-French factor portfolios expanded the investment opportunity set relative to industry-only portfolios.
- GMV portfolios were more robust out-of-sample because they rely on covariance estimates rather than noisy mean estimates.
- Tangency portfolios showed greater out-of-sample degradation, consistent with error maximization.
- Constrained and resampled portfolios sacrificed in-sample Sharpe ratios for more realistic and stable investment allocations.

## Academic Context

This project was completed for FIN41360 Portfolio and Risk Management at UCD Michael Smurfit Graduate Business School. It forms part of the MSc Quantitative Finance curriculum by combining portfolio theory, empirical asset pricing, Python programming, statistical testing, and investment risk management.

Assumption: the academic year is treated as 2025/26 because the dataset extends to December 2025.

## References

1. Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.
2. Jorion, P. (1986). Bayes-Stein Estimation for Portfolio Analysis. Journal of Financial and Quantitative Analysis, 21(3), 279-292.
3. Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2), 365-411.
4. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics, 33(1), 3-56.
5. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1-22.
6. Kenneth R. French Data Library.
