# Project Story

## The Problem

In this project, I wanted to investigate one of the most important practical issues in portfolio management: why portfolios that look optimal in-sample often disappoint out-of-sample. Mean-variance optimization is a powerful theoretical framework, but it is only as reliable as its inputs. Expected returns and covariances are estimated from historical data, and those estimates contain noise. A portfolio optimizer can amplify that noise, producing concentrated allocations and attractive historical Sharpe ratios that may not survive in live investment settings.

## The Approach

I began with the classical Markowitz framework using 30 Kenneth French industry portfolios from January 1980 to December 2025. I constructed efficient frontiers, identified Global Minimum Variance portfolios, and estimated tangency portfolios using the risk-free rate. From there, I introduced Bayes-Stein mean shrinkage and Ledoit-Wolf covariance shrinkage to address parameter uncertainty. These methods were chosen because they directly target the estimation-error problem that affects mean-variance optimization.

I also extended the project into factor investing by comparing industry portfolios with Fama-French 3-factor and 5-factor opportunity sets. This allowed me to evaluate whether systematic factor exposures provide a stronger investment opportunity set than industry portfolios alone. Alternatives such as Black-Litterman or robust optimization would also have been valid, but shrinkage, resampling, and constrained optimization were more directly aligned with the assignment and the estimation-risk theme.

## The Execution

The most technically challenging part was maintaining consistency across different return definitions and sample periods. Industry portfolios are total returns, while Fama-French factors are excess returns, so I converted returns carefully before comparing frontiers. I also split the data into an in-sample period ending in December 2002 and an out-of-sample period ending in December 2025. This made it possible to estimate weights using only historical information and then test whether those weights remained efficient later.

I implemented constrained optimization using SLSQP, evaluated Capital Market Lines, compared ETF proxies with theoretical factor portfolios, and assessed mutual funds using CAPM and multi-factor regressions. The project also included statistical testing through Sharpe ratio comparisons and bootstrap-style robustness logic.

## The Outcome

The results confirmed the core intuition: sample mean-variance optimization is often too optimistic. Bayes-Stein shrinkage created more conservative expected returns, while Ledoit-Wolf covariance shrinkage had a smaller effect because the sample size was relatively large. The Global Minimum Variance portfolio proved more robust out-of-sample because it does not depend on noisy expected return estimates. Factor portfolios improved diversification, while constrained and resampled approaches produced more realistic investment allocations.

If I extended this project, I would add transaction costs, turnover limits, rolling-window rebalancing, and Black-Litterman investor views. These additions would make the framework closer to an institutional portfolio construction process.

## The Skill Signal

This project demonstrates my ability to connect financial theory with practical quantitative implementation. It shows competence in portfolio optimization, empirical asset pricing, statistical testing, Python programming, data cleaning, and investment interpretation. More importantly, it shows that I understand the difference between a mathematically optimal portfolio and a portfolio that is robust enough for real-world risk management.
