"""
Streamlit dashboard for Portfolio Choice and Performance Evaluation.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from main import (
    bayes_stein_shrinkage,
    efficient_frontier,
    find_gmv_portfolio,
    find_tangency_portfolio,
    ledoit_wolf_shrinkage,
    load_project_data,
    portfolio_stats,
)


st.set_page_config(
    page_title="Portfolio Choice Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLORS = {
    "frontier": "#2563eb",
    "gmv": "#f97316",
    "tangency": "#dc2626",
    "cml": "#111827",
    "equal": "#059669",
}


@st.cache_data(show_spinner=False)
def cached_project_data():
    industry_df, ff3_df, ff5_df, rf = load_project_data()
    return industry_df, ff3_df, ff5_df, rf


@st.cache_data(show_spinner=False)
def parse_uploaded_returns(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
    return df.select_dtypes(include=[np.number]).dropna(how="all")


def annualize_return(monthly_return: float) -> float:
    return (1 + monthly_return) ** 12 - 1


def annualize_vol(monthly_vol: float) -> float:
    return monthly_vol * np.sqrt(12)


def annualize_sharpe(monthly_sharpe: float) -> float:
    return monthly_sharpe * np.sqrt(12)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def build_universe(
    universe_name: str,
    industry_df: pd.DataFrame,
    ff3_df: pd.DataFrame,
    ff5_df: pd.DataFrame,
    rf: pd.Series,
    uploaded_returns: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.Series, str]:
    if uploaded_returns is not None:
        rf_custom = pd.Series(0.0, index=uploaded_returns.index, name="RF")
        return uploaded_returns.dropna(), rf_custom, "custom"

    if universe_name == "30 Industry Portfolios":
        return industry_df.copy(), rf.reindex(industry_df.index), "total"

    if universe_name == "FF3 Factors":
        cols = ["Mkt-RF", "SMB", "HML"]
        factors = ff3_df[cols].copy()
        return factors, pd.Series(0.0, index=factors.index, name="RF"), "excess"

    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    factors = ff5_df[cols].copy()
    return factors, pd.Series(0.0, index=factors.index, name="RF"), "excess"


def estimate_inputs(returns: pd.DataFrame, method: str):
    mu = returns.mean().values
    cov = returns.cov().values
    diagnostics = {"Mean estimator": "Sample mean", "Covariance estimator": "Sample covariance"}

    if method in {"Bayes-Stein", "Bayes-Stein + Ledoit-Wolf"}:
        mu, phi = bayes_stein_shrinkage(returns)
        diagnostics["Bayes-Stein phi"] = f"{phi:.4f}"
        diagnostics["Mean estimator"] = "Bayes-Stein shrinkage"

    if method == "Bayes-Stein + Ledoit-Wolf":
        cov, delta = ledoit_wolf_shrinkage(returns)
        diagnostics["Ledoit-Wolf delta"] = f"{delta:.4f}"
        diagnostics["Covariance estimator"] = "Ledoit-Wolf shrinkage"

    return mu, cov, diagnostics


def frontier_frame(frontier_returns, frontier_vols):
    return pd.DataFrame(
        {
            "Monthly Return": frontier_returns,
            "Monthly Volatility": frontier_vols,
            "Annual Return": [annualize_return(x) for x in frontier_returns],
            "Annual Volatility": [annualize_vol(x) for x in frontier_vols],
        }
    )


def weights_frame(asset_names, weights):
    df = pd.DataFrame({"Asset": asset_names, "Weight": weights})
    df["AbsWeight"] = df["Weight"].abs()
    return df.sort_values("AbsWeight", ascending=False).drop(columns="AbsWeight")


def portfolio_return_series(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return pd.Series(returns.values @ weights, index=returns.index)


def metric_table(rows):
    return pd.DataFrame(rows).set_index("Portfolio")


def plot_frontier(frontier_df, points_df, rf_monthly):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier_df["Annual Volatility"],
            y=frontier_df["Annual Return"],
            mode="lines",
            name="Efficient frontier",
            line=dict(color=COLORS["frontier"], width=3),
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    tangency = points_df.loc[points_df["Portfolio"] == "Tangency"].iloc[0]
    cml_vols = np.linspace(0, max(frontier_df["Annual Volatility"].max(), tangency["Annual Volatility"]) * 1.05, 80)
    rf_annual = annualize_return(rf_monthly)
    slope = (tangency["Annual Return"] - rf_annual) / tangency["Annual Volatility"]
    fig.add_trace(
        go.Scatter(
            x=cml_vols,
            y=rf_annual + slope * cml_vols,
            mode="lines",
            name="Capital Market Line",
            line=dict(color=COLORS["cml"], width=2, dash="dash"),
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    for _, row in points_df.iterrows():
        color = COLORS.get(row["Key"], "#6b7280")
        fig.add_trace(
            go.Scatter(
                x=[row["Annual Volatility"]],
                y=[row["Annual Return"]],
                mode="markers+text",
                name=row["Portfolio"],
                text=[row["Portfolio"]],
                textposition="top center",
                marker=dict(size=13, color=color, line=dict(width=1, color="white")),
                hovertemplate=(
                    f"{row['Portfolio']}<br>"
                    "Vol: %{x:.2%}<br>"
                    "Return: %{y:.2%}<br>"
                    f"Sharpe: {row['Annual Sharpe']:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Annualized volatility",
        yaxis_title="Annualized return",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


def plot_weights(weights_df, title):
    shown = weights_df.head(20).sort_values("Weight")
    fig = px.bar(
        shown,
        x="Weight",
        y="Asset",
        orientation="h",
        title=title,
        color="Weight",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20), coloraxis_showscale=False)
    fig.update_xaxes(tickformat=".0%")
    return fig


def download_weights_button(weights_df, filename):
    csv = weights_df[["Asset", "Weight"]].to_csv(index=False)
    st.download_button(
        "Download weights CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


st.title("Portfolio Choice and Performance Evaluation")
st.caption("Interactive dashboard for mean-variance optimization, shrinkage estimators, factor portfolios, and out-of-sample robustness.")

industry_df, ff3_df, ff5_df, rf = cached_project_data()

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Optional: upload monthly returns CSV", type=["csv"])
    uploaded_returns = None
    if uploaded_file is not None:
        uploaded_returns = parse_uploaded_returns(uploaded_file.getvalue())
        st.success(f"Loaded custom returns: {uploaded_returns.shape[0]} rows, {uploaded_returns.shape[1]} assets")

    universe_name = st.selectbox(
        "Asset universe",
        ["30 Industry Portfolios", "FF3 Factors", "FF5 Factors"],
        disabled=uploaded_returns is not None,
    )
    estimator = st.selectbox(
        "Estimator",
        ["Sample", "Bayes-Stein", "Bayes-Stein + Ledoit-Wolf"],
        index=2,
    )
    constraint_mode = st.selectbox(
        "Portfolio constraints",
        ["Allow shorting (-100% to 200%)", "Long-only", "Long-only with max weight"],
        index=1,
    )

returns, rf_series, return_type = build_universe(universe_name, industry_df, ff3_df, ff5_df, rf, uploaded_returns)
asset_names = list(returns.columns)
n_assets = len(asset_names)

with st.sidebar:
    min_cap = float(np.ceil((1 / n_assets) * 100) / 100)
    max_weight = 1.0
    if constraint_mode == "Long-only with max weight":
        max_weight = st.slider(
            "Maximum single-asset weight",
            min_value=min_cap,
            max_value=1.0,
            value=max(min_cap, min(0.20, 1.0)),
            step=0.01,
            format="%.2f",
        )

    frontier_points = st.slider("Frontier points", min_value=25, max_value=120, value=60, step=5)
    split_year = st.slider("OOS split year", min_value=1990, max_value=2020, value=2002, step=1)

if constraint_mode == "Allow shorting (-100% to 200%)":
    bounds = [(-1.0, 2.0)] * n_assets
elif constraint_mode == "Long-only":
    bounds = [(0.0, 1.0)] * n_assets
else:
    bounds = [(0.0, max_weight)] * n_assets

if returns.shape[0] < 36:
    st.error("The selected dataset needs at least 36 monthly observations for a useful portfolio analysis.")
    st.stop()

if sum(high for _, high in bounds) < 1:
    st.error("The selected max-weight constraint is infeasible because total allowed weight is below 100%.")
    st.stop()

rf_mean = float(rf_series.reindex(returns.index).fillna(0.0).mean())
mu, cov, diagnostics = estimate_inputs(returns, estimator)

try:
    frontier_returns, frontier_vols, _ = efficient_frontier(mu, cov, n_points=frontier_points, bounds=bounds)
    w_gmv = find_gmv_portfolio(mu, cov, bounds=bounds)
    w_tan = find_tangency_portfolio(mu, cov, rf=rf_mean, bounds=bounds)
except RuntimeError as exc:
    st.error(f"Optimization failed: {exc}")
    st.stop()

w_equal = np.ones(n_assets) / n_assets

gmv_stats = portfolio_stats(w_gmv, mu, cov, rf_mean)
tan_stats = portfolio_stats(w_tan, mu, cov, rf_mean)
equal_stats = portfolio_stats(w_equal, mu, cov, rf_mean)

points_df = pd.DataFrame(
    [
        {
            "Portfolio": "GMV",
            "Key": "gmv",
            "Monthly Return": gmv_stats[0],
            "Monthly Volatility": gmv_stats[1],
            "Monthly Sharpe": gmv_stats[2],
        },
        {
            "Portfolio": "Tangency",
            "Key": "tangency",
            "Monthly Return": tan_stats[0],
            "Monthly Volatility": tan_stats[1],
            "Monthly Sharpe": tan_stats[2],
        },
        {
            "Portfolio": "Equal Weight",
            "Key": "equal",
            "Monthly Return": equal_stats[0],
            "Monthly Volatility": equal_stats[1],
            "Monthly Sharpe": equal_stats[2],
        },
    ]
)
points_df["Annual Return"] = points_df["Monthly Return"].map(annualize_return)
points_df["Annual Volatility"] = points_df["Monthly Volatility"].map(annualize_vol)
points_df["Annual Sharpe"] = points_df["Monthly Sharpe"].map(annualize_sharpe)

frontier_df = frontier_frame(frontier_returns, frontier_vols)

metric_cols = st.columns(4)
metric_cols[0].metric("Observations", f"{len(returns):,}")
metric_cols[1].metric("Assets", f"{n_assets}")
metric_cols[2].metric("Tangency Sharpe", f"{annualize_sharpe(tan_stats[2]):.2f}")
metric_cols[3].metric("GMV Volatility", format_pct(annualize_vol(gmv_stats[1])))

tabs = st.tabs(["Dashboard", "Weights", "Out-of-Sample", "Theory", "Data"])

with tabs[0]:
    left, right = st.columns([2.2, 1])
    with left:
        st.plotly_chart(plot_frontier(frontier_df, points_df, rf_mean), use_container_width=True)
    with right:
        st.subheader("Current Setup")
        st.write(f"**Universe:** {universe_name if uploaded_returns is None else 'Custom CSV'}")
        st.write(f"**Return basis:** {'Excess returns' if return_type == 'excess' else 'Total returns'}")
        st.write(f"**Estimator:** {estimator}")
        st.write(f"**Constraints:** {constraint_mode}")
        st.divider()
        st.subheader("Estimator Diagnostics")
        st.json(diagnostics)

    st.subheader("Portfolio Metrics")
    metrics = points_df[
        ["Portfolio", "Annual Return", "Annual Volatility", "Annual Sharpe", "Monthly Return", "Monthly Volatility"]
    ].copy()
    st.dataframe(
        metrics.style.format(
            {
                "Annual Return": "{:.2%}",
                "Annual Volatility": "{:.2%}",
                "Annual Sharpe": "{:.2f}",
                "Monthly Return": "{:.2%}",
                "Monthly Volatility": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    col_a, col_b = st.columns(2)
    gmv_weights = weights_frame(asset_names, w_gmv)
    tan_weights = weights_frame(asset_names, w_tan)

    with col_a:
        st.plotly_chart(plot_weights(gmv_weights, "GMV Portfolio Weights"), use_container_width=True)
        download_weights_button(gmv_weights, "gmv_weights.csv")
    with col_b:
        st.plotly_chart(plot_weights(tan_weights, "Tangency Portfolio Weights"), use_container_width=True)
        download_weights_button(tan_weights, "tangency_weights.csv")

    st.subheader("Full Weight Table")
    combined_weights = pd.DataFrame(
        {
            "Asset": asset_names,
            "GMV": w_gmv,
            "Tangency": w_tan,
            "Equal Weight": w_equal,
        }
    )
    st.dataframe(
        combined_weights.style.format({"GMV": "{:.2%}", "Tangency": "{:.2%}", "Equal Weight": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

with tabs[2]:
    split_date = pd.Timestamp(f"{split_year}-12-31")
    train = returns.loc[:split_date]
    test = returns.loc[split_date + pd.offsets.MonthBegin(1):]
    rf_train = rf_series.reindex(train.index).fillna(0.0)
    rf_test = rf_series.reindex(test.index).fillna(0.0)

    if len(train) < 36 or len(test) < 24:
        st.warning("Choose a split with at least 36 training months and 24 test months.")
    else:
        mu_train, cov_train, _ = estimate_inputs(train, estimator)
        w_gmv_train = find_gmv_portfolio(mu_train, cov_train, bounds=bounds)
        w_tan_train = find_tangency_portfolio(mu_train, cov_train, rf=float(rf_train.mean()), bounds=bounds)

        oos_rows = []
        for name, weights in {
            "GMV": w_gmv_train,
            "Tangency": w_tan_train,
            "Equal Weight": w_equal,
        }.items():
            series = portfolio_return_series(test, weights)
            excess = series - rf_test
            monthly_ret = series.mean()
            monthly_vol = series.std(ddof=1)
            monthly_sharpe = excess.mean() / excess.std(ddof=1)
            oos_rows.append(
                {
                    "Portfolio": name,
                    "OOS Annual Return": annualize_return(monthly_ret),
                    "OOS Annual Volatility": annualize_vol(monthly_vol),
                    "OOS Annual Sharpe": annualize_sharpe(monthly_sharpe),
                    "Cumulative Return": (1 + series).prod() - 1,
                }
            )

        oos_df = pd.DataFrame(oos_rows)
        st.subheader(f"Train/Test Split: through {split_year} vs after {split_year}")
        st.dataframe(
            oos_df.style.format(
                {
                    "OOS Annual Return": "{:.2%}",
                    "OOS Annual Volatility": "{:.2%}",
                    "OOS Annual Sharpe": "{:.2f}",
                    "Cumulative Return": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        wealth = pd.DataFrame(index=test.index)
        for name, weights in {
            "GMV": w_gmv_train,
            "Tangency": w_tan_train,
            "Equal Weight": w_equal,
        }.items():
            wealth[name] = (1 + portfolio_return_series(test, weights)).cumprod()

        wealth_long = wealth.reset_index().melt(id_vars=wealth.index.name or "Date", var_name="Portfolio", value_name="Growth of $1")
        fig = px.line(wealth_long, x=wealth_long.columns[0], y="Growth of $1", color="Portfolio", title="Out-of-Sample Growth of $1")
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("What the App Demonstrates")
    st.markdown(
        """
        This dashboard translates the academic notebook into an interactive portfolio-construction tool.

        **Mean-variance optimization** minimizes portfolio variance for a target return:

        $$
        \\min_{\\mathbf{w}} \\mathbf{w}^{\\top}\\boldsymbol{\\Sigma}\\mathbf{w}
        \\quad
        \\text{s.t.}
        \\quad
        \\mathbf{w}^{\\top}\\boldsymbol{\\mu}=\\mu_p,\\;
        \\mathbf{w}^{\\top}\\mathbf{1}=1
        $$

        **Bayes-Stein shrinkage** reduces noisy mean estimates:

        $$
        \\hat{\\boldsymbol{\\mu}}_{BS}
        =
        (1-\\phi)\\hat{\\boldsymbol{\\mu}}
        +
        \\phi\\mu_{GMV}\\mathbf{1}
        $$

        **Ledoit-Wolf shrinkage** stabilizes covariance estimates:

        $$
        \\hat{\\boldsymbol{\\Sigma}}_{LW}
        =
        \\delta\\mathbf{F}+(1-\\delta)\\mathbf{S}
        $$

        The interview signal is practical: users can see how estimation choices and constraints change the frontier,
        portfolio weights, and out-of-sample robustness.
        """
    )

with tabs[4]:
    st.subheader("Dataset Preview")
    st.write(f"Date range: **{returns.index.min().date()}** to **{returns.index.max().date()}**")
    st.dataframe(returns.tail(12).style.format("{:.2%}"), use_container_width=True)

    st.subheader("Summary Statistics")
    stats_df = pd.DataFrame(
        {
            "Annual Return": returns.mean().map(annualize_return),
            "Annual Volatility": returns.std(ddof=1).map(annualize_vol),
            "Monthly Min": returns.min(),
            "Monthly Max": returns.max(),
        }
    )
    st.dataframe(stats_df.style.format("{:.2%}"), use_container_width=True)
