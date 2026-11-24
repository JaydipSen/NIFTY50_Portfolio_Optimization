import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

st.set_page_config(page_title="NIFTY 50 Portfolio Optimization", layout="wide")
st.title("NIFTY 50 Portfolio Optimization")
st.subheader("Minimum Variance Portfolio (MVP) vs Hierarchical Risk Parity (HRP)")

DEFAULT_STOCKS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS", "BHARTIARTL.NS",
    "LT.NS", "SBIN.NS", "AXISBANK.NS", "INFY.NS", "KOTAKBANK.NS", "M&M.NS"
]

with st.sidebar:
    st.header("Portfolio Settings")
    stock_text = st.text_area("Stock symbols (one per line or comma-separated)", value="\n".join(DEFAULT_STOCKS), height=230)
    stocks = list(dict.fromkeys(x.strip().upper() for x in stock_text.replace(",", "\n").splitlines() if x.strip()))
    rf = st.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=30.0, value=6.0, step=0.25) / 100
    st.markdown("---")
    st.caption("Training period — portfolio weights are estimated only from this period")
    train_start = st.date_input("Training start", value=pd.Timestamp("2022-01-01"))
    train_end = st.date_input("Training end", value=pd.Timestamp("2024-12-31"))
    st.markdown("---")
    st.caption("Testing period — weights remain fixed during this period")
    test_start = st.date_input("Testing start", value=pd.Timestamp("2025-01-01"))
    test_end = st.date_input("Testing end", value=pd.Timestamp.today().date())
    frontier_points = st.slider("Efficient frontier points", min_value=20, max_value=120, value=60, step=10)
    run = st.button("Run Portfolio Optimization", type="primary", use_container_width=True)


def download_prices(symbols, start, end):
    raw = yf.download(
        symbols, start=str(start),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d", auto_adjust=True, progress=False, group_by="column", threads=True
    )
    if raw.empty:
        raise ValueError("Yahoo Finance returned no price data. Check symbols and dates.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        elif "Close" in raw.columns.get_level_values(-1):
            prices = raw.xs("Close", axis=1, level=-1).copy()
        else:
            raise ValueError("Could not find adjusted closing prices in the downloaded data.")
    else:
        if "Close" not in raw.columns:
            raise ValueError("Could not find a Close price column.")
        prices = raw[["Close"]].copy()
        prices.columns = [symbols[0]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])
    prices.columns = [str(c).upper() for c in prices.columns]
    available = [s for s in symbols if s in prices.columns and prices[s].notna().any()]
    missing = [s for s in symbols if s not in available]
    if missing:
        st.warning("Skipped symbols with unavailable data: " + ", ".join(missing))
    if len(available) < 2:
        raise ValueError("Fewer than two requested symbols have valid price data.")
    prices = prices[available].apply(pd.to_numeric, errors="coerce").sort_index()
    prices = prices.dropna(how="any")
    if len(prices) < 60:
        raise ValueError("Fewer than 60 common daily price observations are available.")
    return prices


def calculate_mvp(returns):
    cov = returns.cov().to_numpy(dtype=float)
    cov = (cov + cov.T) / 2
    n = len(returns.columns)
    x0 = np.repeat(1.0 / n, n)
    objective = lambda w: float(w @ cov @ w)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(objective, x0, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                      constraints=constraints, options={"maxiter": 3000, "ftol": 1e-12})
    if not result.success:
        raise ValueError("MVP optimization failed: " + result.message)
    w = np.clip(result.x, 0, 1)
    w /= w.sum()
    return pd.Series(w, index=returns.columns, name="MVP")


def cluster_variance(cov, names):
    sub = cov.loc[names, names].to_numpy(dtype=float)
    inv_var = 1.0 / np.clip(np.diag(sub), 1e-12, None)
    w = inv_var / inv_var.sum()
    return float(w @ sub @ w)


def calculate_hrp(returns):
    cov = returns.cov().copy()
    corr = returns.corr().clip(-1, 1).copy()
    distance = np.sqrt(np.maximum(0.0, (1.0 - corr.to_numpy(dtype=float)) / 2.0))
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="single", optimal_ordering=True)
    ordered = list(returns.columns[leaves_list(tree)])
    weights = pd.Series(1.0, index=ordered, dtype=float)
    clusters = [ordered]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            var_left = cluster_variance(cov, left)
            var_right = cluster_variance(cov, right)
            denom = var_left + var_right
            alpha = 0.5 if denom <= 0 else 1.0 - var_left / denom
            weights.loc[left] *= alpha
            weights.loc[right] *= (1.0 - alpha)
            next_clusters.extend([left, right])
        clusters = next_clusters
    weights = weights.reindex(returns.columns)
    weights /= weights.sum()
    weights.name = "HRP"
    return weights


def annualized_stats(weights, mean_daily, cov_daily):
    ret = float(weights @ mean_daily) * 252
    variance = float(weights @ cov_daily @ weights) * 252
    risk = np.sqrt(max(variance, 0.0))
    return ret, risk


def calculate_efficient_frontier(returns, mvp_weights, points=60):
    """Long-only mean-variance frontier, annualized using 252 trading days."""
    mean_daily = returns.mean().to_numpy(dtype=float)
    cov_daily = returns.cov().to_numpy(dtype=float)
    cov_daily = (cov_daily + cov_daily.T) / 2
    mean_annual = mean_daily * 252
    cov_annual = cov_daily * 252
    n = len(mean_annual)
    mvp_array = mvp_weights.reindex(returns.columns).to_numpy(dtype=float)
    mvp_return = float(mvp_array @ mean_annual)
    max_asset_return = float(np.max(mean_annual))
    targets = np.linspace(mvp_return, max_asset_return, points)
    x0 = mvp_array.copy()
    rows = []
    bounds = [(0.0, 1.0)] * n
    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mean_annual) - t}
        ]
        result = minimize(lambda w: float(w @ cov_annual @ w), x0,
                          method="SLSQP", bounds=bounds, constraints=constraints,
                          options={"maxiter": 2500, "ftol": 1e-11})
        if result.success:
            w = np.clip(result.x, 0, 1)
            w /= w.sum()
            ret = float(w @ mean_annual)
            risk = float(np.sqrt(max(w @ cov_annual @ w, 0.0)))
            rows.append({"Return": ret, "Risk": risk})
            x0 = w
    # Include the exact MVP solution as an explicit, clearly marked point.
    mvp_ret, mvp_risk = annualized_stats(mvp_array, mean_daily, cov_daily)
    frontier = pd.DataFrame(rows)
    if frontier.empty:
        frontier = pd.DataFrame([{"Return": mvp_ret, "Risk": mvp_risk}])
    frontier = pd.concat([frontier, pd.DataFrame([{"Return": mvp_ret, "Risk": mvp_risk}])], ignore_index=True)
    frontier = frontier.drop_duplicates().sort_values("Return").reset_index(drop=True)
    return frontier, mvp_ret, mvp_risk


def calculate_metrics(portfolio_returns, risk_free_rate):
    r = portfolio_returns.dropna()
    if r.empty:
        raise ValueError("No portfolio returns are available for the selected period.")
    wealth = (1.0 + r).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    years = len(r) / 252.0
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else np.nan
    volatility = float(r.std() * np.sqrt(252))
    sharpe = float(((r.mean() * 252) - risk_free_rate) / volatility) if volatility > 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    return {"Total Return": total_return, "Annualized Return": cagr,
            "Annualized Risk": volatility, "Sharpe Ratio": sharpe,
            "Maximum Drawdown": float(drawdown.min())}


def fmt_pct(value):
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def fmt_num(value):
    return "N/A" if pd.isna(value) else f"{value:.3f}"


if run:
    if len(stocks) < 2:
        st.error("Please enter at least two stock symbols.")
        st.stop()
    if train_start >= train_end or test_start > test_end:
        st.error("Please check the selected date ranges.")
        st.stop()
    if train_end >= test_start:
        st.error("The training period must end before the testing period starts.")
        st.stop()
    try:
        with st.spinner("Downloading prices and optimizing portfolios..."):
            prices = download_prices(stocks, train_start, test_end)
            daily_returns = prices.pct_change().dropna(how="any")
            train_returns = daily_returns.loc[(daily_returns.index.date >= train_start) & (daily_returns.index.date <= train_end)]
            test_returns = daily_returns.loc[(daily_returns.index.date >= test_start) & (daily_returns.index.date <= test_end)]
            if len(train_returns) < 60:
                raise ValueError("Training period has fewer than 60 daily return observations.")
            if len(test_returns) < 2:
                raise ValueError("Testing period has fewer than two daily return observations.")
            mvp_weights = calculate_mvp(train_returns)
            hrp_weights = calculate_hrp(train_returns)
            frontier, mvp_train_return, mvp_train_risk = calculate_efficient_frontier(train_returns, mvp_weights, frontier_points)
            test_portfolio_returns = pd.DataFrame({
                "MVP": test_returns @ mvp_weights,
                "HRP": test_returns @ hrp_weights
            }).dropna()
            cumulative = (1.0 + test_portfolio_returns).cumprod()
            mvp_metrics = calculate_metrics(test_portfolio_returns["MVP"], rf)
            hrp_metrics = calculate_metrics(test_portfolio_returns["HRP"], rf)
            metrics = pd.DataFrame([mvp_metrics, hrp_metrics], index=["MVP", "HRP"])
            train_portfolio_returns = pd.DataFrame({"MVP": train_returns @ mvp_weights, "HRP": train_returns @ hrp_weights})
        st.success("Portfolio optimization completed.")

        st.header("Out-of-Sample Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MVP total return", fmt_pct(mvp_metrics["Total Return"]))
        c2.metric("HRP total return", fmt_pct(hrp_metrics["Total Return"]))
        c3.metric("MVP Sharpe ratio", fmt_num(mvp_metrics["Sharpe Ratio"]))
        c4.metric("HRP Sharpe ratio", fmt_num(hrp_metrics["Sharpe Ratio"]))
        metrics_display = metrics.copy()
        for col in ["Total Return", "Annualized Return", "Annualized Risk", "Maximum Drawdown"]:
            metrics_display[col] = metrics_display[col].map(fmt_pct)
        metrics_display["Sharpe Ratio"] = metrics_display["Sharpe Ratio"].map(fmt_num)
        st.dataframe(metrics_display, use_container_width=True)

        st.header("Efficient Frontier — Training Period")
        fig_frontier = go.Figure()
        fig_frontier.add_trace(go.Scatter(x=frontier["Risk"], y=frontier["Return"], mode="lines+markers",
                                          name="Efficient frontier", line={"width": 3},
                                          hovertemplate="Annualized risk: %{x:.2%}<br>Annualized return: %{y:.2%}<extra></extra>"))
        fig_frontier.add_trace(go.Scatter(x=[mvp_train_risk], y=[mvp_train_return], mode="markers",
                                          name="Minimum Variance Portfolio (MVP)",
                                          marker={"symbol": "star", "size": 18, "color": "red", "line": {"width": 1, "color": "white"}},
                                          hovertemplate="MVP<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"))
        hrp_train_return = float(train_portfolio_returns["HRP"].mean() * 252)
        hrp_train_risk = float(train_portfolio_returns["HRP"].std() * np.sqrt(252))
        fig_frontier.add_trace(go.Scatter(x=[hrp_train_risk], y=[hrp_train_return], mode="markers", name="HRP",
                                          marker={"symbol": "diamond", "size": 13, "color": "green"}))
        asset_returns = train_returns.mean() * 252
        asset_risks = train_returns.std() * np.sqrt(252)
        fig_frontier.add_trace(go.Scatter(x=asset_risks, y=asset_returns, mode="markers+text", text=asset_returns.index,
                                          textposition="top center", name="Individual stocks", marker={"size": 8, "opacity": 0.65}))
        fig_frontier.update_layout(title="Annualized Risk–Return Trade-off (MVP highlighted)",
                                   xaxis_title="Annualized volatility (risk)", yaxis_title="Annualized expected return",
                                   xaxis_tickformat=".0%", yaxis_tickformat=".0%", hovermode="closest", legend_title="Portfolio")
        st.plotly_chart(fig_frontier, use_container_width=True)
        st.caption("The frontier and MVP marker are estimated from training-period returns. The red star marks the long-only minimum-variance solution.")

        st.header("Portfolio Growth and Risk")
        fig_growth = go.Figure()
        for name in cumulative.columns:
            fig_growth.add_trace(go.Scatter(x=cumulative.index, y=cumulative[name], mode="lines", name=name))
        fig_growth.update_layout(title="Out-of-Sample Growth of ₹1", xaxis_title="Date", yaxis_title="Portfolio value", hovermode="x unified")
        st.plotly_chart(fig_growth, use_container_width=True)

        rolling_vol = test_portfolio_returns.rolling(21).std() * np.sqrt(252)
        fig_vol = px.line(rolling_vol, title="21-Trading-Day Rolling Annualized Volatility", labels={"value": "Annualized volatility", "index": "Date"})
        fig_vol.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_vol, use_container_width=True)

        wealth = (1 + test_portfolio_returns).cumprod()
        drawdowns = wealth / wealth.cummax() - 1
        fig_dd = px.line(drawdowns, title="Out-of-Sample Drawdown", labels={"value": "Drawdown", "index": "Date"})
        fig_dd.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_dd, use_container_width=True)

        st.header("Portfolio Allocation")
        allocations = pd.DataFrame({"MVP": mvp_weights, "HRP": hrp_weights})
        st.dataframe(allocations.style.format("{:.2%}"), use_container_width=True)
        fig_weights = go.Figure()
        for strategy in allocations.columns:
            fig_weights.add_trace(go.Bar(name=strategy, x=allocations.index, y=allocations[strategy]))
        fig_weights.update_layout(title="MVP vs HRP Asset Weights", barmode="group", xaxis_title="Stock", yaxis_title="Portfolio weight", yaxis_tickformat=".0%")
        st.plotly_chart(fig_weights, use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.pie(names=mvp_weights.index, values=mvp_weights.values, hole=0.35, title="MVP Allocation"), use_container_width=True)
        with col_b:
            st.plotly_chart(px.pie(names=hrp_weights.index, values=hrp_weights.values, hole=0.35, title="HRP Allocation"), use_container_width=True)

        st.header("Return and Correlation Diagnostics")
        monthly = (1.0 + test_portfolio_returns).groupby(test_portfolio_returns.index.to_period("M")).prod() - 1.0
        monthly.index = monthly.index.astype(str)
        fig_month = px.imshow(monthly.T, aspect="auto", color_continuous_scale="RdYlGn", origin="lower",
                              labels={"x": "Month", "y": "Strategy", "color": "Monthly return"}, title="Monthly Portfolio Returns")
        fig_month.update_coloraxes(colorbar_tickformat=".0%")
        st.plotly_chart(fig_month, use_container_width=True)
        corr_fig = px.imshow(train_returns.corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, title="Training-Period Stock Return Correlation")
        st.plotly_chart(corr_fig, use_container_width=True)

        st.header("Training-Period Risk–Return Summary")
        train_summary = pd.DataFrame({
            "Annualized Return": train_portfolio_returns.mean() * 252,
            "Annualized Risk": train_portfolio_returns.std() * np.sqrt(252)
        })
        st.dataframe(train_summary.style.format("{:.2%}"), use_container_width=True)

        st.header("Download Results")
        st.download_button("Download portfolio weights", data=allocations.to_csv().encode("utf-8"),
                           file_name="nifty50_portfolio_weights.csv", mime="text/csv")
        backtest = pd.concat([prices.add_prefix("Price_"), test_portfolio_returns.add_prefix("Return_"), cumulative.add_prefix("Wealth_")], axis=1)
        st.download_button("Download backtesting data", data=backtest.to_csv().encode("utf-8"),
                           file_name="nifty50_backtest_results.csv", mime="text/csv")
        st.download_button("Download efficient frontier", data=frontier.to_csv(index=False).encode("utf-8"),
                           file_name="nifty50_efficient_frontier.csv", mime="text/csv")
        st.caption("Prices use Yahoo Finance auto-adjusted daily data. Weights are optimized on the training sample and held constant in the test sample. Annualization assumes 252 trading days. Long-only portfolios; transaction costs and taxes are not included.")
    except Exception as exc:
        st.error(f"An error occurred: {exc}")
else:
    st.info("Configure the symbols and dates in the sidebar, then click Run Portfolio Optimization.")
