import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from alphas import BuyHoldAlpha, MovingAverageCrossAlpha
from backtest import Backtester, PerformanceMetrics, PortfolioBacktester
from data_client import MarketDataClient, MarketDataError
from portfolio import (
    compute_portfolio_returns,
    compute_returns,
    equal_weights,
    normalize_weights,
    correlation_matrix,
    simulate_portfolio,
)

DEFAULT_TICKERS = "AAPL,MSFT,SPY"
AUTO_REFRESH_MS = 5 * 60 * 1000  # 5 minutes


def parse_tickers(s: str) -> list[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def main():
    st.set_page_config(page_title="Quant B – Multi-Asset Portfolio", layout="wide")
    st.title("Quant B – Multi-Asset Portfolio")

    st.markdown(
        "<script>setTimeout(() => { window.location.reload(); }, "
        f"{AUTO_REFRESH_MS});</script>",
        unsafe_allow_html=True,
    )
    st.sidebar.header("Parameters")

    tickers = parse_tickers(
        st.sidebar.text_input("Tickers (comma separated, ≥ 3)", DEFAULT_TICKERS)
    )
    if len(tickers) < 3:
        st.warning("Please enter at least 3 tickers.")
        st.stop()

    today = dt.date.today()
    start = st.sidebar.date_input(
        "Start date", today - dt.timedelta(days=3285), key="start_date"
    )
    end = st.sidebar.date_input("End date", today)

    if start >= end:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    rf_rate = st.sidebar.number_input(
        "Risk free rate (Sharpe)",
        min_value=0.0,
        max_value=0.2,
        value=0.0,
        step=0.001,
        format="%.3f",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Strategy")

    strategy_choice = st.sidebar.selectbox("Strategy", ["Buy & Hold", "MA Crossover"])

    short_window = None
    long_window = None
    if strategy_choice == "MA Crossover":
        short_window = st.sidebar.number_input(
            "Short window", min_value=5, max_value=200, value=20
        )
        long_window = st.sidebar.number_input(
            "Long window", min_value=10, max_value=400, value=50
        )
        if int(short_window) >= int(long_window):
            st.sidebar.error("Short window must be < Long window.")
            st.stop()
    rebalance_label = st.sidebar.selectbox(
        "Rebalancing frequency",
        options=["None", "Weekly", "Monthly"],
        help="Applies to Buy & Hold portfolios.",
    )
    rebalance = rebalance_label.lower()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Portfolio weights")
    mode = st.sidebar.radio("Weight mode", ["Equal weights", "Custom weights"])

    client = MarketDataClient()
    try:
        prices = client.get_multi_asset_prices(tickers, start=start, end=end)
    except MarketDataError as e:
        st.error(str(e))
        st.stop()

    latest_prices = prices.iloc[-1].copy()
    cols = st.columns(len(latest_prices))
    for idx, ticker in enumerate(latest_prices.index):
        cols[idx].metric(ticker, f"{latest_prices[ticker]:.2f}")

    returns = compute_returns(prices)

    if mode == "Equal weights":
        weights = equal_weights(returns.columns)
    else:
        raw = {}
        for c in returns.columns:
            raw[c] = st.sidebar.slider(
                f"{c} (%)",
                min_value=0,
                max_value=100,
                value=int(100 / len(returns.columns)),
                step=1,
            )

        total_weight = sum(raw.values())
        st.sidebar.markdown(f"**Total allocation: {total_weight} %**")

        if total_weight > 100:
            st.sidebar.error("Total allocation cannot exceed 100%.")
            st.stop()

        if total_weight == 0:
            st.sidebar.error("Total allocation must be greater than 0%.")
            st.stop()

        weights = normalize_weights(raw, returns.columns)

    if strategy_choice == "Buy & Hold":
        alpha = BuyHoldAlpha()
    else:
        alpha = MovingAverageCrossAlpha(
            short_window=int(short_window), long_window=int(long_window)
        )

    periods_per_year = 252
    backtester = PortfolioBacktester(risk_free_rate=rf_rate, periods_per_year=periods_per_year)

    use_rebalance = strategy_choice == "Buy & Hold" and rebalance != "none"
    if use_rebalance:
        positions = alpha.generate_positions(prices)
        equity = simulate_portfolio(prices, weights, rebalance=rebalance)
        portfolio_returns = equity.pct_change().fillna(0.0)
        asset_returns = compute_portfolio_returns(compute_returns(prices), weights)
        metrics_engine = PerformanceMetrics(
            risk_free_rate=rf_rate, periods_per_year=periods_per_year
        )
        metrics = metrics_engine.compute_portfolio(
            asset_returns,
            portfolio_returns,
            equity,
            positions=positions,
            weights=weights,
            use_lookahead_safe_shift=False,
        )
    else:
        result = backtester.run(prices, weights, alpha)
        positions = result.positions
        portfolio_returns = result.strategy_returns
        equity = result.equity_curve
        asset_returns = result.asset_returns
        metrics = result.metrics

    equity_aligned = equity.reindex(prices.index)
    if pd.isna(equity_aligned.iloc[0]):
        equity_aligned.iloc[0] = 1.0
    equity_aligned = equity_aligned.ffill()

    comparison_df = prices.div(prices.iloc[0]).copy()
    comparison_df["Portfolio"] = equity_aligned / float(equity_aligned.iloc[0])
    comparison_df.index.name = "Date"

    st.markdown("### Strategy vs Asset Performance")
    st.line_chart(comparison_df)

    st.subheader("Single asset strategy vs portfolio")
    asset_choice = st.selectbox("Asset", options=list(prices.columns))
    single_series = prices[asset_choice].dropna()
    if single_series.empty:
        st.warning("No data available for the selected asset.")
    else:
        single_backtester = Backtester(
            risk_free_rate=rf_rate, periods_per_year=periods_per_year
        )
        single_result = single_backtester.run(single_series, alpha)
        single_strategy = single_result.equity_curve
        single_strategy_norm = single_strategy / float(single_strategy.iloc[0])

        portfolio_norm = comparison_df["Portfolio"]
        single_compare = pd.concat(
            [
                single_strategy_norm.rename(f"{asset_choice} Strategy"),
                portfolio_norm.rename("Portfolio"),
            ],
            axis=1,
        ).dropna()
        st.line_chart(single_compare)

    st.markdown("### Performance metrics")

    m = metrics

    total_asset_return = m["Asset Total Return"]
    total_strategy_return = m["Strategy Total Return"]
    cagr_val = m["Strategy CAGR"]
    vol_val = m["Strategy Volatility (ann.)"]
    sharpe_val = m["Strategy Sharpe"]

    max_dd_val = m["Strategy Max Drawdown"]
    calmar_val = m["Calmar Ratio"]
    upr_val = m["Upside Potential Ratio"]
    omega_val = m["Omega Ratio"]
    turnover_val = m["Strategy Turnover"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total asset return", f"{100 * total_asset_return:.2f} %")
    col2.metric("Total strategy return", f"{100 * total_strategy_return:.2f} %")
    col3.metric("CAGR", f"{100 * cagr_val:.2f} %")
    col4.metric("Volatility (ann.)", f"{100 * vol_val:.2f} %")
    col5.metric("Sharpe", "-" if np.isnan(sharpe_val) else f"{sharpe_val:.2f}")

    col6, col7, col8, col9, col10 = st.columns(5)
    col6.metric("Calmar Ratio", "-" if np.isnan(calmar_val) else f"{calmar_val:.2f}")
    col7.metric("Upside Potential Ratio", "-" if np.isnan(upr_val) else f"{upr_val:.2f}")
    col8.metric("Omega Ratio", "-" if np.isnan(omega_val) else f"{omega_val:.2f}")
    col9.metric("Max Drawdown", f"{100 * max_dd_val:.2f} %")
    col10.metric("Turnover", "-" if np.isnan(turnover_val) else f"{100 * turnover_val:.2f} %")

    st.subheader("Correlation heatmap")

    corr = correlation_matrix(returns).copy()
    corr.index.name = "Ticker"
    corr_long = corr.reset_index().melt(id_vars="Ticker", var_name="Asset", value_name="Corr")

    heat = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("Asset:N", title=""),
            y=alt.Y("Ticker:N", title=""),
            color=alt.Color("Corr:Q", title="Corr", scale=alt.Scale(domain=[-1, 1])),
            tooltip=["Ticker:N", "Asset:N", alt.Tooltip("Corr:Q", format=".2f")],
        )
        .properties(height=350)
    )

    text = (
        alt.Chart(corr_long)
        .mark_text(size=12)
        .encode(
            x="Asset:N",
            y="Ticker:N",
            text=alt.Text("Corr:Q", format=".2f"),
        )
    )

    st.altair_chart((heat + text), use_container_width=True)

    with st.expander("Raw data"):
        st.write("Strategy:", alpha.name)
        st.write("Weights:", weights.to_dict())
        st.write("Prices (tail):")
        st.dataframe(prices.tail())
        st.write("Positions (tail):")
        st.dataframe(positions.tail())
        st.write("Strategy returns (tail):")
        st.dataframe(portfolio_returns.tail())
        st.write("Equity (tail):")
        st.dataframe(equity.tail())


if __name__ == "__main__":
    main()
