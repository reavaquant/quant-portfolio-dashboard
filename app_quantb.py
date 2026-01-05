import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from data_client import MarketDataClient, MarketDataSettings, MarketDataError
from portfolio import (
    compute_returns,
    equal_weights,
    normalize_weights,
    compute_portfolio_returns,
    equity_curve,
    correlation_matrix,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    simulate_portfolio,
)

DEFAULT_TICKERS = "AAPL,MSFT,SPY"


def parse_tickers(s: str) -> list[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def main():
    st.set_page_config(page_title="Quant B – Multi-Asset Portfolio", layout="wide")
    st.title("Quant B – Multi-Asset Portfolio (Yahoo Finance)")

    st.sidebar.header("Parameters")

    tickers = parse_tickers(
        st.sidebar.text_input("Tickers (comma separated, ≥ 3)", DEFAULT_TICKERS)
    )

    if len(tickers) < 3:
        st.warning("Please enter at least 3 tickers.")
        st.stop()

    today = dt.date.today()
    start = st.sidebar.date_input("Start date", today - dt.timedelta(days=365))
    end = st.sidebar.date_input("End date", today)

    rebalance = st.sidebar.selectbox(
        "Rebalancing frequency", ["none", "monthly", "weekly"]
    )

    st.sidebar.markdown("### Portfolio weights")

    mode = st.sidebar.radio("Weight mode", ["Equal weights", "Custom weights"])

    client = MarketDataClient(MarketDataSettings(source="yfinance"))

    try:
        prices = client.get_multi_asset_prices(tickers, start=start, end=end)
    except MarketDataError as e:
        st.error(str(e))
        st.stop()

    returns = compute_returns(prices)

    if mode == "Equal weights":
        weights = equal_weights(returns.columns)
    else:
        raw = {
            c: st.sidebar.slider(f"{c} (%)", 0, 100, int(100 / len(returns.columns)))
            for c in returns.columns
        }
        weights = normalize_weights(raw, returns.columns)

    equity = simulate_portfolio(prices, weights, rebalance=rebalance)
    portfolio_returns = equity.pct_change().dropna()

    st.subheader("Portfolio equity curve")
    st.line_chart(equity)

    st.subheader("Performance metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Annualized return", f"{annualized_return(portfolio_returns)*100:.2f}%")
    col2.metric("Annualized volatility", f"{annualized_volatility(portfolio_returns)*100:.2f}%")
    col3.metric("Max drawdown", f"{max_drawdown(equity)*100:.2f}%")

    st.subheader("Correlation matrix")
    st.dataframe(correlation_matrix(returns).style.format("{:.2f}"))

    with st.expander("Raw data"):
        st.write(prices.tail())
        st.write(weights)


if __name__ == "__main__":
    main()
