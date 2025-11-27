import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from simulation import Backtester
from model import TrendRegressionModel
from data_client import MarketDataClient
from alphas import MovingAverageCrossAlpha, MovingAverageCrossAlpha


def build_strategy(strategy_name: str, short_window: int, long_window: int):
    if strategy_name == "Buy & Hold":
        return MovingAverageCrossAlpha()
    if strategy_name == "MA Crossover":
        return MovingAverageCrossAlpha(short_window=short_window, long_window=long_window)
    raise ValueError("Stratégie inconnue.")


def main():
    st.set_page_config(page_title="Single Asset Module", layout="wide")

    st.title("Single asset module & backtesting")
    st.caption("Backtests, metrics and simple predictive model")

    st.sidebar.header("Parameters")

    default_end = dt.date.today()
    default_start = default_end - dt.timedelta(days=365)

    ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", "AAPL")
    start_date = st.sidebar.date_input("Start date", default_start, max_value=default_end)
    end_date = st.sidebar.date_input(
        "End date", default_end, min_value=start_date, max_value=default_end
    )

    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    strategy_choice = st.sidebar.selectbox("Strategy", options=["Buy & Hold", "MA Crossover"])

    short_window = st.sidebar.number_input(
        "Short window (if MA Crossover)", min_value=5, max_value=100, value=20
    )
    long_window = st.sidebar.number_input(
        "Long window (if MA Crossover)", min_value=10, max_value=300, value=50
    )

    rf_rate = st.sidebar.number_input(
        "Risk free rate (Sharpe)",
        min_value=0.0,
        max_value=0.2,
        value=0.0,
        step=0.001,
        format="%.3f",
    )

    st.sidebar.markdown("---")
    use_forecast = st.sidebar.checkbox("Activate forecast model")
    forecast_horizon = st.sidebar.slider("Forecast horizon", 5, 60, 20)

    data_client = MarketDataClient()
    backtester = Backtester(risk_free_rate=rf_rate)

    try:
        data = data_client.get_history(
            ticker=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
        )
    except Exception as e:
        st.error(f"Error : {e}")
        st.stop()

    close = data["close"].copy()
    last_price = float(close.iloc[-1])

    st.subheader(f"{ticker} - Historical data")
    col_price, col_dates = st.columns(2)
    col_price.metric("Last price", f"{last_price:,.2f}")
    col_dates.write(f"Dates : {close.index[0].date()} → {close.index[-1].date()}")

    try:
        strategy = build_strategy(strategy_choice, short_window, long_window)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    result = backtester.run(close, strategy)

    st.markdown("### Asset price vs strategy equity curve")

    price_normalized = close / close.iloc[0]
    price_normalized.name = "Asset price (normalized)"

    comparison_df = pd.concat([price_normalized, result.equity_curve], axis=1)

    st.line_chart(comparison_df)

    st.markdown("### Performance metrics")

    m = result.metrics

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    col1.metric("Total asset return", f"{100 * m['Asset Total Return']:.2f} %")
    col2.metric("Total strategy return", f"{100 * m['Strategy Total Return']:.2f} %")
    col3.metric(
        "Strategy Volatility (ann.)",
        f"{100 * m['Strategy Volatility (ann.)']:.2f} %",
    )

    sharpe_val = m["Strategy Sharpe"]
    sharpe_str = "-" if np.isnan(sharpe_val) else f"{sharpe_val:.2f}"
    col4.metric("Strategy Sharpe", sharpe_str)
    col5.metric("Strategy Max Drawdown", f"{100 * m['Strategy Max Drawdown']:.2f} %")

    with st.expander("Daily return details"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Asset Return": result.asset_returns,
                    "Strategy Return": result.strategy_returns,
                }
            ).tail(10)
        )

    if use_forecast:
        st.markdown("### Predictive model")

        model = TrendRegressionModel(horizon=forecast_horizon, use_log=True)
        forecast_result = model.fit_predict(close)

        if forecast_result is None:
            st.warning("Model cannot be fit.")
        else:
            hist = close.copy()
            fcast = forecast_result.forecast
            lower = forecast_result.lower
            upper = forecast_result.upper

            df_forecast = pd.concat(
                [
                    hist.rename("Historical Price"),
                    fcast.rename("Forecast"),
                    lower.rename("Lower CI"),
                    upper.rename("Upper CI"),
                ],
                axis=1,
            )

            st.line_chart(df_forecast)

            last_forecast = float(fcast.iloc[-1])
            st.metric(
                "Forecast price at",
                f"{last_forecast:,.2f}",
                delta=f"{100 * (last_forecast / last_price - 1):.2f} % vs dernier prix",
            )

            with st.expander("Predictive data"):
                st.dataframe(df_forecast.tail(10))


if __name__ == "__main__":
    main()
