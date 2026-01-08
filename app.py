import datetime as dt
import os

import numpy as np
import pandas as pd
import streamlit as st

from backtest import Backtester
from model import TrendRegressionModel
from data_client import MarketDataClient, MarketDataSettings, MarketDataError
from alphas import MovingAverageCrossAlpha, BuyHoldAlpha

from dotenv import load_dotenv

load_dotenv()

AUTO_REFRESH_MS = 5 * 60 * 1000  # 5 minutes
DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "AAPL")


def build_strategy(strategy_name: str, short_window: int, long_window: int):
    if strategy_name == "Buy & Hold":
        return BuyHoldAlpha()
    if strategy_name == "MA Crossover":
        return MovingAverageCrossAlpha(short_window=short_window, long_window=long_window)
    raise ValueError("Stratégie inconnue.")



def main():
    st.set_page_config(page_title="Single Asset Module", layout="wide")
    # Keep the app refreshed every 5 minutes to comply with data freshness requirement
    st.markdown(
        "<script>setTimeout(() => { window.location.reload(); }, "
        f"{AUTO_REFRESH_MS});</script>",
        unsafe_allow_html=True,
    )

    st.title("Single asset module & backtesting")
    st.caption("Backtests, metrics and simple predictive model")

    st.sidebar.header("Parameters")

    default_end = dt.date.today()
    default_start = default_end - dt.timedelta(days=3285)

    ticker_input = st.sidebar.text_input("Ticker", DEFAULT_TICKER)
    ticker = ticker_input.strip().upper() or DEFAULT_TICKER
    if ticker_input.strip() == "":
        st.sidebar.caption(f"Using default ticker: {DEFAULT_TICKER}")
    start_date = st.sidebar.date_input("Start date", default_start, max_value=default_end)
    end_date = st.sidebar.date_input(
        "End date", default_end, min_value=start_date, max_value=default_end
    )

    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    strategy_choice = st.sidebar.selectbox("Strategy", options=["Buy & Hold", "MA Crossover"])

    interval_label_map = {
        "5 min": "5m",
        "15 min": "15m",
        "30 min": "30m",
        "1 hour": "1h",
        "1 day": "1d",
    }
    interval_label = st.sidebar.selectbox(
        "Interval",
        options=list(interval_label_map.keys()),
        index=list(interval_label_map.keys()).index("1 day"),
        help="Intraday intervals are subject to Yahoo Finance limits (keep date window short).",
    )
    interval = interval_label_map[interval_label]

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
    forecast_horizon = st.sidebar.slider("Forecast horizon", 1, 60, 20)

    settings = MarketDataSettings(default_ticker=DEFAULT_TICKER)
    data_client = MarketDataClient(settings=settings)

    periods_per_year_map = {
        "5m": 78 * 252,
        "15m": 26 * 252,
        "30m": 13 * 252,
        "1h": int(6.5 * 252),
        "1d": 252,
    }
    periods_per_year = periods_per_year_map.get(interval, 252)
    backtester = Backtester(risk_free_rate=rf_rate, periods_per_year=periods_per_year)

    try:
        data = data_client.get_history(
            ticker=ticker,
            start=start_date,
            end=end_date,
            interval=interval,
        )
    except MarketDataError as e:
        st.error(f"Error : {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error : {e}")
        st.stop()

    close = data["Close"].copy()
    last_price = float(close.iloc[-1])

    try:
        strategy = build_strategy(strategy_choice, short_window, long_window)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    result = backtester.run(close, strategy)

    comparison_raw = pd.concat(
        [
            (close / close.iloc[0]).rename("Asset"),
            (result.equity_curve / result.equity_curve.iloc[0]).rename("Strategy"),
        ],
        axis=1,
    )

    st.markdown("### Strategy vs Asset Performance")
    st.line_chart(comparison_raw)

    st.markdown("### Performance metrics")

    m = result.metrics

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total asset return", f"{100 * m['Asset Total Return']:.2f} %")
    col2.metric("Total strategy return", f"{100 * m['Strategy Total Return']:.2f} %")
    col3.metric("CAGR", f"{100 * m['Strategy CAGR']:.2f} %")
    col4.metric("Volatility (ann.)", f"{100 * m['Strategy Volatility (ann.)']:.2f} %")
    sharpe_val = m["Strategy Sharpe"]
    calmar_val = m["Calmar Ratio"]
    col5.metric("Sharpe", "-" if np.isnan(sharpe_val) else f"{sharpe_val:.2f}")

    col6, col7, col8, col9, col10 = st.columns(5)
    col6.metric("Calmar Ratio", "-" if np.isnan(calmar_val) else f"{calmar_val:.2f}")
    upr_val = m["Upside Potential Ratio"]
    col7.metric("Upside Potential Ratio", "-" if np.isnan(upr_val) else f"{upr_val:.2f}")
    omega_val = m["Omega Ratio"]
    col8.metric("Omega Ratio", "-" if np.isnan(omega_val) else f"{omega_val:.2f}")
    col9.metric("Max Drawdown", f"{100 * m['Strategy Max Drawdown']:.2f} %")
    turnover_val = m["Strategy Turnover"]
    col10.metric(
        "Turnover (avg abs delta pos)",
        "-" if np.isnan(turnover_val) else f"{100 * turnover_val:.2f} %",
    )

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
