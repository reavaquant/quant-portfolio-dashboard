import datetime as dt
import os
from contextlib import contextmanager

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from alphas import BuyHoldAlpha, MovingAverageCrossAlpha
from backtest import Backtester, PerformanceMetrics, PortfolioBacktester
from data_client import MarketDataClient, MarketDataSettings, MarketDataError
from model import TrendRegressionModel
from portfolio import (
    compute_portfolio_returns,
    compute_returns,
    equal_weights,
    normalize_weights,
    correlation_matrix,
    simulate_portfolio,
)

from dotenv import load_dotenv

load_dotenv()

AUTO_REFRESH_MS = 5 * 60 * 1000  # 5 minutes
DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "AAPL")
DEFAULT_TICKERS = "AAPL,MSFT,SPY"
DEFAULT_LOOKBACK_DAYS = 3285


def apply_theme() -> None:
    # Altair theme (applies to altair charts only)
    def _altair_theme():
        return {
            "config": {
                "background": "transparent",
                "font": "Space Grotesk",
                "title": {"font": "Fraunces", "fontSize": 16, "fontWeight": 700},
                "axis": {
                    "labelFont": "Space Grotesk",
                    "titleFont": "Space Grotesk",
                    "labelColor": "#4b5563",
                    "titleColor": "#111827",
                    "gridColor": "rgba(15, 23, 42, 0.08)",
                    "tickColor": "rgba(15, 23, 42, 0.12)",
                },
                "legend": {
                    "labelFont": "Space Grotesk",
                    "titleFont": "Space Grotesk",
                    "labelColor": "#4b5563",
                    "titleColor": "#111827",
                },
                "view": {"stroke": "transparent"},
            }
        }

    alt.themes.register("quant_pro", _altair_theme)
    alt.themes.enable("quant_pro")

    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root{
  --bg0:#fbfaf7;
  --bg1:#f5f4f0;
  --ink:#0f172a;
  --muted:#475569;
  --muted2:#64748b;
  --accent:#0f766e;
  --accent2:#14b8a6;

  --card: rgba(255,255,255,0.78);
  --card2: rgba(255,255,255,0.92);
  --border: rgba(15, 23, 42, 0.08);
  --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
  --shadow2: 0 8px 18px rgba(15, 23, 42, 0.06);
}

html, body, [class*="css"]{
  font-family: "Space Grotesk", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  color: var(--ink);
}

.stApp{
  background:
    radial-gradient(1100px 520px at 10% -8%, #f1ede2 0%, rgba(241,237,226,0) 60%),
    radial-gradient(1100px 640px at 92% -6%, #e6f3f1 0%, rgba(230,243,241,0) 66%),
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
}

div.block-container{
  padding-top: 1.8rem;
  padding-bottom: 2.2rem;
}

h1,h2,h3,h4,h5,h6{
  font-family: "Fraunces", serif;
  letter-spacing: 0.2px;
}
p, li { color: var(--muted); }

/* Hide streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #f2eee4 0%, #f6f5f0 70%, #f6f5f0 100%);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.sidebar-brand{
  padding: 12px 12px;
  border-radius: 16px;
  background: rgba(255,255,255,0.55);
  border: 1px solid var(--border);
  box-shadow: var(--shadow2);
}
.sidebar-brand .t{
  font-family: "Fraunces", serif;
  font-size: 18px;
  font-weight: 700;
  color: var(--ink);
}
.sidebar-brand .s{
  margin-top: 2px;
  font-size: 12px;
  color: var(--muted2);
}

/* Hero */
.hero{
  background: linear-gradient(135deg, rgba(255,255,255,0.85) 0%, rgba(255,255,255,0.68) 100%);
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 20px 22px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(8px);
  animation: rise 0.55s ease-out both;
}
.hero-top{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:12px;
}
.hero-title{
  font-size: 34px;
  font-weight: 700;
  line-height: 1.05;
}
.hero-subtitle{
  color: var(--muted);
  margin-top: 6px;
  font-size: 14px;
}
.badge{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(20, 184, 166, 0.25);
  background: rgba(20, 184, 166, 0.10);
  color: var(--accent);
  font-weight: 700;
  font-size: 12px;
  letter-spacing: 0.3px;
  white-space: nowrap;
}

/* Cards / Sections */
.card{
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: var(--shadow2);
  margin: 14px 0;
}
.card-title{
  font-family: "Fraunces", serif;
  font-weight: 700;
  font-size: 18px;
  margin-bottom: 2px;
}
.card-caption{
  color: var(--muted2);
  font-size: 13px;
  margin-bottom: 10px;
}

/* Metrics */
div[data-testid="metric-container"]{
  background: rgba(255,255,255,0.92);
  border: 1px solid var(--border);
  padding: 14px 14px;
  border-radius: 16px;
  box-shadow: var(--shadow2);
}
div[data-testid="metric-container"] > div { color: var(--muted2); }
div[data-testid="stMetricValue"] { font-weight: 800; }

/* Inputs / Controls */
.stTextInput input, .stDateInput input, .stNumberInput input{
  border-radius: 12px !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  background: rgba(255,255,255,0.85) !important;
}
[data-baseweb="select"] > div{
  border-radius: 12px !important;
  border: 1px solid rgba(15,23,42,0.12) !important;
  background: rgba(255,255,255,0.85) !important;
}
.stSlider [data-baseweb="slider"] > div{
  border-radius: 999px !important;
}
.stRadio, .stCheckbox { padding: 4px 0; }

.stButton > button{
  border-radius: 12px;
  border: 1px solid rgba(15, 118, 110, 0.35);
  background: linear-gradient(180deg, rgba(20,184,166,0.18) 0%, rgba(15,118,110,0.12) 100%);
  color: var(--ink);
  font-weight: 700;
  box-shadow: 0 6px 14px rgba(15, 23, 42, 0.06);
}
.stButton > button:hover{
  border-color: rgba(15, 118, 110, 0.55);
  transform: translateY(-1px);
}

/* Charts containers (best-effort) */
[data-testid="stChart"], [data-testid="stVegaLiteChart"]{
  background: rgba(255,255,255,0.72);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 10px 10px;
  box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
}

/* Expander */
details{
  background: rgba(255,255,255,0.70);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 6px 10px;
}
details summary{
  font-weight: 700;
  color: var(--ink);
}

/* Dataframe */
[data-testid="stDataFrame"]{
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--border);
}

@keyframes rise{
  from{ opacity:0; transform: translateY(6px); }
  to{ opacity:1; transform: translateY(0); }
}
</style>
""",
        unsafe_allow_html=True,
    )


@contextmanager
def card(title: str | None = None, caption: str | None = None):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if title:
        st.markdown(f"<div class='card-title'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<div class='card-caption'>{caption}</div>", unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)


def build_strategy(strategy_name: str, short_window: int, long_window: int):
    if strategy_name == "Buy & Hold":
        return BuyHoldAlpha()
    if strategy_name == "MA Crossover":
        return MovingAverageCrossAlpha(short_window=short_window, long_window=long_window)
    raise ValueError("Unknown strategy.")


def parse_tickers(s: str) -> list[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def render_single_asset() -> None:
    st.markdown("## Single asset module")
    st.caption("Backtests, metrics, and a lightweight predictive model.")

    st.sidebar.subheader("Single asset parameters")

    default_end = dt.date.today()
    default_start = default_end - dt.timedelta(days=DEFAULT_LOOKBACK_DAYS)

    ticker_input = st.sidebar.text_input("Ticker", DEFAULT_TICKER, key="single_ticker")
    ticker = ticker_input.strip().upper() or DEFAULT_TICKER
    if ticker_input.strip() == "":
        st.sidebar.caption(f"Using default ticker: {DEFAULT_TICKER}")

    start_date = st.sidebar.date_input(
        "Start date", default_start, max_value=default_end, key="single_start"
    )
    end_date = st.sidebar.date_input(
        "End date",
        default_end,
        min_value=start_date,
        max_value=default_end,
        key="single_end",
    )

    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    strategy_choice = st.sidebar.selectbox(
        "Strategy", options=["Buy & Hold", "MA Crossover"], key="single_strategy"
    )

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
        key="single_interval",
    )
    interval = interval_label_map[interval_label]

    short_window = st.sidebar.number_input(
        "Short window (if MA Crossover)",
        min_value=5,
        max_value=100,
        value=20,
        key="single_short_window",
    )
    long_window = st.sidebar.number_input(
        "Long window (if MA Crossover)",
        min_value=10,
        max_value=300,
        value=50,
        key="single_long_window",
    )
    if strategy_choice == "MA Crossover" and int(short_window) >= int(long_window):
        st.sidebar.error("Short window must be < Long window.")
        st.stop()

    rf_rate = st.sidebar.number_input(
        "Risk free rate (Sharpe)",
        min_value=0.0,
        max_value=0.2,
        value=0.0,
        step=0.001,
        format="%.3f",
        key="single_rf",
    )

    st.sidebar.markdown("---")
    use_forecast = st.sidebar.checkbox("Activate forecast model", key="single_forecast")
    forecast_horizon = st.sidebar.slider(
        "Forecast horizon", 1, 60, 20, key="single_forecast_horizon"
    )

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
        st.error(f"Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

    close = data["Close"].copy()
    last_price = float(close.iloc[-1])

    with card("Snapshot", "Latest market print for your selected asset."):
        st.metric(f"{ticker} last price", f"{last_price:.2f}")

    try:
        strategy = build_strategy(strategy_choice, int(short_window), int(long_window))
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

    with card("Strategy vs Asset Performance", "Normalized performance comparison."):
        st.line_chart(comparison_raw)

    with card("Performance metrics", "Risk/return overview for the selected configuration."):
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
        with card("Predictive model", "Trend regression forecast with confidence intervals."):
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
                    delta=f"{100 * (last_forecast / last_price - 1):.2f} % vs last price",
                )

                with st.expander("Predictive data"):
                    st.dataframe(df_forecast.tail(10))


def render_multi_asset() -> None:
    st.markdown("## Multi-asset portfolio")
    st.caption("Portfolio analytics, rebalancing, and cross-asset comparisons.")

    st.sidebar.subheader("Multi-asset parameters")

    tickers = parse_tickers(
        st.sidebar.text_input(
            "Tickers (comma separated, min 3)",
            DEFAULT_TICKERS,
            key="multi_tickers",
        )
    )
    if len(tickers) < 3:
        st.warning("Please enter at least 3 tickers.")
        st.stop()

    today = dt.date.today()
    start = st.sidebar.date_input(
        "Start date",
        today - dt.timedelta(days=DEFAULT_LOOKBACK_DAYS),
        key="multi_start_date",
    )
    end = st.sidebar.date_input("End date", today, key="multi_end_date")

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
        key="multi_rf",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Strategy")
    strategy_choice = st.sidebar.selectbox(
        "Strategy",
        ["Buy & Hold", "MA Crossover"],
        key="multi_strategy",
    )

    short_window = 20
    long_window = 50
    if strategy_choice == "MA Crossover":
        short_window = st.sidebar.number_input(
            "Short window", min_value=5, max_value=200, value=20, key="multi_short"
        )
        long_window = st.sidebar.number_input(
            "Long window", min_value=10, max_value=400, value=50, key="multi_long"
        )
        if int(short_window) >= int(long_window):
            st.sidebar.error("Short window must be < Long window.")
            st.stop()

    rebalance_label = st.sidebar.selectbox(
        "Rebalancing frequency",
        options=["None", "Weekly", "Monthly"],
        help="Applies to Buy & Hold portfolios.",
        key="multi_rebalance",
    )
    rebalance = rebalance_label.lower()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Portfolio weights")
    mode = st.sidebar.radio(
        "Weight mode", ["Equal weights", "Custom weights"], key="multi_weight_mode"
    )

    client = MarketDataClient()
    try:
        prices = client.get_multi_asset_prices(tickers, start=start, end=end)
    except MarketDataError as e:
        st.error(str(e))
        st.stop()

    latest_prices = prices.iloc[-1].copy()
    with card("Latest prices", "Most recent close for each selected asset."):
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
                key=f"weight_{c}",
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
    backtester = PortfolioBacktester(
        risk_free_rate=rf_rate, periods_per_year=periods_per_year
    )

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

    with card("Strategy vs Asset Performance", "Normalized comparison across assets and portfolio."):
        st.line_chart(comparison_df)

    with card("Single asset strategy vs portfolio", "Compare one asset strategy equity curve to the portfolio."):
        asset_choice = st.selectbox("Asset", options=list(prices.columns), key="multi_asset")
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

    with card("Performance metrics", "Portfolio risk/return indicators for the selected strategy."):
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
        col10.metric(
            "Turnover", "-" if np.isnan(turnover_val) else f"{100 * turnover_val:.2f} %"
        )

    with card("Correlation heatmap", "Pairwise correlations from returns (same data, nicer frame)."):
        corr = correlation_matrix(returns).copy()
        corr.index.name = "Ticker"
        corr_long = corr.reset_index().melt(
            id_vars="Ticker", var_name="Asset", value_name="Corr"
        )

        heat = (
            alt.Chart(corr_long)
            .mark_rect(cornerRadius=4)
            .encode(
                x=alt.X("Asset:N", title=""),
                y=alt.Y("Ticker:N", title=""),
                color=alt.Color("Corr:Q", title="Corr", scale=alt.Scale(domain=[-1, 1])),
                tooltip=["Ticker:N", "Asset:N", alt.Tooltip("Corr:Q", format=".2f")],
            )
            .properties(height=360)
        )

        text = (
            alt.Chart(corr_long)
            .mark_text(size=12)
            .encode(
                x="Asset:N",
                y="Ticker:N",
                text=alt.Text("Corr:Q", format=".2f"),
                color=alt.value("#0f172a"),
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


def main():
    st.set_page_config(page_title="Quant Portfolio Dashboard", layout="wide")
    apply_theme()

    # Sidebar brand (appearance only)
    st.sidebar.markdown(
        """
<div class="sidebar-brand">
  <div class="t">Quant Dashboard</div>
  <div class="s">Backtests • Portfolio • Metrics</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("")

    st.markdown(
        "<script>setTimeout(() => { window.location.reload(); }, "
        f"{AUTO_REFRESH_MS});</script>",
        unsafe_allow_html=True,
    )

    # Hero
    st.markdown(
        """
<div class="hero">
  <div class="hero-top">
    <div>
      <div class="hero-title">Quant Portfolio Dashboard</div>
      <div class="hero-subtitle">Single asset backtests and multi-asset portfolio analytics in one place.</div>
    </div>
    <div class="badge">● Live refresh • 5 min</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Workspace")
    mode = st.sidebar.radio(
        "Module", ["Single Asset", "Multi-Asset Portfolio"], key="workspace"
    )
    st.sidebar.markdown("---")

    if mode == "Single Asset":
        render_single_asset()
    else:
        render_multi_asset()


if __name__ == "__main__":
    main()
