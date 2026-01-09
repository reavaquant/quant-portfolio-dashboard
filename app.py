import datetime as dt
import os
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

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

# -----------------------
# ENV / CONSTANTS
# -----------------------
load_dotenv()

AUTO_REFRESH_MS = 5 * 60 * 1000  # 5 min (constant)
DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "AAPL")
DEFAULT_TICKERS = os.getenv("DEFAULT_TICKERS", "AAPL,MSFT,SPY")
DEFAULT_LOOKBACK_DAYS = int(os.getenv("DEFAULT_LOOKBACK_DAYS", "3285"))


# -----------------------
# THEME (dark, pro, minimal)
# -----------------------
def apply_minimal_dark() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap');
:root{
  --bg:#0e1117;
  --panel:#111827;
  --panel-2:#0f172a;
  --border: rgba(148,163,184,0.22);
  --text:#e5e7eb;
  --muted:#94a3b8;
  --accent:#38bdf8;
  --accent-2:#22c55e;
  --radius: 10px;
}

html, body, [class*="css"] { color: var(--text); font-family: "Manrope", "Segoe UI", sans-serif; }
h1, h2, h3, h4, h5 { font-family: "Space Grotesk", "Segoe UI", sans-serif; letter-spacing: -0.015em; }
.stApp { background: var(--bg); }
div.block-container { max-width: 1400px; padding-top: 1.0rem; padding-bottom: 2.0rem; }

#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"]{
  background: transparent;
  box-shadow: none;
}

[data-testid="stSidebar"]{
  background: var(--bg);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

.stTextInput input, .stDateInput input, .stNumberInput input{
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  background: var(--panel) !important;
  color: var(--text) !important;
}
.stTextInput input::placeholder,
.stNumberInput input::placeholder{
  color: rgba(148,163,184,0.7) !important;
}
[data-baseweb="select"] > div{
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  background: var(--panel) !important;
  color: var(--text) !important;
}

.stButton > button{
  border-radius: var(--radius);
  border: 1px solid rgba(56,189,248,0.45);
  background: rgba(56,189,248,0.12);
  color: var(--text);
  font-weight: 600;
  transition: all 160ms ease;
}
.stButton > button:hover{
  border-color: rgba(56,189,248,0.75);
  transform: translateY(-1px);
}

div[data-testid="metric-container"]{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 12px;
}

[data-testid="stChart"], [data-testid="stVegaLiteChart"]{
  border-radius: var(--radius);
  border: none;
  background: var(--panel);
}

[data-testid="stDataFrame"]{
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--panel);
}

[data-testid="stExpander"]{
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--panel);
}

.page-header{ margin-bottom: 0.6rem; }
.page-title-row{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
}
.page-title{
  font-size: 1.9rem;
  font-weight: 600;
  color: var(--text);
}
.page-subtitle{
  color: var(--muted);
  font-size: 0.95rem;
}
.live-pill{
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  border: 1px solid rgba(56,189,248,0.55);
  color: #e2e8f0;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  background: rgba(56,189,248,0.12);
}
.live-dot{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #22d3ee;
  box-shadow: 0 0 0 0 rgba(34,211,238,0.7);
  animation: livePulse 1.6s ease-out infinite;
}

.price-banner{
  margin: 0.6rem 0 0.75rem 0;
  padding: 0.55rem 0.85rem;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--panel-2);
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem 1rem;
  font-weight: 600;
}
.price-banner span{ color: var(--text); }

.nav-wrap{
  margin: 0.35rem 0 1.0rem 0;
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}
.nav-top, .nav-sub{
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}
.nav-link{
  padding: 0.45rem 0.9rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--muted);
  text-decoration: none;
  font-weight: 600;
  background: var(--panel);
  transition: all 140ms ease;
}
.nav-link:hover{
  color: var(--text);
  border-color: rgba(56,189,248,0.65);
}
.nav-link.active{
  color: #f8fafc;
  border-color: rgba(56,189,248,0.95);
  background: rgba(56,189,248,0.2);
}

@keyframes livePulse{
  0% { box-shadow: 0 0 0 0 rgba(34,211,238,0.6); }
  70% { box-shadow: 0 0 0 6px rgba(34,211,238,0.0); }
  100% { box-shadow: 0 0 0 0 rgba(34,211,238,0.0); }
}
</style>
""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="page-header">
  <div class="page-title-row">
    <div class="page-title">{title}</div>
    <div class="live-pill"><span class="live-dot"></span> Live</div>
  </div>
  <div class="page-subtitle">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def inject_auto_refresh_constant() -> None:
    # Constant auto refresh (like your original)
    st.markdown(
        "<script>setTimeout(() => { window.location.reload(); }, "
        f"{AUTO_REFRESH_MS});</script>",
        unsafe_allow_html=True,
    )

# -----------------------
# CACHE DATA
# -----------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str, start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
    settings = MarketDataSettings(default_ticker=DEFAULT_TICKER)
    client = MarketDataClient(settings=settings)
    return client.get_history(ticker=ticker, start=start, end=end, interval=interval)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_multi_prices(tickers: tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    client = MarketDataClient()
    return client.get_multi_asset_prices(list(tickers), start=start, end=end)


# -----------------------
# HELPERS
# -----------------------
def parse_tickers(s: str) -> list[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def build_strategy(strategy_name: str, short_window: int, long_window: int):
    if strategy_name == "Buy & Hold":
        return BuyHoldAlpha()
    if strategy_name == "MA Crossover":
        return MovingAverageCrossAlpha(short_window=short_window, long_window=long_window)
    raise ValueError("Unknown strategy.")


def periods_per_year_for_interval(interval: str) -> int:
    return {
        "5m": 78 * 252,
        "15m": 26 * 252,
        "30m": 13 * 252,
        "1h": int(6.5 * 252),
        "1d": 252,
    }.get(interval, 252)


def fmt_or_dash(x, fmt=".2f") -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return format(float(x), fmt)
    except Exception:
        return "-"


SA_STATE_KEY = "sa_state"
PF_STATE_KEY = "pf_state"


def _ensure_state_defaults(state_key: str, defaults: dict) -> None:
    state = st.session_state.get(state_key)
    if not isinstance(state, dict):
        st.session_state[state_key] = defaults.copy()
        return
    for key, value in defaults.items():
        state.setdefault(key, value)


def init_session_defaults() -> None:
    today = dt.date.today()
    default_start = today - dt.timedelta(days=DEFAULT_LOOKBACK_DAYS)
    _ensure_state_defaults(
        SA_STATE_KEY,
        {
            "ticker": DEFAULT_TICKER,
            "start": default_start,
            "end": today,
            "interval_label": "1 day",
            "rf": 0.0,
            "strategy": "Buy & Hold",
            "short_window": 20,
            "long_window": 50,
            "forecast_horizon": 20,
            "use_log": True,
        },
    )
    _ensure_state_defaults(
        PF_STATE_KEY,
        {
            "tickers": DEFAULT_TICKERS,
            "start": default_start,
            "end": today,
            "rf": 0.0,
            "strategy": "Buy & Hold",
            "short_window": 20,
            "long_window": 50,
            "rebalance": "None",
            "weight_mode": "Equal weights",
            "custom_weights": {},
        },
    )


def metrics_grid(m: dict) -> None:
    # Same metrics as your original (2 rows of 5)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total asset return", f"{100 * float(m['Asset Total Return']):.2f} %")
    col2.metric("Total strategy return", f"{100 * float(m['Strategy Total Return']):.2f} %")
    col3.metric("CAGR", f"{100 * float(m['Strategy CAGR']):.2f} %")
    col4.metric("Volatility (ann.)", f"{100 * float(m['Strategy Volatility (ann.)']):.2f} %")
    col5.metric("Sharpe", fmt_or_dash(m.get("Strategy Sharpe"), ".2f"))

    col6, col7, col8, col9, col10 = st.columns(5)
    col6.metric("Calmar Ratio", fmt_or_dash(m.get("Calmar Ratio"), ".2f"))
    col7.metric("Upside Potential Ratio", fmt_or_dash(m.get("Upside Potential Ratio"), ".2f"))
    col8.metric("Omega Ratio", fmt_or_dash(m.get("Omega Ratio"), ".2f"))
    col9.metric("Max Drawdown", f"{100 * float(m['Strategy Max Drawdown']):.2f} %")
    turnover_val = m.get("Strategy Turnover", np.nan)
    col10.metric("Turnover (avg abs delta pos)", "-" if np.isnan(turnover_val) else f"{100 * float(turnover_val):.2f} %")


def render_price_banner(prices: dict[str, Optional[float]]) -> None:
    if not prices:
        return
    items = []
    for ticker, value in prices.items():
        display = "-"
        try:
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                display = f"{float(value):,.2f}"
        except Exception:
            display = "-"
        items.append(f"<span><strong>{ticker}</strong> : {display}</span>")
    st.markdown(f"<div class=\"price-banner\">{''.join(items)}</div>", unsafe_allow_html=True)


# -----------------------
# SIDEBARS (single / portfolio)
# -----------------------
def sidebar_single_common():
    st.sidebar.subheader("Single asset parameters")

    today = dt.date.today()
    state = st.session_state[SA_STATE_KEY]

    ticker_input = st.sidebar.text_input("Ticker", value=state["ticker"], key="sa_ticker_input")
    ticker_clean = ticker_input.strip().upper()
    if not ticker_clean:
        ticker = state["ticker"]
        st.session_state["sa_ticker_input"] = ticker
    else:
        ticker = ticker_clean
    state["ticker"] = ticker

    start_default = min(state["start"], today)
    start_date = st.sidebar.date_input("Start date", value=start_default, max_value=today, key="sa_start")
    if "sa_end" in st.session_state and st.session_state["sa_end"] < start_date:
        st.session_state["sa_end"] = start_date
    end_default = state["end"]
    if end_default < start_date:
        end_default = start_date
    if end_default > today:
        end_default = today
    end_date = st.sidebar.date_input("End date", value=end_default, min_value=start_date, max_value=today, key="sa_end")

    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    interval_label_map = {"5 min": "5m", "15 min": "15m", "30 min": "30m", "1 hour": "1h", "1 day": "1d"}
    interval_labels = list(interval_label_map.keys())
    interval_label = state.get("interval_label", "1 day")
    if interval_label not in interval_labels:
        interval_label = "1 day"
    interval_label = st.sidebar.selectbox(
        "Interval",
        options=interval_labels,
        help="Intraday intervals are subject to Yahoo Finance limits (keep date window short).",
        index=interval_labels.index(interval_label),
        key="sa_interval_label",
    )
    interval = interval_label_map[interval_label]

    rf_rate = st.sidebar.number_input(
        "Risk free rate (Sharpe)",
        min_value=0.0,
        max_value=0.2,
        step=0.001,
        format="%.3f",
        value=float(state["rf"]),
        key="sa_rf",
    )

    state["start"] = start_date
    state["end"] = end_date
    state["interval_label"] = interval_label
    state["rf"] = float(rf_rate)
    return ticker, start_date, end_date, interval, float(rf_rate)


def sidebar_single_strategy():
    state = st.session_state[SA_STATE_KEY]
    strategy_options = ["Buy & Hold", "MA Crossover"]
    strategy_default = state.get("strategy", "Buy & Hold")
    if strategy_default not in strategy_options:
        strategy_default = "Buy & Hold"
    strategy_choice = st.sidebar.selectbox(
        "Strategy",
        options=strategy_options,
        index=strategy_options.index(strategy_default),
        key="sa_strategy_choice",
    )

    short_window = st.sidebar.number_input(
        "Short window (if MA Crossover)",
        min_value=5,
        max_value=100,
        value=int(state["short_window"]),
        key="sa_short_window",
    )
    long_window = st.sidebar.number_input(
        "Long window (if MA Crossover)",
        min_value=10,
        max_value=300,
        value=int(state["long_window"]),
        key="sa_long_window",
    )
    if strategy_choice == "MA Crossover" and int(short_window) >= int(long_window):
        st.sidebar.error("Short window must be < Long window.")
        st.stop()

    state["strategy"] = strategy_choice
    state["short_window"] = int(short_window)
    state["long_window"] = int(long_window)
    return strategy_choice, int(short_window), int(long_window)


def sidebar_portfolio_common():
    st.sidebar.subheader("Multi-asset parameters")
    state = st.session_state[PF_STATE_KEY]

    raw_tickers = st.sidebar.text_input(
        "Tickers (comma separated, min 3)",
        value=state["tickers"],
        key="pf_tickers_input",
    )
    candidate = parse_tickers(raw_tickers)
    persisted_list = parse_tickers(state["tickers"])
    if len(candidate) >= 3:
        tickers = candidate
        state["tickers"] = raw_tickers
    elif len(persisted_list) >= 3:
        tickers = persisted_list
        st.sidebar.warning("Please enter at least 3 tickers.")
    else:
        tickers = persisted_list if len(persisted_list) >= 3 else parse_tickers(DEFAULT_TICKERS)
        state["tickers"] = ",".join(tickers)
        st.sidebar.warning("Please enter at least 3 tickers.")

    start = st.sidebar.date_input(
        "Start date",
        value=state["start"],
        key="pf_start",
    )
    if "pf_end" in st.session_state and st.session_state["pf_end"] < start:
        st.session_state["pf_end"] = start
    end_default = state["end"]
    if end_default < start:
        end_default = start
    end = st.sidebar.date_input("End date", value=end_default, key="pf_end", min_value=start)

    if start >= end:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    rf_rate = st.sidebar.number_input(
        "Risk free rate (Sharpe)",
        min_value=0.0,
        max_value=0.2,
        step=0.001,
        format="%.3f",
        value=float(state["rf"]),
        key="pf_rf",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Strategy")
    strategy_options = ["Buy & Hold", "MA Crossover"]
    strategy_default = state.get("strategy", "Buy & Hold")
    if strategy_default not in strategy_options:
        strategy_default = "Buy & Hold"
    strategy_choice = st.sidebar.selectbox(
        "Strategy",
        strategy_options,
        index=strategy_options.index(strategy_default),
        key="pf_strategy",
    )

    short_window = int(state["short_window"])
    long_window = int(state["long_window"])
    if strategy_choice == "MA Crossover":
        short_window = st.sidebar.number_input(
            "Short window",
            min_value=5,
            max_value=200,
            step=1,
            value=short_window,
            key="pf_short",
        )
        long_window = st.sidebar.number_input(
            "Long window",
            min_value=10,
            max_value=400,
            step=1,
            value=long_window,
            key="pf_long",
        )
        if int(short_window) >= int(long_window):
            st.sidebar.error("Short window must be < Long window.")
            st.stop()

    rebalance_options = ["None", "Weekly", "Monthly"]
    rebalance_default = state.get("rebalance", "None")
    if rebalance_default not in rebalance_options:
        rebalance_default = "None"
    rebalance_label = st.sidebar.selectbox(
        "Rebalancing frequency",
        options=rebalance_options,
        help="Applies to Buy & Hold portfolios.",
        index=rebalance_options.index(rebalance_default),
        key="pf_rebalance",
    )
    rebalance = rebalance_label.lower()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Portfolio weights")
    mode_options = ["Equal weights", "Custom weights"]
    mode_default = state.get("weight_mode", "Equal weights")
    if mode_default not in mode_options:
        mode_default = "Equal weights"
    mode = st.sidebar.radio(
        "Weight mode",
        mode_options,
        index=mode_options.index(mode_default),
        key="pf_weight_mode",
    )

    custom_raw = None
    if mode == "Custom weights":
        prior_weights = state.get("custom_weights", {})
        if not isinstance(prior_weights, dict):
            prior_weights = {}
        raw = {}
        for t in tickers:
            raw[t] = st.sidebar.slider(
                f"{t} (%)",
                min_value=0,
                max_value=100,
                value=int(prior_weights.get(t, 100 / len(tickers))),
                step=1,
                key=f"pf_w_{t}",
            )

        total_weight = sum(raw.values())
        st.sidebar.markdown(f"**Total allocation: {total_weight} %**")

        if total_weight > 100:
            st.sidebar.error("Total allocation cannot exceed 100%.")
            st.stop()
        if total_weight == 0:
            st.sidebar.error("Total allocation must be greater than 0%.")
            st.stop()

        custom_raw = raw
        state["custom_weights"] = raw

    state["start"] = start
    state["end"] = end
    state["rf"] = float(rf_rate)
    state["strategy"] = strategy_choice
    state["short_window"] = int(short_window)
    state["long_window"] = int(long_window)
    state["rebalance"] = rebalance_label
    state["weight_mode"] = mode

    return (
        tickers,
        start,
        end,
        float(rf_rate),
        strategy_choice,
        int(short_window),
        int(long_window),
        rebalance,
        mode,
        custom_raw,
    )


# -----------------------
# PAGES: SINGLE ASSET
# -----------------------
def page_single_strat_perf():
    page_header("Single Asset Strategy", "Strategy & metrics")

    ticker, start_date, end_date, interval, rf_rate = sidebar_single_common()
    strategy_choice, short_window, long_window = sidebar_single_strategy()
    periods_per_year = periods_per_year_for_interval(interval)
    backtester = Backtester(risk_free_rate=rf_rate, periods_per_year=periods_per_year)

    try:
        data = fetch_history(ticker=ticker, start=start_date, end=end_date, interval=interval)
    except MarketDataError as e:
        st.error(f"Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

    close = data["Close"].copy()
    strategy = build_strategy(strategy_choice, short_window, long_window)
    result = backtester.run(close, strategy)

    render_price_banner({ticker: close.iloc[-1]})

    comparison = pd.concat(
        [
            (close / close.iloc[0]).rename("Asset"),
            (result.equity_curve / result.equity_curve.iloc[0]).rename("Strategy"),
        ],
        axis=1,
    )

    st.subheader("Performance metrics")
    metrics_grid(result.metrics)

    st.subheader("Strategy vs Asset Performance")
    st.line_chart(comparison, height=380)


def page_single_forecast():
    page_header("Single Asset Strategy", "Forecast")

    ticker, start_date, end_date, interval, rf_rate = sidebar_single_common()
    st.sidebar.markdown("---")
    st.sidebar.subheader("Forecast model")
    state = st.session_state[SA_STATE_KEY]
    forecast_horizon = st.sidebar.slider(
        "Forecast horizon",
        1,
        60,
        value=int(state["forecast_horizon"]),
        key="sa_forecast_horizon",
    )
    use_log = st.sidebar.checkbox("Use log prices", value=bool(state["use_log"]), key="sa_use_log")
    state["forecast_horizon"] = int(forecast_horizon)
    state["use_log"] = bool(use_log)

    try:
        data = fetch_history(ticker=ticker, start=start_date, end=end_date, interval=interval)
    except MarketDataError as e:
        st.error(f"Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

    close = data["Close"].copy()
    last_price = float(close.iloc[-1])

    st.subheader("Predictive model (trend regression + CI)")
    model = TrendRegressionModel(horizon=forecast_horizon, use_log=use_log)
    forecast_result = model.fit_predict(close)

    if forecast_result is None:
        st.warning("Model cannot be fit.")
        return

    hist = close.rename("Historical Price")
    fcast = forecast_result.forecast.rename("Forecast")
    lower = forecast_result.lower.rename("Lower CI")
    upper = forecast_result.upper.rename("Upper CI")

    df_forecast = pd.concat([hist, fcast, lower, upper], axis=1)
    st.line_chart(df_forecast, height=420)

    last_forecast = float(forecast_result.forecast.iloc[-1])
    st.metric(
        "Forecast price at horizon",
        f"{last_forecast:,.2f}",
        delta=f"{100 * (last_forecast / last_price - 1):.2f} % vs last price",
    )

    with st.expander("Predictive data"):
        st.dataframe(df_forecast.tail(10), use_container_width=True)


# -----------------------
# PAGES: PORTFOLIO
# -----------------------
def run_portfolio_engine(
    tickers: list[str],
    start: dt.date,
    end: dt.date,
    rf_rate: float,
    strategy_choice: str,
    short_window: int,
    long_window: int,
    rebalance: str,
    weight_mode: str,
    custom_raw: Optional[dict[str, float]],
):
    prices = fetch_multi_prices(tuple(tickers), start=start, end=end)
    returns = compute_returns(prices)

    if weight_mode == "Equal weights":
        weights = equal_weights(returns.columns)
    else:
        weights = normalize_weights(custom_raw, returns.columns)

    if strategy_choice == "Buy & Hold":
        alpha = BuyHoldAlpha()
    else:
        alpha = MovingAverageCrossAlpha(short_window=short_window, long_window=long_window)

    periods_per_year = 252
    backtester = PortfolioBacktester(risk_free_rate=rf_rate, periods_per_year=periods_per_year)

    use_rebalance = (strategy_choice == "Buy & Hold") and (rebalance != "none")
    if use_rebalance:
        positions = alpha.generate_positions(prices)
        equity = simulate_portfolio(prices, weights, rebalance=rebalance)
        portfolio_returns = equity.pct_change().fillna(0.0)
        asset_returns = compute_portfolio_returns(compute_returns(prices), weights)

        metrics_engine = PerformanceMetrics(risk_free_rate=rf_rate, periods_per_year=periods_per_year)
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

    return prices, returns, weights, positions, portfolio_returns, equity, metrics, alpha


def page_portfolio_main_metrics():
    page_header("Portfolio", "Portfolio & metrics")

    (
        tickers,
        start,
        end,
        rf_rate,
        strategy_choice,
        short_window,
        long_window,
        rebalance,
        weight_mode,
        custom_raw,
    ) = sidebar_portfolio_common()

    try:
        prices, returns, weights, positions, portfolio_returns, equity, metrics, alpha = run_portfolio_engine(
            tickers, start, end, rf_rate, strategy_choice, short_window, long_window, rebalance, weight_mode, custom_raw
        )
    except MarketDataError as e:
        st.error(str(e))
        st.stop()

    latest_prices = prices.iloc[-1]
    render_price_banner({ticker: latest_prices.get(ticker) for ticker in tickers})

    st.subheader("Performance metrics")
    metrics_grid(metrics)

    # Normalized comparison
    equity_aligned = equity.reindex(prices.index)
    if pd.isna(equity_aligned.iloc[0]):
        equity_aligned.iloc[0] = 1.0
    equity_aligned = equity_aligned.ffill()

    comparison_df = prices.div(prices.iloc[0]).copy()
    comparison_df["Portfolio"] = equity_aligned / float(equity_aligned.iloc[0])
    comparison_df.index.name = "Date"

    st.subheader("Strategy vs Asset Performance")
    st.line_chart(comparison_df, height=420)


def page_portfolio_single_vs():
    page_header("Portfolio", "Single vs portfolio")

    (
        tickers,
        start,
        end,
        rf_rate,
        strategy_choice,
        short_window,
        long_window,
        rebalance,
        weight_mode,
        custom_raw,
    ) = sidebar_portfolio_common()

    try:
        prices, returns, weights, positions, portfolio_returns, equity, metrics, alpha = run_portfolio_engine(
            tickers, start, end, rf_rate, strategy_choice, short_window, long_window, rebalance, weight_mode, custom_raw
        )
    except MarketDataError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Single asset strategy vs portfolio")
    asset_choice = st.selectbox("Asset", options=list(prices.columns), key="pf_asset_choice")

    single_series = prices[asset_choice].dropna()
    if single_series.empty:
        st.warning("No data available for the selected asset.")
        return

    single_backtester = Backtester(risk_free_rate=rf_rate, periods_per_year=252)
    single_result = single_backtester.run(single_series, alpha)
    single_strategy = single_result.equity_curve
    single_strategy_norm = single_strategy / float(single_strategy.iloc[0])

    portfolio_norm = equity.reindex(prices.index)
    if pd.isna(portfolio_norm.iloc[0]):
        portfolio_norm.iloc[0] = 1.0
    portfolio_norm = portfolio_norm.ffill()
    portfolio_norm = portfolio_norm / float(portfolio_norm.iloc[0])

    single_compare = pd.concat(
        [
            single_strategy_norm.rename(f"{asset_choice} Strategy"),
            portfolio_norm.rename("Portfolio"),
        ],
        axis=1,
    ).dropna()

    st.line_chart(single_compare, height=420)


def page_portfolio_correlation():
    page_header("Portfolio", "Correlation matrix")

    (
        tickers,
        start,
        end,
        rf_rate,
        strategy_choice,
        short_window,
        long_window,
        rebalance,
        weight_mode,
        custom_raw,
    ) = sidebar_portfolio_common()

    try:
        prices, returns, weights, positions, portfolio_returns, equity, metrics, alpha = run_portfolio_engine(
            tickers, start, end, rf_rate, strategy_choice, short_window, long_window, rebalance, weight_mode, custom_raw
        )
    except MarketDataError as e:
        st.error(str(e))
        st.stop()

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

    st.altair_chart((heat + text).configure_view(stroke=None), use_container_width=True)

    with st.expander("Raw correlation table"):
        st.dataframe(corr.round(3), use_container_width=True)


NAV_SECTIONS = {
    "single": {
        "label": "Single Asset — Quant A",
        "pages": {
            "strat-perf": ("Strat + Perf", page_single_strat_perf),
            "forecast": ("Forecast", page_single_forecast),
        },
    },
    "portfolio": {
        "label": "Portfolio",
        "pages": {
            "main-metrics": ("Main graph + metrics", page_portfolio_main_metrics),
            "single-vs": ("Single vs portfolio", page_portfolio_single_vs),
            "correlation": ("Correlation matrix", page_portfolio_correlation),
        },
    },
}


def _get_query_params() -> dict[str, list[str]]:
    if hasattr(st, "experimental_get_query_params"):
        return st.experimental_get_query_params()
    if hasattr(st, "query_params"):
        params: dict[str, list[str]] = {}
        for key in st.query_params:
            value = st.query_params.get(key)
            if isinstance(value, list):
                params[key] = value
            elif value is None:
                params[key] = []
            else:
                params[key] = [str(value)]
        return params
    return {}


def _get_param(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key, [])
    if not values:
        return default
    return values[0]


def _render_nav_html(section_key: str, page_key: str) -> None:
    top_links = []
    for key, meta in NAV_SECTIONS.items():
        default_page_key = next(iter(meta["pages"]))
        href = f"?section={key}&view={default_page_key}"
        cls = "nav-link active" if key == section_key else "nav-link"
        top_links.append(f'<a class="{cls}" href="{href}">{meta["label"]}</a>')

    sub_links = []
    for key, (label, _) in NAV_SECTIONS[section_key]["pages"].items():
        href = f"?section={section_key}&view={key}"
        cls = "nav-link active" if key == page_key else "nav-link"
        sub_links.append(f'<a class="{cls}" href="{href}">{label}</a>')

    st.markdown(
        f"""
<div class="nav-wrap">
  <div class="nav-top">{''.join(top_links)}</div>
  <div class="nav-sub">{''.join(sub_links)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_fallback_nav() -> dict[str, object]:
    params = _get_query_params()
    default_section = next(iter(NAV_SECTIONS))
    section_key = _get_param(params, "section", default_section)
    if section_key not in NAV_SECTIONS:
        section_key = default_section

    default_page = next(iter(NAV_SECTIONS[section_key]["pages"]))
    page_key = _get_param(params, "view", default_page)
    if page_key not in NAV_SECTIONS[section_key]["pages"]:
        page_key = default_page

    _render_nav_html(section_key, page_key)
    page_fn = NAV_SECTIONS[section_key]["pages"][page_key][1]
    return {"section": section_key, "page": page_key, "page_fn": page_fn}


# -----------------------
# MAIN ROUTER (pages/subpages)
# -----------------------
def main():
    st.set_page_config(
        page_title="Quant Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_minimal_dark()
    inject_auto_refresh_constant()
    init_session_defaults()

    # Navigation (fix: url_path WITHOUT slashes)
    if hasattr(st, "Page") and hasattr(st, "navigation"):
        pages = {
            "Single Asset": [
                st.Page(
                    page_single_strat_perf,
                    title="Strategy & metrics",
                    url_path="single-strat-perf",  
                    default=True,
                ),
                st.Page(
                    page_single_forecast,
                    title="Forecast",
                    url_path="single-forecast",  
                ),
            ],
            "Portfolio": [
                st.Page(
                    page_portfolio_main_metrics,
                    title="Portfolio & metrics",
                    url_path="portfolio-main",  
                ),
                st.Page(
                    page_portfolio_single_vs,
                    title="Single vs portfolio",
                    url_path="portfolio-single-vs",  
                ),
                st.Page(
                    page_portfolio_correlation,
                    title="Correlation matrix",
                    url_path="portfolio-correlation",  
                ),
            ],
        }
        nav = st.navigation(pages, position="top")
        nav.run()
    else:
        nav_state = render_fallback_nav()
        nav_state["page_fn"]()


if __name__ == "__main__":
    main()
