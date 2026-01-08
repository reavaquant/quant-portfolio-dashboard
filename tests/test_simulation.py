import numpy as np
import pandas as pd

from alphas import BuyHoldAlpha, MovingAverageCrossAlpha
from simulation import Backtester, PerformanceMetrics


def test_moving_average_cross_alpha_positions():
    prices = pd.DataFrame(
        {"Asset": [1, 2, 3, 2, 1, 2]},
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )
    alpha = MovingAverageCrossAlpha(short_window=1, long_window=2)
    positions = alpha.generate_positions(prices)["Asset"]
    # Should flip when short MA crosses above long MA
    assert positions.iloc[0] == 0.0
    assert positions.iloc[2] == 1.0
    assert positions.iloc[-2] == 0.0


def test_backtester_metrics_use_calendar_time():
    # Same total return (10%) but different calendar durations -> longer duration => lower CAGR
    short_dates = pd.date_range("2024-01-01", periods=2, freq="D")
    long_dates = pd.date_range("2024-01-01", periods=2, freq="30D")

    prices_short = pd.Series([100, 110], index=short_dates)
    prices_long = pd.Series([100, 110], index=long_dates)

    bt = Backtester()
    cagr_short = bt.run(prices_short, BuyHoldAlpha()).metrics["Strategy CAGR"]
    cagr_long = bt.run(prices_long, BuyHoldAlpha()).metrics["Strategy CAGR"]

    assert cagr_long < cagr_short


def test_turnover_calculation():
    pm = PerformanceMetrics()
    positions = pd.Series([0, 1, 0.5, 0.5], index=pd.date_range("2024-01-01", periods=4, freq="D"))
    turnover = pm._turnover(positions)
    expected = (1 + 0.5) / 3  # abs deltas: [1,0.5,0]
    assert np.isclose(turnover, expected)
