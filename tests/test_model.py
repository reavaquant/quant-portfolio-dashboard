import numpy as np
import pandas as pd

from model import TrendRegressionModel


def test_trend_regression_requires_min_length():
    model = TrendRegressionModel(horizon=5, use_log=True)
    short_series = pd.Series([1, 2, 3, 4], index=pd.date_range("2024-01-01", periods=4, freq="D"))
    assert model.fit_predict(short_series) is None


def test_trend_regression_outputs_forecast_series():
    model = TrendRegressionModel(horizon=5, use_log=True)
    prices = pd.Series(
        np.linspace(100, 120, 50),
        index=pd.date_range("2024-01-01", periods=50, freq="D"),
    )
    result = model.fit_predict(prices)
    assert result is not None
    assert len(result.forecast) == 5
    assert result.forecast.index.is_monotonic_increasing
