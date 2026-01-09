from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    forecast: pd.Series
    lower: pd.Series
    upper: pd.Series


class TrendRegressionModel:
    def __init__(self, horizon: int = 10, use_log: bool = True):
        self.horizon = horizon
        self.use_log = use_log

    def fit_predict(self, prices: pd.Series):
        y = prices.dropna()
        if len(y) < 20:
            return None

        if self.use_log:
            y_trans = np.log(y.values)
        else:
            y_trans = y.values

        x = np.arange(len(y_trans))
        coeffs = np.polyfit(x, y_trans, 1)
        trend = np.poly1d(coeffs)

        x_future = np.arange(len(y_trans), len(y_trans) + self.horizon)
        y_future = trend(x_future)

        residuals = y_trans - trend(x)
        sigma = residuals.std(ddof=2)
        ci = 2 * sigma

        future_index = pd.date_range(
            start=y.index[-1] + pd.Timedelta(days=1),
            periods=self.horizon,
            freq=pd.infer_freq(y.index) or "D",
        )

        if self.use_log:
            forecast_vals = np.exp(y_future)
            lower_vals = np.exp(y_future - ci)
            upper_vals = np.exp(y_future + ci)
        else:
            forecast_vals = y_future
            lower_vals = y_future - ci
            upper_vals = y_future + ci

        forecast = pd.Series(forecast_vals, index=future_index, name="forecast")
        lower = pd.Series(lower_vals, index=future_index, name="lower")
        upper = pd.Series(upper_vals, index=future_index, name="upper")

        return ForecastResult(forecast=forecast, lower=lower, upper=upper)
