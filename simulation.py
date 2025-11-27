from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from alphas import Alpha


class PerformanceMetrics:
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def compute(
        self,
        asset_returns: pd.Series,
        strategy_returns: pd.Series,
        equity_curve: pd.Series,
    ) -> Dict[str, float]:
        return {
            "Asset Total Return": self._total_return(asset_returns),
            "Strategy Total Return": equity_curve.iloc[-1] - 1.0,
            "Strategy Volatility (ann.)": self._volatility(strategy_returns),
            "Strategy Sharpe": self._sharpe(strategy_returns),
            "Strategy Max Drawdown": self._max_drawdown(equity_curve),
        }

    def _total_return(self, returns: pd.Series) -> float:
        return float((1 + returns).prod() - 1.0)

    def _volatility(self, returns: pd.Series) -> float:
        return float(returns.std() * np.sqrt(self.periods_per_year))

    def _sharpe(self, returns: pd.Series) -> float:
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess = returns - rf_per_period
        vol = excess.std()
        if vol == 0 or np.isnan(vol):
            return float("nan")
        return float(np.sqrt(self.periods_per_year) * excess.mean() / vol)

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0
        return float(drawdown.min())


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    asset_returns: pd.Series
    metrics: Dict[str, float]


class Backtester:
    def __init__(self, risk_free_rate: float = 0.0):
        self.metrics_engine = PerformanceMetrics(risk_free_rate=risk_free_rate)

    def run(self, close_prices: pd.Series, strategy: Alpha) -> BacktestResult:
        """
        Run a backtest on a given strategy and set of close prices.

        Parameters
        ----------
        close_prices : pd.Series
            A series of close prices.
        strategy : Alpha
            The strategy to backtest.

        Returns
        -------
        BacktestResult
            A BacktestResult object containing the equity curve, strategy returns,
            asset returns and metrics for the backtest.
        """
        prices = close_prices.dropna()
        positions = strategy.generate_positions(prices)
        positions = positions.reindex(prices.index).ffill().fillna(0.0)

        asset_returns = prices.pct_change().fillna(0.0)
        strategy_returns = positions.shift(1).fillna(0.0) * asset_returns
        equity_curve = (1 + strategy_returns).cumprod()
        equity_curve.name = "Strategy Equity"

        metrics = self.metrics_engine.compute(asset_returns, strategy_returns, equity_curve)

        return BacktestResult(
            equity_curve=equity_curve,
            strategy_returns=strategy_returns,
            asset_returns=asset_returns,
            metrics=metrics,
        )
