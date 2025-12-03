from dataclasses import dataclass
from typing import Dict, Optional

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
        positions: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        cagr = self._cagr(equity_curve)
        max_dd = self._max_drawdown(equity_curve)

        return {
            "Asset Total Return": self._total_return(asset_returns),
            "Strategy Total Return": equity_curve.iloc[-1] - 1.0,
            "Strategy CAGR": cagr,
            "Strategy Volatility (ann.)": self._volatility(strategy_returns),
            "Strategy Sharpe": self._sharpe(strategy_returns),
            "Strategy Sortino": self._sortino(strategy_returns),
            "Upside Potential Ratio": self._upside_potential_ratio(strategy_returns),
            "Omega Ratio": self._omega_ratio(strategy_returns),
            "Strategy Max Drawdown": max_dd,
            "Sterling Ratio": self._sterling_ratio(cagr, max_dd),
            "Calmar Ratio": self._calmar_ratio(cagr, max_dd),
            "Strategy Turnover": self._turnover(positions),
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

    def _sortino(self, returns: pd.Series) -> float:
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess = returns - rf_per_period
        downside = excess[excess < 0]
        downside_dev = np.sqrt((downside**2).mean()) if len(downside) > 0 else 0.0
        if downside_dev == 0:
            return float("nan")
        return float(np.sqrt(self.periods_per_year) * excess.mean() / downside_dev)

    def _upside_potential_ratio(self, returns: pd.Series) -> float:
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess = returns - rf_per_period
        upside = excess[excess > 0]
        downside = excess[excess < 0]
        downside_dev = np.sqrt((downside**2).mean()) if len(downside) > 0 else 0.0
        if downside_dev == 0:
            return float("nan")
        if len(upside) == 0:
            return 0.0
        return float(upside.mean() / downside_dev)

    def _omega_ratio(self, returns: pd.Series) -> float:
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess = returns - rf_per_period
        numerator = excess[excess > 0].sum()
        denominator = (-excess[excess < 0]).sum()
        if denominator == 0:
            return float("nan")
        return float(numerator / denominator)

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0
        return float(drawdown.min())

    def _cagr(self, equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return float("nan")
        years = len(equity_curve) / self.periods_per_year
        if years <= 0 or equity_curve.iloc[0] <= 0:
            return float("nan")
        return float(equity_curve.iloc[-1] ** (1 / years) - 1.0)

    def _sterling_ratio(self, cagr: float, max_drawdown: float) -> float:
        if max_drawdown >= 0:
            return float("nan")
        return float((cagr - self.risk_free_rate) / abs(max_drawdown))

    def _calmar_ratio(self, cagr: float, max_drawdown: float) -> float:
        if max_drawdown >= 0:
            return float("nan")
        return float(cagr / abs(max_drawdown))

    def _turnover(self, positions: Optional[pd.Series]) -> float:
        if positions is None or len(positions) < 2:
            return float("nan")
        changes = positions.diff().abs().fillna(0.0)
        # average absolute position change per period
        return float(changes.sum() / (len(changes) - 1))


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    asset_returns: pd.Series
    metrics: Dict[str, float]
    positions: pd.Series


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

        metrics = self.metrics_engine.compute(asset_returns, strategy_returns, equity_curve, positions)

        return BacktestResult(
            equity_curve=equity_curve,
            strategy_returns=strategy_returns,
            asset_returns=asset_returns,
            metrics=metrics,
            positions=positions,
        )
