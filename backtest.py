from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from alphas import Alpha
from portfolio import (
    annualized_volatility,
    compute_portfolio_returns,
    compute_returns,
    equity_curve,
    max_drawdown,
    strategy_portfolio_returns,
)


class PerformanceMetrics:
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def compute(self, asset_returns, strategy_returns, equity, positions = None):
        turnover = self._turnover(positions)
        return self._compute_base(asset_returns, strategy_returns, equity, turnover)

    def compute_portfolio(self, asset_returns, strategy_returns, equity,positions = None, weights = None,use_lookahead_safe_shift = True):
        turnover = self._turnover_portfolio(positions, weights, use_lookahead_safe_shift)
        return self._compute_base(asset_returns, strategy_returns, equity, turnover)

    def _compute_base(
        self,
        asset_returns: pd.Series,
        strategy_returns: pd.Series,
        equity: pd.Series,
        turnover: float,
    ):
        cagr = self._cagr(equity)
        max_dd = max_drawdown(equity)

        return {
            "Asset Total Return": self._total_return(asset_returns),
            "Strategy Total Return": equity.iloc[-1] - 1.0,
            "Strategy CAGR": cagr,
            "Strategy Volatility (ann.)": self._volatility(strategy_returns),
            "Strategy Sharpe": self._sharpe(strategy_returns),
            "Strategy Sortino": self._sortino(strategy_returns),
            "Upside Potential Ratio": self._upside_potential_ratio(strategy_returns),
            "Omega Ratio": self._omega_ratio(strategy_returns),
            "Strategy Max Drawdown": max_dd,
            "Sterling Ratio": self._sterling_ratio(cagr, max_dd),
            "Calmar Ratio": self._calmar_ratio(cagr, max_dd),
            "Strategy Turnover": turnover,
        }

    def _total_return(self, returns: pd.Series):
        return float((1 + returns).prod() - 1.0)

    def _volatility(self, returns: pd.Series):
        return annualized_volatility(returns, periods_per_year=self.periods_per_year)

    def _sharpe(self, returns: pd.Series):
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess = returns - rf_per_period
        vol = excess.std()
        if vol == 0 or np.isnan(vol):
            return float("nan")
        return float(np.sqrt(self.periods_per_year) * excess.mean() / vol)

    def _sortino(self, returns: pd.Series):
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess = returns - rf_per_period
        downside = excess[excess < 0]
        downside_dev = np.sqrt((downside**2).mean()) if len(downside) > 0 else 0.0
        if downside_dev == 0:
            return float("nan")
        return float(np.sqrt(self.periods_per_year) * excess.mean() / downside_dev)

    def _upside_potential_ratio(self, returns: pd.Series):
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

    def _omega_ratio(self, returns: pd.Series):
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess = returns - rf_per_period
        numerator = excess[excess > 0].sum()
        denominator = (-excess[excess < 0]).sum()
        if denominator == 0:
            return float("nan")
        return float(numerator / denominator)

    def _cagr(self, equity: pd.Series):
        if len(equity) < 2:
            return float("nan")

        idx = pd.to_datetime(equity.index)
        delta_days = (idx[-1] - idx[0]).days + (idx[-1] - idx[0]).seconds / 86400
        years = delta_days / 365.25
        if years <= 0 or equity.iloc[0] <= 0:
            return float("nan")
        return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)

    def _sterling_ratio(self, cagr: float, max_drawdown_val: float):
        if max_drawdown_val >= 0:
            return float("nan")
        return float((cagr - self.risk_free_rate) / abs(max_drawdown_val))

    def _calmar_ratio(self, cagr: float, max_drawdown_val: float):
        if max_drawdown_val >= 0:
            return float("nan")
        return float(cagr / abs(max_drawdown_val))

    def _turnover(self, positions: Optional[pd.Series]):
        if positions is None or len(positions) < 2:
            return float("nan")
        changes = positions.diff().abs().fillna(0.0)
        return float(changes.sum() / (len(changes) - 1))

    def _turnover_portfolio(
        self,
        positions: Optional[pd.DataFrame],
        weights: Optional[pd.Series],
        use_lookahead_safe_shift: bool,
    ):
        if positions is None or positions.empty or len(positions) < 2:
            return float("nan")
        if weights is None or weights.empty:
            return float("nan")

        pos = positions.copy()
        if use_lookahead_safe_shift:
            pos = pos.shift(1).fillna(0.0)

        w = weights.reindex(pos.columns).fillna(0.0)
        if float(w.sum()) == 0.0:
            return float("nan")
        w = w / float(w.sum())

        w_eff = pos.mul(w, axis=1)
        changes = w_eff.diff().abs().sum(axis=1).fillna(0.0)
        return float(changes.sum() / (len(changes) - 1))


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    asset_returns: pd.Series
    metrics: Dict[str, float]
    positions: pd.Series


@dataclass
class PortfolioBacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    asset_returns: pd.Series
    metrics: Dict[str, float]
    positions: pd.DataFrame
    weights: pd.Series


class Backtester:
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.metrics_engine = PerformanceMetrics(
            risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        )

    def run(self, close_prices: pd.Series, strategy: Alpha) -> BacktestResult:
        """
        Run a backtest on a given strategy and set of close prices.
        """
        prices = close_prices.dropna()
        prices_df = prices.to_frame(name=prices.name or "price")
        positions_df = strategy.generate_positions(prices_df)
        positions = positions_df.iloc[:, 0].reindex(prices.index).ffill().fillna(0.0)
        positions.name = "position"

        asset_returns = prices.pct_change().fillna(0.0)
        strategy_returns = positions.shift(1).fillna(0.0) * asset_returns
        equity = (1 + strategy_returns).cumprod()
        equity.name = "Strategy Equity"

        metrics = self.metrics_engine.compute(asset_returns, strategy_returns, equity, positions)

        return BacktestResult(
            equity_curve=equity,
            strategy_returns=strategy_returns,
            asset_returns=asset_returns,
            metrics=metrics,
            positions=positions,
        )


class PortfolioBacktester:
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.metrics_engine = PerformanceMetrics(
            risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        )

    def run(
        self,
        prices: pd.DataFrame,
        base_weights: pd.Series,
        strategy: Alpha,
        use_lookahead_safe_shift: bool = True,
    ) -> PortfolioBacktestResult:
        if prices is None or prices.empty:
            raise ValueError("prices is empty")

        prices = prices.dropna(how="all")
        positions = strategy.generate_positions(prices)
        positions = positions.reindex(prices.index).reindex(columns=prices.columns).ffill().fillna(0.0)

        weights = base_weights.reindex(prices.columns).fillna(0.0)
        if float(weights.sum()) == 0.0:
            raise ValueError("weights sum to 0")
        weights = weights / float(weights.sum())

        portfolio_returns = strategy_portfolio_returns(
            prices, weights, positions, use_lookahead_safe_shift=use_lookahead_safe_shift
        )
        equity = equity_curve(portfolio_returns)

        asset_returns = compute_portfolio_returns(compute_returns(prices), weights)

        metrics = self.metrics_engine.compute_portfolio(
            asset_returns,
            portfolio_returns,
            equity,
            positions=positions,
            weights=weights,
            use_lookahead_safe_shift=use_lookahead_safe_shift,
        )

        return PortfolioBacktestResult(
            equity_curve=equity,
            strategy_returns=portfolio_returns,
            asset_returns=asset_returns,
            metrics=metrics,
            positions=positions,
            weights=weights,
        )
