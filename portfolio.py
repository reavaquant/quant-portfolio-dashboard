from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Optional

import numpy as np
import pandas as pd


class PortfolioError(Exception):
    pass


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple returns from a price matrix (date x assets).
    """
    if prices is None or prices.empty:
        raise PortfolioError("Prices are empty.")

    returns = prices.pct_change().dropna(how="all")
    # replace potential inf/-inf (shouldn't happen often, but safe)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return returns


def normalize_weights(weights: Dict[str, float], columns: Sequence[str]) -> pd.Series:
    """
    Normalize a dict of weights to match asset columns and sum to 1.
    Missing assets get 0. Extra keys are ignored.
    """
    w = pd.Series({c: float(weights.get(c, 0.0)) for c in columns})

    total = float(w.sum())
    if total == 0.0:
        raise PortfolioError("Sum of weights is 0. Provide at least one non-zero weight.")
    return w / total


def equal_weights(columns: Sequence[str]) -> pd.Series:
    """
    Equal-weight vector for given asset names.
    """
    n = len(columns)
    if n < 2:
        raise PortfolioError("Need at least 2 assets for a portfolio.")
    return pd.Series(1.0 / n, index=list(columns))


def compute_portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Weighted portfolio returns: r_p(t) = sum_i w_i * r_i(t)
    """
    if returns is None or returns.empty:
        raise PortfolioError("Returns are empty.")

    weights = weights.reindex(returns.columns).fillna(0.0)
    if float(weights.sum()) == 0.0:
        raise PortfolioError("Weights sum to 0 after alignment with returns columns.")

    pr = (returns * weights).sum(axis=1)
    pr.name = "Portfolio Return"
    return pr


def equity_curve(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """
    Convert returns into an equity curve (cumulative value).
    """
    if returns is None or returns.empty:
        raise PortfolioError("Returns are empty.")
    eq = initial_value * (1.0 + returns).cumprod()
    eq.name = "Portfolio Equity"
    return eq

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    if returns is None or returns.empty:
        raise PortfolioError("Returns are empty.")
    return returns.corr()


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns is None or returns.empty:
        raise PortfolioError("Returns are empty.")
    return float(returns.std() * np.sqrt(periods_per_year))


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Geometric annualized return from periodic returns.
    """
    if returns is None or returns.empty:
        raise PortfolioError("Returns are empty.")
    total = float((1.0 + returns).prod())
    n = len(returns)
    if n == 0:
        raise PortfolioError("Returns length is 0.")
    return float(total ** (periods_per_year / n) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        raise PortfolioError("Equity curve is empty.")
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())

def simulate_portfolio(
    prices: pd.DataFrame,
    target_weights: pd.Series,
    rebalance: str = "none",
    initial_value: float = 1.0,
) -> pd.Series:
    """
    Simulate a portfolio equity curve with optional rebalancing.
    rebalance: "none", "weekly", "monthly"
    """
    if prices is None or prices.empty:
        raise PortfolioError("Prices are empty.")

    prices = prices.dropna(how="all").copy()
    rets = compute_returns(prices)

    w = target_weights.reindex(rets.columns).fillna(0.0)
    if float(w.sum()) == 0.0:
        raise PortfolioError("Target weights sum to 0 after alignment.")
    w = w / float(w.sum())

    rebalance = rebalance.lower().strip()
    if rebalance not in {"none", "weekly", "monthly"}:
        raise PortfolioError("rebalance must be one of: none, weekly, monthly")

    equity = pd.Series(index=rets.index, dtype=float, name="Portfolio Equity")


    holdings = initial_value * w
    equity.iloc[0] = initial_value


    if rebalance == "none":
        rebalance_dates = set()
    elif rebalance == "weekly":
        rebalance_dates = set(rets.resample("W").first().index)
    else: 
        rebalance_dates = set(rets.resample("M").first().index)

    for i in range(1, len(rets)):
        date = rets.index[i]
        r = rets.iloc[i]

        holdings = holdings * (1.0 + r)
        total_value = float(holdings.sum())
        equity.iloc[i] = total_value

        if date in rebalance_dates:
            holdings = total_value * w

    equity = equity.ffill()
    return equity

def strategy_portfolio_returns(
    prices: pd.DataFrame,
    base_weights: pd.Series,
    positions: pd.DataFrame,
    use_lookahead_safe_shift: bool = True,
) -> pd.Series:
    """
    Convert strategy positions (0/1 per asset) into portfolio returns.
    Uninvested part stays in cash (0% return).
    """
    if prices is None or prices.empty:
        raise PortfolioError("Prices are empty.")
    if positions is None or positions.empty:
        raise PortfolioError("Positions are empty.")

    rets = compute_returns(prices)

    positions = positions.reindex(rets.index).reindex(columns=rets.columns).fillna(0.0)
    w = base_weights.reindex(rets.columns).fillna(0.0)
    if float(w.sum()) == 0.0:
        raise PortfolioError("Base weights sum to 0.")

    w = w / float(w.sum())

    if use_lookahead_safe_shift:
        positions = positions.shift(1).fillna(0.0)

    w_eff = positions.mul(w, axis=1)

    pr = (rets * w_eff).sum(axis=1)
    pr.name = "Portfolio Return"
    return pr
