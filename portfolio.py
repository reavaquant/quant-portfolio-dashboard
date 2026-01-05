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
