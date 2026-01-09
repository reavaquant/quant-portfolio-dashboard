from abc import ABC, abstractmethod

import pandas as pd


def _validate_prices(prices: pd.DataFrame) -> None:
    if prices is None or prices.empty:
        raise ValueError("prices is empty")


class Alpha(ABC):
    name: str

    @abstractmethod
    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class BuyHoldAlpha(Alpha):
    def __init__(self):
        self.name = "Buy & Hold"

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        _validate_prices(prices)
        return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)


class MovingAverageCrossAlpha(Alpha):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly smaller than long_window.")
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"MA Cross {short_window}/{long_window}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        _validate_prices(prices)
        short_ma = prices.rolling(self.short_window).mean()
        long_ma = prices.rolling(self.long_window).mean()
        raw_signal = (short_ma > long_ma).astype(float)
        positions = raw_signal.ffill().fillna(0.0)
        return positions
