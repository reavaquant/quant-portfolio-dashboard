from abc import ABC, abstractmethod

import pandas as pd


class Alpha(ABC):
    name: str

    @abstractmethod
    def generate_positions(self, prices: pd.Series) -> pd.Series:
        pass


class BuyHoldAlpha(Alpha):
    def __init__(self):
        self.name = "Buy & Hold"

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=prices.index, name="position")


class MovingAverageCrossAlpha(Alpha):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise ValueError("short_window need to be strictly smaller than long_window.")
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"MA Cross {short_window}/{long_window}"

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        short_ma = prices.rolling(self.short_window).mean()
        long_ma = prices.rolling(self.long_window).mean()
        raw_signal = (short_ma > long_ma).astype(float)
        positions = raw_signal.ffill().fillna(0.0)
        positions.name = "position"
        return positions
