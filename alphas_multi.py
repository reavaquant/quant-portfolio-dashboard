from abc import ABC, abstractmethod
import pandas as pd


class Alpha(ABC):
    name: str

    @abstractmethod
    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        prices: DataFrame (Date x Assets)
        returns: positions DataFrame (Date x Assets) with values in {0,1} (or [0,1])
        """
        raise NotImplementedError


class BuyHoldAlpha(Alpha):
    def __init__(self):
        self.name = "Buy & Hold"

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices is None or prices.empty:
            raise ValueError("prices is empty")
        return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)


class MovingAverageCrossAlpha(Alpha):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly smaller than long_window.")
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"MA Cross {short_window}/{long_window}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices is None or prices.empty:
            raise ValueError("prices is empty")

        short_ma = prices.rolling(self.short_window).mean()
        long_ma = prices.rolling(self.long_window).mean()

        raw_signal = (short_ma > long_ma).astype(float)

        # forward-fill signals, then fill initial NaNs with 0 (out of market)
        positions = raw_signal.ffill().fillna(0.0)
        return positions
