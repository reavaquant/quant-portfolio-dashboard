import datetime as dt
import re

import pandas as pd
import yfinance as yf


class MarketDataClient:
    def __init__(self, source: str = "yfinance"):
        self.source = source

    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize vendor-specific column names to open/high/low/close/volume."""
        def normalize(col: str) -> str:
            return re.sub(r"[^a-z]", "", col.lower())

        aliases = {
            "open": {"open", "o", "priceopen"},
            "high": {"high", "h", "pricehigh"},
            "low": {"low", "l", "pricelow"},
            "close": {"close", "c", "priceclose", "adjclose"},
            "volume": {"volume", "v", "vol"},
        }

        normalized_lookup = {normalize(col): col for col in data.columns}
        rename_map = {}
        missing = []

        for target, candidates in aliases.items():
            match = next((normalized_lookup[c] for c in candidates if c in normalized_lookup), None)
            if match is None:
                missing.append(target)
                continue
            rename_map[match] = target

        if missing:
            raise ValueError(
                f"Missing columns or ambiguous column names : {', '.join(missing)}."
            )

        standardized = data.rename(columns=rename_map)
        return standardized[["open", "high", "low", "close", "volume"]]

    def get_history(self, ticker: str, start: dt.date, end: dt.date, interval: str = "1d") -> pd.DataFrame:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            raise ValueError("No data found.")

        data = self._standardize_columns(data)
        data.dropna(inplace=True)
        return data
