import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import pandas as pd
import requests


class MarketDataError(Exception):
    """Market data error."""


@dataclass
class MarketDataSettings:
    default_ticker: Optional[str] = None


class MarketDataClient:
    _interval_map = {
        "1d": "1d",
        "1day": "1d",
        "daily": "1d",
        "1h": "1h",
        "60m": "1h",
        "30m": "30m",
        "15m": "15m",
        "5m": "5m",
    }

    def __init__(self, settings: Optional[MarketDataSettings] = None):
        self.settings = settings or MarketDataSettings()

    def get_history(
        self,
        ticker: Optional[str] = None,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
        interval: str = "1d",
    ):
        ticker = self._resolve_ticker(ticker)
        interval = self._validate_interval(interval)
        start, end = self._normalize_dates(start, end)

        df = self._download_yfinance(ticker, start, end, interval)
        frame = self._select_ticker_frame(df, ticker)
        frame = self._ensure_datetime_index(frame)
        frame = self._slice_date_range(frame, start, end)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise MarketDataError(
                f"Yahoo Finance missing columns for {ticker}: {missing}"
            )

        ordered = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in frame.columns]
        frame = frame[ordered].sort_index()

        if frame.empty:
            raise MarketDataError(f"No data for {ticker} between {start} and {end}")

        return frame

    def get_multi_asset_prices(
        self,
        tickers: Iterable[str],
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
        interval: str = "1d",
    ):
        tickers = self._normalize_tickers(tickers)
        if len(tickers) < 2:
            raise MarketDataError("At least two tickers are required.")

        interval = self._validate_interval(interval)
        start, end = self._normalize_dates(start, end)

        df = self._download_yfinance(tickers, start, end, interval)
        df = self._ensure_datetime_index(df)

        prices = df.xs("Close", level=1, axis=1)
        prices = prices[tickers]
        prices = self._slice_date_range(prices, start, end)
        prices = prices.dropna(how="all").sort_index()

        if prices.empty:
            raise MarketDataError("Multi-asset price matrix is empty after cleaning.")

        return prices

    def get_last_price(self, ticker: Optional[str] = None):
        df = self.get_history(ticker=ticker)
        return float(df["Close"].iloc[-1])

    def get_latest_price(self, ticker: Optional[str] = None):
        return self.get_last_price(ticker)

    def _normalize_tickers(self, tickers: Iterable[str]):
        if tickers is None:
            return []
        if isinstance(tickers, str):
            raw = [tickers]
        else:
            raw = list(tickers)
        cleaned = [str(t).strip().upper() for t in raw if str(t).strip()]
        return sorted(set(cleaned))

    def _resolve_ticker(self, ticker: Optional[str]):
        resolved = ticker or self.settings.default_ticker
        if resolved is None:
            raise MarketDataError("Ticker is required.")
        return resolved.strip().upper()

    def _validate_interval(self, interval: str):
        normalized = self._interval_map.get(interval.lower())
        if normalized is None:
            raise MarketDataError(
                "Unsupported interval. Use 5m, 15m, 30m, 1h or 1d."
            )
        return normalized

    def _normalize_dates(
        self, start: Optional[dt.date], end: Optional[dt.date]
    ):
        today = dt.date.today()
        if end is None or end > today:
            end = today
        if start is None:
            start = end - dt.timedelta(days=365)
        if start > end:
            raise MarketDataError("Start date must be <= end date.")
        return start, end

    def _ensure_datetime_index(self, df: pd.DataFrame):
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df

    def _slice_date_range(
        self, df: pd.DataFrame, start: dt.date, end: dt.date
    ):
        if df.empty:
            return df
        mask = (df.index.date >= start) & (df.index.date <= end)
        return df.loc[mask]

    def _select_ticker_frame(self, df: pd.DataFrame, ticker: str):
        if not isinstance(df.columns, pd.MultiIndex):
            return df

        if df.columns.nlevels != 2:
            raise MarketDataError("Unexpected Yahoo Finance column format.")

        level0 = df.columns.get_level_values(0)
        if ticker not in level0:
            raise MarketDataError("Unexpected Yahoo Finance column format.")

        return df[ticker]

    def _download_yfinance(
        self,
        tickers,
        start: dt.date,
        end: dt.date,
        interval: str,
    ):
        try:
            import yfinance as yf
        except ImportError as exc:
            raise MarketDataError("yfinance is not installed.") from exc

        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                end=end + dt.timedelta(days=1),
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except requests.exceptions.RequestException as exc:
            raise MarketDataError(f"Yahoo Finance request failed: {exc}") from exc
        except Exception as exc:
            raise MarketDataError(f"Yahoo Finance error: {exc}") from exc

        if df is None or df.empty:
            raise MarketDataError("Yahoo Finance returned no data.")

        return df
