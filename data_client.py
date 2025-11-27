import datetime as dt
import os
import re
from typing import Optional

import pandas as pd


class MarketDataClient:
    """
    Fetch market data from a public source.

    Default source is Finnhub (listed in the professor's suggested sources).
    Set the environment variable `FINNHUB_API_KEY` to enable. For environments
    without a key, you can pass source="stooq" to use the public CSV fallback.
    """

    def __init__(self, source: str = "finnhub"):
        self.source = source.lower()

    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the column names of a given DataFrame based on a set of aliases.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to standardize.

        Returns
        -------
        pd.DataFrame
            A DataFrame with standardized column names.

        Raises
        ------
        ValueError
            If any of the columns in the aliases are missing or ambiguous.
        """
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

    def _fetch_from_stooq(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """
        Public Stooq CSV (daily data, no key required).
        """
        symbol = ticker.lower()
        if not symbol.endswith(".us") and len(symbol) <= 5:
            symbol = f"{symbol}.us"

        url = f"https://stooq.pl/q/d/l/?s={symbol}&i=d"
        df = pd.read_csv(url)
        if df.empty or "Data" not in df.columns:
            raise ValueError("No data returned from Stooq.")

        df = df.rename(
            columns={
                "Data": "date",
                "Otwarcie": "open",
                "Najwyzszy": "high",
                "Najnizszy": "low",
                "Zamkniecie": "close",
                "Wolumen": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        min_date = df.index.date.min()
        max_date = df.index.date.max()
        start = max(min(start, max_date), min_date)
        end = max(min(end, max_date), min_date)
        if start > end:
            start, end = min_date, max_date

        df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
        if df.empty:
            raise ValueError("No data found in Stooq for the requested range.")
        return df[["open", "high", "low", "close", "volume"]]

    def _fetch_from_finnhub(
        self,
        ticker: str,
        start: dt.date,
        end: dt.date,
        interval: str,
    ) -> pd.DataFrame:
        """
        Finnhub source (daily candles). Requires FINNHUB_API_KEY in env.
        """
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            raise ValueError("FINNHUB_API_KEY is not set.")

        import finnhub  # lazy import

        if interval != "1d":
            raise ValueError("Finnhub client currently supports only interval='1d'.")

        start_ts = int(dt.datetime.combine(start, dt.time()).timestamp())
        end_ts = int(dt.datetime.combine(end, dt.time(23, 59)).timestamp())

        res = finnhub.Client(api_key=api_key).stock_candles(
            symbol=ticker,
            resolution="D",
            _from=start_ts,
            to=end_ts,
        )

        if res.get("s") != "ok" or not res.get("t"):
            raise ValueError(f"Finnhub returned status {res.get('s')}")

        df = pd.DataFrame(res)
        df["date"] = pd.to_datetime(df["t"], unit="s")
        df = df.rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )
        df = df.set_index("date").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    def get_history(self, ticker: str, start: dt.date, end: dt.date, interval: str = "1d") -> pd.DataFrame:
        """
        Retrieve historical data for a given ticker between a start and end date.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the stock.
        start : dt.date
            The start date of the data to retrieve.
        end : dt.date
            The end date of the data to retrieve.
        interval : str, optional
            The interval at which the data is retrieved.
            Defaults to "1d".

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the historical data.

        Raises
        ------
        ValueError
            If no data is found for the given ticker and date range.
        """
        today = dt.date.today()
        end = min(end, today)
        if start > end:
            start = end - dt.timedelta(days=365)

        data: Optional[pd.DataFrame] = None
        errors = []

        if self.source == "finnhub":
            try:
                data = self._fetch_from_finnhub(ticker, start, end, interval)
            except Exception as exc:
                errors.append(f"Finnhub error: {exc}")
        elif self.source == "stooq":
            if interval != "1d":
                raise ValueError("Stooq supports only daily data; use interval='1d'.")
            try:
                data = self._fetch_from_stooq(ticker, start, end)
            except Exception as exc:
                errors.append(f"Stooq error: {exc}")
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

        if data is None or data.empty:
            raise ValueError(" | ".join(errors) if errors else "No data found.")

        data = self._standardize_columns(data)

        min_date = data.index.date.min()
        max_date = data.index.date.max()

        # Keep the requested window within the available data range.
        start = max(min(start, max_date), min_date)
        end = max(min(end, max_date), min_date)
        if start > end:
            start, end = min_date, max_date

        data = data.loc[(data.index.date >= start) & (data.index.date <= end)]
        data.dropna(inplace=True)

        if data.empty:
            raise ValueError("No data found in the requested range.")

        return data
