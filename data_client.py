import os
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import requests


class MarketDataError(Exception):
    """Erreur métier pour les accès marché."""


@dataclass
class MarketDataSettings:
    source: str = "yfinance"      # "yfinance", "csv", "bloomberg"
    default_ticker: Optional[str] = None
    csv_path: Optional[str] = None    # dossier où tu ranges tes CSV
    bloomberg_host: str = "localhost"
    bloomberg_port: int = 8194
    bloomberg_service: str = "//blp/refdata"


class MarketDataClient:
    def __init__(self, settings: Optional[MarketDataSettings] = None):
        self.settings = settings or MarketDataSettings()
        # cache par source + ticker pour limiter les appels (surtout Yahoo Finance)
        self._cache: Dict[Tuple, pd.DataFrame] = {}

    def get_history(self, ticker: Optional[str] = None, start: Optional[dt.date] = None, end: Optional[dt.date] = None, interval: str = "1d") -> pd.DataFrame:
        """
        Retrieve a pandas DataFrame containing historical market data for a given ticker.

        Parameters
        ----------
        ticker : Optional[str]
            The ticker to retrieve data for. If not provided, the default ticker from the settings will be used.
        start : Optional[dt.date]
            The start date for the data. If not provided, data will be retrieved from the earliest available date.
        end : Optional[dt.date]
            The end date for the data. If not provided, data will be retrieved up to the latest available date.
        interval : str
            The interval at which to retrieve data. Must be one of the following: "1m", "2m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "5d", "1wk", "1mo", "3mo". Defaults to "1d".

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the historical market data for the given ticker and interval.

        Raises
        -------
        MarketDataError
            If there is an error retrieving the data (e.g. invalid ticker, invalid interval, etc.).
        """
        ticker = self._resolve_ticker(ticker)
        self._validate_interval(interval)
        start, end = self._normalize_dates(start, end)

        source = self.settings.source.lower()
        if source == "yfinance":
            df = self._get_from_yfinance(ticker, start, end)
        elif source == "csv":
            df = self._get_from_csv(ticker)
        elif source == "bloomberg":
            df = self._get_from_bloomberg(ticker, start, end)
        else:
            raise MarketDataError(f"Source inconnue: {source}")

        df = self._standardize_columns(df)
        mask = (df.index.date >= start) & (df.index.date <= end)
        df = df.loc[mask]

        if df.empty:
            raise MarketDataError(f"Aucune donnée pour {ticker} entre {start} et {end}")

        return df.sort_index()

    def get_last_price(self, ticker: Optional[str] = None) -> float:
        """
        Return the last available price for the given ticker.

        Parameters
        ----------
        ticker : Optional[str]
            The ticker for which to retrieve the last price. If not provided, uses the default ticker.

        Returns
        -------
        float
            The last available price for the given ticker.

        Raises
        -------
        MarketDataError
            If there is an error retrieving the data (e.g. invalid ticker, etc.).
        """
        df = self.get_history(ticker=ticker)
        return float(df["Close"].iloc[-1])

    def get_latest_price(self, ticker: Optional[str] = None) -> float:
        # alias utilisé par l'app Streamlit
        """
        Alias for get_last_price used by the Streamlit app.

        Parameters
        ----------
        ticker : Optional[str]
            The ticker for which to retrieve the last price. If not provided, uses the default ticker.

        Returns
        -------
        float
            The last available price for the given ticker.

        Raises
        -------
        MarketDataError
            If there is an error retrieving the data (e.g. invalid ticker, etc.).
        """
        return self.get_last_price(ticker)
    
    def _col_to_str(self, col) -> Optional[str]:
        """
        Convertit un label de colonne (str, tuple, autre) en nom de colonne exploitable.
        - Si c'est déjà une str : on renvoie tel quel.
        - Si c'est un tuple (MultiIndex) : on cherche la partie qui ressemble à open/high/low/close/volume.
        - Sinon : on cast en str.
        """
        if isinstance(col, str):
            return col

        if isinstance(col, tuple):
            candidates = [p for p in col if isinstance(p, str)]
            if not candidates:
                return None

            priority_words = ["open", "high", "low", "close", "last", "adj", "vol", "volume"]
            for p in candidates:
                low = p.lower()
                if any(word in low for word in priority_words):
                    return p

            return candidates[-1]

        return str(col)


    def _resolve_ticker(self, ticker: Optional[str]) -> str:
        """
        Resolve the given ticker to a valid ticker.

        If the ticker is None, use the default ticker from the settings.
        If the resolved ticker is None, raise a MarketDataError.

        Returns
        -------
        str
            The resolved ticker.
        """
        resolved = ticker or self.settings.default_ticker
        if resolved is None:
            raise MarketDataError("Ticker manquant")
        return resolved

    def _validate_interval(self, interval: str) -> None:
        """
        Validate that the given interval is valid.

        Parameters
        ----------
        interval : str
            The interval to validate.

        Raises
        -------
        MarketDataError
            If the interval is not valid (i.e. not one of "1d", "1day", "daily").
        """
        if interval.lower() not in {"1d", "1day", "daily"}:
            raise MarketDataError("Seul l'intervalle daily (1d) est supporté ici.")

    def _normalize_dates(self, start: Optional[dt.date], end: Optional[dt.date]) -> Tuple[dt.date, dt.date]:
        """
        Normalize the given start and end dates to be within valid range.

        If the end date is None or greater than today, set it to today.
        If the start date is None, set it to the end date minus one year.
        If the start date is greater than the end date, raise a MarketDataError.

        Parameters
        ----------
        start : Optional[dt.date]
            The start date to normalize.
        end : Optional[dt.date]
            The end date to normalize.

        Returns
        -------
        Tuple[dt.date, dt.date]
            A tuple containing the normalized start and end dates.
        """
        today = dt.date.today()
        if end is None or end > today:
            end = today
        if start is None:
            start = end - dt.timedelta(days=365)
        if start > end:
            raise MarketDataError("La date de début doit être <= date de fin.")
        return start, end

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # index en datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Si MultiIndex → on aplanit en strings "lisibles"
        if isinstance(df.columns, pd.MultiIndex):
            flat_cols = []
            for col in df.columns:
                name = self._col_to_str(col)
                flat_cols.append(name if name is not None else str(col))
            df = df.copy()
            df.columns = flat_cols

        # Si ce n'est pas un MultiIndex mais des types chelous, on convertit aussi
        else:
            new_cols = []
            for col in df.columns:
                name = self._col_to_str(col)
                new_cols.append(name if name is not None else str(col))
            df = df.copy()
            df.columns = new_cols

        # Mapping souple vers Open / High / Low / Close / Adj Close / Volume
        rename_map = {}
        for col in df.columns:
            if not isinstance(col, str):
                continue
            low = col.lower()

            if "open" in low and "adj" not in low:
                rename_map[col] = "Open"
            elif "high" in low:
                rename_map[col] = "High"
            elif "low" in low and "close" not in low:
                rename_map[col] = "Low"
            elif ("close" in low or "last" in low) and "adj" not in low:
                rename_map[col] = "Close"
            elif "adj" in low and "close" in low:
                rename_map[col] = "Adj Close"
            elif "vol" in low:
                rename_map[col] = "Volume"

        df = df.rename(columns=rename_map)

        ordered_cols = [
            c
            for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            if c in df.columns
        ]
        if not ordered_cols:
            # on affiche les colonnes dispo pour débug si besoin
            raise MarketDataError(
                f"Impossible de standardiser les colonnes de prix. Colonnes trouvées : {list(df.columns)}"
            )

        for col in ordered_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[ordered_cols]



    def _get_from_yfinance(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """
        Retrieve historical market data for a given ticker from Yahoo Finance.

        Parameters
        ----------
        ticker : str
            The ticker to retrieve data for.
        start : dt.date
            The start date for the data.
        end : dt.date
            The end date for the data.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the historical market data for the given ticker and interval.

        Raises
        -------
        MarketDataError
            If there is an error retrieving the data (e.g. invalid ticker, invalid interval, etc.).
        """
        key = ("yfinance", ticker.upper(), start, end)
        if key in self._cache:
            return self._cache[key].copy()

        try:
            import yfinance as yf
        except ImportError as exc:
            raise MarketDataError("yfinance n'est pas installé. Lance `pip install yfinance`.") from exc

        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end + dt.timedelta(days=1),  # Yahoo exclut la date de fin
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except requests.exceptions.HTTPError as exc:
            msg = str(exc)
            if "429" in msg or "Too Many Requests" in msg:
                raise MarketDataError(
                    "Rate limit Yahoo Finance (HTTP 429). Wait a few minutes, or change IP/VPN."
                ) from exc
            if "999" in msg:
                raise MarketDataError(
                    "Yahoo Finance blocked connexion (HTTP 999)"
                ) from exc
            raise MarketDataError(f"HTTP Error Yahoo Finance: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise MarketDataError(f"Request Error Yahoo Finance: {exc}") from exc
        except Exception as exc:
            raise MarketDataError(f"Exception Yahoo Finance: {exc}") from exc

        if df is None or df.empty:
            raise MarketDataError(
                "Yahoo Finance failed to retrieve data."
            )

        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        self._cache[key] = df
        return df

    def _get_from_csv(self, ticker: str) -> pd.DataFrame:
        """
        Retrieve a pandas DataFrame containing historical market data for a given ticker
        from a CSV file.

        Parameters
        ----------
        ticker : str
            The ticker to retrieve data for.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the historical market data for the given ticker.

        Raises
        -------
        MarketDataError
            If the csv_path is not defined in MarketDataSettings or if the CSV file is not found.
        """
        if not self.settings.csv_path:
            raise MarketDataError("csv_path not defined in MarketDataSettings.")

        path = os.path.join(self.settings.csv_path, f"{ticker}.csv")
        if not os.path.exists(path):
            raise MarketDataError(f"CSV file not found: {path}")

        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.set_index("Date")
        return df

    def _get_from_bloomberg(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """
        Retrieve a pandas DataFrame containing historical market data for a given ticker
        from Bloomberg Terminal.

        Parameters
        ----------
        ticker : str
            The ticker to retrieve data for.
        start : dt.date
            The start date for the data.
        end : dt.date
            The end date for the data.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the historical market data for the given ticker.

        Raises
        -------
        MarketDataError
            If there is an error retrieving the data (e.g. invalid ticker, invalid interval, etc.).
        """
        key = ("bloomberg", ticker.upper(), start, end)
        if key in self._cache:
            return self._cache[key].copy()

        try:
            import blpapi  # type: ignore
        except ImportError as exc:
            raise MarketDataError(
                "blpapi not installed. Run `pip install blpapi`."
            ) from exc

        options = blpapi.SessionOptions()
        options.setServerHost(self.settings.bloomberg_host)
        options.setServerPort(self.settings.bloomberg_port)

        session = blpapi.Session(options)
        if not session.start():
            raise MarketDataError("Impossible to connect to Bloomberg.")
        if not session.openService(self.settings.bloomberg_service):
            session.stop()
            raise MarketDataError(f"Service Bloomberg not found: {self.settings.bloomberg_service}")

        service = session.getService(self.settings.bloomberg_service)
        request = service.createRequest("HistoricalDataRequest")
        request.getElement("securities").appendValue(ticker)
        fields = request.getElement("fields")
        for f in ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "VOLUME"]:
            fields.appendValue(f)

        request.set("startDate", start.strftime("%Y%m%d"))
        request.set("endDate", end.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")

        session.sendRequest(request)

        records = []
        try:
            while True:
                ev = session.nextEvent(500)
                for msg in ev:
                    if msg.hasElement("responseError"):
                        err = msg.getElement("responseError")
                        raise MarketDataError(f"Erreur Bloomberg: {err.getElementAsString('message')}")

                    if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                        sec_data = msg.getElement("securityData")
                        if sec_data.hasElement("securityError"):
                            sec_err = sec_data.getElement("securityError")
                            raise MarketDataError(f"Erreur Bloomberg sur {ticker}: {sec_err.getElementAsString('message')}")

                        field_data = sec_data.getElement("fieldData")
                        for i in range(field_data.numValues()):
                            bar = field_data.getValueAsElement(i)
                            records.append(
                                {
                                    "date": pd.to_datetime(bar.getElementAsDatetime("date")),
                                    "open": bar.getElementAsFloat("PX_OPEN") if bar.hasElement("PX_OPEN") else None,
                                    "high": bar.getElementAsFloat("PX_HIGH") if bar.hasElement("PX_HIGH") else None,
                                    "low": bar.getElementAsFloat("PX_LOW") if bar.hasElement("PX_LOW") else None,
                                    "close": bar.getElementAsFloat("PX_LAST") if bar.hasElement("PX_LAST") else None,
                                    "volume": bar.getElementAsFloat("VOLUME") if bar.hasElement("VOLUME") else None,
                                }
                            )

                if ev.eventType() == blpapi.Event.RESPONSE:
                    break
        finally:
            session.stop()

        if not records:
            raise MarketDataError(f"No data found for {ticker}.")

        df = pd.DataFrame(records).set_index("date").sort_index()
        self._cache[key] = df
        return df
