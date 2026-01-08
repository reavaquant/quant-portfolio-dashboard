import datetime as dt

import pandas as pd
import pytest

from data_client import MarketDataClient, MarketDataError


def test_validate_interval_normalizes_and_blocks_unknown():
    client = MarketDataClient()
    assert client._validate_interval("1day") == "1d"
    assert client._validate_interval("60m") == "1h"
    with pytest.raises(MarketDataError):
        client._validate_interval("2d")


def test_select_ticker_frame_handles_multiindex():
    df = pd.DataFrame(
        {
            ("AAPL", "Open"): [0.9, 1.8],
            ("AAPL", "High"): [1.1, 2.1],
            ("AAPL", "Low"): [0.8, 1.7],
            ("AAPL", "Close"): [1.0, 2.0],
            ("AAPL", "Volume"): [100, 200],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    client = MarketDataClient()
    out = client._select_ticker_frame(df, "AAPL")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.iloc[0]["Close"] == 1.0


def test_normalize_dates_caps_end_to_today():
    client = MarketDataClient()
    today = dt.date.today()
    start, end = client._normalize_dates(None, today + dt.timedelta(days=2))
    assert end == today
    assert start <= end
