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


def test_standardize_columns_handles_multiindex():
    df = pd.DataFrame(
        {
            ("AAPL", "close"): [1.0, 2.0],
            ("AAPL", "open"): [0.9, 1.8],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    client = MarketDataClient()
    out = client._standardize_columns(df)
    assert list(out.columns) == ["Open", "Close"]
    assert out.iloc[0]["Close"] == 1.0


def test_normalize_dates_caps_end_to_today():
    client = MarketDataClient()
    today = dt.date.today()
    start, end = client._normalize_dates(None, today + dt.timedelta(days=2))
    assert end == today
    assert start <= end
