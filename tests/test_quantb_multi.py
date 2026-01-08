import datetime as dt
from data_client import MarketDataClient

client = MarketDataClient()

start = dt.date.today() - dt.timedelta(days=365)
end = dt.date.today()

prices = client.get_multi_asset_prices(
    ["AAPL", "MSFT", "SPY"],
    start=start,
    end=end,
    interval="1d",
)

print(prices.head())
print(prices.tail())
print("shape:", prices.shape)
print("columns:", prices.columns.tolist())
