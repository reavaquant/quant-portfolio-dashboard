import datetime as dt
from data_client import MarketDataClient, MarketDataSettings

settings = MarketDataSettings(source="yfinance", default_ticker="AAPL")
client = MarketDataClient(settings)

start = dt.date(2024, 1, 1)
end = dt.date.today()

df = client.get_history(start=start, end=end)
print(df.head())
print(df.tail())
print(df.columns)
print("Dernier prix :", client.get_last_price())
