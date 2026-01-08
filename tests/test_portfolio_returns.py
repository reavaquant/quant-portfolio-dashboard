import datetime as dt
from data_client import MarketDataClient
from portfolio import compute_returns, equal_weights, compute_portfolio_returns, equity_curve

client = MarketDataClient()

start = dt.date.today() - dt.timedelta(days=365)
end = dt.date.today()

prices = client.get_multi_asset_prices(["AAPL", "MSFT", "SPY"], start=start, end=end, interval="1d")

rets = compute_returns(prices)
w = equal_weights(rets.columns)
port_rets = compute_portfolio_returns(rets, w)
eq = equity_curve(port_rets)

print("returns shape:", rets.shape)
print("weights:", w.to_dict())
print("portfolio returns head:\n", port_rets.head())
print("equity tail:\n", eq.tail())
