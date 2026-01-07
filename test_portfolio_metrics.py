import datetime as dt
from data_client import MarketDataClient, MarketDataSettings
from portfolio import (
    compute_returns,
    equal_weights,
    compute_portfolio_returns,
    equity_curve,
    correlation_matrix,
    annualized_return,
    annualized_volatility,
    max_drawdown,
)

settings = MarketDataSettings(source="yfinance")
client = MarketDataClient(settings)

start = dt.date.today() - dt.timedelta(days=365)
end = dt.date.today()

prices = client.get_multi_asset_prices(["AAPL", "MSFT", "SPY"], start=start, end=end, interval="1d")
rets = compute_returns(prices)

w = equal_weights(rets.columns)
port_rets = compute_portfolio_returns(rets, w)
eq = equity_curve(port_rets)

corr = correlation_matrix(rets)

print("Correlation matrix:\n", corr)
print("Portfolio ann return:", annualized_return(port_rets))
print("Portfolio ann vol:", annualized_volatility(port_rets))
print("Portfolio max drawdown:", max_drawdown(eq))
