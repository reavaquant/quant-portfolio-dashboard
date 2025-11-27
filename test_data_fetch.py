import datetime as dt

from data_client import MarketDataClient


def main():
    ticker = "AAPL"
    client = MarketDataClient(source="stooq")
    today = dt.date.today()
    start = today - dt.timedelta(days=30)

    print(f"Requesting {ticker} from {start} to {today} via Stooq")
    try:
        data = client.get_history(ticker, start=start, end=today)
    except Exception as exc:
        print("Download failed:", exc)
        return

    print(f"Rows: {len(data)}, first: {data.index.min().date()}, last: {data.index.max().date()}")
    print(data.tail())


if __name__ == "__main__":
    main()
