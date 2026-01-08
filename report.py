import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, Any

from alphas import BuyHoldAlpha
from data_client import MarketDataClient, MarketDataSettings, MarketDataError
from backtest import Backtester


PERIODS_PER_YEAR = {
    "5m": 78 * 252,
    "15m": 26 * 252,
    "30m": 13 * 252,
    "1h": int(6.5 * 252),
    "1d": 252,
}


def generate_report(
    ticker: str,
    window_days: int = 365,
    interval: str = "1d",
    risk_free_rate: float = 0.0,
    output_dir: str = "reports",
) -> Path:
    end = dt.date.today()
    start = end - dt.timedelta(days=window_days)

    settings = MarketDataSettings(default_ticker=ticker)
    client = MarketDataClient(settings=settings)

    prices = client.get_history(ticker=ticker, start=start, end=end, interval=interval)
    periods_per_year = PERIODS_PER_YEAR.get(interval, 252)
    backtester = Backtester(risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)

    strategy = BuyHoldAlpha()
    result = backtester.run(prices["Close"], strategy)

    report: Dict[str, Any] = {
        "as_of": dt.datetime.utcnow().isoformat() + "Z",
        "ticker": ticker,
        "interval": interval,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "last_price": float(prices["Close"].iloc[-1]),
        "strategy": strategy.name,
        "metrics": result.metrics,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"daily_report_{ticker}_{end.isoformat()}_{interval}.json"
    with filename.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return filename


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate daily performance report.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to download")
    parser.add_argument(
        "--window-days",
        type=int,
        default=365,
        help="Lookback window in days for the report (default: 365)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        choices=list(PERIODS_PER_YEAR.keys()),
        help="Data interval (5m, 15m, 30m, 1h, 1d)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annual risk-free rate used in metrics (e.g. 0.02 for 2%)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Folder where the JSON report will be written (default: reports/)",
    )
    args = parser.parse_args()

    try:
        path = generate_report(
            ticker=args.ticker,
            window_days=args.window_days,
            interval=args.interval,
            risk_free_rate=args.risk_free_rate,
            output_dir=args.output_dir,
        )
    except MarketDataError as exc:
        print(f"Market data error: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - unexpected errors
        print(f"Unexpected error: {exc}")
        return 1

    print(f"Report saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
