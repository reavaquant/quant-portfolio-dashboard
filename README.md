## Single Asset Module (Quant A)

Streamlit dashboard for single-asset backtesting and a daily reporting CLI.

### Setup
1. Python 3.10+ recommended.
2. Install deps: `pip install -r requirements.txt`
3. Optional: set `DEFAULT_TICKER` in your env (`.env` supported).

### Run the dashboard
```bash
streamlit run app.py
```
- Supports Yahoo Finance intervals: 5m, 15m, 30m, 1h, 1d. Intraday windows are limited by Yahoo (keep date range short).
- Auto-refresh every 5 minutes is injected via a lightweight JS timer.
- Two strategies: Buy & Hold, Moving Average Crossover (configurable windows).
- Metrics: total returns, CAGR (calendar time), vol, Sharpe/Sortino/Omega/Upside Potential, max drawdown, turnover, Calmar/Sterling.
- Optional trend regression forecast with CIs.

### Daily report + cron
- Generate a JSON report (default 1-year Buy & Hold) into `reports/`:
```bash
python report.py --ticker AAPL --window-days 365 --interval 1d --risk-free-rate 0.0
```
- Report fields include open/close price, volatility, and max drawdown.
- Sample cron entry: see `cron_daily_report.cron`. Replace `/path/to/repo` with the absolute repo path and ensure `python3` and `report.py` paths are correct. Cron output is redirected to `logs/daily_report.log` (create `logs/` and `reports/` or adjust the path).

### Keep the app running 24/7
- Run under a process supervisor (systemd, supervisor, or tmux/screen) on the Linux VM, e.g.:
```bash
nohup streamlit run app.py --server.port 8501 > logs/streamlit.log 2>&1 &
```
- For VM deployment steps, cron setup, and a ready-to-edit systemd unit, see `DEPLOYMENT.md`.

### Tests
```bash
pytest
```

### Project structure
- `app.py` : Streamlit UI + backtests + forecast
- `data_client.py` : market data accessors (Yahoo Finance by default)
- `backtest.py` : backtesting and metrics engine
- `alphas.py` : strategies (Buy & Hold, MA crossover)
- `model.py` : trend regression forecaster
- `report.py` : CLI daily report generator
- `cron_daily_report.cron` : cron example
- `tests/` : unit tests (offline, synthetic data)
