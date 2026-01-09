# Quant Dashboard

Streamlit dashboard for single-asset and portfolio backtesting plus a daily reporting.

## Overview
- Quant A (single asset): Yahoo Finance backtests with 5m/15m/30m/1h/1d data, Buy & Hold or MA crossover, and optional trend regression forecast.
- Quant B (portfolio): multi-asset analysis with equal/custom weights, optional weekly/monthly rebalancing, single-vs-portfolio comparison, and correlation heatmap.
- Metrics engine: total return, CAGR (calendar time), vol, Sharpe/Sortino/Omega/Upside Potential, max drawdown, turnover, Calmar/Sterling.
- CLI generates JSON reports for single-asset Buy & Hold runs (cron-friendly).

## Requirements
- Python 3.10+ recommended
- `pip` (or `pip3`)

## Installation
```bash
pip install -r requirements.txt
```

## Run the dashboard
```bash
streamlit run app.py
```
- Yahoo Finance intervals supported: 5m, 15m, 30m, 1h, 1d.
- Intraday data windows are limited by Yahoo; keep date ranges short.
- Auto-refresh every 5 minutes is injected via a lightweight JS timer.
- Use the top navigation to switch between Single Asset and Portfolio pages.

## Daily report CLI
Generate a JSON report (default 1-year Buy & Hold) into `reports/`:
```bash
python report.py --ticker AAPL --window-days 365 --interval 1d --risk-free-rate 0.0 --output-dir reports
```
- Report includes price snapshot and the strategy metrics.
- Cron example: see `cron_daily_report.cron`. Replace `/path/to/repo` with the absolute repo path and ensure `python3` and `report.py` paths are correct. Cron output is redirected to `logs/daily_report.log` (create `logs/` and `reports/` or adjust the path).

## Tests
```bash
pytest
```

## Project structure
- `app.py`: Streamlit UI + backtests + forecast
- `data_client.py`: market data accessors (Yahoo Finance by default)
- `backtest.py`: backtesting and metrics engine
- `alphas.py`: strategies (Buy & Hold, MA crossover)
- `portfolio.py`: portfolio math (returns, weights, rebalancing, correlation)
- `model.py`: trend regression forecaster
- `report.py`: CLI daily report generator (single asset)
- `tests/`: unit tests (offline, synthetic data)

## Deployment (Hetzner VPS / Ubuntu) — Streamlit + systemd + cron

This Streamlit app is deployed on a VPS (Hetzner) and exposed publicly on port **8501**.  
A **cron job** generates a daily JSON report on the VM.

### Steps

On the VPS:
```bash
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3 python3-venv python3-pip git ufw
```

### 2) Open network access (Option A: direct access to Streamlit)

On the VPS (UFW):

```bash
sudo ufw allow OpenSSH
sudo ufw allow 8501/tcp
sudo ufw --force enable
sudo ufw status
```

Paths used:

* Code: `/opt/quant-portfolio-dashboard/quant-portfolio-dashboard`
* Virtualenv: `/opt/quant-portfolio-dashboard/.venv`

```bash
sudo mkdir -p /opt/quant-portfolio-dashboard
sudo chown -R $USER:$USER /opt/quant-portfolio-dashboard
cd /opt/quant-portfolio-dashboard

git clone <REPO_URL> quant-portfolio-dashboard
cd /opt/quant-portfolio-dashboard/quant-portfolio-dashboard

python3 -m venv /opt/quant-portfolio-dashboard/.venv
/opt/quant-portfolio-dashboard/.venv/bin/pip install -U pip
/opt/quant-portfolio-dashboard/.venv/bin/pip install -r requirements.txt
```

Create the service:

```bash
sudo nano /etc/systemd/system/quant-portfolio-dashboard.service
```

Content:

```ini
[Unit]
Description=Streamlit App
After=network.target

[Service]
User=root
WorkingDirectory=/opt/quant-portfolio-dashboard/quant-portfolio-dashboard
Environment="PATH=/opt/quant-portfolio-dashboard/.venv/bin"
ExecStart=/opt/quant-portfolio-dashboard/.venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now quant-portfolio-dashboard
sudo systemctl status quant-portfolio-dashboard --no-pager -l
```

From your browser:

* `http://135.181.145.104:8501`

Cron job — generate a daily JSON report

The `report.py` script generates a JSON report into `reports/` and writes logs into `logs/`.

Create output folders:

```bash
mkdir -p /opt/quant-portfolio-dashboard/quant-portfolio-dashboard/reports
mkdir -p /opt/quant-portfolio-dashboard/quant-portfolio-dashboard/logs
```

Install a system cron (recommended: `/etc/cron.d/`):

```bash
sudo nano /etc/cron.d/quant-portfolio-dashboard
```

Content (runs every day at 20:00 Paris time):

```cron
CRON_TZ=Europe/Paris
0 20 * * * root cd /opt/quant-portfolio-dashboard/quant-portfolio-dashboard && /opt/quant-portfolio-dashboard/.venv/bin/python report.py --ticker AAPL --interval 1d --output-dir /opt/quant-portfolio-dashboard/quant-portfolio-dashboard/reports >> /opt/quant-portfolio-dashboard/quant-portfolio-dashboard/logs/report.log 2>&1
```

Deploy updates (after a `git push`)

Deploy script:

```bash
/opt/quant-portfolio-dashboard/deploy.sh
```
