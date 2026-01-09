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

## Deployment (Hetzner VPS / Ubuntu) — Streamlit + systemd + cron

This Streamlit app is deployed on a VPS (Hetzner) and exposed publicly on port **8501** (Option A: direct access).  
A **cron job** generates a daily JSON report on the VM.

### Steps

On the VPS:
```bash
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3 python3-venv python3-pip git ufw
````

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
