# VM Deployment Guide

This repo includes:
- `report.py` and `cron_daily_report.cron` for the daily JSON report.
- `systemd/streamlit.service` for running the Streamlit app 24/7.

## 1) Provision a small Linux VM
- Use a minimal Ubuntu/Debian image (1 vCPU, 1-2 GB RAM to start).
- Open TCP port 8501 in your firewall/security group.
- Install system packages: `python3-venv`, `git`, and `cron`.

## 2) Install the app
```bash
git clone <your-repo-url>
cd /path/to/repo/project
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
mkdir -p logs reports
```

## 3) Daily report via cron (20:00)
1. Edit `cron_daily_report.cron` to replace `/path/to/repo` with your VM path.
2. Optional: set `CRON_TZ=Europe/Paris` (or your timezone).
3. Install the job:
```bash
crontab -e
```
4. Verify:
```bash
crontab -l
tail -f logs/daily_report.log
```

## 4) 24/7 Streamlit via systemd
1. Copy the unit file:
```bash
sudo cp systemd/streamlit.service /etc/systemd/system/pgl-streamlit.service
```
2. Edit `/etc/systemd/system/pgl-streamlit.service`:
   - `User=...`
   - `WorkingDirectory=...`
   - `ExecStart=...` (point to your `.venv/bin/streamlit`)
3. Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now pgl-streamlit.service
```
4. Check status/logs:
```bash
systemctl status pgl-streamlit.service
journalctl -u pgl-streamlit.service -f
```

## 5) Cost control tips
- Use the smallest VM that still runs Streamlit + yfinance smoothly.
- Avoid Docker if you do not need it; systemd keeps overhead low.
- Rotate logs periodically to avoid disk growth.
- Prefer reserved/savings plans or low-cost providers for 24/7 uptime.
