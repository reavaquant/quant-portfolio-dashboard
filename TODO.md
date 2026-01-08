1. Provide user access to portfolio strategy parameters => rebalancing
frequency

2. Provide visual comparisons between single assets and the portfolio.

3. The main chart must show multiple asset prices together with the
cumulative value of the portfolio. regarde app.py comment j'ai fait et fait pareil !

ensuite merge avec app_quantb en un grand app.py

4. Automatically refresh data every 5 minutes.

5. Provide a daily report (e.g., volatility, open/close price, max drawdown)
generated at a fixed time (e.g., 8pm) via cron, stored locally on the VM.
The cron job configuration and scripts must also be included and
documented in the GitHub repository. Ca c'est fdait normalmeent dans report.py

6. Ensure the app is always running (24/7) on the hosted Linux machine,
while minimizing cloud costs.