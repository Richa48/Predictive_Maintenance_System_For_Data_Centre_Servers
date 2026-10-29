# Predictive Maintenance System for Data Centre Servers: run guide

## Where to implement
Use ONE Linux machine or VM (Ubuntu 22.04/24.04, 4 GB RAM, Python 3.10+, Docker). On Windows use WSL2.
No physical server? Use `--simulate` (step 5), it works everywhere.

Project layout (everything lives in one folder, e.g. `~/pdm`):
```
pdm/
  data/            <- put Backblaze CSV files here
  models/          <- trained model is saved here (model.joblib + metrics.json)
  common.py        <- feature list shared by training and monitoring
  make_synthetic.py<- fake data for a dry run
  train.py         <- trains + validates + SAVES the model
  monitor.py       <- collects telemetry, LOADS the model, predicts, writes to InfluxDB
  docker-compose.yml  <- InfluxDB (storage) + Grafana (dashboard)
```

## Step 1: Setup
```bash
cd ~/pdm
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
sudo apt install -y smartmontools        # only needed for real disks
```

## Step 2: Start storage + dashboard
```bash
docker compose up -d
```
InfluxDB: http://localhost:8086 (admin / adminadmin123, org `pdm`, bucket `health`, token `pdm-token-change-me`)
Grafana:  http://localhost:3000 (admin / admin)

## Step 3: Get training data
Real data (use this for your results): https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
Download one quarter, unzip, and copy the daily `YYYY-MM-DD.csv` files into `data/`.
Dry run only: `python make_synthetic.py` (fake data; do NOT quote its numbers in your report).

## Step 4: Train and store the model
```bash
python train.py --horizon 3                                  # predict failure within 3 days (72 h)
python train.py --horizon 3 --model-filter ST4000DM000       # one drive model, uses less RAM
python train.py --horizon 3 --algo xgb                       # XGBoost instead of Random Forest
```
Output: `models/model.joblib` (the model, feature list and alert threshold in ONE file) and
`models/metrics.json` (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
To deploy on another machine, copy `model.joblib` to the same `models/` folder.

## Step 5: Run end to end
```bash
python monitor.py --simulate --once                  # one test reading
python monitor.py --simulate --degrade --interval 5  # fake failing disk: risk climbs, alert turns on
lsblk                                                # find your disk name
sudo ./venv/bin/python monitor.py --device /dev/sda --interval 60   # real SATA disk
```
Each loop: collect SMART (+ CPU/memory/temperature, + Redfish if `REDFISH_URL` is set) -> model gives
`failure_risk` (0-1) -> `alert` = 1 when risk >= threshold -> written to InfluxDB.
Redfish (optional): `export REDFISH_URL=https://<bmc-ip> REDFISH_USER=... REDFISH_PASS=...`

## Step 6: Dashboard (Grafana)
1. Connections -> Data sources -> Add -> InfluxDB. Query language: Flux. URL: `http://influxdb:8086`.
   Organization `pdm`, Token `pdm-token-change-me`, Default bucket `health`. Save & test.
2. New dashboard -> Add visualization -> Flux query:
```
from(bucket: "health")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "server_health" and r._field == "failure_risk")
```
   Repeat with `_field == "smart_5_raw"`, `"smart_197_raw"`, `"cpu_percent"`, `"max_temp_c"` for more panels.
3. Alerts: Alerting -> Alert rules -> condition `failure_risk > 0.5` -> add a contact point (email/Slack).

## Step 7 (optional): keep it running as a service
Copy the project to `/opt/pdm`, then create `/etc/systemd/system/pdm.service`:
```
[Unit]
Description=PdM monitor
After=network.target docker.service
[Service]
WorkingDirectory=/opt/pdm
ExecStart=/opt/pdm/venv/bin/python monitor.py --device /dev/sda --interval 60
Restart=always
User=root
[Install]
WantedBy=multi-user.target
```
`sudo systemctl daemon-reload && sudo systemctl enable --now pdm`

## What to report (Phase II results)
- Failures are rare, so plain accuracy is always ~99% and proves little. Report **recall, precision, F1, balanced accuracy, ROC-AUC and the confusion matrix** (all in `metrics.json`). State your 85% target against recall / balanced accuracy.
- Compare Random Forest vs XGBoost (run both, keep a table).
- Compare against scheduled maintenance: count failures caught early and healthy drives flagged needlessly (false positives) vs replacing every drive on a fixed calendar.
- Limitations to state: public Backblaze data (not your own servers), SATA/ATA SMART only (NVMe not handled), raw-value features only (no trend features yet).

## Troubleshooting
- `smartctl` permission error -> run with `sudo`; USB/virtual disks may not expose SMART (use `--simulate`).
- Out of memory in training -> lower `--healthy-frac` (e.g. 0.03) or use `--model-filter`.
- `[warn] InfluxDB unavailable` -> run `docker compose up -d` and `pip install influxdb-client`.
