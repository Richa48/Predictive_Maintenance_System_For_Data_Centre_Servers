"""Live monitor: collect telemetry -> run trained model -> write to InfluxDB (Grafana reads it).
Examples:
  python monitor.py --simulate --once            # quick test, no real disk needed
  python monitor.py --simulate --degrade         # simulated failing drive (risk climbs)
  sudo python monitor.py --device /dev/sda       # real SATA disk via smartctl
Env (optional): INFLUX_URL INFLUX_TOKEN INFLUX_ORG INFLUX_BUCKET REDFISH_URL REDFISH_USER REDFISH_PASS"""
import argparse, json, os, re, socket, subprocess, time, random
from collections import deque
import joblib, psutil, pandas as pd
from common import MODEL_PATH, TREND_WINDOW

# ---------- 1. COLLECTORS ----------
def collect_smart(device):
    """Read SMART attributes with smartctl (needs: sudo apt install smartmontools, run as root)."""
    out = subprocess.run(["smartctl", "-j", "-A", device], capture_output=True, text=True)
    data = json.loads(out.stdout)
    table = data.get("ata_smart_attributes", {}).get("table", [])
    return {f"smart_{a['id']}_raw": a["raw"]["value"] for a in table}

def collect_system():
    """OS-level performance counters (psutil)."""
    d = {"cpu_percent": psutil.cpu_percent(interval=1),
         "mem_percent": psutil.virtual_memory().percent,
         "disk_percent": psutil.disk_usage("/").percent}
    if hasattr(os, "getloadavg"):
    
        d["load1"] = os.getloadavg()[0]
    try:
        temps = [t.current for lst in psutil.sensors_temperatures().values() for t in lst]
        if temps: d["max_temp_c"] = max(temps)
    except Exception:
        pass
    return d

def collect_redfish():
    """Optional BMC data via Redfish. Path differs by vendor - adjust THERMAL_PATH."""
    url = os.getenv("REDFISH_URL")
    if not url: return {}
    import requests
    THERMAL_PATH = "/redfish/v1/Chassis/1/Thermal"
    r = requests.get(url + THERMAL_PATH, auth=(os.getenv("REDFISH_USER", ""), os.getenv("REDFISH_PASS", "")),
                     verify=False, timeout=10).json()
    temps = [t["ReadingCelsius"] for t in r.get("Temperatures", []) if t.get("ReadingCelsius") is not None]
    fans = [f["Reading"] for f in r.get("Fans", []) if f.get("Reading") is not None]
    return {"bmc_max_temp_c": max(temps) if temps else 0, "bmc_avg_fan_rpm": sum(fans) / len(fans) if fans else 0}

_step = {"n": 0}
def simulate(degrade):
    """Fake SMART values. With --degrade the error counters keep growing (like a dying disk)."""
    _step["n"] += 1
    n = _step["n"] if degrade else 0
    return {"smart_5_raw": n * 6, "smart_9_raw": 20000 + _step["n"], "smart_187_raw": n * 2,
            "smart_188_raw": 0, "smart_192_raw": random.randint(0, 2),
            "smart_194_raw": 30 + n // 2, "smart_197_raw": n * 3, "smart_198_raw": n * 2}

# ---------- 2. INFLUXDB WRITER ----------
def make_writer():
    try:
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS
        c = InfluxDBClient(url=os.getenv("INFLUX_URL", "http://localhost:8086"),
                           token=os.getenv("INFLUX_TOKEN", "pdm-token-change-me"),
                           org=os.getenv("INFLUX_ORG", "pdm"))
        return c.write_api(write_options=SYNCHRONOUS), os.getenv("INFLUX_BUCKET", "health"), os.getenv("INFLUX_ORG", "pdm")
    except Exception as e:
        print(f"[warn] InfluxDB unavailable ({e}); printing only")
        return None, None, None

def write_point(writer, bucket, org, host, fields):
    if writer is None: return
    from influxdb_client import Point
    p = Point("server_health").tag("host", host)
    for k, v in fields.items():
        p = p.field(k, float(v))
    try:
        writer.write(bucket=bucket, org=org, record=p)
    except Exception as e:
        print(f"[warn] write failed: {e}")

# ---------- 3. MAIN LOOP ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/sda")
    ap.add_argument("--interval", type=int, default=60, help="seconds between readings")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--degrade", action="store_true", help="with --simulate: mimic a failing drive")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--host", default=socket.gethostname())
    a = ap.parse_args()

    bundle = joblib.load(MODEL_PATH)                       # <- load the stored trained model
    clf, feats, thr = bundle["model"], bundle["features"], bundle["threshold"]
    print(f"Loaded model from {MODEL_PATH} (threshold {thr:.3f}, horizon {bundle['horizon_days']}d)")
    writer, bucket, org = make_writer()
    hist = deque(maxlen=TREND_WINDOW + 1)                  # last readings, for trend features

    while True:
        smart = simulate(a.degrade) if a.simulate else collect_smart(a.device)
        extra = {**collect_system(), **collect_redfish()}
        hist.append(smart)
        row = {}
        for f in feats:
            mt = re.match(r"(smart_\d+_raw)$", f)
            md = re.match(r"(smart_\d+)_delta\d+$", f)
            if md:   # trend = newest reading minus oldest reading in the window
                key = md.group(1) + "_raw"
                row[f] = hist[-1].get(key, 0) - hist[0].get(key, 0)
            else:
                row[f] = smart.get(f, 0)
        X = pd.DataFrame([row])[feats]                     # same feature order as training
        risk = float(clf.predict_proba(X)[0, 1])
        alert = int(risk >= thr)
        print(f"{time.strftime('%H:%M:%S')} risk={risk:.3f} alert={'YES' if alert else 'no'} "
              f"smart5={smart.get('smart_5_raw')} smart197={smart.get('smart_197_raw')}")
        write_point(writer, bucket, org, a.host, {**smart, **extra, "failure_risk": risk, "alert": alert})
        if a.once: break
        time.sleep(a.interval)

if __name__ == "__main__":
    main()
