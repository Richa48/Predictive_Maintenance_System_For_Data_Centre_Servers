"""Local web dashboard (no Docker needed). Open http://localhost:8000

  python dashboard.py --replay        # RECOMMENDED: replays REAL drives from data/ (blind test)
  python dashboard.py --simulate      # fake values, only for quick UI testing
  sudo python dashboard.py --device /dev/sda   # real SATA disk on a Linux server

The alert is raised ONLY by the trained model: when the failure risk stays at or above the
threshold for --confirm consecutive readings (default 2). Nothing is scripted in --replay mode.
"""
import argparse, csv, glob, json, random, re, threading, time, webbrowser
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import joblib, pandas as pd
from common import BASE, DATA_DIR, MODEL_DIR, MODEL_PATH, METRICS_PATH, TREND_WINDOW, RAW_FEATURES, TREND_IDS
import monitor                      # reuse the collectors (SMART, system, Redfish)

LOCK = threading.Lock()
STATE = {"history": deque(maxlen=300), "alerts": deque(maxlen=20), "mode": "healthy", "threshold": 0.5,
         "step": 0, "tick": 0, "last_alert": False, "streak": 0, "drive": 1, "seq": 0, "error": ""}
INFO = {}

# ---------------- simulated disk (testing only) ----------------
def sim_smart(mode):
    STATE["tick"] += 1
    STATE["step"] = STATE["step"] + 1 if mode == "degrade" else 0
    n = STATE["step"]
    return {"smart_5_raw": n * 6 + random.randint(0, 1), "smart_9_raw": 20000,
            "smart_187_raw": n * 2, "smart_188_raw": 0, "smart_192_raw": random.randint(0, 2),
            "smart_194_raw": 30 + n // 3 + random.randint(0, 1),
            "smart_197_raw": n * 3, "smart_198_raw": n * 2}

# ---------------- replay of REAL drives from your Backblaze data ----------------
CACHE = BASE / "replay_cache.csv"

def _trend(df):
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    for i in TREND_IDS:
        df[f"smart_{i}_delta{TREND_WINDOW}"] = df.groupby("serial_number")[f"smart_{i}_raw"].diff(TREND_WINDOW).fillna(0)
    return df

def build_replay(bundle, model_filter, rebuild, n_drives=10):
    """Picks n real drives (about 60% that failed, 40% that did not) from data/ and caches them."""
    if CACHE.exists() and not rebuild:
        c = pd.read_csv(CACHE, parse_dates=["date"])
        if "drive_id" in c.columns and "req" in c.columns and int(c["req"].iloc[0]) == n_drives: return c
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files in {DATA_DIR}. Replay needs your Backblaze data.")
    print("Building replay from real data (first run only, takes a few minutes)...")
    failed, by_model = {}, Counter()
    for f in files:
        d = pd.read_csv(f, usecols=["serial_number", "failure", "model"])
        bad = d[d.failure == 1]
        failed.update(dict(zip(bad.serial_number, bad.model))); by_model.update(bad.model)
    target = by_model.most_common(1)[0][0] if model_filter in (None, "auto") else model_filter
    cand_fail = {s for s, m in failed.items() if m == target}
    last = pd.read_csv(files[-1], usecols=["serial_number", "failure", "model"])
    cand_ok = set(last[(last.model == target) & (last.failure == 0) & (~last.serial_number.isin(failed))].serial_number.head(60))
    cols = {"date", "serial_number", "model", "failure", *RAW_FEATURES}
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=lambda c: c in cols)
        for c in RAW_FEATURES:
            if c not in d: d[c] = 0
        parts.append(d[d.serial_number.isin(cand_fail | cand_ok)])
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]); df[RAW_FEATURES] = df[RAW_FEATURES].fillna(0)
    df = _trend(df)
    clf, feats, thr = bundle["model"], bundle["features"], bundle["threshold"]
    df["risk"] = clf.predict_proba(df[feats])[:, 1]

    fails, oks = [], []
    for sn, g in df[df.serial_number.isin(cand_fail)].groupby("serial_number"):
        g = g.sort_values("date").tail(25)
        if len(g) < 20: continue
        r = g.risk.values
        late, early = r[-3:].max(), (r[:-8].max() if len(r) > 8 else r[:-3].max())
        fails.append(((late - early) + (1 if early < thr <= late else 0), g))
    for sn, g in df[df.serial_number.isin(cand_ok)].groupby("serial_number"):
        g = g.sort_values("date")
        if len(g) >= 40: oks.append((g.risk.max(), g.tail(40)))
    if not fails or not oks:
        raise SystemExit("Could not find suitable drives in data/.")
    fails.sort(key=lambda x: -x[0]); oks.sort(key=lambda x: x[0])
    def spread(lst, k):                       # k evenly spaced picks: clearest ... hardest
        if len(lst) <= k: return list(range(len(lst)))
        return [0] if k == 1 else sorted(set(round(i * (len(lst) - 1) / (k - 1)) for i in range(k)))
    nf = max(1, round(n_drives * 0.6)); no = max(1, n_drives - nf)
    items = [(fails[i][1], "failing") for i in spread(fails, nf)] + [(oks[i][1], "healthy") for i in spread(oks, no)]
    random.Random(7).shuffle(items)           # mixed order = blind test
    parts = [g.assign(kind=kind, drive_id=k + 1) for k, (g, kind) in enumerate(items)]
    out = pd.concat(parts, ignore_index=True)
    out["req"] = n_drives
    out.to_csv(CACHE, index=False)
    print(f"Replay ready: {len(parts)} real drives of model {target}")
    return out

class Replay:
    def __init__(self, df):
        self.d = {int(k): g.sort_values("date").reset_index(drop=True) for k, g in df.groupby("drive_id")}
        self.cur, self.seq, self.i = None, -1, 0
    def ids(self): return sorted(self.d)
    def next(self, drive, seq):
        if drive != self.cur or seq != self.seq: self.cur, self.seq, self.i = drive, seq, 0
        g = self.d[drive]; k = min(self.i, len(g) - 1); row = g.iloc[k]; self.i += 1
        day = f"{row.date:%Y-%m-%d}"
        if k == len(g) - 1:
            cap = f"Replay finished ({day}). Actual outcome: " + (f"this drive failed on {day}." if row.kind == "failing" else "this drive stayed healthy.")
        else:
            cap = f"Drive {drive} · serial {row.serial_number} · {day} · day {k + 1} of {len(g)}"
        return {c: float(row[c]) for c in RAW_FEATURES}, cap

# ---------------- background worker ----------------
def worker(args, bundle, replay=None):
    clf, feats = bundle["model"], bundle["features"]
    hist, prev = deque(maxlen=TREND_WINDOW + 1), None
    log_path = BASE / "logs" / "readings.csv"
    log_path.parent.mkdir(exist_ok=True)
    while True:
        try:
            with LOCK:
                mode, thr, drive, seq = STATE["mode"], STATE["threshold"], STATE["drive"], STATE["seq"]
            key = (drive, seq) if replay is not None else mode
            if key != prev:                                  # switched drive / mode: start fresh
                hist.clear(); prev = key
                with LOCK:
                    STATE["streak"] = 0; STATE["last_alert"] = False
                    if replay is not None: STATE["history"].clear(); STATE["alerts"].clear()
            cap = ""
            if replay is not None: smart, cap = replay.next(drive, seq)
            else: smart = sim_smart(mode) if args.simulate else monitor.collect_smart(args.device)
            extra = {**monitor.collect_system(), **monitor.collect_redfish()}
            hist.append(smart)
            row = {}
            for f in feats:
                m = re.match(r"(smart_\d+)_delta\d+$", f)
                row[f] = (hist[-1].get(m.group(1) + "_raw", 0) - hist[0].get(m.group(1) + "_raw", 0)) if m else smart.get(f, 0)
            risk = float(clf.predict_proba(pd.DataFrame([row])[feats])[0, 1])
            t194 = smart.get("smart_194_raw", 0)
            with LOCK:
                STATE["streak"] = STATE["streak"] + 1 if risk >= thr else 0
                alert = STATE["streak"] >= args.confirm      # alert only after N readings in a row above threshold
                pt = {"t": time.strftime("%H:%M:%S"), "risk": risk, "alert": alert, "caption": cap,
                      "smart5": smart.get("smart_5_raw", 0), "smart187": smart.get("smart_187_raw", 0),
                      "smart197": smart.get("smart_197_raw", 0), "smart198": smart.get("smart_198_raw", 0),
                      "temp": (int(t194) & 0xFF) if t194 > 255 else t194,
                      "cpu": extra.get("cpu_percent", 0), "mem": extra.get("mem_percent", 0)}
                STATE["history"].append(pt); STATE["error"] = ""
                if alert and not STATE["last_alert"]:
                    STATE["alerts"].appendleft({"t": pt["t"], "risk": risk, "thr": thr})
                STATE["last_alert"] = alert
            names = sorted(pt.keys() - {"t", "risk", "alert", "caption"})
            new = not log_path.exists()
            with open(log_path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new: w.writerow(["time", "risk", "alert", *names])
                w.writerow([pt["t"], round(risk, 4), int(alert), *[pt[k] for k in names]])
        except Exception as e:                              # keep running, show the error on the page
            with LOCK: STATE["error"] = str(e)
        time.sleep(args.interval)

# ---------------- model metrics for the "Model performance" panel ----------------
def _row(name, m):
    r = m["report"]["1"]
    return {"name": name, "recall": r["recall"], "precision": r["precision"], "f1": r["f1-score"],
            "bal_acc": m["balanced_accuracy"], "auc": m["roc_auc"], "cm": m["confusion_matrix"]}

def load_metrics():
    rows = []
    for name, fn in [("Random Forest", "metrics_rf.json"), ("Gradient Boosting", "metrics_hgb.json"),
                     ("XGBoost", "metrics_xgb.json")]:
        p = MODEL_DIR / fn
        if p.exists(): rows.append(_row(name, json.loads(p.read_text())))
    if not rows and METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text()); rows.append(_row(m.get("algo", "model"), m))
    return rows

# ---------------- web server ----------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body, ctype="application/json", code=200):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == "/api/state":
            with LOCK:
                s = {"history": list(STATE["history"]), "alerts": list(STATE["alerts"]), "mode": STATE["mode"],
                     "threshold": STATE["threshold"], "drive": STATE["drive"], "error": STATE["error"], **INFO}
            self._send(json.dumps(s))
        else:
            self._send(PAGE, "text/html; charset=utf-8")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
        with LOCK:
            if self.path == "/api/mode" and body.get("mode") in ("healthy", "degrade"):
                STATE["mode"] = body["mode"]
            elif self.path == "/api/threshold":
                STATE["threshold"] = min(0.99, max(0.01, float(body["threshold"])))
            elif self.path == "/api/drive":
                STATE["drive"] = int(body["id"]); STATE["seq"] += 1
        self._send("{}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--replay", action="store_true", help="replay REAL drives from data/ (recommended)")
    ap.add_argument("--model-filter", default=None, help="drive model for --replay (default: auto)")
    ap.add_argument("--drives", type=int, default=10, help="how many real drives to replay (default 10)")
    ap.add_argument("--rebuild-replay", action="store_true", help="re-pick the replay drives (use after retraining)")
    ap.add_argument("--device", default="/dev/sda")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between readings")
    ap.add_argument("--confirm", type=int, default=2, help="readings in a row above threshold before ALERT")
    ap.add_argument("--threshold", type=float, default=None, help="override the model's alert threshold")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    bundle = joblib.load(MODEL_PATH)
    STATE["threshold"] = args.threshold if args.threshold is not None else bundle["threshold"]
    replay = Replay(build_replay(bundle, args.model_filter, args.rebuild_replay, args.drives)) if args.replay else None
    if replay: STATE["drive"] = replay.ids()[0]
    INFO.update({"source": "replay" if args.replay else ("sim" if args.simulate else "live"),
                 "drives": replay.ids() if replay else [], "confirm": args.confirm,
                 "horizon": bundle["horizon_days"], "algo": type(bundle["model"]).__name__,
                 "metrics": load_metrics(), "default_threshold": bundle["threshold"]})
    threading.Thread(target=worker, args=(args, bundle, replay), daemon=True).start()
    url = f"http://localhost:{args.port}"
    print(f"Dashboard running at {url}  (Ctrl+C to stop)")
    if not args.no_browser: webbrowser.open(url)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predictive Maintenance Dashboard</title>
<style>
:root{--bg:#f4f6fb;--card:#fff;--text:#1c2333;--muted:#6b7385;--line:#e3e7f0;--ok:#16a34a;--warn:#d97706;--bad:#dc2626;--accent:#3b5bdb;--track:#e8ecf5}
@media(prefers-color-scheme:dark){:root{--bg:#0f1420;--card:#171d2d;--text:#e8ecf7;--muted:#97a0b6;--line:#263049;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--accent:#7c93ff;--track:#243049}}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text)}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding:18px 28px;background:var(--card);border-bottom:1px solid var(--line)}
h1{font-size:20px;margin:0}.sub{color:var(--muted);font-size:13px;margin-top:2px}
.pill{padding:7px 16px;border-radius:999px;font-weight:700;font-size:14px;color:#fff;background:var(--ok)}
body[data-s=WARNING] .pill{background:var(--warn)}body[data-s=ALERT] .pill{background:var(--bad)}
main{padding:22px 28px;display:grid;gap:18px;max-width:1280px;margin:0 auto}
.row{display:grid;gap:18px}.r1{grid-template-columns:minmax(260px,1fr) 2fr}.r3{grid-template-columns:1fr 1.4fr}
@media(max-width:860px){.r1,.r3{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
.card h2{font-size:13px;margin:0 0 12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.gauge{text-align:center}.gauge svg{width:100%;max-width:280px}
.big{font-size:44px;font-weight:700;margin-top:-34px}.big small{font-size:20px;color:var(--muted)}
.state{font-weight:700;margin-top:4px}body[data-s=HEALTHY] .state{color:var(--ok)}body[data-s=WARNING] .state{color:var(--warn)}body[data-s=ALERT] .state{color:var(--bad)}
.hint{color:var(--muted);font-size:12px;margin-top:6px}
#arc{stroke:var(--ok);transition:stroke-dashoffset .6s,stroke .3s}body[data-s=WARNING] #arc{stroke:var(--warn)}body[data-s=ALERT] #arc{stroke:var(--bad)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px}
.tile .v{font-size:26px;font-weight:700}.tile .l{font-size:12px;color:var(--muted);margin-top:2px}.tile.flag .v{color:var(--warn)}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
button,select{font:inherit;padding:8px 14px;border-radius:9px;border:1px solid var(--line);background:var(--card);color:var(--text);cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 6px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
.alert-item{padding:9px 0;border-bottom:1px solid var(--line);font-size:13px}.alert-item b{color:var(--bad)}.empty{color:var(--muted);font-size:13px}
.err{color:var(--bad);font-size:13px}svg text{fill:var(--muted);font-size:11px}
</style></head>
<body data-s="HEALTHY">
<header><div><h1>Predictive Maintenance Dashboard</h1>
<div class="sub" id="sub">Data centre server health, failure risk forecast</div></div>
<div class="pill" id="pill">HEALTHY</div></header>
<main>
<div class="card controls" id="controls">
 <span id="repctl"><span class="sub">Replay real drive:</span> <select id="drv"></select> <button onclick="restart()">Restart</button></span>
 <span id="simctl"><span class="sub">Test controls:</span> <button id="bh" onclick="setMode('healthy')">Healthy disk</button> <button id="bd" onclick="setMode('degrade')">Simulate failing disk</button></span>
 <span class="sub" style="margin-left:14px">Alert threshold</span>
 <input type="range" id="thr" min="0.1" max="0.95" step="0.01"><b id="thrv"></b>
 <span class="err" id="err"></span>
</div>
<div class="row r1">
 <div class="card gauge"><h2 id="gt">Failure risk</h2>
  <svg viewBox="0 0 200 115"><path d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke="var(--track)" stroke-width="16" stroke-linecap="round"/>
  <path id="arc" d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke-width="16" stroke-linecap="round" stroke-dasharray="282.74" stroke-dashoffset="282.74"/></svg>
  <div class="big"><span id="risk">0</span><small>%</small></div>
  <div class="state" id="state">HEALTHY</div><div class="hint" id="hint"></div><div class="hint" id="cap"></div></div>
 <div class="card"><h2>Failure risk over time</h2><svg id="chart" viewBox="0 0 640 240" style="width:100%"></svg></div>
</div>
<div class="card"><h2>Live telemetry</h2><div class="tiles" id="tiles"></div></div>
<div class="card" id="sbcard" style="display:none"><h2>Replay scoreboard (fills as each drive finishes)</h2><div id="sb"></div></div>
<div class="row r3">
 <div class="card"><h2>Alert log</h2><div id="alerts"><div class="empty">No alerts yet</div></div></div>
 <div class="card" id="mcard"><h2>Model performance (test data)</h2><div id="metrics"></div></div>
</div>
</main>
<script>
const $=id=>document.getElementById(id);
const TILES=[["smart5","Reallocated sectors (SMART 5)",v=>v>0],["smart197","Pending sectors (SMART 197)",v=>v>0],
["smart198","Uncorrectable sectors (SMART 198)",v=>v>0],["smart187","Reported uncorrectable (SMART 187)",v=>v>0],
["temp","Disk temperature °C",v=>v>45],["cpu","CPU load % (this computer)",v=>v>90],["mem","Memory used % (this computer)",v=>v>90]];
$("tiles").innerHTML=TILES.map(t=>`<div class="tile" id="t_${t[0]}"><div class="v">-</div><div class="l">${t[1]}</div></div>`).join("");
async function post(u,b){await fetch(u,{method:"POST",body:JSON.stringify(b)});poll()}
function setMode(m){post("/api/mode",{mode:m})}
function restart(){post("/api/drive",{id:+$("drv").value})}
$("drv").addEventListener("change",restart);
$("thr").addEventListener("change",e=>post("/api/threshold",{threshold:e.target.value}));
function chart(h,thr){
 const W=640,H=240,L=40,R=10,T=10,B=26,n=h.length,x=i=>L+(n<2?0:i*(W-L-R)/(n-1)),y=v=>T+(1-v)*(H-T-B);
 let s="";
 [0,.25,.5,.75,1].forEach(g=>{s+=`<line x1="${L}" x2="${W-R}" y1="${y(g)}" y2="${y(g)}" stroke="var(--line)"/><text x="${L-6}" y="${y(g)+4}" text-anchor="end">${Math.round(g*100)}%</text>`});
 if(n){const pts=h.map((p,i)=>x(i)+","+y(p.risk)).join(" ");
  s+=`<polygon points="${x(0)},${y(0)} ${pts} ${x(n-1)},${y(0)}" fill="var(--accent)" opacity=".12"/>`;
  s+=`<polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round"/>`;
  s+=`<circle cx="${x(n-1)}" cy="${y(h[n-1].risk)}" r="4.5" fill="var(--accent)"/>`;
  s+=`<text x="${L}" y="${H-6}">${h[0].t}</text><text x="${W-R}" y="${H-6}" text-anchor="end">${h[n-1].t}</text>`}
 s+=`<line x1="${L}" x2="${W-R}" y1="${y(thr)}" y2="${y(thr)}" stroke="var(--bad)" stroke-dasharray="6 4" stroke-width="1.5"/><text x="${W-R}" y="${y(thr)-5}" text-anchor="end" style="fill:var(--bad)">alert threshold ${Math.round(thr*100)}%</text>`;
 $("chart").innerHTML=s}
const RES={};
function scoreboard(s,last){
 if(last&&last.caption.startsWith("Replay finished"))RES[s.drive]={failed:last.caption.includes("failed on"),alerted:s.alerts.length>0};
 const ids=Object.keys(RES).map(Number).sort((a,b)=>a-b);$("sbcard").style.display=ids.length?"block":"none";if(!ids.length)return;
 let f=0,c=0,h=0,fa=0;
 const rows=ids.map(i=>{const r=RES[i];let v;
  if(r.failed){f++;if(r.alerted){c++;v="Caught"}else v="Missed"}else{h++;if(r.alerted){fa++;v="False alarm"}else v="Correct (no alert)"}
  return`<tr><td>Drive ${i}</td><td>${r.failed?"Failed":"Stayed healthy"}</td><td>${r.alerted?"Alert":"No alert"}</td><td>${v}</td></tr>`}).join("");
 $("sb").innerHTML=`<table><tr><th>Drive</th><th>Actual outcome</th><th>Model</th><th>Verdict</th></tr>${rows}</table><div class="hint">Caught ${c} of ${f} failing drives · ${fa} false alarm(s) on ${h} healthy drives</div>`}
function render(s){
 const h=s.history,last=h[h.length-1],thr=s.threshold,rep=s.source==="replay",sim=s.source==="sim";
 $("sub").textContent=(rep?"Replay of real Backblaze drives (blind test)":sim?"Simulation mode":"Live mode")+" · model: "+s.algo+" · forecast window: "+s.horizon+" days · alert after "+s.confirm+" readings above threshold";
 $("gt").textContent="Failure risk (next "+s.horizon+" days)";
 $("controls").style.display=(rep||sim)?"flex":"none";
 $("repctl").style.display=rep?"inline":"none";$("simctl").style.display=sim?"inline":"none";
 if(rep&&!$("drv").options.length)$("drv").innerHTML=s.drives.map(d=>`<option value="${d}">Drive ${d}</option>`).join("");
 if(rep&&document.activeElement!==$("drv"))$("drv").value=s.drive;
 $("bh").className=s.mode==="healthy"?"on":"";$("bd").className=s.mode==="degrade"?"on":"";
 if(document.activeElement!==$("thr"))$("thr").value=thr;$("thrv").textContent=Math.round(thr*100)+"%";
 $("err").textContent=s.error?("Error: "+s.error):"";
 chart(h,thr);
 if(last){const r=last.risk,st=last.alert?"ALERT":r>=thr*0.6?"WARNING":"HEALTHY";
  document.body.dataset.s=st;$("pill").textContent=st==="ALERT"?"ALERT: FAILURE LIKELY":st;$("state").textContent=st;
  $("risk").textContent=(r*100).toFixed(1);$("arc").style.strokeDashoffset=282.74*(1-r);
  $("cap").textContent=(last.caption||"")+(last.caption&&last.caption.startsWith("Replay finished")?(s.alerts.length?" The model raised an alert.":" The model raised NO alert."):"");scoreboard(s,last);
  $("hint").textContent=st==="ALERT"?"Schedule maintenance for this drive now":(r>=thr?"Above threshold, confirming...":r>=thr*0.6?"Risk rising, keep watching":"No action needed");
  TILES.forEach(t=>{const e=$("t_"+t[0]),v=last[t[0]];e.firstChild.textContent=typeof v==="number"?(Number.isInteger(v)?v:v.toFixed(1)):v;e.className="tile"+(t[2](v)?" flag":"")})}
 $("alerts").innerHTML=s.alerts.length?s.alerts.map(a=>`<div class="alert-item"><b>${a.t}</b> risk ${(a.risk*100).toFixed(1)}% crossed threshold ${Math.round(a.thr*100)}%</div>`).join(""):'<div class="empty">No alerts yet</div>';
 const m=s.metrics||[];$("mcard").style.display=m.length?"block":"none";
 if(m.length)$("metrics").innerHTML="<table><tr><th>Model</th><th>Recall</th><th>Precision</th><th>F1</th><th>Bal. acc.</th><th>ROC-AUC</th></tr>"+
  m.map(r=>`<tr><td>${r.name}</td><td>${(r.recall*100).toFixed(0)}%</td><td>${(r.precision*100).toFixed(0)}%</td><td>${r.f1.toFixed(2)}</td><td>${r.bal_acc.toFixed(3)}</td><td>${r.auc.toFixed(3)}</td></tr>`).join("")+"</table>"}
async function poll(){try{render(await (await fetch("/api/state")).json())}catch(e){$("err").textContent="Connection lost"}}
poll();setInterval(poll,1000);
</script></body></html>"""

if __name__ == "__main__":
    main()
