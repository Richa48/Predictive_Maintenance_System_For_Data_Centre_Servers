"""Creates FAKE Backblaze-format daily CSVs in data/ so you can test the pipeline
before downloading the real dataset.  Usage: python make_synthetic.py"""
import numpy as np, pandas as pd
from common import DATA_DIR, FEATURES

rng = np.random.default_rng(42)
DATA_DIR.mkdir(exist_ok=True)
N_DRIVES, N_DAYS, N_FAIL = 400, 40, 40
days = pd.date_range("2024-01-01", periods=N_DAYS)
fail_day = {f"SN{i:04d}": int(rng.integers(12, N_DAYS)) for i in range(N_FAIL)}

for d, day in enumerate(days):
    rows = []
    for i in range(N_DRIVES):
        sn = f"SN{i:04d}"
        fd = fail_day.get(sn)
        if fd is not None and d > fd:
            continue                                  # drive is gone after failure
        left = (fd - d) if fd is not None else 999    # days until failure
        bad = max(0, 8 - left) if left < 8 else 0     # degradation ramps up before failure
        row = {"date": day.date(), "serial_number": sn, "model": "SYNTH-4TB",
               "capacity_bytes": 4_000_787_030_016, "failure": int(fd is not None and d == fd)}
        row.update({
            "smart_5_raw": int(rng.poisson(0.2) + bad * rng.integers(5, 20)),
            "smart_9_raw": int(20000 + d * 24 + i),
            "smart_187_raw": int(rng.poisson(0.1) + bad * rng.integers(1, 6)),
            "smart_188_raw": int(rng.poisson(0.3)),
            "smart_192_raw": int(rng.poisson(1)),
            "smart_194_raw": int(rng.normal(30 + bad * 1.5, 3)),
            "smart_197_raw": int(rng.poisson(0.1) + bad * rng.integers(2, 10)),
            "smart_198_raw": int(rng.poisson(0.1) + bad * rng.integers(1, 8)),
        })
        rows.append(row)
    pd.DataFrame(rows).to_csv(DATA_DIR / f"{day.date()}.csv", index=False)
print(f"Wrote {N_DAYS} synthetic daily files to {DATA_DIR}")
