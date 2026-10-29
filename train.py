"""Train the failure-prediction model from Backblaze-style daily CSVs in data/.
Label = 1 if the drive fails within the next HORIZON days (3 days = 72 h).
Usage: python train.py [--horizon 3] [--algo rf|xgb] [--healthy-frac 0.1]"""
import argparse, json, glob, hashlib
from collections import Counter
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             balanced_accuracy_score, precision_recall_curve)
from common import (DATA_DIR, MODEL_DIR, MODEL_PATH, METRICS_PATH, FEATURES,
                    RAW_FEATURES, TREND_IDS, TREND_WINDOW)

ap = argparse.ArgumentParser()
ap.add_argument("--horizon", type=int, default=3, help="predict failure within N days")
ap.add_argument("--algo", choices=["rf", "xgb", "hgb"], default="rf")
ap.add_argument("--healthy-frac", type=float, default=0.1, help="share of healthy drives kept (saves RAM)")
ap.add_argument("--model-filter", default=None, help="keep only this drive model, e.g. ST4000DM000, or 'auto' = model with most failures")
args = ap.parse_args()

files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
if not files:
    raise SystemExit(f"No CSV files in {DATA_DIR}. Put Backblaze files there or run make_synthetic.py")
print(f"Found {len(files)} daily files")

# ---- pass 1: which drives ever fail? ----
failed, fail_by_model = set(), Counter()
for f in files:
    d = pd.read_csv(f, usecols=["serial_number", "failure", "model"])
    bad = d[d.failure == 1]
    failed.update(bad.serial_number)
    fail_by_model.update(bad.model)
print(f"Failed drives: {len(failed)}")
if args.model_filter == "auto":
    args.model_filter = fail_by_model.most_common(1)[0][0]
    print(f"Auto-selected drive model: {args.model_filter} ({fail_by_model[args.model_filter]} failures)")

def keep_healthy(sn):  # deterministic sampling of healthy drives
    return int(hashlib.md5(sn.encode()).hexdigest(), 16) % 1000 < args.healthy_frac * 1000

# ---- pass 2: load needed rows ----
cols = {"date", "serial_number", "model", "failure", *RAW_FEATURES}
parts = []
for f in files:
    d = pd.read_csv(f, usecols=lambda c: c in cols)
    for c in RAW_FEATURES:
        if c not in d: d[c] = 0
    if args.model_filter:
        d = d[d.model == args.model_filter]
    mask = d.serial_number.isin(failed) | d.serial_number.map(keep_healthy)
    parts.append(d[mask])
df = pd.concat(parts, ignore_index=True)
df["date"] = pd.to_datetime(df["date"])
df[RAW_FEATURES] = df[RAW_FEATURES].fillna(0)

# ---- trend features: change over the last TREND_WINDOW days, per drive ----
df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
for i in TREND_IDS:
    df[f"smart_{i}_delta{TREND_WINDOW}"] = (
        df.groupby("serial_number")[f"smart_{i}_raw"].diff(TREND_WINDOW).fillna(0))

# ---- labelling ----
fail_date = df.loc[df.failure == 1].groupby("serial_number")["date"].min()
df["fail_date"] = df.serial_number.map(fail_date)
df["days_to_fail"] = (df.fail_date - df.date).dt.days
df = df[~(df.days_to_fail < 0)]                       # drop rows after failure
df["label"] = ((df.days_to_fail >= 0) & (df.days_to_fail <= args.horizon)).astype(int)
print(f"Rows: {len(df)}  positives: {df.label.sum()} ({df.label.mean():.2%})")

# ---- split BY DRIVE (no leakage): train / validation / test ----
X, y, g = df[FEATURES], df.label, df.serial_number
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
tr_idx, te_idx = next(gss.split(X, y, g))
Xtr, ytr, gtr = X.iloc[tr_idx], y.iloc[tr_idx], g.iloc[tr_idx]
Xte, yte = X.iloc[te_idx], y.iloc[te_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2)
a, v = next(gss2.split(Xtr, ytr, gtr))
Xfit, yfit, Xval, yval = Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[v], ytr.iloc[v]

# ---- model ----
if args.algo == "xgb":
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=(yfit == 0).sum() / max(1, (yfit == 1).sum()),
                        eval_metric="logloss", n_jobs=-1)
elif args.algo == "hgb":
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_depth=6,
                                         class_weight="balanced", random_state=0)
else:
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                 class_weight="balanced_subsample", n_jobs=-1, random_state=0)
clf.fit(Xfit, yfit)

# ---- pick alert threshold on validation set (best F1), report on untouched test set ----
p, r, t = precision_recall_curve(yval, clf.predict_proba(Xval)[:, 1])
f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
thr = float(t[np.argmax(f1[:-1])]) if len(t) else 0.5
proba = clf.predict_proba(Xte)[:, 1]
pred = (proba >= thr).astype(int)
metrics = {
    "algo": args.algo, "horizon_days": args.horizon, "threshold": round(thr, 4),
    "accuracy": float((pred == yte).mean()),
    "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
    "roc_auc": float(roc_auc_score(yte, proba)),
    "report": classification_report(yte, pred, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(yte, pred).tolist(),
}
print(classification_report(yte, pred, zero_division=0))
print("Confusion matrix [[TN FP],[FN TP]]:", metrics["confusion_matrix"])
print(f"Accuracy {metrics['accuracy']:.3f} | Balanced acc {metrics['balanced_accuracy']:.3f} | ROC-AUC {metrics['roc_auc']:.3f} | threshold {thr:.3f}")

# ---- STORE the trained model (one file = model + features + threshold) ----
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump({"model": clf, "features": FEATURES, "threshold": thr, "horizon_days": args.horizon}, MODEL_PATH)
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
print(f"Saved model -> {MODEL_PATH}\nSaved metrics -> {METRICS_PATH}")
