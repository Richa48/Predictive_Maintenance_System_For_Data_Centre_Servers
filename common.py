"""Shared settings. The SAME feature list is used for training and live monitoring."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# SMART attribute IDs used as features (Backblaze column = smart_<id>_raw)
SMART_IDS = [5, 9, 187, 188, 192, 194, 197, 198]
RAW_FEATURES = [f"smart_{i}_raw" for i in SMART_IDS]
# Trend features: change in an error counter over the last TREND_WINDOW readings (days)
TREND_IDS = [5, 187, 197, 198]
TREND_WINDOW = 7
TREND_FEATURES = [f"smart_{i}_delta{TREND_WINDOW}" for i in TREND_IDS]
FEATURES = RAW_FEATURES + TREND_FEATURES
