"""
src/paths.py
Central path definitions for the Climate Crop Radar project.
All paths resolve relative to the project root, so the app works even when
launched with an absolute path from another folder.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = ROOT / "synthetic"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

MERGED_CSV = PROCESSED_DATA_DIR / "merged_zones.csv"
CROP_INFO_CSV = RAW_DATA_DIR / "crop_info.csv"
WEATHER_CSV = RAW_DATA_DIR / "weather_data.csv"
SOIL_CSV = RAW_DATA_DIR / "soil_data.csv"
ZONE_CSV = RAW_DATA_DIR / "zones.csv"
SYNTHETIC_CSV = SYNTHETIC_DATA_DIR / "synthetic_zones.csv"

MODEL_PATH = MODELS_DIR / "risk_model.joblib"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
HISTORY_PATH = MODELS_DIR / "training_history.json"
META_PATH = MODELS_DIR / "model_meta.json"

for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR, ASSETS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
