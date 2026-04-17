"""
src/auto_updater.py
Background data fetcher + model retrainer for Climate Crop Radar.

Fetches fresh weather data from Open-Meteo (free, no API key) every N hours,
then retrains the model. Status is written to a JSON file so Streamlit can
display it in the sidebar.

Usage (auto-starts when imported in app.py):
    from src.auto_updater import AutoUpdater
    AutoUpdater.start(interval_hours=2)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Paths ────────────────────────────────────────────────────────────────────
import os, sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import (
    MERGED_CSV, PROCESSED_DATA_DIR, RAW_DATA_DIR,
    WEATHER_CSV, SOIL_CSV, ZONE_CSV, CROP_INFO_CSV,
)

STATUS_PATH = ROOT / "models" / "updater_status.json"
LOG_PATH    = ROOT / "models" / "updater.log"

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [AutoUpdater] %(levelname)s %(message)s",
    handlers= [logging.FileHandler(LOG_PATH, encoding="utf-8"),
               logging.StreamHandler()],
)
log = logging.getLogger("auto_updater")

ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

ZONES = [
    {"zone_id": 0,  "zone_name": "Punjab Wheat Belt",      "lat": 30.9, "lon": 75.8, "crop": "Wheat"},
    {"zone_id": 1,  "zone_name": "Haryana Rice Zone",       "lat": 29.1, "lon": 76.4, "crop": "Rice"},
    {"zone_id": 2,  "zone_name": "UP Sugarcane Belt",       "lat": 26.8, "lon": 80.9, "crop": "Sugarcane"},
    {"zone_id": 3,  "zone_name": "Bihar Maize Zone",        "lat": 25.6, "lon": 85.1, "crop": "Maize"},
    {"zone_id": 4,  "zone_name": "WB Jute Region",          "lat": 22.6, "lon": 88.4, "crop": "Jute"},
    {"zone_id": 5,  "zone_name": "Odisha Rice Delta",       "lat": 20.5, "lon": 85.8, "crop": "Rice"},
    {"zone_id": 6,  "zone_name": "AP Cotton Zone",          "lat": 16.5, "lon": 80.6, "crop": "Cotton"},
    {"zone_id": 7,  "zone_name": "TN Paddy Region",         "lat": 10.8, "lon": 78.7, "crop": "Rice"},
    {"zone_id": 8,  "zone_name": "Karnataka Coffee Zone",   "lat": 13.3, "lon": 75.7, "crop": "Coffee"},
    {"zone_id": 9,  "zone_name": "Kerala Coconut Region",   "lat": 10.5, "lon": 76.2, "crop": "Coconut"},
    {"zone_id": 10, "zone_name": "Maharashtra Soybean",     "lat": 19.7, "lon": 75.3, "crop": "Soybean"},
    {"zone_id": 11, "zone_name": "Gujarat Groundnut",       "lat": 22.3, "lon": 72.6, "crop": "Groundnut"},
    {"zone_id": 12, "zone_name": "Rajasthan Millet Zone",   "lat": 26.9, "lon": 73.9, "crop": "Millet"},
    {"zone_id": 13, "zone_name": "MP Soybean Belt",         "lat": 23.2, "lon": 77.4, "crop": "Soybean"},
    {"zone_id": 14, "zone_name": "Chhattisgarh Rice",       "lat": 21.3, "lon": 81.6, "crop": "Rice"},
    {"zone_id": 15, "zone_name": "Jharkhand Maize",         "lat": 23.6, "lon": 85.5, "crop": "Maize"},
    {"zone_id": 16, "zone_name": "Assam Tea Garden",        "lat": 26.2, "lon": 92.9, "crop": "Tea"},
    {"zone_id": 17, "zone_name": "Nagaland Horticulture",   "lat": 25.7, "lon": 94.1, "crop": "Horticulture"},
    {"zone_id": 18, "zone_name": "Manipur Vegetables",      "lat": 24.7, "lon": 93.9, "crop": "Vegetables"},
    {"zone_id": 19, "zone_name": "Meghalaya Potatoes",      "lat": 25.5, "lon": 91.4, "crop": "Potato"},
    {"zone_id": 20, "zone_name": "Himachal Apple Zone",     "lat": 31.1, "lon": 77.2, "crop": "Apple"},
    {"zone_id": 21, "zone_name": "Uttarakhand Wheat",       "lat": 30.1, "lon": 79.1, "crop": "Wheat"},
    {"zone_id": 22, "zone_name": "J&K Saffron Zone",        "lat": 33.7, "lon": 74.8, "crop": "Saffron"},
    {"zone_id": 23, "zone_name": "Telangana Cotton",        "lat": 17.4, "lon": 78.5, "crop": "Cotton"},
    {"zone_id": 24, "zone_name": "Goa Cashew Region",       "lat": 15.3, "lon": 74.1, "crop": "Cashew"},
    {"zone_id": 25, "zone_name": "Sikkim Cardamom",         "lat": 27.5, "lon": 88.5, "crop": "Cardamom"},
    {"zone_id": 26, "zone_name": "Arunachal Ginger",        "lat": 27.1, "lon": 93.6, "crop": "Ginger"},
    {"zone_id": 27, "zone_name": "Tripura Pineapple",       "lat": 23.8, "lon": 91.3, "crop": "Pineapple"},
    {"zone_id": 28, "zone_name": "Mizoram Turmeric",        "lat": 23.2, "lon": 92.9, "crop": "Turmeric"},
    {"zone_id": 29, "zone_name": "Punjab Mustard Zone",     "lat": 31.5, "lon": 74.3, "crop": "Mustard"},
]

CROP_INFO = [
    {"crop": "Wheat",       "water_need_mm": 450,  "heat_tolerance_c": 32, "drought_tolerance": 0.6},
    {"crop": "Rice",        "water_need_mm": 1200, "heat_tolerance_c": 35, "drought_tolerance": 0.3},
    {"crop": "Maize",       "water_need_mm": 600,  "heat_tolerance_c": 38, "drought_tolerance": 0.5},
    {"crop": "Sugarcane",   "water_need_mm": 1500, "heat_tolerance_c": 38, "drought_tolerance": 0.4},
    {"crop": "Cotton",      "water_need_mm": 700,  "heat_tolerance_c": 40, "drought_tolerance": 0.5},
    {"crop": "Soybean",     "water_need_mm": 500,  "heat_tolerance_c": 35, "drought_tolerance": 0.5},
    {"crop": "Groundnut",   "water_need_mm": 500,  "heat_tolerance_c": 36, "drought_tolerance": 0.6},
    {"crop": "Millet",      "water_need_mm": 350,  "heat_tolerance_c": 42, "drought_tolerance": 0.8},
    {"crop": "Jute",        "water_need_mm": 1000, "heat_tolerance_c": 35, "drought_tolerance": 0.3},
    {"crop": "Coffee",      "water_need_mm": 1600, "heat_tolerance_c": 30, "drought_tolerance": 0.3},
    {"crop": "Coconut",     "water_need_mm": 1800, "heat_tolerance_c": 38, "drought_tolerance": 0.4},
    {"crop": "Tea",         "water_need_mm": 1500, "heat_tolerance_c": 32, "drought_tolerance": 0.3},
    {"crop": "Horticulture","water_need_mm": 800,  "heat_tolerance_c": 35, "drought_tolerance": 0.5},
    {"crop": "Vegetables",  "water_need_mm": 700,  "heat_tolerance_c": 33, "drought_tolerance": 0.4},
    {"crop": "Potato",      "water_need_mm": 500,  "heat_tolerance_c": 28, "drought_tolerance": 0.4},
    {"crop": "Apple",       "water_need_mm": 1200, "heat_tolerance_c": 30, "drought_tolerance": 0.4},
    {"crop": "Saffron",     "water_need_mm": 300,  "heat_tolerance_c": 25, "drought_tolerance": 0.6},
    {"crop": "Cashew",      "water_need_mm": 900,  "heat_tolerance_c": 38, "drought_tolerance": 0.5},
    {"crop": "Cardamom",    "water_need_mm": 1500, "heat_tolerance_c": 30, "drought_tolerance": 0.3},
    {"crop": "Ginger",      "water_need_mm": 1200, "heat_tolerance_c": 33, "drought_tolerance": 0.3},
    {"crop": "Pineapple",   "water_need_mm": 1100, "heat_tolerance_c": 36, "drought_tolerance": 0.4},
    {"crop": "Turmeric",    "water_need_mm": 1300, "heat_tolerance_c": 35, "drought_tolerance": 0.4},
    {"crop": "Mustard",     "water_need_mm": 350,  "heat_tolerance_c": 30, "drought_tolerance": 0.6},
]


# ── Status helpers ────────────────────────────────────────────────────────────

def _write_status(state: str, message: str, extra: dict | None = None):
    payload = {
        "state":      state,        # idle | fetching | training | done | error
        "message":    message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_status() -> dict:
    try:
        if STATUS_PATH.exists():
            return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"state": "idle", "message": "Not started yet.", "updated_at": "—"}


# ── Weather fetch (Open-Meteo, free, no key) ──────────────────────────────────

def _fetch_weather(lat: float, lon: float, days_back: int = 30) -> dict:
    """Fetch recent weather from Open-Meteo Archive API."""
    from datetime import date, timedelta
    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=days_back - 1)
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": str(start),
        "end_date":   str(end),
        "daily":      "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                      "windspeed_10m_max,et0_fao_evapotranspiration",
        "hourly":     "soil_moisture_0_to_7cm",
        "timezone":   "Asia/Kolkata",
    }
    try:
        r = requests.get(ARCHIVE_URL, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning(f"Weather fetch failed ({lat},{lon}): {exc}")
        return {}


def _weather_to_agg(data: dict) -> dict:
    if not data or "daily" not in data:
        return {}
    d = data["daily"]

    def _mean(lst):
        a = np.array(lst, dtype=float)
        return float(np.nanmean(a)) if a.size else np.nan

    def _sum(lst):
        a = np.array(lst, dtype=float)
        return float(np.nansum(a[~np.isnan(a)])) if a.size else np.nan

    tmax = d.get("temperature_2m_max", [])
    tmin = d.get("temperature_2m_min", [])
    nd   = len(d.get("time", []))

    sm_hourly = data.get("hourly", {}).get("soil_moisture_0_to_7cm", [])
    if sm_hourly and len(sm_hourly) >= nd * 24:
        sm = np.array(sm_hourly[:nd * 24], dtype=float).reshape(nd, 24)
        soil_moisture = float(np.nanmean(sm))
    else:
        soil_moisture = np.nan

    return {
        "temp_max_mean":      _mean(tmax),
        "temp_min_mean":      _mean(tmin),
        "temp_avg":           (_mean(tmax) + _mean(tmin)) / 2,
        "precip_total_mm":    _sum(d.get("precipitation_sum", [])),
        "precip_avg_mm":      _mean(d.get("precipitation_sum", [])),
        "windspeed_mean":     _mean(d.get("windspeed_10m_max", [])),
        "et0_mean":           _mean(d.get("et0_fao_evapotranspiration", [])),
        "soil_moisture_mean": soil_moisture,
    }


def _fetch_soil(lat: float, lon: float) -> dict:
    """Fetch soil properties from SoilGrids (free)."""
    params = {
        "lat":      lat, "lon": lon,
        "property": ["phh2o", "clay", "soc", "bdod", "sand", "silt"],
        "depth":    "0-5cm",
        "value":    "mean",
    }
    scale_map = {"phh2o": 0.1, "clay": 0.1, "soc": 0.1,
                 "bdod": 0.01, "sand": 0.1, "silt": 0.1}
    try:
        r = requests.get(SOILGRIDS_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning(f"SoilGrids fetch failed ({lat},{lon}): {exc}")
        return {}
    result = {}
    for prop in data.get("properties", {}).get("layers", []):
        name   = prop.get("name", "")
        depths = prop.get("depths", [])
        if depths:
            val = depths[0].get("values", {}).get("mean")
            if val is not None:
                result[f"soil_{name}"] = val * scale_map.get(name, 1.0)
    return result


# ── Feature engineering ───────────────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    T  = df["temp_avg"].fillna(30)
    RH = (df["soil_moisture_mean"].fillna(0.3) * 100).clip(0, 100)
    df["heat_index"] = (
        -8.78469 + 1.61139411 * T + 2.338549 * RH
        - 0.14611605 * T * RH - 0.01230894 * T**2
        - 0.01642482 * RH**2 + 0.00221173 * T**2 * RH
        + 0.00072546 * T * RH**2 - 0.00000358 * T**2 * RH**2
    )
    df["temp_rain_ratio"] = T / (df["precip_avg_mm"].fillna(1) + 1)
    df["drought_index"]   = (df["et0_mean"].fillna(3) * 30) / (df["precip_total_mm"].fillna(1) + 1)
    soc  = df.get("soil_soc",  pd.Series(10.0, index=df.index)).fillna(10)
    bdod = df.get("soil_bdod", pd.Series(1.3,  index=df.index)).fillna(1.3)
    df["soil_stress"] = (bdod / 1.3) - (soc / 20)
    return df


def _assign_risk_labels(df: pd.DataFrame, crop_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        crop_df[["crop", "water_need_mm", "heat_tolerance_c", "drought_tolerance"]],
        on="crop", how="left",
    )
    labels = []
    for _, r in df.iterrows():
        precip = r.get("precip_total_mm", 0) or 0
        temp   = r.get("temp_max_mean", 30) or 30
        soil_m = r.get("soil_moisture_mean", 0.3) or 0.3
        drIdx  = r.get("drought_index", 1.0) or 1.0
        soilSt = r.get("soil_stress", 0.0) or 0.0
        heat_t = r.get("heat_tolerance_c", 35) or 35
        water_n= r.get("water_need_mm", 600) or 600
        drTol  = r.get("drought_tolerance", 0.5) or 0.5
        if precip > water_n * 1.5 or soil_m > 0.45:
            label = 3    # Flood
        elif temp > heat_t + 3:
            label = 2    # Heat Stress
        elif drIdx > (2.0 / drTol) or soil_m < 0.15:
            label = 1    # Drought
        elif soilSt > 0.4:
            label = 4    # Soil Risk
        else:
            label = 0    # Safe
        labels.append(label)
    df["risk_label"] = labels
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(soil_cache: dict | None = None) -> bool:
    """
    Full fetch + engineer + label + save pipeline.
    Returns True on success.

    soil_cache: dict mapping zone_id → soil dict (to avoid re-fetching stable soil data)
    """
    _write_status("fetching", "Fetching weather data from Open-Meteo…")
    log.info("Starting data fetch pipeline")

    crop_df = pd.DataFrame(CROP_INFO)
    weather_records = []

    for i, z in enumerate(ZONES):
        _write_status("fetching", f"Fetching weather zone {i+1}/{len(ZONES)}: {z['zone_name']}")
        raw = _fetch_weather(z["lat"], z["lon"], days_back=30)
        agg = _weather_to_agg(raw)
        row = {"zone_id": z["zone_id"], "zone_name": z["zone_name"],
               "lat": z["lat"], "lon": z["lon"], "crop": z["crop"]}
        row.update(agg)
        weather_records.append(row)
        time.sleep(0.25)   # gentle rate-limit

    weather_df = pd.DataFrame(weather_records)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(WEATHER_CSV, index=False)
    log.info(f"Weather saved → {WEATHER_CSV}")

    # ── Soil (fetch only on first run; soil doesn't change hourly) ────────────
    if SOIL_CSV.exists():
        soil_df = pd.read_csv(SOIL_CSV)
        log.info("Reusing cached soil data")
    else:
        _write_status("fetching", "Fetching soil data from SoilGrids (one-time)…")
        soil_records = []
        for i, z in enumerate(ZONES):
            _write_status("fetching", f"Fetching soil zone {i+1}/{len(ZONES)}: {z['zone_name']}")
            soil = _fetch_soil(z["lat"], z["lon"])
            soil["zone_id"] = z["zone_id"]
            soil_records.append(soil)
            time.sleep(0.4)
        soil_df = pd.DataFrame(soil_records)
        soil_df.to_csv(SOIL_CSV, index=False)
        log.info(f"Soil saved → {SOIL_CSV}")

    # ── Merge & engineer ──────────────────────────────────────────────────────
    _write_status("training", "Merging data and engineering features…")
    merged = weather_df.merge(soil_df, on="zone_id", how="left")
    num_cols = merged.select_dtypes(include=[np.number]).columns
    merged[num_cols] = merged[num_cols].fillna(merged[num_cols].median())
    merged = merged.fillna(0)
    merged = _engineer_features(merged)
    merged = _assign_risk_labels(merged, crop_df)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False)
    log.info(f"Merged data saved → {MERGED_CSV} | {len(merged)} zones")
    log.info(f"Risk distribution:\n{merged['risk_label'].value_counts().to_string()}")

    # ── Retrain model ─────────────────────────────────────────────────────────
    _write_status("training", "Retraining model on fresh data…")
    try:
        from src.graph_builder import build_graph
        from src.trainer import RiskTrainer
        X, adj_norm, adj_mask, y, df2, feat_cols = build_graph(fit_scaler=True, fallback=False)
        trainer = RiskTrainer(random_state=42)
        trainer.train(X, adj_norm, y, feat_cols)
        log.info("Model retrained successfully")
    except Exception as exc:
        log.error(f"Model retraining failed: {exc}\n{traceback.format_exc()}")
        _write_status("error", f"Model retrain failed: {exc}")
        return False

    dist = {int(k): int(v) for k, v in merged["risk_label"].value_counts().items()}
    _write_status("done", "Data refreshed and model retrained successfully.", {
        "risk_distribution": dist,
        "n_zones": len(merged),
    })
    return True


# ── Scheduler ────────────────────────────────────────────────────────────────

class AutoUpdater:
    _thread: threading.Thread | None = None
    _stop_event = threading.Event()
    _interval_sec: int = 7200   # default 2 hours

    @classmethod
    def start(cls, interval_hours: float = 2.0, run_immediately: bool = True):
        """Start the background update thread (idempotent)."""
        if cls._thread is not None and cls._thread.is_alive():
            return   # already running
        cls._interval_sec = int(interval_hours * 3600)
        cls._stop_event.clear()
        cls._thread = threading.Thread(
            target=cls._loop,
            args=(run_immediately,),
            daemon=True,
            name="ClimateRadar-AutoUpdater",
        )
        cls._thread.start()
        log.info(f"AutoUpdater started — interval {interval_hours}h, "
                 f"immediate={run_immediately}")

    @classmethod
    def stop(cls):
        cls._stop_event.set()

    @classmethod
    def is_running(cls) -> bool:
        return cls._thread is not None and cls._thread.is_alive()

    @classmethod
    def _loop(cls, run_immediately: bool):
        if run_immediately:
            cls._safe_run()
        while not cls._stop_event.wait(timeout=cls._interval_sec):
            cls._safe_run()

    @classmethod
    def _safe_run(cls):
        try:
            run_pipeline()
        except Exception as exc:
            log.error(f"Pipeline error: {exc}\n{traceback.format_exc()}")
            _write_status("error", str(exc))
