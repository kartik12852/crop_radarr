"""
data/fetch_data.py — Climate Crop Radar Data Pipeline
Run: python data/fetch_data.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from tqdm import tqdm

from src.paths import (RAW_DATA_DIR, PROCESSED_DATA_DIR,
                       ZONE_CSV, CROP_INFO_CSV, WEATHER_CSV,
                       SOIL_CSV, MERGED_CSV)

# ── Zone definitions ─────────────────────────────────────────────────────────
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
    {"crop": "Wheat",       "water_need_mm": 450,  "heat_tolerance_c": 32, "drought_tolerance": 0.6, "sowing_month": 10, "harvest_month": 4},
    {"crop": "Rice",        "water_need_mm": 1200, "heat_tolerance_c": 35, "drought_tolerance": 0.3, "sowing_month": 6,  "harvest_month": 11},
    {"crop": "Maize",       "water_need_mm": 600,  "heat_tolerance_c": 38, "drought_tolerance": 0.5, "sowing_month": 5,  "harvest_month": 10},
    {"crop": "Sugarcane",   "water_need_mm": 1500, "heat_tolerance_c": 38, "drought_tolerance": 0.4, "sowing_month": 2,  "harvest_month": 12},
    {"crop": "Cotton",      "water_need_mm": 700,  "heat_tolerance_c": 40, "drought_tolerance": 0.5, "sowing_month": 5,  "harvest_month": 11},
    {"crop": "Soybean",     "water_need_mm": 500,  "heat_tolerance_c": 35, "drought_tolerance": 0.5, "sowing_month": 6,  "harvest_month": 10},
    {"crop": "Groundnut",   "water_need_mm": 500,  "heat_tolerance_c": 36, "drought_tolerance": 0.6, "sowing_month": 6,  "harvest_month": 10},
    {"crop": "Millet",      "water_need_mm": 350,  "heat_tolerance_c": 42, "drought_tolerance": 0.8, "sowing_month": 6,  "harvest_month": 10},
    {"crop": "Jute",        "water_need_mm": 1000, "heat_tolerance_c": 35, "drought_tolerance": 0.3, "sowing_month": 4,  "harvest_month": 9},
    {"crop": "Coffee",      "water_need_mm": 1600, "heat_tolerance_c": 30, "drought_tolerance": 0.3, "sowing_month": 6,  "harvest_month": 1},
    {"crop": "Coconut",     "water_need_mm": 1800, "heat_tolerance_c": 38, "drought_tolerance": 0.4, "sowing_month": 6,  "harvest_month": 12},
    {"crop": "Tea",         "water_need_mm": 1500, "heat_tolerance_c": 32, "drought_tolerance": 0.3, "sowing_month": 3,  "harvest_month": 11},
    {"crop": "Horticulture","water_need_mm": 800,  "heat_tolerance_c": 35, "drought_tolerance": 0.5, "sowing_month": 4,  "harvest_month": 10},
    {"crop": "Vegetables",  "water_need_mm": 700,  "heat_tolerance_c": 33, "drought_tolerance": 0.4, "sowing_month": 3,  "harvest_month": 9},
    {"crop": "Potato",      "water_need_mm": 500,  "heat_tolerance_c": 28, "drought_tolerance": 0.4, "sowing_month": 10, "harvest_month": 3},
    {"crop": "Apple",       "water_need_mm": 1200, "heat_tolerance_c": 30, "drought_tolerance": 0.4, "sowing_month": 3,  "harvest_month": 10},
    {"crop": "Saffron",     "water_need_mm": 300,  "heat_tolerance_c": 25, "drought_tolerance": 0.6, "sowing_month": 7,  "harvest_month": 11},
    {"crop": "Cashew",      "water_need_mm": 900,  "heat_tolerance_c": 38, "drought_tolerance": 0.5, "sowing_month": 3,  "harvest_month": 6},
    {"crop": "Cardamom",    "water_need_mm": 1500, "heat_tolerance_c": 30, "drought_tolerance": 0.3, "sowing_month": 6,  "harvest_month": 12},
    {"crop": "Ginger",      "water_need_mm": 1200, "heat_tolerance_c": 33, "drought_tolerance": 0.3, "sowing_month": 4,  "harvest_month": 12},
    {"crop": "Pineapple",   "water_need_mm": 1100, "heat_tolerance_c": 36, "drought_tolerance": 0.4, "sowing_month": 2,  "harvest_month": 8},
    {"crop": "Turmeric",    "water_need_mm": 1300, "heat_tolerance_c": 35, "drought_tolerance": 0.4, "sowing_month": 5,  "harvest_month": 1},
    {"crop": "Mustard",     "water_need_mm": 350,  "heat_tolerance_c": 30, "drought_tolerance": 0.6, "sowing_month": 10, "harvest_month": 3},
]

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
SOILGRIDS_URL= "https://rest.isric.org/soilgrids/v2.0/properties/query"


def fetch_weather_zone(lat, lon, days_back=90):
    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=days_back - 1)

    daily_vars  = ["temperature_2m_max", "temperature_2m_min",
                   "precipitation_sum", "windspeed_10m_max",
                   "et0_fao_evapotranspiration", "weathercode"]
    hourly_vars = ["soil_moisture_0_to_7cm"]

    params = {
        "latitude":   lat, "longitude": lon,
        "start_date": str(start), "end_date": str(end),
        "daily":      ",".join(daily_vars),
        "hourly":     ",".join(hourly_vars),
        "timezone":   "Asia/Kolkata",
    }
    try:
        r = requests.get(ARCHIVE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ⚠ Weather fetch failed ({lat},{lon}): {e}")
        return {}

    daily_soil = []
    if "hourly" in data and "soil_moisture_0_to_7cm" in data["hourly"]:
        sm   = np.array(data["hourly"]["soil_moisture_0_to_7cm"], dtype=float)
        nd   = len(data["daily"].get("time", []))
        if len(sm) >= nd * 24:
            sm = sm[:nd * 24].reshape(nd, 24)
            daily_soil = np.nanmean(sm, axis=1).tolist()

    nd = len(data["daily"].get("time", []))
    return {
        "time":        data["daily"].get("time", []),
        "temp_max":    data["daily"].get("temperature_2m_max", []),
        "temp_min":    data["daily"].get("temperature_2m_min", []),
        "precipitation": data["daily"].get("precipitation_sum", []),
        "windspeed":   data["daily"].get("windspeed_10m_max", []),
        "et0":         data["daily"].get("et0_fao_evapotranspiration", []),
        "weathercode": data["daily"].get("weathercode", []),
        "soil_moisture": daily_soil if daily_soil else [np.nan] * nd,
    }


def weather_to_agg(raw: dict) -> dict:
    if not raw or not raw.get("time"):
        return {}
    def _mean(lst):
        a = np.array(lst, dtype=float)
        return float(np.nanmean(a)) if a.size else np.nan
    def _sum(lst):
        a = np.array(lst, dtype=float)
        return float(np.nansum(a[~np.isnan(a)])) if a.size else np.nan
    return {
        "temp_max_mean":      _mean(raw["temp_max"]),
        "temp_min_mean":      _mean(raw["temp_min"]),
        "temp_avg":           (_mean(raw["temp_max"]) + _mean(raw["temp_min"])) / 2,
        "precip_total_mm":    _sum(raw["precipitation"]),
        "precip_avg_mm":      _mean(raw["precipitation"]),
        "windspeed_mean":     _mean(raw["windspeed"]),
        "et0_mean":           _mean(raw["et0"]),
        "soil_moisture_mean": _mean(raw["soil_moisture"]),
    }


def fetch_soil_zone(lat, lon) -> dict:
    params = {"lat": lat, "lon": lon,
              "property": ["phh2o","clay","soc","bdod","sand","silt"],
              "depth": "0-5cm", "value": "mean"}
    try:
        r = requests.get(SOILGRIDS_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ⚠ SoilGrids fetch failed ({lat},{lon}): {e}")
        return {}
    result = {}
    scale_map = {"phh2o": 0.1, "clay": 0.1, "soc": 0.1,
                 "bdod": 0.01, "sand": 0.1, "silt": 0.1}
    for prop in data.get("properties", {}).get("layers", []):
        name = prop.get("name", "")
        depths = prop.get("depths", [])
        if depths:
            val = depths[0].get("values", {}).get("mean")
            if val is not None:
                result[f"soil_{name}"] = val * scale_map.get(name, 1.0)
    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    T  = df["temp_avg"].fillna(30)
    RH = (df["soil_moisture_mean"].fillna(0.3) * 100).clip(0, 100)
    df["heat_index"] = (
        -8.78469 + 1.61139411*T + 2.338549*RH
        - 0.14611605*T*RH - 0.01230894*T**2
        - 0.01642482*RH**2 + 0.00221173*T**2*RH
        + 0.00072546*T*RH**2 - 0.00000358*T**2*RH**2
    )
    df["temp_rain_ratio"] = T / (df["precip_avg_mm"].fillna(1) + 1)
    df["drought_index"]   = (df["et0_mean"].fillna(3) * 90) / (df["precip_total_mm"].fillna(1) + 1)
    soc  = df.get("soil_soc",  pd.Series(10.0, index=df.index)).fillna(10)
    bdod = df.get("soil_bdod", pd.Series(1.3,  index=df.index)).fillna(1.3)
    df["soil_stress"] = (bdod / 1.3) - (soc / 20)
    return df


def assign_risk_labels(df: pd.DataFrame, crop_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        crop_df[["crop","water_need_mm","heat_tolerance_c","drought_tolerance"]],
        on="crop", how="left")
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
            label = 3
        elif temp > heat_t + 3:
            label = 2
        elif drIdx > (2.0 / drTol) or soil_m < 0.15:
            label = 1
        elif soilSt > 0.4:
            label = 4
        else:
            label = 0
        labels.append(label)
    df["risk_label"] = labels
    return df


def main():
    print("=" * 60)
    print("  Climate Radar — Data Fetch Pipeline")
    print("=" * 60)

    zone_df = pd.DataFrame(ZONES)
    zone_df.to_csv(ZONE_CSV, index=False)
    print(f"✓ Zones → {ZONE_CSV}")

    crop_df = pd.DataFrame(CROP_INFO)
    crop_df.to_csv(CROP_INFO_CSV, index=False)
    print(f"✓ Crop info → {CROP_INFO_CSV}")

    print("\n[1/3] Fetching weather from Open-Meteo Archive …")
    weather_records = []
    for z in tqdm(ZONES, desc="Weather"):
        raw = fetch_weather_zone(z["lat"], z["lon"])
        agg = weather_to_agg(raw)
        if agg:
            agg.update({"zone_id": z["zone_id"], "zone_name": z["zone_name"],
                        "lat": z["lat"], "lon": z["lon"], "crop": z["crop"]})
            weather_records.append(agg)
        else:
            weather_records.append({
                "zone_id": z["zone_id"], "zone_name": z["zone_name"],
                "lat": z["lat"], "lon": z["lon"], "crop": z["crop"],
                "temp_max_mean": np.nan, "temp_min_mean": np.nan, "temp_avg": np.nan,
                "precip_total_mm": np.nan, "precip_avg_mm": np.nan,
                "windspeed_mean": np.nan, "et0_mean": np.nan, "soil_moisture_mean": np.nan,
            })
        time.sleep(0.4)

    weather_df = pd.DataFrame(weather_records)
    weather_df.to_csv(WEATHER_CSV, index=False)
    print(f"✓ Weather saved → {WEATHER_CSV} ({len(weather_df)} zones)")

    print("\n[2/3] Fetching soil from SoilGrids …")
    soil_records = []
    for z in tqdm(ZONES, desc="Soil"):
        soil = fetch_soil_zone(z["lat"], z["lon"])
        soil["zone_id"] = z["zone_id"]
        soil_records.append(soil)
        time.sleep(0.5)
    soil_df = pd.DataFrame(soil_records)
    soil_df.to_csv(SOIL_CSV, index=False)
    print(f"✓ Soil saved → {SOIL_CSV}")

    print("\n[3/3] Merging and engineering features …")
    merged = weather_df.merge(soil_df, on="zone_id", how="left")
    num_cols = merged.select_dtypes(include=[np.number]).columns
    merged[num_cols] = merged[num_cols].fillna(merged[num_cols].median())
    merged = merged.fillna(0)
    merged = engineer_features(merged)
    merged = assign_risk_labels(merged, crop_df)
    merged.to_csv(MERGED_CSV, index=False)
    print(f"\n✅ Merged dataset saved → {MERGED_CSV}")
    print(f"   Rows: {len(merged)} | Columns: {len(merged.columns)}")
    print(f"   Risk distribution:\n{merged['risk_label'].value_counts().to_string()}")
    print("\nDone! Next: python train_model.py")


if __name__ == "__main__":
    main()
