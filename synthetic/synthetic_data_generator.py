"""
synthetic/synthetic_data_generator.py
Generates realistic synthetic climate + soil data for 30 Indian zones.
Used as offline fallback when API fetch fails.
Run: python synthetic/synthetic_data_generator.py

FIX: Zones now have climate-realistic risk labels instead of equal distribution.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.paths import MERGED_CSV, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR

np.random.seed(42)

ZONES = [
    (0,  "Punjab Wheat Belt",      30.9, 75.8, "Wheat"),
    (1,  "Haryana Rice Zone",       29.1, 76.4, "Rice"),
    (2,  "UP Sugarcane Belt",       26.8, 80.9, "Sugarcane"),
    (3,  "Bihar Maize Zone",        25.6, 85.1, "Maize"),
    (4,  "WB Jute Region",          22.6, 88.4, "Jute"),
    (5,  "Odisha Rice Delta",       20.5, 85.8, "Rice"),
    (6,  "AP Cotton Zone",          16.5, 80.6, "Cotton"),
    (7,  "TN Paddy Region",         10.8, 78.7, "Rice"),
    (8,  "Karnataka Coffee Zone",   13.3, 75.7, "Coffee"),
    (9,  "Kerala Coconut Region",   10.5, 76.2, "Coconut"),
    (10, "Maharashtra Soybean",     19.7, 75.3, "Soybean"),
    (11, "Gujarat Groundnut",       22.3, 72.6, "Groundnut"),
    (12, "Rajasthan Millet Zone",   26.9, 73.9, "Millet"),
    (13, "MP Soybean Belt",         23.2, 77.4, "Soybean"),
    (14, "Chhattisgarh Rice",       21.3, 81.6, "Rice"),
    (15, "Jharkhand Maize",         23.6, 85.5, "Maize"),
    (16, "Assam Tea Garden",        26.2, 92.9, "Tea"),
    (17, "Nagaland Horticulture",   25.7, 94.1, "Horticulture"),
    (18, "Manipur Vegetables",      24.7, 93.9, "Vegetables"),
    (19, "Meghalaya Potatoes",      25.5, 91.4, "Potato"),
    (20, "Himachal Apple Zone",     31.1, 77.2, "Apple"),
    (21, "Uttarakhand Wheat",       30.1, 79.1, "Wheat"),
    (22, "J&K Saffron Zone",        33.7, 74.8, "Saffron"),
    (23, "Telangana Cotton",        17.4, 78.5, "Cotton"),
    (24, "Goa Cashew Region",       15.3, 74.1, "Cashew"),
    (25, "Sikkim Cardamom",         27.5, 88.5, "Cardamom"),
    (26, "Arunachal Ginger",        27.1, 93.6, "Ginger"),
    (27, "Tripura Pineapple",       23.8, 91.3, "Pineapple"),
    (28, "Mizoram Turmeric",        23.2, 92.9, "Turmeric"),
    (29, "Punjab Mustard Zone",     31.5, 74.3, "Mustard"),
]

# ── Realistic risk assignments based on actual Indian agricultural geography ──
# 0=Safe, 1=Drought, 2=Heat Stress, 3=Flood, 4=Soil Risk
ZONE_RISK_MAP = {
    0:  0,   # Punjab Wheat Belt       → Safe (good irrigation, fertile)
    1:  2,   # Haryana Rice Zone       → Heat Stress (hot summers)
    2:  3,   # UP Sugarcane Belt       → Flood (Ganga basin)
    3:  3,   # Bihar Maize Zone        → Flood (frequently flooded)
    4:  3,   # WB Jute Region          → Flood (deltaic, waterlogged)
    5:  3,   # Odisha Rice Delta       → Flood (cyclone + deltaic floods)
    6:  1,   # AP Cotton Zone          → Drought (rain-shadow region)
    7:  0,   # TN Paddy Region         → Safe (canal irrigation)
    8:  0,   # Karnataka Coffee Zone   → Safe (good rainfall, cool)
    9:  3,   # Kerala Coconut Region   → Flood (heavy monsoon)
    10: 1,   # Maharashtra Soybean     → Drought (Vidarbha drought belt)
    11: 1,   # Gujarat Groundnut       → Drought (semi-arid)
    12: 1,   # Rajasthan Millet Zone   → Drought (Thar desert fringe)
    13: 0,   # MP Soybean Belt         → Safe (moderate rainfall)
    14: 0,   # Chhattisgarh Rice       → Safe (good rainfall)
    15: 4,   # Jharkhand Maize         → Soil Risk (laterite soils)
    16: 3,   # Assam Tea Garden        → Flood (Brahmaputra flooding)
    17: 0,   # Nagaland Horticulture   → Safe (terrace farming)
    18: 0,   # Manipur Vegetables      → Safe (valley farming)
    19: 3,   # Meghalaya Potatoes      → Flood (highest rainfall globally)
    20: 0,   # Himachal Apple Zone     → Safe (cool, irrigated)
    21: 2,   # Uttarakhand Wheat       → Heat Stress (valley heat)
    22: 0,   # J&K Saffron Zone        → Safe (cool plateau)
    23: 2,   # Telangana Cotton        → Heat Stress (very hot dry summer)
    24: 0,   # Goa Cashew Region       → Safe (coastal moderate)
    25: 4,   # Sikkim Cardamom         → Soil Risk (steep slopes, erosion)
    26: 4,   # Arunachal Ginger        → Soil Risk (high slope erosion)
    27: 0,   # Tripura Pineapple       → Safe (moderate hills)
    28: 2,   # Mizoram Turmeric        → Heat Stress (valley heat)
    29: 1,   # Punjab Mustard Zone     → Drought (rabi, low water)
}

CROP_INFO = {
    "Wheat":       {"water_need_mm": 450,  "heat_tolerance_c": 32, "drought_tolerance": 0.6},
    "Rice":        {"water_need_mm": 1200, "heat_tolerance_c": 35, "drought_tolerance": 0.3},
    "Maize":       {"water_need_mm": 600,  "heat_tolerance_c": 38, "drought_tolerance": 0.5},
    "Sugarcane":   {"water_need_mm": 1500, "heat_tolerance_c": 38, "drought_tolerance": 0.4},
    "Cotton":      {"water_need_mm": 700,  "heat_tolerance_c": 40, "drought_tolerance": 0.5},
    "Soybean":     {"water_need_mm": 500,  "heat_tolerance_c": 35, "drought_tolerance": 0.5},
    "Groundnut":   {"water_need_mm": 500,  "heat_tolerance_c": 36, "drought_tolerance": 0.6},
    "Millet":      {"water_need_mm": 350,  "heat_tolerance_c": 42, "drought_tolerance": 0.8},
    "Jute":        {"water_need_mm": 1000, "heat_tolerance_c": 35, "drought_tolerance": 0.3},
    "Coffee":      {"water_need_mm": 1600, "heat_tolerance_c": 30, "drought_tolerance": 0.3},
    "Coconut":     {"water_need_mm": 1800, "heat_tolerance_c": 38, "drought_tolerance": 0.4},
    "Tea":         {"water_need_mm": 1500, "heat_tolerance_c": 32, "drought_tolerance": 0.3},
    "Horticulture":{"water_need_mm": 800,  "heat_tolerance_c": 35, "drought_tolerance": 0.5},
    "Vegetables":  {"water_need_mm": 700,  "heat_tolerance_c": 33, "drought_tolerance": 0.4},
    "Potato":      {"water_need_mm": 500,  "heat_tolerance_c": 28, "drought_tolerance": 0.4},
    "Apple":       {"water_need_mm": 1200, "heat_tolerance_c": 30, "drought_tolerance": 0.4},
    "Saffron":     {"water_need_mm": 300,  "heat_tolerance_c": 25, "drought_tolerance": 0.6},
    "Cashew":      {"water_need_mm": 900,  "heat_tolerance_c": 38, "drought_tolerance": 0.5},
    "Cardamom":    {"water_need_mm": 1500, "heat_tolerance_c": 30, "drought_tolerance": 0.3},
    "Ginger":      {"water_need_mm": 1200, "heat_tolerance_c": 33, "drought_tolerance": 0.3},
    "Pineapple":   {"water_need_mm": 1100, "heat_tolerance_c": 36, "drought_tolerance": 0.4},
    "Turmeric":    {"water_need_mm": 1300, "heat_tolerance_c": 35, "drought_tolerance": 0.4},
    "Mustard":     {"water_need_mm": 350,  "heat_tolerance_c": 30, "drought_tolerance": 0.6},
}

# scenario_idx → (precip_multiplier, temp_delta, soil_moisture_delta, soil_stress_delta)
RISK_SCENARIOS = {
    0: (1.0,  0.0,  0.0,   0.0),    # Safe
    1: (0.3, +1.5, -0.15,  0.1),    # Drought
    2: (0.7, +6.0,  0.0,   0.0),    # Heat Stress
    3: (3.5,  0.0, +0.18,  0.0),    # Flood
    4: (0.9,  1.0, -0.05,  0.5),    # Soil Risk
}


def generate_zone_record(zone_id, zone_name, lat, lon, crop, scenario_idx):
    cd = CROP_INFO.get(crop, CROP_INFO["Wheat"])
    sc = RISK_SCENARIOS[scenario_idx]
    base_temp   = 20 + 0.1 * (lat - 10) * (-1) + np.random.normal(0, 2)
    base_precip = 300 + cd["water_need_mm"] * 0.3 + np.random.normal(0, 50)
    temp_max    = base_temp + 8 + sc[1] + np.random.normal(0, 1.5)
    temp_min    = base_temp - 4 + np.random.normal(0, 1)
    temp_avg    = (temp_max + temp_min) / 2
    precip_total= max(5, base_precip * sc[0] + np.random.normal(0, 30))
    precip_avg  = precip_total / 90
    et0_mean    = 2.5 + 0.05 * temp_avg + np.random.normal(0, 0.3)
    soil_moisture = np.clip(0.28 + sc[2] + np.random.normal(0, 0.05), 0.05, 0.55)
    windspeed   = np.random.uniform(5, 20)
    soil_phh2o  = np.clip(6.5 + np.random.normal(0, 0.5), 5.0, 8.5)
    soil_clay   = np.clip(25 + np.random.normal(0, 8), 5, 60)
    soil_soc    = np.clip(12 + np.random.normal(0, 4) - sc[3] * 5, 2, 35)
    soil_bdod   = np.clip(1.3 + sc[3] * 0.2 + np.random.normal(0, 0.1), 0.9, 1.9)
    soil_sand   = np.clip(100 - soil_clay + np.random.normal(0, 5), 10, 80)
    soil_silt   = np.clip(100 - soil_clay - soil_sand, 5, 50)
    rh          = soil_moisture * 100
    heat_index  = (
        -8.78469 + 1.61139411 * temp_avg + 2.338549 * rh
        - 0.14611605 * temp_avg * rh - 0.01230894 * temp_avg**2
        - 0.01642482 * rh**2 + 0.00221173 * temp_avg**2 * rh
        + 0.00072546 * temp_avg * rh**2 - 0.00000358 * temp_avg**2 * rh**2
    )
    temp_rain_ratio = temp_avg / (precip_avg + 1)
    drought_index   = (et0_mean * 90) / (precip_total + 1)
    soil_stress     = (soil_bdod / 1.3) - (soil_soc / 20)
    return {
        "zone_id": zone_id,
        "zone_name": zone_name,
        "lat": lat,
        "lon": lon,
        "crop": crop,
        "temp_max_mean":      round(temp_max, 2),
        "temp_min_mean":      round(temp_min, 2),
        "temp_avg":           round(temp_avg, 2),
        "precip_total_mm":    round(precip_total, 1),
        "precip_avg_mm":      round(precip_avg, 3),
        "windspeed_mean":     round(windspeed, 1),
        "et0_mean":           round(et0_mean, 3),
        "soil_moisture_mean": round(soil_moisture, 4),
        "soil_phh2o":         round(soil_phh2o, 2),
        "soil_clay":          round(soil_clay, 1),
        "soil_soc":           round(soil_soc, 2),
        "soil_bdod":          round(soil_bdod, 3),
        "soil_sand":          round(soil_sand, 1),
        "soil_silt":          round(soil_silt, 1),
        "heat_index":         round(heat_index, 2),
        "temp_rain_ratio":    round(temp_rain_ratio, 3),
        "drought_index":      round(drought_index, 3),
        "soil_stress":        round(soil_stress, 4),
        "water_need_mm":      cd["water_need_mm"],
        "heat_tolerance_c":   cd["heat_tolerance_c"],
        "drought_tolerance":  cd["drought_tolerance"],
        "risk_label":         int(scenario_idx),
    }


def main(n_samples_per_zone: int = 10):
    print("=" * 55)
    print("  Synthetic Data Generator — Climate Radar")
    print("=" * 55)
    records = []
    for zone in ZONES:
        zone_id, zone_name, lat, lon, crop = zone
        # Use realistic geographic risk as dominant label
        dominant = ZONE_RISK_MAP[zone_id]
        # Dominant label gets majority of samples; minority noise from other classes
        noise_classes = [c for c in range(5) if c != dominant]
        noise_counts  = max(0, n_samples_per_zone - 7)
        noise_labels  = np.random.choice(noise_classes, size=noise_counts).tolist() if noise_counts > 0 else []
        risks = [dominant] * 7 + noise_labels
        np.random.shuffle(risks)
        for r in risks:
            records.append(generate_zone_record(zone_id, zone_name, lat, lon, crop, int(r)))

    df = pd.DataFrame(records)
    out_path = SYNTHETIC_DATA_DIR / "synthetic_zones.csv"
    df.to_csv(out_path, index=False)
    print(f"✓ Synthetic data -> {out_path}  ({len(df)} records)")
    print(f"  Sample risk distribution:\n{df['risk_label'].value_counts().sort_index().to_string()}")

    def _mode(series):
        m = series.mode(dropna=True)
        return int(m.iloc[0]) if not m.empty else int(series.iloc[0])

    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    num_no_id = [c for c in num_cols if c not in {"zone_id", "risk_label"}]
    agg = df.groupby("zone_id")[num_no_id].mean().reset_index()
    for col in ["zone_name", "crop", "lat", "lon"]:
        agg[col] = agg["zone_id"].map(df.groupby("zone_id")[col].first())
    agg["risk_label"] = [ZONE_RISK_MAP[int(zid)] for zid in agg["zone_id"]]

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(MERGED_CSV, index=False)
    print(f"✓ Aggregated per-zone copy -> {MERGED_CSV}")
    print(f"  Zone risk distribution:")
    dist = agg["risk_label"].value_counts().sort_index()
    names = {0: "Safe", 1: "Drought", 2: "Heat Stress", 3: "Flood", 4: "Soil Risk"}
    for label, count in dist.items():
        print(f"    {names.get(int(label), label)}: {count} zones")
    print("\nDone! Next: python train_model.py")


if __name__ == "__main__":
    main()
