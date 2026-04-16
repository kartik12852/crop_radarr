"""
src/graph_builder.py
Builds the feature matrix and the spatial graph used across the app.
No torch dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from src.paths import MERGED_CSV, SYNTHETIC_CSV, SCALER_PATH
from src.gnn_model import build_norm_adjacency

FEATURE_COLS = [
    "temp_avg", "temp_max_mean", "temp_min_mean",
    "precip_total_mm", "precip_avg_mm",
    "windspeed_mean", "et0_mean",
    "soil_moisture_mean",
    "soil_phh2o", "soil_clay", "soil_soc",
    "soil_bdod", "soil_sand", "soil_silt",
    "heat_index", "temp_rain_ratio",
    "drought_index", "soil_stress",
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    f1, f2 = np.radians(lat1), np.radians(lat2)
    df, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(df / 2) ** 2 + np.cos(f1) * np.cos(f2) * np.sin(dl / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def build_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    lats = df["lat"].to_numpy(dtype=float)
    lons = df["lon"].to_numpy(dtype=float)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            dist[i, j] = dist[j, i] = d
    return dist


def load_data(fallback_to_synthetic: bool = True) -> pd.DataFrame:
    if MERGED_CSV.exists():
        return pd.read_csv(MERGED_CSV)
    if fallback_to_synthetic and SYNTHETIC_CSV.exists():
        df = pd.read_csv(SYNTHETIC_CSV)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_no_id = [c for c in num_cols if c != "zone_id"]
        agg = df.groupby("zone_id")[num_no_id].mean().reset_index()
        for col in ["zone_name", "crop", "lat", "lon"]:
            agg[col] = agg["zone_id"].map(df.groupby("zone_id")[col].first())
        agg["risk_label"] = agg["risk_label"].round().astype(int)
        return agg
    raise FileNotFoundError(
        "No data found. Run python synthetic/synthetic_data_generator.py or python data/fetch_data.py first."
    )


def build_graph(threshold_km: float = 800.0, fit_scaler: bool = True, fallback: bool = True):
    df = load_data(fallback_to_synthetic=fallback).copy()
    avail_feats = [c for c in FEATURE_COLS if c in df.columns]
    if len(avail_feats) < 4:
        raise ValueError(f"Not enough feature columns found: {avail_feats}")

    X_raw = df[avail_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df["risk_label"].fillna(0).round().astype(int).to_numpy(dtype=int)

    scaler = StandardScaler()
    if fit_scaler or not SCALER_PATH.exists():
        X = scaler.fit_transform(X_raw).astype(np.float32)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X = scaler.transform(X_raw).astype(np.float32)

    dist = build_distance_matrix(df)
    adj_bin = (dist <= threshold_km).astype(np.float32)
    np.fill_diagonal(adj_bin, 0.0)
    adj_norm = build_norm_adjacency(adj_bin)
    return X, adj_norm.astype(np.float32), adj_bin.astype(np.float32), y, df, avail_feats
