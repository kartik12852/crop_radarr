"""
src/graph_builder.py
Builds the feature matrix (X), normalised adjacency (A_norm), and label vector (y)
from raw zone data.  Falls back to synthetic data automatically if the real
merged CSV is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from src.paths import MERGED_CSV, SYNTHETIC_CSV, SCALER_PATH
from src.gnn_model import build_norm_adjacency

# ------------------------------------------------------------------
# Feature columns used by the model
# Subset actually present in the DataFrame is selected at runtime.
# ------------------------------------------------------------------
FEATURE_COLS = [
    "temp_avg",
    "temp_max_mean",
    "temp_min_mean",
    "precip_total_mm",
    "precip_avg_mm",
    "windspeed_mean",
    "et0_mean",
    "soil_moisture_mean",
    "soil_phh2o",
    "soil_clay",
    "soil_soc",
    "soil_bdod",
    "soil_sand",
    "soil_silt",
    "heat_index",
    "temp_rain_ratio",
    "drought_index",
    "soil_stress",
]


# ------------------------------------------------------------------
# Distance helpers
# ------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate great-circle distance in kilometres."""
    r   = 6371.0
    f1, f2 = np.radians(lat1), np.radians(lat2)
    df, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(df / 2) ** 2 + np.cos(f1) * np.cos(f2) * np.sin(dl / 2) ** 2
    return r * 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))


def build_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return pairwise Haversine distance matrix (km) for all zones."""
    n    = len(df)
    lats = df["lat"].to_numpy(dtype=float)
    lons = df["lon"].to_numpy(dtype=float)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d               = haversine_km(lats[i], lons[i], lats[j], lons[j])
            dist[i, j]      = d
            dist[j, i]      = d
    return dist


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_data(fallback_to_synthetic: bool = True) -> pd.DataFrame:
    """
    Load zone data from merged CSV (real API data) or synthetic CSV.
    If synthetic data has multiple rows per zone, aggregate to one row per zone.
    """
    if MERGED_CSV.exists():
        return pd.read_csv(MERGED_CSV)

    if fallback_to_synthetic and SYNTHETIC_CSV.exists():
        df      = pd.read_csv(SYNTHETIC_CSV)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_no_id = [c for c in num_cols if c != "zone_id"]

        def _mode(s: pd.Series) -> int:
            m = s.mode(dropna=True)
            return int(m.iloc[0]) if not m.empty else int(s.iloc[0])

        agg = df.groupby("zone_id")[num_no_id].mean().reset_index()
        for col in ["zone_name", "crop", "lat", "lon"]:
            if col in df.columns:
                agg[col] = agg["zone_id"].map(df.groupby("zone_id")[col].first())
        agg["risk_label"] = df.groupby("zone_id")["risk_label"].agg(_mode).values
        return agg

    raise FileNotFoundError(
        "No data found. Run:\n"
        "  python synthetic/synthetic_data_generator.py\n"
        "or\n"
        "  python data/fetch_data.py"
    )


# ------------------------------------------------------------------
# Main graph construction
# ------------------------------------------------------------------

def build_graph(
    threshold_km: float = 800.0,
    fit_scaler: bool    = True,
    fallback: bool      = True,
):
    """
    Build the full graph for the model.

    Args:
        threshold_km: Zones within this distance share an edge.
        fit_scaler:   If True, fit a new StandardScaler and save it.
                      If False, load existing scaler (inference mode).
        fallback:     Allow synthetic data fallback.

    Returns:
        X          – scaled feature matrix (n_zones, n_features)
        adj_norm   – normalised adjacency matrix (n_zones, n_zones)
        adj_bin    – binary adjacency matrix (n_zones, n_zones)
        y          – integer label array (n_zones,)
        df         – raw DataFrame (all columns, one row per zone)
        avail_feats– list of feature column names actually used
    """
    df          = load_data(fallback_to_synthetic=fallback).copy()
    avail_feats = [c for c in FEATURE_COLS if c in df.columns]

    if len(avail_feats) < 4:
        raise ValueError(
            f"Too few feature columns found ({avail_feats}). "
            "Check that your data has climate and soil fields."
        )

    X_raw = (
        df[avail_feats]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    y = df["risk_label"].fillna(0).round().astype(int).to_numpy(dtype=int)

    # ── Normalise features ──────────────────────────────────────────
    if fit_scaler or not SCALER_PATH.exists():
        scaler = StandardScaler()
        X      = scaler.fit_transform(X_raw).astype(np.float32)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X      = scaler.transform(X_raw).astype(np.float32)

    # ── Build adjacency ─────────────────────────────────────────────
    dist    = build_distance_matrix(df)
    adj_bin = (dist <= threshold_km).astype(np.float32)
    np.fill_diagonal(adj_bin, 0.0)
    adj_norm = build_norm_adjacency(adj_bin)

    return (
        X,
        adj_norm.astype(np.float32),
        adj_bin.astype(np.float32),
        y,
        df,
        avail_feats,
    )