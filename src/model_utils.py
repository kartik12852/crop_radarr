"""
src/model_utils.py
Inference helpers shared by the Streamlit pages.
"""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from src.gnn_model import GraphRiskModel
from src.graph_builder import build_graph
from src.paths import METRICS_PATH, MODEL_PATH

LABEL_NAMES = {0: "Safe", 1: "Drought", 2: "Heat Stress", 3: "Flood", 4: "Soil Risk"}
LABEL_COLORS = {0: "#34D399", 1: "#F59E0B", 2: "#F97316", 3: "#38BDF8", 4: "#A78BFA"}
LABEL_EMOJIS = {0: "✅", 1: "🌵", 2: "🔥", 3: "🌊", 4: "🪨"}
NUM_CLASSES = 5


def _fit_temporary_model(X: np.ndarray, y: np.ndarray, adj_norm: np.ndarray, feature_names: list[str]) -> GraphRiskModel:
    model = GraphRiskModel(random_state=42)
    model.fit(X, y, adj_norm=adj_norm, feature_names=feature_names)
    return model


def load_inference_bundle():
    X, adj_norm, adj_mask, y, df, feat_cols = build_graph(fit_scaler=False)
    model = None
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None
    if model is None:
        model = _fit_temporary_model(X, y, adj_norm, feat_cols)
    return model, X, adj_norm, adj_mask, y, df, feat_cols


def predict_all(model: GraphRiskModel, X: np.ndarray, adj_norm: np.ndarray):
    probs = np.asarray(model.predict_proba(X, adj_norm), dtype=np.float32)
    preds = probs.argmax(axis=1).astype(int)
    return preds, probs


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def zone_summary_df(df: pd.DataFrame, preds: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    out = df[["zone_id", "zone_name", "lat", "lon", "crop"]].copy()
    out["pred_label"] = preds.astype(int)
    out["risk_name"] = [LABEL_NAMES.get(int(p), "Unknown") for p in preds]
    out["confidence_pct"] = np.round(probs.max(axis=1) * 100, 1)
    out["emoji"] = [LABEL_EMOJIS.get(int(p), "❓") for p in preds]
    out["color"] = [LABEL_COLORS.get(int(p), "#94A3B8") for p in preds]
    return out.sort_values(["pred_label", "confidence_pct"], ascending=[False, False]).reset_index(drop=True)
