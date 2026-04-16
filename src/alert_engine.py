"""
src/alert_engine.py
Generates structured farmer-friendly alerts from GNN predictions.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import numpy as np
import pandas as pd
from src.model_utils import LABEL_NAMES, LABEL_COLORS, LABEL_EMOJIS

ALERT_TEMPLATES = {
    0: {
        "headline": "Conditions are favourable for {crop} cultivation.",
        "severity": "Low", "severity_icon": "🟢",
        "actions": [
            "Continue regular monitoring",
            "Maintain scheduled irrigation",
            "Apply balanced fertilisation",
        ],
    },
    1: {
        "headline": "DROUGHT WARNING: Insufficient rainfall for {crop}.",
        "severity": "High", "severity_icon": "🟠",
        "actions": [
            "Activate supplemental irrigation immediately",
            "Apply mulch to reduce soil moisture loss",
            "Consider drought-resistant cover crops",
            "Monitor soil moisture daily",
        ],
    },
    2: {
        "headline": "HEAT STRESS ALERT: Temperature exceeds tolerance for {crop}.",
        "severity": "High", "severity_icon": "🔴",
        "actions": [
            "Irrigate during cooler morning/evening hours",
            "Apply reflective mulch to reduce soil temperature",
            "Delay transplanting until cooler conditions",
            "Shade netting recommended for sensitive crops",
        ],
    },
    3: {
        "headline": "FLOOD RISK: Excess rainfall / waterlogging detected.",
        "severity": "Critical", "severity_icon": "🚨",
        "actions": [
            "Open drainage channels and check bunds",
            "Harvest mature crops immediately if possible",
            "Avoid field operations until water recedes",
            "Apply fungicide post-flood to prevent root rot",
        ],
    },
    4: {
        "headline": "SOIL DEGRADATION: Poor soil health detected for {crop}.",
        "severity": "Medium", "severity_icon": "🟡",
        "actions": [
            "Conduct a detailed soil test",
            "Add organic compost to improve structure",
            "Reduce tillage to preserve soil biology",
            "Rotate with legumes to restore nitrogen",
        ],
    },
}

SEVERITY_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def generate_alert(zone_row: pd.Series, pred_label: int, confidence: float,
                   timestamp: datetime = None) -> dict:
    if timestamp is None:
        timestamp = datetime.now()
    tmpl = ALERT_TEMPLATES.get(pred_label, ALERT_TEMPLATES[0])
    crop = zone_row.get("crop", "the crop")
    return {
        "zone_id":       int(zone_row.get("zone_id", -1)),
        "zone_name":     str(zone_row.get("zone_name", "Unknown")),
        "lat":           float(zone_row.get("lat", 0.0)),
        "lon":           float(zone_row.get("lon", 0.0)),
        "crop":          crop,
        "risk_id":       pred_label,
        "risk_name":     LABEL_NAMES.get(pred_label, "Unknown"),
        "emoji":         LABEL_EMOJIS.get(pred_label, "❓"),
        "color":         LABEL_COLORS.get(pred_label, "#ccc"),
        "severity":      tmpl["severity"],
        "severity_icon": tmpl["severity_icon"],
        "headline":      tmpl["headline"].format(crop=crop),
        "actions":       tmpl["actions"],
        "confidence":    round(confidence * 100, 1),
        "timestamp":     timestamp.strftime("%Y-%m-%d %H:%M"),
    }


def generate_all_alerts(df: pd.DataFrame, preds: np.ndarray,
                         probs: np.ndarray) -> list:
    ts = datetime.now()
    return [
        generate_alert(row, int(preds[i]), float(probs[i].max()), ts)
        for i, (_, row) in enumerate(df.iterrows())
    ]


def sort_alerts_by_severity(alerts: list) -> list:
    return sorted(alerts, key=lambda a: SEVERITY_ORDER.get(a["severity"], 0), reverse=True)


def filter_alerts(alerts: list, min_severity: str = "Low",
                  risk_types: list = None) -> list:
    min_order = SEVERITY_ORDER.get(min_severity, 0)
    result = [a for a in alerts if SEVERITY_ORDER.get(a["severity"], 0) >= min_order]
    if risk_types:
        result = [a for a in result if a["risk_name"] in risk_types]
    return result


def alerts_to_dataframe(alerts: list) -> pd.DataFrame:
    cols = ["zone_name", "crop", "emoji", "risk_name",
            "severity", "confidence", "headline", "timestamp"]
    return pd.DataFrame([{c: a.get(c, "") for c in cols} for a in alerts])
