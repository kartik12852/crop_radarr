"""
src/alert_engine.py
Generates structured, farmer-friendly alerts from model predictions.
Each alert contains a headline, severity level, recommended actions, and metadata.
"""

from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd

from src.model_utils import LABEL_COLORS, LABEL_EMOJIS, LABEL_NAMES

# ------------------------------------------------------------------
# Alert templates per risk class
# ------------------------------------------------------------------
ALERT_TEMPLATES = {
    0: {
        "headline":      "Conditions are favourable for {crop} cultivation.",
        "severity":      "Low",
        "severity_icon": "🟢",
        "actions": [
            "Continue regular monitoring schedule",
            "Maintain scheduled irrigation",
            "Apply balanced fertilisation",
        ],
    },
    1: {
        "headline":      "DROUGHT WARNING: Insufficient rainfall detected for {crop}.",
        "severity":      "High",
        "severity_icon": "🟠",
        "actions": [
            "Activate supplemental drip/sprinkler irrigation immediately",
            "Apply mulch (5–7 cm) to reduce soil moisture evaporation",
            "Consider drought-resistant cover crops as temporary measure",
            "Monitor soil moisture at 10 cm and 30 cm depth daily",
        ],
    },
    2: {
        "headline":      "HEAT STRESS ALERT: Temperature exceeds crop tolerance for {crop}.",
        "severity":      "High",
        "severity_icon": "🔴",
        "actions": [
            "Irrigate during cooler morning or evening hours only",
            "Install reflective mulch to reduce soil surface temperature",
            "Delay transplanting until temperatures drop below tolerance",
            "Shade netting recommended for sensitive crop stages",
        ],
    },
    3: {
        "headline":      "FLOOD RISK: Excess rainfall and waterlogging detected.",
        "severity":      "Critical",
        "severity_icon": "🚨",
        "actions": [
            "Open and clear all drainage channels immediately",
            "Harvest any mature crops before flooding worsens",
            "Avoid field operations until standing water recedes",
            "Apply fungicide post-flood to prevent root rot and disease",
        ],
    },
    4: {
        "headline":      "SOIL DEGRADATION: Poor soil health conditions detected for {crop}.",
        "severity":      "Medium",
        "severity_icon": "🟡",
        "actions": [
            "Send soil sample for comprehensive nutrient and pH test",
            "Add organic compost (2–3 T/ha) to improve soil structure",
            "Reduce tillage depth to preserve beneficial soil biology",
            "Rotate with nitrogen-fixing legumes next season",
        ],
    },
}

SEVERITY_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


# ------------------------------------------------------------------
# Single alert generation
# ------------------------------------------------------------------

def generate_alert(
    zone_row: pd.Series,
    pred_label: int,
    confidence: float,
    timestamp: datetime | None = None,
) -> dict:
    """Build one alert dict for a single zone."""
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
        "color":         LABEL_COLORS.get(pred_label, "#94A3B8"),
        "severity":      tmpl["severity"],
        "severity_icon": tmpl["severity_icon"],
        "headline":      tmpl["headline"].format(crop=crop),
        "actions":       tmpl["actions"],
        "confidence":    round(confidence * 100, 1),
        "timestamp":     timestamp.strftime("%Y-%m-%d %H:%M"),
    }


# ------------------------------------------------------------------
# Batch alert generation
# ------------------------------------------------------------------

def generate_all_alerts(
    df: pd.DataFrame,
    preds: np.ndarray,
    probs: np.ndarray,
) -> list[dict]:
    """Generate alerts for all zones at once."""
    ts = datetime.now()
    return [
        generate_alert(row, int(preds[i]), float(probs[i].max()), ts)
        for i, (_, row) in enumerate(df.iterrows())
    ]


def sort_alerts_by_severity(alerts: list[dict]) -> list[dict]:
    return sorted(
        alerts,
        key=lambda a: SEVERITY_ORDER.get(a["severity"], 0),
        reverse=True,
    )


def filter_alerts(
    alerts: list[dict],
    min_severity: str = "Low",
    risk_types: list[str] | None = None,
) -> list[dict]:
    min_order = SEVERITY_ORDER.get(min_severity, 0)
    result = [a for a in alerts if SEVERITY_ORDER.get(a["severity"], 0) >= min_order]
    if risk_types:
        result = [a for a in result if a["risk_name"] in risk_types]
    return result


def alerts_to_dataframe(alerts: list[dict]) -> pd.DataFrame:
    cols = ["zone_name", "crop", "emoji", "risk_name", "severity", "confidence", "headline", "timestamp"]
    return pd.DataFrame([{c: a.get(c, "") for c in cols} for a in alerts])