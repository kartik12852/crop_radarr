"""
src/recommendation.py
Risk-specific mitigation recommendations with crop-specific agronomic tips.
Each recommendation has three temporal horizons:
  - Immediate (within 24-48 hours)
  - Short-term (within 2 weeks)
  - Long-term (seasonal / annual planning)
"""

from __future__ import annotations

import pandas as pd
from src.model_utils import LABEL_NAMES

# ------------------------------------------------------------------
# Risk mitigation playbooks
# ------------------------------------------------------------------
RISK_MITIGATIONS = {
    0: {
        "title":      "✅ Optimal Conditions — Maintenance Mode",
        "immediate":  [
            "Continue standard field monitoring schedule",
            "Maintain irrigation scheduling as planned",
        ],
        "short_term": [
            "Optimise fertiliser application timing based on soil test",
            "Scout for early pest or disease indicators",
            "Update crop calendar records",
        ],
        "long_term":  [
            "Plan next season sowing dates for optimal climate windows",
            "Consider precision agriculture sensors for continued optimisation",
        ],
    },
    1: {
        "title":      "🌵 Drought Mitigation Plan",
        "immediate":  [
            "🚰 Activate drip or sprinkler irrigation immediately",
            "🌿 Apply 5–7 cm organic mulch to reduce evaporation",
            "⏰ Irrigate only during early morning hours (5–7 AM)",
            "📊 Measure soil moisture at 10 cm and 30 cm depth daily",
        ],
        "short_term": [
            "Harvest mature crops early to reduce water demand",
            "Apply anti-transpirant sprays (kaolin clay) if available",
            "Consider drought-tolerant cover crops such as cowpea or sorghum",
        ],
        "long_term":  [
            "Install rainwater harvesting ponds or farm tanks",
            "Shift to SRI (System of Rice Intensification) or precision drip",
            "Select drought-tolerant crop varieties for next season",
            "Build groundwater recharge structures (bunds, check dams)",
        ],
    },
    2: {
        "title":      "🔥 Heat Stress Response Plan",
        "immediate":  [
            "🌡 Provide 30–50% shade nets for sensitive crop stages",
            "💧 Increase irrigation frequency using short, frequent applications",
            "🕔 Schedule field operations before 10 AM and after 4 PM only",
            "Apply potassium-rich foliar spray to improve heat tolerance",
        ],
        "short_term": [
            "Apply kaolin clay spray to reduce leaf surface temperature",
            "Delay pollination-sensitive stages if agronomically feasible",
            "Ensure adequate boron and zinc micronutrient supply",
        ],
        "long_term":  [
            "Introduce heat-tolerant crop varieties for future seasons",
            "Establish windbreaks and agroforestry shelter belts",
            "Shift sowing windows earlier to avoid peak summer stress",
            "Install automated temperature and humidity monitoring sensors",
        ],
    },
    3: {
        "title":      "🌊 Flood and Waterlogging Response Plan",
        "immediate":  [
            "🔓 Open all drainage channels and field bunds immediately",
            "Harvest any mature or near-mature crops before flooding worsens",
            "Avoid entering waterlogged fields with heavy machinery",
            "📷 Document crop damage for insurance claims",
        ],
        "short_term": [
            "Apply copper oxychloride fungicide once water recedes",
            "Replant with short-duration varieties if the season permits",
            "Apply nitrogen top-dressing once soil drains adequately",
            "Check for nutrient leaching — apply potassium and phosphorus",
        ],
        "long_term":  [
            "Construct raised bed systems for future flood-prone seasons",
            "Introduce submergence-tolerant varieties (e.g., Swarna Sub1)",
            "Map field drainage patterns and install tile drainage",
            "Subscribe to weather insurance and flood early-warning services",
        ],
    },
    4: {
        "title":      "🪨 Soil Health Restoration Plan",
        "immediate":  [
            "🧪 Send soil sample for comprehensive nutrient and pH testing",
            "Stop deep tillage — switch to zero-till or minimum tillage immediately",
            "Apply compost or vermicompost top-dressing at 2–3 T/ha",
        ],
        "short_term": [
            "Introduce green manure legume crop (dhaincha or sesbania)",
            "Apply biofertilisers: Rhizobium, Azospirillum, PSB",
            "Correct pH: use lime for acidic soils, gypsum for saline soils",
        ],
        "long_term":  [
            "Rotate with nitrogen-fixing legumes every 2–3 years",
            "Establish a soil health card monitoring programme",
            "Adopt conservation agriculture with minimum tillage",
            "Build long-term organic matter with annual compost programme",
        ],
    },
}

# ------------------------------------------------------------------
# Crop-specific agronomic tips
# ------------------------------------------------------------------
CROP_TIPS = {
    "Wheat": {
        "drought": "Use SRI-style wide spacing; apply 10 cm mulch; prefer HD-2967 variety.",
        "heat":    "Sow 15 days early to avoid terminal heat. Irrigate at boot stage.",
        "flood":   "Ensure field slope > 1%. Sow on ridges in flood-prone areas.",
        "soil":    "Maintain pH 6.5–7.5. Balanced NPK + zinc application recommended.",
    },
    "Rice": {
        "drought": "Adopt Alternate Wetting and Drying (AWD) — saves 30% water.",
        "heat":    "Use heat-tolerant IR varieties. Apply silica spray at panicle initiation.",
        "flood":   "Plant Swarna Sub1 or FL478 (tolerates 2 weeks of submergence).",
        "soil":    "Incorporate green manure (sesbania) before transplanting for N-fix.",
    },
    "Maize": {
        "drought": "Plant drought-tolerant DMAIZE series hybrids. Irrigate at silking stage.",
        "heat":    "Schedule planting to avoid pollination during peak summer heat.",
        "flood":   "Maize is highly sensitive to waterlogging — install sub-surface drainage.",
        "soil":    "Heavy feeder — ensure adequate N, P, K and sulphur supply.",
    },
    "Cotton": {
        "drought": "Use drip irrigation. Avoid water stress at flowering and boll formation.",
        "heat":    "Spray 2% potassium nitrate solution during heat waves.",
        "flood":   "Plant on raised beds (15–20 cm). Improve field drainage capacity.",
        "soil":    "Sensitive to soil compaction — deep plough once every 3 years.",
    },
    "Soybean": {
        "drought": "Irrigate at flowering (R1–R2) and pod fill (R5–R6) growth stages.",
        "heat":    "Use shade nets at R1–R5 stages. Select heat-tolerant varieties.",
        "flood":   "Tolerates less than 2 days of flooding. Install tile drainage.",
        "soil":    "Inoculate seed with Bradyrhizobium. Maintain soil pH 6.0–6.8.",
    },
    "Millet": {
        "drought": "Naturally drought-tolerant up to 200 mm rainfall zone.",
        "heat":    "Heat-tolerant up to 42°C — minimal intervention needed.",
        "flood":   "Avoid waterlogged soils — roots rot quickly when submerged.",
        "soil":    "Grows well in poor sandy soils. Minimal fertiliser input needed.",
    },
    "default": {
        "drought": "Implement water conservation. Use mulching and deficit irrigation protocols.",
        "heat":    "Shade netting, increased irrigation frequency, and cooler working hours.",
        "flood":   "Improve field drainage. Harvest mature crops before potential flooding.",
        "soil":    "Conduct soil testing. Add organic matter. Plan crop rotation.",
    },
}

RISK_TO_CROP_KEY = {1: "drought", 2: "heat", 3: "flood", 4: "soil"}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_recommendations(zone_name: str, crop: str, risk_id: int) -> dict:
    """Get the full mitigation package for one zone."""
    mitigation = RISK_MITIGATIONS.get(risk_id, RISK_MITIGATIONS[0])
    risk_key   = RISK_TO_CROP_KEY.get(risk_id)
    crop_data  = CROP_TIPS.get(crop, CROP_TIPS["default"])
    crop_tip   = crop_data.get(risk_key, "") if risk_key else ""

    return {
        "zone_name":  zone_name,
        "crop":       crop,
        "risk_id":    risk_id,
        "risk_name":  LABEL_NAMES.get(risk_id, "Unknown"),
        "title":      mitigation["title"],
        "immediate":  mitigation["immediate"],
        "short_term": mitigation["short_term"],
        "long_term":  mitigation["long_term"],
        "crop_tip":   crop_tip,
    }


def get_all_recommendations(df: pd.DataFrame, preds) -> list[dict]:
    """Generate recommendations for every zone in the DataFrame."""
    return [
        get_recommendations(
            zone_name = row.get("zone_name", f"Zone {i}"),
            crop      = row.get("crop", "default"),
            risk_id   = int(preds[i]),
        )
        for i, (_, row) in enumerate(df.iterrows())
    ]