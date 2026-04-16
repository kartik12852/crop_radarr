"""
src/recommendation.py
Risk-specific mitigation recommendations + crop-specific agronomic tips.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.model_utils import LABEL_NAMES

RISK_MITIGATIONS = {
    0: {
        "title": "✅ Optimal Conditions — Maintenance Mode",
        "immediate":  ["Continue standard field monitoring schedule",
                       "Maintain irrigation scheduling as planned"],
        "short_term": ["Optimise fertiliser application timing",
                       "Scout for early pest/disease indicators",
                       "Update crop calendar records"],
        "long_term":  ["Plan for next season's sowing date",
                       "Consider precision agriculture sensors for continued optimisation"],
    },
    1: {
        "title": "🌵 Drought Mitigation Plan",
        "immediate":  ["🚰 Activate drip/sprinkler irrigation immediately",
                       "🌿 Apply 5–7 cm organic mulch to reduce evaporation",
                       "⏰ Irrigate only during early morning (5–7 AM)",
                       "📊 Measure soil moisture at 10 cm and 30 cm daily"],
        "short_term": ["Harvest mature crops to reduce water demand",
                       "Apply anti-transpirant sprays (kaolin clay) if available",
                       "Consider drought-tolerant cover crops (cowpea, sorghum)"],
        "long_term":  ["Install rainwater harvesting and farm ponds",
                       "Shift to SRI or precision irrigation techniques",
                       "Select drought-tolerant crop varieties",
                       "Develop groundwater recharge structures (bunds, check dams)"],
    },
    2: {
        "title": "🔥 Heat Stress Response Plan",
        "immediate":  ["🌡 Provide shade nets (30–50% shade) for sensitive stages",
                       "💧 Increase irrigation frequency — short, frequent applications",
                       "🕔 Schedule field operations before 10 AM and after 4 PM",
                       "Apply potassium-rich foliar spray to improve heat tolerance"],
        "short_term": ["Apply kaolin clay spray to reduce leaf temperature",
                       "Delay pollination-sensitive stages if possible",
                       "Ensure adequate boron & zinc micronutrient supply"],
        "long_term":  ["Introduce heat-tolerant varieties",
                       "Establish windbreaks and agroforestry systems",
                       "Consider shifting sowing windows to cooler months",
                       "Install automated temperature monitoring sensors"],
    },
    3: {
        "title": "🌊 Flood & Waterlogging Response Plan",
        "immediate":  ["🔓 Open all drainage channels and field bunds urgently",
                       "Harvest any mature or near-mature crops immediately",
                       "Avoid entering waterlogged fields with machinery",
                       "📷 Document crop damage for insurance claims"],
        "short_term": ["Apply fungicide (copper oxychloride) after water recedes",
                       "Replant with short-duration varieties if season allows",
                       "Apply nitrogen topdressing once soil drains",
                       "Check for nutrient leaching — apply potassium and phosphorus"],
        "long_term":  ["Construct raised bed systems for future flood years",
                       "Introduce submergence-tolerant varieties (e.g., Swarna Sub1)",
                       "Map field drainage patterns using GIS tools",
                       "Invest in weather insurance and flood early-warning subscriptions"],
    },
    4: {
        "title": "🪨 Soil Health Restoration Plan",
        "immediate":  ["🧪 Send soil sample for comprehensive nutrient testing",
                       "Stop deep tillage — switch to zero-till or minimum tillage",
                       "Apply compost / vermicompost top-dressing (2–3 T/ha)"],
        "short_term": ["Introduce green manure legume crop (dhaincha, sesbania)",
                       "Apply biofertilisers (Rhizobium, Azospirillum, PSB)",
                       "Correct pH with lime (acidic) or gypsum (saline/sodic soil)"],
        "long_term":  ["Rotate with nitrogen-fixing legumes every 2–3 years",
                       "Establish a soil health card monitoring programme",
                       "Adopt conservation agriculture principles",
                       "Enrich organic matter with long-term compost programme"],
    },
}

CROP_TIPS = {
    "Wheat":   {"drought": "Use SRI-style wide spacing; apply 10 cm mulch; use HD-2967 variety.",
                "heat":    "Sow 15 days early to avoid terminal heat. Irrigate at boot stage.",
                "flood":   "Ensure field slope > 1%. Sow on ridges in flood-prone areas.",
                "soil":    "Wheat responds well to balanced NPK + zinc. Maintain pH 6.5–7.5."},
    "Rice":    {"drought": "Adopt Alternate Wetting and Drying (AWD) irrigation — saves 30% water.",
                "heat":    "Use heat-tolerant IR varieties. Apply silica spray at panicle initiation.",
                "flood":   "Plant Swarna Sub1 or FL478 (tolerates 2 weeks submergence).",
                "soil":    "Incorporate green manure (sesbania) before transplanting."},
    "Maize":   {"drought": "Plant drought-tolerant hybrids (DMAIZE series). Irrigate at silking.",
                "heat":    "Schedule planting to avoid pollination during peak summer.",
                "flood":   "Maize is sensitive to waterlogging. Install sub-surface drainage.",
                "soil":    "Maize is a heavy feeder — ensure adequate N, P, K and S."},
    "Cotton":  {"drought": "Use drip irrigation; avoid stress at flowering and boll formation.",
                "heat":    "Spray 2% potassium nitrate solution during heat waves.",
                "flood":   "Plant on raised beds (15–20 cm); improve field drainage.",
                "soil":    "Sensitive to soil compaction. Deep ploughing once every 3 years."},
    "Soybean": {"drought": "Irrigate at flowering (R1–R2) and pod fill (R5–R6) stages.",
                "heat":    "Shade nets at R1–R5 stages; use heat-tolerant varieties.",
                "flood":   "Tolerates < 2 days flooding. Install tile drainage.",
                "soil":    "Inoculate with Bradyrhizobium; maintain pH 6.0–6.8."},
    "Millet":  {"drought": "Naturally drought-tolerant. Maintain 200 mm rainfall for good yields.",
                "heat":    "Heat-tolerant up to 42°C. No major concern.",
                "flood":   "Avoid waterlogged soils. Roots rot quickly when submerged.",
                "soil":    "Grows well in poor sandy soils. Minimal fertiliser input needed."},
    "default": {"drought": "Implement water conservation. Use mulching and deficit irrigation.",
                "heat":    "Shade, increase irrigation frequency, work in cooler hours.",
                "flood":   "Improve drainage. Harvest mature crops before flooding.",
                "soil":    "Conduct soil testing. Add organic matter. Rotate crops."},
}

RISK_TO_CROP_KEY = {1: "drought", 2: "heat", 3: "flood", 4: "soil"}


def get_recommendations(zone_name: str, crop: str, risk_id: int) -> dict:
    mitigation = RISK_MITIGATIONS.get(risk_id, RISK_MITIGATIONS[0])
    risk_key   = RISK_TO_CROP_KEY.get(risk_id)
    crop_data  = CROP_TIPS.get(crop, CROP_TIPS["default"])
    crop_tip   = crop_data.get(risk_key, "") if risk_key else ""
    return {
        "zone_name": zone_name, "crop": crop,
        "risk_id": risk_id, "risk_name": LABEL_NAMES.get(risk_id, "Unknown"),
        "title": mitigation["title"],
        "immediate": mitigation["immediate"],
        "short_term": mitigation["short_term"],
        "long_term":  mitigation["long_term"],
        "crop_tip":   crop_tip,
    }


def get_all_recommendations(df: pd.DataFrame, preds) -> list:
    return [
        get_recommendations(
            zone_name=row.get("zone_name", f"Zone {i}"),
            crop=row.get("crop", "default"),
            risk_id=int(preds[i]),
        )
        for i, (_, row) in enumerate(df.iterrows())
    ]
