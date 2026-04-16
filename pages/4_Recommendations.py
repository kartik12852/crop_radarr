"""pages/4_Recommendations.py — recommendations"""

from __future__ import annotations

import os
import sys
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.theme import hero, inject_theme

st.set_page_config(page_title="Recommendations", page_icon="💡", layout="wide")
inject_theme()
hero("💡 Agricultural Recommendations", "Actionable mitigation steps tailored to the predicted risk and crop type.", "Decision support")


@st.cache_resource
def _load():
    from src.model_utils import LABEL_NAMES, load_inference_bundle, predict_all
    from src.recommendation import get_all_recommendations
    model, X, adj_norm, adj_mask, y, df, fc = load_inference_bundle()
    preds, probs = predict_all(model, X, adj_norm)
    recs = get_all_recommendations(df, preds)
    return df, preds, probs, recs, LABEL_NAMES


df, preds, probs, recs, LABEL_NAMES = _load()
with st.sidebar:
    risk_filter = st.multiselect("Risk filter", list(LABEL_NAMES.values()), default=[v for k, v in LABEL_NAMES.items() if k != 0])
    crop_filter = st.multiselect("Crop filter", sorted(df["crop"].unique().tolist()), default=sorted(df["crop"].unique().tolist()))

filtered = [r for r in recs if r["risk_name"] in risk_filter and r["crop"] in crop_filter]
st.caption(f"Showing {len(filtered)} recommendation cards")

t1, t2 = st.tabs(["Recommendation Cards", "Export"])
with t1:
    for rec in filtered:
        zone_idx = df.index[df["zone_name"] == rec["zone_name"]][0]
        conf = probs[zone_idx].max() * 100
        st.markdown(
            f"<div class='glass-card' style='margin-bottom:0.85rem;'>"
            f"<h4 style='margin-top:0;'>{rec['title']}</h4>"
            f"<p><b>{rec['zone_name']}</b> · {rec['crop']} · Confidence {conf:.1f}%</p>"
            f"<p><b>Immediate:</b> {' • '.join(rec['immediate'])}</p>"
            f"<p><b>Short-term:</b> {' • '.join(rec['short_term'])}</p>"
            f"<p><b>Long-term:</b> {' • '.join(rec['long_term'])}</p>"
            f"<p><b>Crop tip:</b> {rec.get('crop_tip', '')}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
with t2:
    exp_df = pd.DataFrame([{
        "Zone": r["zone_name"],
        "Crop": r["crop"],
        "Risk": r["risk_name"],
        "Immediate": " | ".join(r["immediate"]),
        "Short-term": " | ".join(r["short_term"]),
        "Long-term": " | ".join(r["long_term"]),
        "Crop Tip": r.get("crop_tip", ""),
    } for r in filtered])
    st.download_button("Download CSV", exp_df.to_csv(index=False).encode(), "climate_radar_recommendations.csv", "text/csv")
    st.dataframe(exp_df, use_container_width=True, hide_index=True)
