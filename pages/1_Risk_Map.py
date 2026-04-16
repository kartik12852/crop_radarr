"""pages/1_Risk_Map.py — Interactive spatial risk map"""

from __future__ import annotations

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import plotly.express as px
import streamlit as st

from utils.theme import hero, inject_theme, pills

st.set_page_config(page_title="Risk Map", page_icon="🗺️", layout="wide")
inject_theme()
hero("🗺️ Crop Risk Map", "Spatial view of predicted climate and soil risk across 30 Indian agricultural zones.", "Interactive Map")


@st.cache_resource
def _load():
    from src.model_utils import LABEL_NAMES, load_inference_bundle, predict_all, zone_summary_df
    model, X, adj_norm, adj_mask, y, df, fc = load_inference_bundle()
    preds, probs = predict_all(model, X, adj_norm)
    return zone_summary_df(df, preds, probs), LABEL_NAMES


summary, LABEL_NAMES = _load()

# ── Sidebar filters ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")
    risk_filter = st.multiselect(
        "Risk Types", list(LABEL_NAMES.values()),
        default=list(LABEL_NAMES.values())
    )
    min_conf = st.slider("Minimum Confidence %", 0, 100, 0, 5)
    marker_sz = st.slider("Marker Size", 8, 36, 18, 2)

pills([f"Showing: {len(summary[summary['risk_name'].isin(risk_filter)])}/{len(summary)} zones", f"Confidence floor: {min_conf}%"])

filtered = summary[
    (summary["risk_name"].isin(risk_filter)) &
    (summary["confidence_pct"] >= min_conf)
].copy()

# ── Map ────────────────────────────────────────────────────────────
COLOR_MAP = {
    "Safe":       "#34D399",
    "Drought":    "#F59E0B",
    "Heat Stress":"#F97316",
    "Flood":      "#38BDF8",
    "Soil Risk":  "#A78BFA",
}

fig = px.scatter_geo(
    filtered,
    lat       = "lat",
    lon       = "lon",
    color     = "risk_name",
    hover_name= "zone_name",
    hover_data= {
        "crop":           True,
        "confidence_pct": True,
        "lat":            False,
        "lon":            False,
    },
    color_discrete_map = COLOR_MAP,
    projection = "natural earth",
    title      = "Agricultural Zone Risk Map — India",
)
fig.update_traces(
    marker=dict(size=marker_sz, line=dict(width=1, color="#F8FAFC"))
)
fig.update_geos(
    visible       = False,
    showcountries = True,
    countrycolor  = "rgba(255,255,255,0.28)",
    lataxis_range = [6, 38],
    lonaxis_range = [67, 98],
    bgcolor       = "rgba(0,0,0,0)",
)
fig.update_layout(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font_color    = "#E5F7EE",
    legend_title_text = "Predicted Risk",
    height = 540,
)
st.plotly_chart(fig, use_container_width=True)

# ── Zone detail table ──────────────────────────────────────────────
st.markdown("### 📋 Filtered Zone Details")
st.dataframe(
    filtered[["zone_name", "crop", "emoji", "risk_name", "confidence_pct", "lat", "lon"]].rename(columns={
        "zone_name":"Zone", "crop":"Crop", "emoji":"", "risk_name":"Risk",
        "confidence_pct":"Confidence %", "lat":"Lat", "lon":"Lon",
    }),
    use_container_width=True,
    hide_index=True,
)