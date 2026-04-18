"""pages/1_Risk_Map.py — Interactive spatial risk map"""

from __future__ import annotations

import os
import sys
import datetime

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.theme import hero, inject_theme, pills

st.set_page_config(page_title="Risk Map", page_icon="🗺️", layout="wide")
inject_theme()

hero(
    "🗺️ Crop Risk Map",
    "Spatial view of predicted climate and soil risk across 30 Indian agricultural zones. "
    "Click a zone on the map for details.",
    "🛰️ Interactive Map",
)


@st.cache_resource
def _load():
    from src.model_utils import LABEL_NAMES, LABEL_COLORS, LABEL_EMOJIS, load_inference_bundle, predict_all, zone_summary_df
    model, X, adj_norm, adj_mask, y, df, fc = load_inference_bundle()
    preds, probs = predict_all(model, X, adj_norm)
    return zone_summary_df(df, preds, probs), LABEL_NAMES, LABEL_COLORS, LABEL_EMOJIS, df, fc, X


summary, LABEL_NAMES, LABEL_COLORS, LABEL_EMOJIS, df, fc, X = _load()

# ── Data source info ──────────────────────────────────────────────────────────
from src.paths import MERGED_CSV, WEATHER_CSV
if MERGED_CSV.exists():
    mtime = datetime.datetime.fromtimestamp(MERGED_CSV.stat().st_mtime)
    age_h = (datetime.datetime.now() - mtime).total_seconds() / 3600
    src_label = "🛰️ Real API data" if WEATHER_CSV.exists() else "🧬 Synthetic data"
    freshness_color = "#34D399" if age_h < 3 else "#F59E0B" if age_h < 12 else "#F87171"
    pills([
        f"{src_label}",
        f"Updated {age_h:.1f}h ago",
        f"{len(summary)} zones",
        "Auto-refresh every 2h",
    ])

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Map Filters")

    risk_filter = st.multiselect(
        "Risk Types",
        list(LABEL_NAMES.values()),
        default=list(LABEL_NAMES.values()),
    )
    min_conf = st.slider("Min Confidence %", 0, 100, 0, 5)
    marker_sz = st.slider("Marker Size", 8, 36, 20, 2)

    st.divider()
    st.markdown("### 🎨 Map Style")
    map_style = st.radio(
        "Projection",
        ["natural earth", "mercator", "orthographic"],
        index=0,
        horizontal=True,
    )
    show_labels = st.checkbox("Show zone labels", value=False)

    st.divider()
    # Summary counts in sidebar
    st.markdown("### 📊 Current counts")
    for risk_id, risk_name in LABEL_NAMES.items():
        cnt = (summary["pred_label"] == risk_id).sum()
        emoji = LABEL_EMOJIS.get(risk_id, "?")
        color = LABEL_COLORS.get(risk_id, "#94A3B8")
        st.markdown(
            f'<div style="border-left:3px solid {color};padding:2px 8px;margin:2px 0;">'
            f'<b style="color:{color};">{emoji} {risk_name}</b>: {cnt} zones</div>',
            unsafe_allow_html=True,
        )


# ── Filter data ───────────────────────────────────────────────────────────────
filtered = summary[
    (summary["risk_name"].isin(risk_filter)) &
    (summary["confidence_pct"] >= min_conf)
].copy()

n_total    = len(summary)
n_filtered = len(filtered)
pills([
    f"Showing {n_filtered}/{n_total} zones",
    f"Confidence ≥ {min_conf}%",
])

# ── Map ───────────────────────────────────────────────────────────────────────
COLOR_MAP = {
    "Safe":       "#34D399",
    "Drought":    "#F59E0B",
    "Heat Stress":"#F97316",
    "Flood":      "#38BDF8",
    "Soil Risk":  "#A78BFA",
}

fig = px.scatter_geo(
    filtered,
    lat        = "lat",
    lon        = "lon",
    color      = "risk_name",
    hover_name = "zone_name",
    hover_data = {
        "crop":           True,
        "confidence_pct": True,
        "lat":            False,
        "lon":            False,
        "emoji":          False,
        "pred_label":     False,
    },
    text              = "zone_name" if show_labels else None,
    color_discrete_map= COLOR_MAP,
    projection        = map_style,
    title             = "Agricultural Zone Risk Map — India",
    size_max          = marker_sz,
)

fig.update_traces(
    marker=dict(size=marker_sz, line=dict(width=1.5, color="#F8FAFC")),
    textfont=dict(color="#E5F7EE", size=8),
)
fig.update_geos(
    visible        = False,
    showcountries  = True,
    showsubunits   = True,
    countrycolor   = "rgba(255,255,255,0.28)",
    subunitcolor   = "rgba(255,255,255,0.12)",
    lataxis_range  = [6, 38],
    lonaxis_range  = [67, 98],
    bgcolor        = "rgba(0,0,0,0)",
    landcolor      = "rgba(17,34,17,0.60)",
    oceancolor     = "rgba(7,18,35,0.80)",
    showocean      = True,
    showland       = True,
    showrivers     = True,
    rivercolor     = "rgba(56,189,248,0.4)",
)
fig.update_layout(
    paper_bgcolor     = "rgba(0,0,0,0)",
    plot_bgcolor      = "rgba(0,0,0,0)",
    font_color        = "#E5F7EE",
    legend_title_text = "Predicted Risk",
    legend            = dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(124,255,178,0.2)"),
    height            = 560,
    margin            = dict(l=0, r=0, t=40, b=0),
)
st.plotly_chart(fig, use_container_width=True)


# ── Zone detail cards ─────────────────────────────────────────────────────────
st.markdown("### 🔍 Zone Details")
tab_tbl, tab_cards, tab_risk = st.tabs(["📋 Table", "🃏 Cards", "⚠️ Risk Zones Only"])

with tab_tbl:
    view = filtered[[
        "zone_name", "crop", "emoji", "risk_name", "confidence_pct", "lat", "lon"
    ]].rename(columns={
        "zone_name":      "Zone",
        "crop":           "Crop",
        "emoji":          "",
        "risk_name":      "Risk",
        "confidence_pct": "Confidence %",
        "lat":            "Lat",
        "lon":            "Lon",
    })
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download CSV",
        view.to_csv(index=False).encode(),
        "climate_radar_zones.csv",
        "text/csv",
    )

with tab_cards:
    # Show cards in 3-column grid
    cols_per_row = 3
    rows = [
        filtered.iloc[i:i+cols_per_row]
        for i in range(0, len(filtered), cols_per_row)
    ]
    for row_df in rows:
        cols = st.columns(cols_per_row)
        for col, (_, zone_row) in zip(cols, row_df.iterrows()):
            color = COLOR_MAP.get(zone_row["risk_name"], "#94A3B8")
            with col:
                st.markdown(
                    f'<div style="background:rgba(15,23,42,0.85);border:1px solid {color};'
                    f'border-radius:14px;padding:0.8rem;margin-bottom:0.5rem;">'
                    f'<b style="color:{color};">{zone_row["emoji"]} {zone_row["zone_name"]}</b><br>'
                    f'<small style="color:#A7F3D0;">🌾 {zone_row["crop"]}</small><br>'
                    f'<b style="color:{color};">{zone_row["risk_name"]}</b> '
                    f'<small style="color:#CBD5E1;">({zone_row["confidence_pct"]:.1f}%)</small><br>'
                    f'<small style="color:#64748B;">📍 {zone_row["lat"]:.1f}°N {zone_row["lon"]:.1f}°E</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

with tab_risk:
    # Show only non-safe zones sorted by confidence
    risk_zones = summary[summary["pred_label"] != 0].sort_values(
        "confidence_pct", ascending=False
    )
    if risk_zones.empty:
        st.success("✅ No at-risk zones currently detected.")
    else:
        st.warning(f"⚠️ {len(risk_zones)} zones at risk — sorted by severity confidence")
        for _, rz in risk_zones.iterrows():
            color = COLOR_MAP.get(rz["risk_name"], "#94A3B8")
            st.markdown(
                f'<div style="background:rgba(15,23,42,0.8);border-left:4px solid {color};'
                f'border-radius:10px;padding:0.6rem 1rem;margin:4px 0;">'
                f'<b style="color:{color};">{rz["emoji"]} {rz["zone_name"]}</b> '
                f'· {rz["crop"]} '
                f'· <b style="color:{color};">{rz["risk_name"]}</b> '
                f'<span style="color:#94A3B8;">({rz["confidence_pct"]:.1f}% confidence)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
