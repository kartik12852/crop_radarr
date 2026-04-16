"""
app.py — Climate Crop Radar Alert System
Main Streamlit dashboard entry point.

Run with:
    streamlit run app.py

Pages:
  Dashboard (this file) + 5 sub-pages in pages/
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.theme import hero, inject_theme, metric_tile, pills

# ------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ------------------------------------------------------------------
st.set_page_config(
    page_title           = "🌾 Climate Crop Radar",
    page_icon            = "🌾",
    layout               = "wide",
    initial_sidebar_state= "expanded",
)
inject_theme()


# ------------------------------------------------------------------
# Cached data bundle
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="🔄 Loading climate intelligence…")
def load_bundle():
    from src.alert_engine     import generate_all_alerts, sort_alerts_by_severity
    from src.model_utils      import load_inference_bundle, load_metrics, predict_all, zone_summary_df
    from src.recommendation   import get_all_recommendations

    model, X, adj_norm, adj_mask, y, df, feat_cols = load_inference_bundle()
    preds, probs  = predict_all(model, X, adj_norm)
    summary       = zone_summary_df(df, preds, probs)
    alerts        = sort_alerts_by_severity(generate_all_alerts(df, preds, probs))
    recs          = get_all_recommendations(df, preds)
    metrics       = load_metrics()

    return {
        "model":    model,
        "X":        X,
        "adj_norm": adj_norm,
        "adj_mask": adj_mask,
        "y":        y,
        "df":       df,
        "feat_cols":feat_cols,
        "preds":    preds,
        "probs":    probs,
        "summary":  summary,
        "alerts":   alerts,
        "recs":     recs,
        "metrics":  metrics,
    }


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌾 Climate Radar")
    st.caption("Graph Neural Networks + Explainable AI")
    st.divider()
    pills([
        "30 Indian Zones",
        "5 Risk Classes",
        "SHAP Explanations",
        "PDF Reports",
    ])
    st.divider()
    st.markdown("### 🚀 Run Order")
    st.code(
        "python doctor.py\n"
        "python synthetic/synthetic_data_generator.py\n"
        "python train_model.py\n"
        "streamlit run app.py",
        language="bash",
    )
    st.divider()
    st.markdown("### 📄 Pages")
    st.markdown(
        "- 🏠 **Dashboard** (this page)\n"
        "- 🗺️ Risk Map\n"
        "- 🔗 Graph Explorer\n"
        "- 🧪 XAI Explanations\n"
        "- 💡 Recommendations\n"
        "- 📊 Reports & Metrics",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Hero header
# ------------------------------------------------------------------
hero(
    "🌾 Climate Crop Radar — Intelligent Agricultural Risk Intelligence",
    "GNN-powered early warning system for 30 Indian agricultural zones. "
    "Predicts drought, heat stress, flood, and soil risk with full explainability.",
    "🛰️ Live Dashboard",
)
pills(["Graph-aware Model", "30 Indian Zones", "5 Risk Categories", "SHAP XAI", "PDF Export"])


# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"❌ Failed to load the project bundle: {exc}")
    st.info(
        "Run these commands first:\n\n"
        "```\npython doctor.py\n"
        "python synthetic/synthetic_data_generator.py\n"
        "python train_model.py\n```"
    )
    st.stop()

summary = bundle["summary"]
alerts  = bundle["alerts"]
metrics = bundle["metrics"]
preds   = bundle["preds"]

# ------------------------------------------------------------------
# Metric tiles row
# ------------------------------------------------------------------
import numpy as np
counts = {int(k): int(v) for k, v in pd.Series(preds).value_counts().items()}

c1, c2, c3, c4, c5 = st.columns(5)
tiles = [
    ("✅ Safe Zones",  counts.get(0, 0)),
    ("🌵 Drought",     counts.get(1, 0)),
    ("🔥 Heat Stress", counts.get(2, 0)),
    ("🌊 Flood Risk",  counts.get(3, 0)),
    ("🪨 Soil Risk",   counts.get(4, 0)),
]
for col, (label, value) in zip([c1, c2, c3, c4, c5], tiles):
    col.markdown(metric_tile(label, str(value)), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Charts row
# ------------------------------------------------------------------
left, right = st.columns([1.45, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Risk Distribution")
    from utils.visualizer import plot_risk_distribution
    st.plotly_chart(plot_risk_distribution(summary), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🥧 Zone Breakdown")
    from utils.visualizer import plot_risk_pie
    st.plotly_chart(plot_risk_pie(summary), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Priority alerts
# ------------------------------------------------------------------
st.markdown("### 🚨 Priority Alerts")

priority = [a for a in alerts if a["severity"] in ("High", "Critical")][:6]
if not priority:
    st.success("✅ No high-priority alerts at this time. All monitored zones appear stable.")
else:
    cols = st.columns(min(3, len(priority)))
    for i, alert in enumerate(priority):
        with cols[i % 3]:
            st.markdown(
                f"<div class='glass-card' style='border-left:4px solid {alert['color']};'>"
                f"<b>{alert['severity_icon']} {alert['zone_name']}</b><br>"
                f"<small>{alert['crop']}</small><br>"
                f"<b>{alert['risk_name']}</b> — {alert['severity']}<br>"
                f"<small style='color:#A7F3D0;'>Confidence: {alert['confidence']}%</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------
# Full zone table
# ------------------------------------------------------------------
st.markdown("### 📋 All 30 Zones — Risk Summary")
view = summary[[
    "zone_name", "crop", "emoji", "risk_name", "confidence_pct", "lat", "lon"
]].copy()
view.columns = ["Zone", "Crop", "", "Risk", "Confidence %", "Lat", "Lon"]
st.dataframe(view, use_container_width=True, hide_index=True, height=460)

# ------------------------------------------------------------------
# Validation metrics (if available)
# ------------------------------------------------------------------
if metrics:
    st.markdown("### 🏆 Model Validation Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    metric_pairs = [
        ("Accuracy",  metrics.get("accuracy", "–")),
        ("F1-Score",  metrics.get("f1_score", "–")),
        ("Precision", metrics.get("precision", "–")),
        ("Recall",    metrics.get("recall", "–")),
        ("ROC-AUC",   metrics.get("roc_auc") or "–"),
    ]
    for col, (name, val) in zip([m1, m2, m3, m4, m5], metric_pairs):
        col.metric(name, val)

    with st.expander("🔍 Full metrics JSON"):
        st.json(metrics)