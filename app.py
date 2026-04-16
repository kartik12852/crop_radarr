
from __future__ import annotations

import os
import sys
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.theme import hero, inject_theme, metric_tile, pills

st.set_page_config(page_title="Climate Radar — Crop Risk Intelligence", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")
inject_theme()


@st.cache_resource(show_spinner="Loading climate intelligence bundle...")
def load_bundle():
    from src.alert_engine import generate_all_alerts, sort_alerts_by_severity
    from src.model_utils import load_inference_bundle, load_metrics, predict_all, zone_summary_df
    from src.recommendation import get_all_recommendations

    model, X, adj_norm, adj_mask, y, df, feat_cols = load_inference_bundle()
    preds, probs = predict_all(model, X, adj_norm)
    summary = zone_summary_df(df, preds, probs)
    alerts = sort_alerts_by_severity(generate_all_alerts(df, preds, probs))
    recs = get_all_recommendations(df, preds)
    metrics = load_metrics()
    return {
        "model": model,
        "X": X,
        "adj_norm": adj_norm,
        "adj_mask": adj_mask,
        "y": y,
        "df": df,
        "feat_cols": feat_cols,
        "preds": preds,
        "probs": probs,
        "summary": summary,
        "alerts": alerts,
        "recs": recs,
        "metrics": metrics,
    }


with st.sidebar:
    st.markdown("## 🌾 Climate Radar")
    st.caption("Stable Windows-safe build")
    pills(["No PyTorch", "Graph-aware model", "30 Indian zones", "Streamlit dashboard"])
    st.divider()
    st.markdown("### Run order")
    st.code("python doctor.py\npython synthetic/synthetic_data_generator.py\npython train_model.py\nstreamlit run app.py", language="bash")
    st.divider()
    st.markdown("### Pages")
    st.markdown("- Dashboard\n- Risk Map\n- Graph Explorer\n- XAI Panel\n- Recommendations\n- Reports")

hero("🌾 Climate Radar — Crop Risk Intelligence", "A cleaner, safer build that avoids the Windows torch/fbgemm crash and keeps the full analytics workflow running.", "Production-ready demo")
pills(["Fresh UI refresh", "Graph-aware ensemble", "Offline synthetic fallback", "PDF reporting"])

try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Failed to load the project: {e}")
    st.info("Run `python doctor.py`, then `python synthetic/synthetic_data_generator.py`, then `python train_model.py`.")
    st.stop()

summary = bundle["summary"]
alerts = bundle["alerts"]
metrics = bundle["metrics"]
preds = bundle["preds"]

a, b, c, d, e = st.columns(5)
counts = pd.Series(preds).value_counts()
for col, label, value in [
    (a, "✅ Safe", int(counts.get(0, 0))),
    (b, "🌵 Drought", int(counts.get(1, 0))),
    (c, "🔥 Heat Stress", int(counts.get(2, 0))),
    (d, "🌊 Flood", int(counts.get(3, 0))),
    (e, "🪨 Soil Risk", int(counts.get(4, 0))),
]:
    col.markdown(metric_tile(label, str(value)), unsafe_allow_html=True)

left, right = st.columns([1.45, 1])
with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Risk distribution")
    from utils.visualizer import plot_risk_distribution
    st.plotly_chart(plot_risk_distribution(summary), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Zone breakdown")
    from utils.visualizer import plot_risk_pie
    st.plotly_chart(plot_risk_pie(summary), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### 🚨 Priority alerts")
priority = [a for a in alerts if a["severity"] != "Low"][:6]
if not priority:
    st.success("No high-priority alerts right now.")
else:
    for alert in priority:
        st.markdown(
            f"<div class='glass-card' style='margin-bottom:0.7rem; border-left:4px solid {alert['color']};'>"
            f"<b>{alert['severity_icon']} {alert['zone_name']}</b> · {alert['crop']} · <b>{alert['risk_name']}</b><br>"
            f"{alert['headline']}<br><span style='color:#A7F3D0;'>Confidence: {alert['confidence']}% · {alert['timestamp']}</span></div>",
            unsafe_allow_html=True,
        )

st.markdown("### 📋 All zones")
view = summary[["zone_name", "crop", "emoji", "risk_name", "confidence_pct", "lat", "lon"]].copy()
view.columns = ["Zone", "Crop", "", "Risk", "Confidence %", "Lat", "Lon"]
st.dataframe(view, use_container_width=True, hide_index=True, height=420)

if metrics:
    st.markdown("### 🏆 Validation metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", metrics.get("accuracy", 0))
    m2.metric("F1", metrics.get("f1_score", 0))
    m3.metric("Precision", metrics.get("precision", 0))
    m4.metric("Recall", metrics.get("recall", 0))
    m5.metric("ROC-AUC", metrics.get("roc_auc") or "N/A")
    with st.expander("Metric details"):
        st.json(metrics)
