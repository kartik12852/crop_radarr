"""pages/5_Reports.py — reports"""

from __future__ import annotations

import json
import os
import sys
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.theme import hero, inject_theme

st.set_page_config(page_title="Reports", page_icon="📊", layout="wide")
inject_theme()
hero("📊 Model Reports & Analytics", "Validation metrics, confusion matrix, training curves, and exportable PDF reports.", "Reporting")


@st.cache_resource
def _load():
    from src.model_utils import load_inference_bundle, load_metrics, predict_all, zone_summary_df
    from src.recommendation import get_all_recommendations
    model, X, adj_norm, adj_mask, y, df, fc = load_inference_bundle()
    preds, probs = predict_all(model, X, adj_norm)
    summary = zone_summary_df(df, preds, probs)
    recs = get_all_recommendations(df, preds)
    metrics = load_metrics()
    return model, X, adj_norm, y, df, fc, preds, probs, summary, recs, metrics


model, X, adj_norm, y, df, fc, preds, probs, summary, recs, metrics = _load()
t1, t2, t3, t4 = st.tabs(["Metrics", "Confusion Matrix", "Training Curves", "PDF Export"])

with t1:
    if metrics:
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", metrics.get("accuracy", 0))
        b.metric("F1", metrics.get("f1_score", 0))
        c.metric("Precision", metrics.get("precision", 0))
        d.metric("Recall", metrics.get("recall", 0))
        e.metric("ROC-AUC", metrics.get("roc_auc") or "N/A")
        st.json(metrics)
    else:
        st.warning("Run python train_model.py to generate validation metrics.")
    from utils.visualizer import plot_risk_distribution
    st.plotly_chart(plot_risk_distribution(summary), use_container_width=True)

with t2:
    from utils.visualizer import plot_confusion_matrix_bytes
    st.image(plot_confusion_matrix_bytes(y, preds), caption="Confusion matrix", use_column_width=False)
    rep = classification_report(y, preds, labels=list(range(5)), target_names=["Safe", "Drought", "Heat Stress", "Flood", "Soil Risk"], output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T.reset_index()
    rep_df.columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    st.dataframe(rep_df, use_container_width=True, hide_index=True)

with t3:
    from src.paths import HISTORY_PATH
    from utils.visualizer import plot_training_curves
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, encoding="utf-8") as f:
            history = json.load(f)
        st.plotly_chart(plot_training_curves(history), use_container_width=True)
    else:
        st.info("Training history not found yet.")

with t4:
    include_recs = st.checkbox("Include recommendations", value=True)
    if st.button("Generate PDF", type="primary"):
        try:
            from utils.pdf_exporter import generate_pdf_report
            pdf_bytes = generate_pdf_report(df, preds, probs, metrics, recommendations=recs if include_recs else None)
            st.success("PDF generated")
            st.download_button("Download PDF", pdf_bytes, "climate_radar_report.pdf", "application/pdf")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
