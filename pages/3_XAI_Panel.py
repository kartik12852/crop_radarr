"""pages/3_XAI_Panel.py — explainability"""

from __future__ import annotations

import os
import sys
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.theme import hero, inject_theme

st.set_page_config(page_title="XAI Panel", page_icon="🧪", layout="wide")
inject_theme()
hero("🧪 XAI / SHAP Panel", "Explain why the model marked a zone as drought, heat, flood, soil risk, or safe.", "Explainability")


@st.cache_resource
def _load():
    from src.model_utils import LABEL_NAMES, load_inference_bundle, predict_all
    model, X, adj_norm, adj_mask, y, df, fc = load_inference_bundle()
    preds, probs = predict_all(model, X, adj_norm)
    return model, X, adj_norm, df, fc, preds, probs, LABEL_NAMES


model, X, adj_norm, df, fc, preds, probs, LABEL_NAMES = _load()
zone_names = df["zone_name"].tolist()
zone = st.selectbox("Zone", zone_names)
zone_idx = zone_names.index(zone)
pred_cls = int(preds[zone_idx])
st.info(f"Prediction: {LABEL_NAMES[pred_cls]} | Confidence: {probs[zone_idx].max() * 100:.1f}%")

c1, c2 = st.columns([1, 1])
with c1:
    if st.button("Explain selected zone", type="primary"):
        try:
            from src.xai_explainer import explain_node, plot_shap_bar
            result = explain_node(model, X, adj_norm, zone_idx, fc)
            st.image(plot_shap_bar(result), caption=f"SHAP explanation — {zone}", use_column_width=True)
            sv = result["shap_values"][pred_cls]
            shap_df = pd.DataFrame({
                "Feature": fc,
                "SHAP Value": sv,
                "Effect": ["↑ pushes toward predicted risk" if v > 0 else "↓ pulls away from predicted risk" for v in sv],
            }).sort_values("SHAP Value", key=lambda s: s.abs(), ascending=False)
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")
            st.info("Install shap if needed: pip install shap")

with c2:
    st.subheader("Fast global importance")
    if getattr(model, "feature_importances_", None) is not None:
        from utils.visualizer import plot_feature_importance
        st.plotly_chart(plot_feature_importance(fc, model.feature_importances_, top_n=14), use_container_width=True)
    if st.button("Compute global SHAP"):
        try:
            from src.xai_explainer import plot_global_shap
            img = plot_global_shap(model, X, adj_norm, fc, top_n=14)
            st.image(img, caption="Global SHAP importance", use_column_width=True)
        except Exception as e:
            st.error(f"Global SHAP failed: {e}")
