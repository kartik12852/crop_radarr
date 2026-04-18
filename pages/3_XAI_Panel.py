"""pages/3_XAI_Panel.py — Explainability dashboard"""

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

hero(
    "🧪 XAI — Explainability Panel",
    "Understand why each zone received its risk prediction. "
    "Uses SHAP when available, or a built-in feature-impact approximation as fallback.",
    "🔍 Explainability",
)

# ── SHAP availability banner ──────────────────────────────────────────────────
try:
    import shap as _shap_check  # noqa
    st.success("✅ SHAP is installed — full TreeExplainer explanations available.")
    SHAP_OK = True
except ImportError:
    st.info(
        "ℹ️ **SHAP not installed** — using built-in feature-impact approximation instead.\n\n"
        "For full SHAP values run: `pip install shap` then restart Streamlit.",
        icon="📊",
    )
    SHAP_OK = False


# ── Load bundle ───────────────────────────────────────────────────────────────
@st.cache_resource
def _load():
    from src.model_utils import LABEL_NAMES, load_inference_bundle, predict_all
    model, X, adj_norm, adj_mask, y, df, fc = load_inference_bundle()
    preds, probs = predict_all(model, X, adj_norm)
    return model, X, adj_norm, df, fc, preds, probs, LABEL_NAMES


with st.spinner("Loading model…"):
    model, X, adj_norm, df, fc, preds, probs, LABEL_NAMES = _load()

# ── Zone selector ─────────────────────────────────────────────────────────────
zone_names = df["zone_name"].tolist()
RISK_COLORS = {0: "#34D399", 1: "#F59E0B", 2: "#F97316", 3: "#38BDF8", 4: "#A78BFA"}
RISK_EMOJIS = {0: "✅", 1: "🌵", 2: "🔥", 3: "🌊", 4: "🪨"}

col_sel, col_info = st.columns([2, 3])
with col_sel:
    zone = st.selectbox("Select Zone", zone_names, key="xai_zone_select")

zone_idx  = zone_names.index(zone)
pred_cls  = int(preds[zone_idx])
conf      = float(probs[zone_idx].max()) * 100
risk_name = LABEL_NAMES.get(pred_cls, "Unknown")
risk_color= RISK_COLORS.get(pred_cls, "#94A3B8")
risk_emoji= RISK_EMOJIS.get(pred_cls, "❓")

with col_info:
    st.markdown(
        f"""
        <div style="background:rgba(15,23,42,0.80);border:1px solid {risk_color};
                    border-radius:14px;padding:0.85rem 1.1rem;margin-top:0.5rem;">
            <span style="font-size:1.4rem;">{risk_emoji}</span>
            <b style="color:{risk_color};font-size:1.1rem;"> {risk_name}</b>
            &nbsp;|&nbsp; <span style="color:#A7F3D0;">Confidence: {conf:.1f}%</span>
            <br><small style="color:#CBD5E1;">Zone: {zone}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Class probability breakdown ───────────────────────────────────────────────
with st.expander("📊 Class probability breakdown for this zone", expanded=False):
    prob_data = {
        "Risk Class": [f"{RISK_EMOJIS.get(i,'?')} {LABEL_NAMES.get(i, i)}" for i in range(5)],
        "Probability %": [round(float(probs[zone_idx, i]) * 100, 1) for i in range(5)],  # numeric, not string
    }
    prob_df = pd.DataFrame(prob_data)
    st.dataframe(
        prob_df.style.background_gradient(
            subset=["Probability %"], cmap="Greens"   # now works — column is float
        ).format({"Probability %": "{:.1f}%"}),       # display with % sign
        use_container_width=True, hide_index=True,
    )

st.divider()

# ── Main two-column layout ────────────────────────────────────────────────────
c_left, c_right = st.columns([1, 1], gap="large")

with c_left:
    st.markdown("#### 🔬 Per-Zone Feature Impact")
    st.caption(
        "Why did the model predict this risk for the selected zone? "
        "Orange bars push toward the predicted risk; blue bars reduce it."
    )

    if st.button("🔍 Explain this zone", type="primary", use_container_width=True):
        with st.spinner("Computing explanation…"):
            try:
                from src.xai_explainer import explain_node, plot_shap_bar
                result = explain_node(
                    model, X, adj_norm, zone_idx, fc,
                    use_shap=SHAP_OK,
                )
                method = result.get("method", "shap")

                if method == "shap":
                    st.success("✅ SHAP TreeExplainer used")
                else:
                    st.info("📊 Built-in feature-impact approximation used (install shap for exact SHAP values)")

                img = plot_shap_bar(result, top_n=14)
                st.image(img, caption=f"Feature impact — {zone}", use_column_width=True)

                sv       = result["shap_values"][pred_cls]
                shap_df  = pd.DataFrame({
                    "Feature": fc,
                    "Impact Score": [round(float(v), 5) for v in sv],
                    "Direction": [
                        "↑ increases risk" if v > 0 else "↓ reduces risk"
                        for v in sv
                    ],
                    "Zone Value": [
                        f"{float(X[zone_idx, i]):.3f}" if i < X.shape[1] else "—"
                        for i in range(len(fc))
                    ],
                }).sort_values("Impact Score", key=lambda s: s.abs(), ascending=False)

                st.dataframe(shap_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Explanation failed: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")


with c_right:
    st.markdown("#### 🌐 Global Feature Importance")
    st.caption(
        "Which features matter most across ALL zones? "
        "Longer bar = more influential in risk classification."
    )

    # Always show RF-based fast importance
    if getattr(model, "feature_importances_", None) is not None:
        from utils.visualizer import plot_feature_importance
        st.plotly_chart(
            plot_feature_importance(fc, model.feature_importances_, top_n=14),
            use_container_width=True,
        )
    else:
        st.warning("Feature importances not available — retrain the model.")

    st.divider()

    if st.button("🧮 Compute Global SHAP Importance", use_container_width=True):
        with st.spinner("Computing global importance (may take ~10 seconds)…"):
            try:
                from src.xai_explainer import plot_global_shap
                img = plot_global_shap(model, X, adj_norm, fc, top_n=14)
                method_label = "SHAP" if SHAP_OK else "RF Importance"
                st.image(img, caption=f"Global {method_label} importance", use_column_width=True)
            except Exception as e:
                st.error(f"❌ Global importance failed: {e}")

st.divider()

# ── Feature value table for selected zone ─────────────────────────────────────
st.markdown("#### 📋 Raw Feature Values — Selected Zone")
import numpy as np
feat_vals = pd.DataFrame({
    "Feature":     fc,
    "Zone Value":  [round(float(X[zone_idx, i]), 4) for i in range(len(fc))],
    "All Zones Mean": [round(float(X[:, i].mean()), 4) for i in range(len(fc))],
    "All Zones Std":  [round(float(X[:, i].std()), 4) for i in range(len(fc))],
    "Z-Score":     [
        round(float((X[zone_idx, i] - X[:, i].mean()) / (X[:, i].std() + 1e-8)), 2)
        for i in range(len(fc))
    ],
})
feat_vals["Deviation"] = feat_vals["Z-Score"].apply(
    lambda z: "🔴 Very High" if z > 2 else ("🟠 High" if z > 1 else
              ("🔵 Very Low" if z < -2 else ("🟦 Low" if z < -1 else "🟢 Normal")))
)
st.dataframe(feat_vals, use_container_width=True, hide_index=True, height=420)
