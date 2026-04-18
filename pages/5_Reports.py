"""pages/5_Reports.py — Model Reports & Analytics"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.theme import hero, inject_theme

st.set_page_config(page_title="Reports", page_icon="📊", layout="wide")
inject_theme()
hero(
    "📊 Model Reports & Analytics",
    "Validation metrics, confusion matrix, training curves, and exportable PDF reports.",
    "📈 Reporting",
)


@st.cache_resource
def _load():
    from src.model_utils import load_inference_bundle, load_metrics, predict_all, zone_summary_df
    from src.recommendation import get_all_recommendations
    model, X, adj_norm, adj_mask, y, df, fc = load_inference_bundle()
    preds, probs = predict_all(model, X, adj_norm)
    summary      = zone_summary_df(df, preds, probs)
    recs         = get_all_recommendations(df, preds)
    metrics      = load_metrics()
    return model, X, adj_norm, y, df, fc, preds, probs, summary, recs, metrics


model, X, adj_norm, y, df, fc, preds, probs, summary, recs, metrics = _load()

t1, t2, t3, t4, t5 = st.tabs([
    "📊 Metrics", "🔢 Confusion Matrix", "📈 Training Curves", "🗺️ Zone Analysis", "📄 PDF Export"
])

# ── Tab 1: Metrics ────────────────────────────────────────────────────────────
with t1:
    if metrics:
        # Overfitting warning
        warn = metrics.get("overfit_warning")
        if warn:
            st.warning(
                "**⚠️ Suspiciously perfect metrics detected**\n\n"
                f"{warn}\n\n"
                "**Why this happens:** The model is trained on only 30 zones (one row per zone). "
                "With such a small dataset a RandomForest can memorise the labels perfectly. "
                "This does **not** mean the model is broken — it means you need more data.\n\n"
                "**Fix:** Run the live data fetch to get real 30-day weather data:\n"
                "```\npython data/fetch_data.py\npython train_model.py\n```\n"
                "Or click **Fetch Data Now** in the sidebar."
            )
        else:
            st.success("✅ Model trained on sufficient data — metrics are reliable.")

        # Metrics row
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy",  metrics.get("accuracy", 0))
        b.metric("F1-Score",  metrics.get("f1_score", 0))
        c.metric("Precision", metrics.get("precision", 0))
        d.metric("Recall",    metrics.get("recall", 0))
        e.metric("ROC-AUC",   metrics.get("roc_auc") or "N/A")

        # Eval strategy callout
        strategy = metrics.get("eval_strategy", "unknown")
        n_samp   = metrics.get("n_samples", "?")
        st.info(
            f"**Evaluation strategy:** {strategy} | "
            f"**Training samples:** {n_samp} zones | "
            f"**Model:** {metrics.get('model_family', '?')}"
        )

        with st.expander("🔍 Full metrics JSON"):
            st.json(metrics)
    else:
        st.warning("Run `python train_model.py` to generate validation metrics.")

    st.divider()
    from utils.visualizer import plot_risk_distribution, plot_risk_pie
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(plot_risk_distribution(summary), use_container_width=True)
    with col_right:
        st.plotly_chart(plot_risk_pie(summary), use_container_width=True)

    # Risk distribution table
    st.markdown("#### Zone count by risk class")
    risk_counts = pd.Series(preds).value_counts().sort_index()
    from src.model_utils import LABEL_NAMES, LABEL_EMOJIS, LABEL_COLORS
    dist_df = pd.DataFrame({
        "Risk Class":   [f"{LABEL_EMOJIS.get(int(i),'?')} {LABEL_NAMES.get(int(i),str(i))}" for i in risk_counts.index],
        "Zones":        risk_counts.values,
        "% of Total":  [f"{v/len(preds)*100:.1f}%" for v in risk_counts.values],
        "Avg Confidence %": [
            f"{np.mean(probs[preds == i].max(axis=1)) * 100:.1f}%"
            if (preds == i).sum() > 0 else "—"
            for i in risk_counts.index
        ],
    })
    st.dataframe(dist_df, use_container_width=True, hide_index=True)


# ── Tab 2: Confusion Matrix ───────────────────────────────────────────────────
with t2:
    st.markdown("#### Confusion Matrix")
    st.caption(
        "Each cell shows how many zones predicted as **column class** actually belong to **row class**. "
        "A perfect model has all values on the diagonal."
    )

    from utils.visualizer import plot_confusion_matrix_bytes
    img_bytes = plot_confusion_matrix_bytes(y, preds)
    col_img, col_info = st.columns([2, 1])
    with col_img:
        st.image(img_bytes, caption="Confusion matrix (true labels vs predicted labels)", use_column_width=True)
    with col_info:
        # Per-class accuracy
        st.markdown("**Per-class accuracy**")
        for cls in range(5):
            mask    = y == cls
            if mask.sum() == 0:
                continue
            correct = (preds[mask] == cls).sum()
            total   = mask.sum()
            pct     = correct / total * 100
            color   = "#34D399" if pct >= 80 else "#F59E0B" if pct >= 60 else "#F87171"
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.92);border-left:3px solid {color};'
                f'border-radius:8px;padding:4px 10px;margin:3px 0;">'
                f'<b style="color:{color};">{LABEL_EMOJIS.get(cls,"?")} {LABEL_NAMES.get(cls,"?")}</b> '
                f'— {pct:.0f}% ({correct}/{total})</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### Classification Report")
    present = sorted(set(y.tolist()) | set(preds.tolist()))
    rep = classification_report(
        y, preds,
        labels=present,
        target_names=[LABEL_NAMES.get(i, str(i)) for i in present],
        output_dict=True,
        zero_division=0,
    )
    rep_df = pd.DataFrame(rep).T.reset_index()
    rep_df.columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rep_df[["Precision", "Recall", "F1-Score"]] = rep_df[["Precision", "Recall", "F1-Score"]].round(4)
    st.dataframe(rep_df, use_container_width=True, hide_index=True)


# ── Tab 3: Training Curves ────────────────────────────────────────────────────
with t3:
    from src.paths import HISTORY_PATH
    from utils.visualizer import plot_training_curves

    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, encoding="utf-8") as f:
            history = json.load(f)

        st.plotly_chart(plot_training_curves(history), use_container_width=True)

        # Learning curve interpretation
        final_train = history["train_loss"][-1] if history["train_loss"] else None
        final_val   = history["val_loss"][-1]   if history["val_loss"]   else None
        final_acc   = history["val_acc"][-1]     if history["val_acc"]    else None

        if final_train is not None and final_val is not None:
            gap = final_val - final_train
            if gap > 0.5:
                st.warning(f"⚠️ High overfitting detected — train/val loss gap: {gap:.3f}. Fetch more real data.")
            elif gap > 0.1:
                st.info(f"ℹ️ Slight overfitting — gap: {gap:.3f}. Model is reasonable.")
            else:
                st.success(f"✅ Curves converged well — gap: {gap:.3f}.")

        col_h1, col_h2, col_h3 = st.columns(3)
        col_h1.metric("Final Train Loss",    f"{final_train:.4f}" if final_train else "—")
        col_h2.metric("Final Val Loss",      f"{final_val:.4f}"   if final_val   else "—")
        col_h3.metric("Final Val Accuracy",  f"{final_acc:.4f}"   if final_acc   else "—")
    else:
        st.info("Training history not found. Run `python train_model.py` to generate it.")


# ── Tab 4: Zone Analysis ──────────────────────────────────────────────────────
with t4:
    st.markdown("#### 🌡️ Feature Statistics by Risk Class")
    if len(fc) > 0:
        feat_stats = pd.DataFrame(X, columns=fc)
        feat_stats["Risk"] = [LABEL_NAMES.get(int(p), "?") for p in preds]

        selected_feat = st.selectbox("Select feature to compare across risk classes", fc)
        if selected_feat:
            import plotly.express as px
            fig_box = px.box(
                feat_stats,
                x="Risk",
                y=selected_feat,
                color="Risk",
                color_discrete_map={LABEL_NAMES[k]: v for k, v in LABEL_COLORS.items()},
                title=f"Distribution of {selected_feat} by Risk Class",
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(255,255,255,0.92)",
                font_color   ="#1a2e1a",
            )
            st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    st.markdown("#### 📋 Full Zone Data")
    full_view = df.copy()
    full_view["Predicted Risk"]  = [LABEL_NAMES.get(int(p), "?") for p in preds]
    full_view["Confidence %"]    = np.round(probs.max(axis=1) * 100, 1)
    display_cols = (
        ["zone_name", "crop", "Predicted Risk", "Confidence %"]
        + [c for c in fc if c in full_view.columns]
    )
    st.dataframe(
        full_view[[c for c in display_cols if c in full_view.columns]],
        use_container_width=True, hide_index=True, height=450,
    )


# ── Tab 5: PDF Export ─────────────────────────────────────────────────────────
with t5:
    st.markdown("#### 📄 Generate PDF Report")
    st.caption(
        "Exports an A4 PDF containing the executive summary, zone risk table, "
        "model metrics, and mitigation recommendations."
    )

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        include_recs = st.checkbox("Include mitigation recommendations", value=True)
    with col_opt2:
        risk_filter_pdf = st.multiselect(
            "Include only these risk classes",
            list(LABEL_NAMES.values()),
            default=list(LABEL_NAMES.values()),
        )

    st.divider()
    if st.button("🖨️ Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating PDF…"):
            try:
                from utils.pdf_exporter import generate_pdf_report
                # Filter recs to selected risk classes
                filtered_recs = (
                    [r for r in recs if r["risk_name"] in risk_filter_pdf]
                    if include_recs else None
                )
                pdf_bytes = generate_pdf_report(
                    df, preds, probs, metrics or {},
                    recommendations=filtered_recs,
                )
                from datetime import datetime
                fname = f"climate_radar_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.success(f"✅ PDF generated — {len(pdf_bytes)//1024} KB")
                st.download_button(
                    "⬇️ Download PDF Report",
                    pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                )
            except ImportError:
                st.error("❌ fpdf2 not installed. Run: `pip install fpdf2`")
            except Exception as e:
                st.error(f"❌ PDF generation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
