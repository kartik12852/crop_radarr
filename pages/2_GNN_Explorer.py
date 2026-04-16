"""pages/2_GNN_Explorer.py — graph explorer"""

from __future__ import annotations

import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.theme import hero, inject_theme

st.set_page_config(page_title="Graph Explorer", page_icon="🔗", layout="wide")
inject_theme()
hero("🔗 Graph Explorer", "Inspect zone connectivity and graph-smoothed embeddings without any PyTorch dependency.", "Network analytics")


@st.cache_resource
def _load():
    from src.model_utils import LABEL_COLORS, LABEL_NAMES, load_inference_bundle, predict_all, zone_summary_df
    model, X, adj_norm, adj_mask, y, df, fc = load_inference_bundle()
    preds, probs = predict_all(model, X, adj_norm)
    summary = zone_summary_df(df, preds, probs)
    embeds = model.get_embeddings(X, adj_norm)
    return model, X, adj_norm, adj_mask, y, df, fc, preds, probs, summary, embeds, LABEL_NAMES, LABEL_COLORS


model, X, adj_norm, adj_mask, y, df, fc, preds, probs, summary, embeds, LABEL_NAMES, LABEL_COLORS = _load()
t1, t2, t3 = st.tabs(["🕸️ Connectivity Graph", "🧬 Embeddings", "📊 Feature Explorer"])

with t1:
    from utils.visualizer import build_network_figure
    st.plotly_chart(build_network_figure(df, preds, adj_mask), use_container_width=True)
    st.info(f"Nodes: {len(df)} | Edges: {int(adj_mask.sum()) // 2} | Avg degree: {adj_mask.sum(axis=1).mean():.1f}")

with t2:
    proj = PCA(n_components=2, random_state=42).fit_transform(embeds)
    emb_df = pd.DataFrame({
        "x": proj[:, 0],
        "y": proj[:, 1],
        "zone": df["zone_name"],
        "crop": df["crop"],
        "risk": [LABEL_NAMES[int(p)] for p in preds],
        "confidence": [f"{probs[i].max() * 100:.1f}%" for i in range(len(preds))],
    })
    fig = px.scatter(
        emb_df,
        x="x",
        y="y",
        color="risk",
        hover_data=["zone", "crop", "confidence"],
        color_discrete_map={LABEL_NAMES[k]: v for k, v in LABEL_COLORS.items()},
    )
    fig.update_traces(marker=dict(size=13, line=dict(width=1, color="#F8FAFC")))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.55)", font_color="#E5F7EE")
    st.plotly_chart(fig, use_container_width=True)

with t3:
    zone = st.selectbox("Select zone", df["zone_name"].tolist())
    row = df[df["zone_name"] == zone].iloc[0]
    from utils.visualizer import plot_feature_importance, plot_zone_radar
    st.plotly_chart(plot_zone_radar(row, fc), use_container_width=True)
    feat_df = pd.DataFrame({"Feature": fc, "Value": [float(row.get(c, 0)) for c in fc]})
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
    if getattr(model, "feature_importances_", None) is not None:
        st.plotly_chart(plot_feature_importance(fc, model.feature_importances_, top_n=12), use_container_width=True)
