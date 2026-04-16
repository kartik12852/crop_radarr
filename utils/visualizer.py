"""
utils/visualizer.py
Visual helpers for charts, graphs, and report figures.
"""

from __future__ import annotations

from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.model_utils import LABEL_COLORS, LABEL_NAMES

PALETTE = [LABEL_COLORS[i] for i in sorted(LABEL_COLORS)]
sns.set_theme(style="whitegrid")


def glass_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        font_color="#E5F7EE",
        legend_font_color="#E5F7EE",
        margin=dict(l=30, r=20, t=55, b=30),
    )
    return fig


def plot_risk_distribution(df: pd.DataFrame, pred_col: str = "pred_label") -> go.Figure:
    counts = df[pred_col].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=[LABEL_NAMES.get(i, str(i)) for i in counts.index],
        y=counts.values,
        text=counts.values,
        textposition="outside",
        marker_color=[LABEL_COLORS.get(i, "#94A3B8") for i in counts.index],
    ))
    fig.update_yaxes(title="Zones")
    fig.update_xaxes(title="Predicted Risk")
    return glass_layout(fig, "Risk distribution across zones")


def plot_risk_pie(df: pd.DataFrame, pred_col: str = "pred_label") -> go.Figure:
    counts = df[pred_col].value_counts().sort_index()
    fig = go.Figure(go.Pie(
        labels=[LABEL_NAMES.get(i, str(i)) for i in counts.index],
        values=counts.values,
        hole=0.48,
        marker_colors=[LABEL_COLORS.get(i, "#94A3B8") for i in counts.index],
        textinfo="label+percent",
    ))
    return glass_layout(fig, "Zone risk breakdown")


def plot_training_curves(history: dict) -> go.Figure:
    x = history.get("estimators") or list(range(1, len(history.get("train_loss", [])) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=history.get("train_loss", []), name="Train log-loss", line=dict(color="#F97316", width=3)))
    fig.add_trace(go.Scatter(x=x, y=history.get("val_loss", []), name="Validation log-loss", line=dict(color="#38BDF8", width=3)))
    fig.add_trace(go.Scatter(x=x, y=history.get("val_acc", []), name="Validation accuracy", line=dict(color="#34D399", width=3), yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1]),
        xaxis=dict(title="Trees / training step"),
    )
    return glass_layout(fig, "Training progress")


def plot_confusion_matrix_bytes(y_true, y_pred) -> bytes:
    labels = [LABEL_NAMES.get(i, str(i)) for i in range(5)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#08131f")
    ax.set_facecolor("#08131f")
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", xticklabels=labels, yticklabels=labels, linewidths=0.5, ax=ax, cbar=True)
    ax.set_xlabel("Predicted", color="#E5F7EE")
    ax.set_ylabel("Actual", color="#E5F7EE")
    ax.set_title("Confusion Matrix", color="#F8FFFB", pad=12)
    ax.tick_params(colors="#D1FAE5")
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_feature_importance(feature_names: list, importances: np.ndarray, top_n: int = 12) -> go.Figure:
    idx = np.argsort(importances)[::-1][:top_n]
    fig = go.Figure(go.Bar(
        x=importances[idx][::-1],
        y=[feature_names[i] for i in idx][::-1],
        orientation="h",
        marker_color="#55D6BE",
    ))
    fig.update_xaxes(title="Importance")
    return glass_layout(fig, f"Top {top_n} feature drivers")


def plot_zone_radar(zone_row: pd.Series, feature_names: list) -> go.Figure:
    cols = [c for c in feature_names if c in zone_row.index][:8]
    vals = np.array([float(zone_row[c]) for c in cols], dtype=float)
    vals = (vals - vals.min()) / (vals.ptp() + 1e-9)
    fig = go.Figure(go.Scatterpolar(
        r=vals.tolist() + [float(vals[0])],
        theta=cols + [cols[0]],
        fill="toself",
        line=dict(color="#7CFFB2", width=3),
        fillcolor="rgba(124,255,178,0.22)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.12)")),
    )
    return glass_layout(fig, f"Zone feature radar — {zone_row.get('zone_name', '')}")


def build_network_figure(df: pd.DataFrame, preds: np.ndarray, adj_mask: np.ndarray) -> go.Figure:
    g = nx.Graph()
    for i, row in df.reset_index(drop=True).iterrows():
        g.add_node(i, name=row.get("zone_name", ""), crop=row.get("crop", ""))
    n = len(df)
    adj = np.asarray(adj_mask)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                g.add_edge(i, j)
    pos = nx.spring_layout(g, seed=42, k=0.9 / np.sqrt(max(n, 1)))

    edge_x, edge_y = [], []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="rgba(203,213,225,0.35)", width=1), hoverinfo="none")
    node_x = [pos[i][0] for i in g.nodes()]
    node_y = [pos[i][1] for i in g.nodes()]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(i) for i in g.nodes()],
        textposition="top center",
        marker=dict(size=18, color=[LABEL_COLORS.get(int(preds[i]), "#94A3B8") for i in g.nodes()], line=dict(color="#F8FAFC", width=1.2)),
        hovertext=[f"{g.nodes[i]['name']}<br>{g.nodes[i]['crop']}<br>{LABEL_NAMES.get(int(preds[i]), '?')}" for i in g.nodes()],
        hoverinfo="text",
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return glass_layout(fig, "Spatial connectivity graph")
