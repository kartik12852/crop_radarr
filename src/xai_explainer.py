"""
src/xai_explainer.py
SHAP (SHapley Additive exPlanations) wrapper for the graph-aware model.

What SHAP does:
  For each zone, SHAP answers "which climate/soil features pushed
  the model towards predicting Drought (or Heat, Flood, etc.)?"
  Positive SHAP = feature increased predicted risk.
  Negative SHAP = feature reduced predicted risk.

We use TreeExplainer (fast, exact for tree-based models) on the
RandomForest's augmented feature space, then collapse the three
views (raw + smoothed + diff) back to the original feature space
for interpretability.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.gnn_model import GraphRiskModel
from src.model_utils import LABEL_NAMES


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _normalize_shap_values(
    shap_values,
    n_classes: int,
    n_aug_features: int,
) -> np.ndarray:
    """
    Normalise SHAP output to shape (n_classes, n_aug_features).
    SHAP returns different shapes depending on version — this handles all.
    """
    if isinstance(shap_values, list):
        return np.stack([np.asarray(v)[0] for v in shap_values], axis=0)

    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return arr[0][None, :]
    if arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[1] == n_aug_features:
            return np.moveaxis(arr[0], -1, 0)
        if arr.shape[0] == n_classes:
            return arr[:, 0, :]
        if arr.shape[-1] == n_classes:
            return np.moveaxis(arr[0], -1, 0)

    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


# ------------------------------------------------------------------
# Per-node explanation
# ------------------------------------------------------------------

def explain_node(
    model: GraphRiskModel,
    X: np.ndarray,
    adj_norm: np.ndarray,
    node_idx: int,
    feature_names: list[str],
) -> dict:
    """
    Compute SHAP values for one zone node.

    Returns a dict with:
      - shap_values: (5, n_raw_features) array
      - pred_class:  integer
      - feature_names: list
      - node_idx:    int
      - probs:       (5,) probability array
    """
    X_aug   = GraphRiskModel.augment_features(X, adj_norm)
    X_row   = X_aug[node_idx : node_idx + 1]

    explainer   = shap.TreeExplainer(model.estimator)
    shap_values = explainer.shap_values(X_row, check_additivity=False)

    probs      = model.predict_proba(X, adj_norm)
    pred_class = int(probs[node_idx].argmax())

    raw_n  = len(feature_names)
    sv     = _normalize_shap_values(shap_values, len(model.estimator.classes_), X_aug.shape[1])

    # Align to 5 classes
    sv_aligned = np.zeros((5, sv.shape[1]), dtype=np.float32)
    for i, cls in enumerate(model.estimator.classes_):
        sv_aligned[int(cls)] = sv[i]

    # Collapse back to original features
    if sv_aligned.shape[1] >= raw_n * 3:
        sv_collapsed = (
            sv_aligned[:, :raw_n]
            + sv_aligned[:, raw_n : 2 * raw_n]
            + sv_aligned[:, 2 * raw_n : 3 * raw_n]
        )
    else:
        sv_collapsed = sv_aligned[:, :raw_n]

    return {
        "shap_values":   sv_collapsed,
        "pred_class":    pred_class,
        "feature_names": feature_names,
        "node_idx":      node_idx,
        "probs":         probs[node_idx],
    }


# ------------------------------------------------------------------
# Visualisation helpers
# ------------------------------------------------------------------

def plot_shap_bar(shap_result: dict, top_n: int = 12) -> bytes:
    """Horizontal bar chart of SHAP values for the predicted class."""
    pred_cls   = shap_result["pred_class"]
    feat_names = shap_result["feature_names"]
    sv         = shap_result["shap_values"][pred_cls]
    n          = min(top_n, len(feat_names), len(sv))
    idx        = np.argsort(np.abs(sv))[::-1][:n]

    names  = [feat_names[i] for i in idx][::-1]
    vals   = sv[idx][::-1]
    colors = ["#F97316" if v > 0 else "#38BDF8" for v in vals]

    fig, ax = plt.subplots(figsize=(8.5, max(4, n * 0.45)))
    fig.patch.set_facecolor("#08131f")
    ax.set_facecolor("#08131f")
    ax.barh(names, vals, color=colors, edgecolor="none", height=0.65)
    ax.axvline(0, color="#E2E8F0", linewidth=1, alpha=0.6)
    ax.set_title(
        f"SHAP — Why {LABEL_NAMES.get(pred_cls, pred_cls)} was predicted",
        color="white", pad=12, fontsize=13,
    )
    ax.set_xlabel("Impact on predicted risk score", color="#D1FAE5", fontsize=10)
    ax.tick_params(colors="#E5F7EE", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1F2937")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_global_shap(
    model: GraphRiskModel,
    X: np.ndarray,
    adj_norm: np.ndarray,
    feature_names: list[str],
    top_n: int = 12,
) -> bytes:
    """Global SHAP importance averaged over all zones."""
    X_aug       = GraphRiskModel.augment_features(X, adj_norm)
    explainer   = shap.TreeExplainer(model.estimator)
    shap_values = explainer.shap_values(X_aug, check_additivity=False)

    sv     = _normalize_shap_values(shap_values, len(model.estimator.classes_), X_aug.shape[1])
    sv_aligned = np.zeros((5, sv.shape[1]), dtype=np.float32)
    for i, cls in enumerate(model.estimator.classes_):
        sv_aligned[int(cls)] = sv[i]

    raw_n = len(feature_names)
    if sv_aligned.shape[1] >= raw_n * 3:
        sv_c = (
            sv_aligned[:, :raw_n]
            + sv_aligned[:, raw_n : 2 * raw_n]
            + sv_aligned[:, 2 * raw_n : 3 * raw_n]
        )
    else:
        sv_c = sv_aligned[:, :raw_n]

    imp = np.mean(np.abs(sv_c), axis=0)
    idx = np.argsort(imp)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)))
    fig.patch.set_facecolor("#08131f")
    ax.set_facecolor("#08131f")
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        imp[idx][::-1],
        color="#55D6BE", edgecolor="none",
    )
    ax.set_title("Global Feature Importance (SHAP)", color="white", pad=12, fontsize=13)
    ax.set_xlabel("Mean |SHAP value|", color="#D1FAE5", fontsize=10)
    ax.tick_params(colors="#E5F7EE", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1F2937")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()