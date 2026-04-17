"""
src/xai_explainer.py
SHAP (SHapley Additive exPlanations) wrapper for the graph-aware model,
with a fully functional sklearn-based fallback when shap is not installed.

Fallback explanation (no shap required):
  - Uses RandomForest leaf-node impurity decrease to approximate local importance
  - Weights each feature's importance by how extreme the zone's value is vs. the mean
  - Produces a directional "push" score similar to SHAP

This means the XAI panel ALWAYS works, even without pip install shap.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.gnn_model import GraphRiskModel
from src.model_utils import LABEL_NAMES

# ── Check SHAP availability ───────────────────────────────────────────────────
try:
    import shap as _shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_shap_values(shap_values, n_classes: int, n_aug_features: int) -> np.ndarray:
    """Normalize SHAP output to shape (n_classes, n_aug_features)."""
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


# ── Fallback local importance (no SHAP) ──────────────────────────────────────

def _local_importance_fallback(
    model: GraphRiskModel,
    X: np.ndarray,
    adj_norm: np.ndarray,
    node_idx: int,
    feature_names: list[str],
) -> np.ndarray:
    """
    Approximate per-class local feature importance without SHAP.

    Method:
      1. Global feature importance from RF (already computed)
      2. Multiply by sign * magnitude of each feature vs the training mean
         → features far from average in the risk-relevant direction get higher scores
      3. Return (5, n_features) signed importance array
    """
    X_aug    = GraphRiskModel.augment_features(X, adj_norm)
    x_zone   = X_aug[node_idx]           # augmented row for this zone
    x_mean   = X_aug.mean(axis=0)        # training mean
    x_std    = X_aug.std(axis=0) + 1e-8  # std for normalisation

    # Deviation of this zone from the mean (in standard deviations)
    deviation = (x_zone - x_mean) / x_std    # shape (n_aug,)

    # Global feature importance per class: proxy via forest class probabilities
    probs   = model.predict_proba(X, adj_norm)   # (n_zones, 5)
    raw_n   = len(feature_names)

    # For each class, weight global feature importance by zone deviation
    global_imp = np.asarray(model.estimator.feature_importances_, dtype=np.float32)

    # Build aligned (5, n_aug_features) importance
    sv_per_class = np.zeros((5, global_imp.size), dtype=np.float32)
    for cls in range(5):
        # Scale global importance by class probability difference vs base rate
        class_prob = float(probs[node_idx, cls])
        base_rate  = float(probs[:, cls].mean())
        scale      = class_prob - base_rate      # positive if zone is elevated risk
        sv_per_class[cls] = global_imp * deviation * scale

    # Collapse augmented views back to original feature count
    if sv_per_class.shape[1] >= raw_n * 3:
        sv_collapsed = (
            sv_per_class[:, :raw_n]
            + sv_per_class[:, raw_n : 2 * raw_n]
            + sv_per_class[:, 2 * raw_n : 3 * raw_n]
        )
    else:
        sv_collapsed = sv_per_class[:, :raw_n]

    return sv_collapsed     # (5, n_features)


# ── Per-node explanation ──────────────────────────────────────────────────────

def explain_node(
    model: GraphRiskModel,
    X: np.ndarray,
    adj_norm: np.ndarray,
    node_idx: int,
    feature_names: list[str],
    use_shap: bool = True,
) -> dict:
    """
    Compute feature-importance explanation for one zone node.

    Uses SHAP TreeExplainer if available (and use_shap=True), otherwise falls
    back to the local-importance approximation which always works.

    Returns:
        shap_values:   (5, n_raw_features) array — positive = pushes toward that risk
        pred_class:    int
        feature_names: list
        node_idx:      int
        probs:         (5,) probability array
        method:        "shap" | "local_importance"
    """
    probs      = model.predict_proba(X, adj_norm)
    pred_class = int(probs[node_idx].argmax())
    raw_n      = len(feature_names)

    if use_shap and SHAP_AVAILABLE:
        try:
            X_aug    = GraphRiskModel.augment_features(X, adj_norm)
            X_row    = X_aug[node_idx : node_idx + 1]
            explainer   = _shap.TreeExplainer(model.estimator)
            shap_values = explainer.shap_values(X_row, check_additivity=False)

            sv      = _normalize_shap_values(
                shap_values, len(model.estimator.classes_), X_aug.shape[1]
            )
            sv_aligned = np.zeros((5, sv.shape[1]), dtype=np.float32)
            for i, cls in enumerate(model.estimator.classes_):
                sv_aligned[int(cls)] = sv[i]

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
                "method":        "shap",
            }
        except Exception:
            pass   # fall through to fallback

    # ── Fallback ──────────────────────────────────────────────────────────────
    sv_collapsed = _local_importance_fallback(model, X, adj_norm, node_idx, feature_names)
    return {
        "shap_values":   sv_collapsed,
        "pred_class":    pred_class,
        "feature_names": feature_names,
        "node_idx":      node_idx,
        "probs":         probs[node_idx],
        "method":        "local_importance",
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_shap_bar(shap_result: dict, top_n: int = 12) -> bytes:
    """Horizontal bar chart of SHAP / local-importance values for the predicted class."""
    pred_cls   = shap_result["pred_class"]
    feat_names = shap_result["feature_names"]
    sv         = shap_result["shap_values"][pred_cls]
    method     = shap_result.get("method", "shap")
    n          = min(top_n, len(feat_names), len(sv))
    idx        = np.argsort(np.abs(sv))[::-1][:n]

    names  = [feat_names[i] for i in idx][::-1]
    vals   = sv[idx][::-1]
    colors = ["#F97316" if v > 0 else "#38BDF8" for v in vals]

    title_prefix = "SHAP" if method == "shap" else "Feature Impact"
    label_name   = LABEL_NAMES.get(pred_cls, str(pred_cls))

    fig, ax = plt.subplots(figsize=(8.5, max(4, n * 0.45)))
    fig.patch.set_facecolor("#08131f")
    ax.set_facecolor("#08131f")
    ax.barh(names, vals, color=colors, edgecolor="none", height=0.65)
    ax.axvline(0, color="#E2E8F0", linewidth=1, alpha=0.6)
    ax.set_title(
        f"{title_prefix} — Why '{label_name}' was predicted",
        color="white", pad=12, fontsize=13,
    )
    xlabel = ("SHAP value (impact on model output)"
              if method == "shap"
              else "Feature impact score (orange=risk-increasing, blue=risk-reducing)")
    ax.set_xlabel(xlabel, color="#D1FAE5", fontsize=9)
    ax.tick_params(colors="#E5F7EE", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1F2937")

    if method != "shap":
        ax.text(
            0.99, 0.01,
            "📊 Approximation (install shap for exact values)",
            transform=ax.transAxes,
            color="#A7F3D0", fontsize=7.5, ha="right", va="bottom",
        )
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
    """Global feature importance — SHAP if available, else RF importances."""
    if SHAP_AVAILABLE:
        try:
            X_aug       = GraphRiskModel.augment_features(X, adj_norm)
            explainer   = _shap.TreeExplainer(model.estimator)
            shap_values = explainer.shap_values(X_aug, check_additivity=False)
            sv          = _normalize_shap_values(
                shap_values, len(model.estimator.classes_), X_aug.shape[1]
            )
            sv_aligned  = np.zeros((5, sv.shape[1]), dtype=np.float32)
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
            imp    = np.mean(np.abs(sv_c), axis=0)
            method = "SHAP"
        except Exception:
            imp    = np.asarray(model.feature_importances_, dtype=np.float32)
            method = "RF Importance"
    else:
        imp    = np.asarray(model.feature_importances_, dtype=np.float32) if model.feature_importances_ is not None else np.ones(len(feature_names))
        method = "RF Importance"

    idx  = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)))
    fig.patch.set_facecolor("#08131f")
    ax.set_facecolor("#08131f")
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        imp[idx][::-1],
        color="#55D6BE", edgecolor="none",
    )
    ax.set_title(f"Global Feature Importance ({method})", color="white", pad=12, fontsize=13)
    ax.set_xlabel("Mean |importance|", color="#D1FAE5", fontsize=10)
    ax.tick_params(colors="#E5F7EE", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1F2937")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
