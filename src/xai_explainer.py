"""
src/xai_explainer.py
SHAP explainability for the graph-aware ensemble model.
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



def _normalize_shap_values(shap_values, n_classes: int, n_features: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.stack([np.asarray(v)[0] for v in shap_values], axis=0)
    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return arr[0][None, :]
    if arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[1] == n_features:
            return np.moveaxis(arr[0], -1, 0)
        if arr.shape[0] == n_classes:
            return arr[:, 0, :]
        if arr.shape[-1] == n_classes:
            return np.moveaxis(arr[0], -1, 0)
    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")



def explain_node(model: GraphRiskModel, X: np.ndarray, adj_norm: np.ndarray, node_idx: int, feature_names: list[str]) -> dict:
    X_aug = GraphRiskModel.augment_features(X, adj_norm)
    X_row = X_aug[node_idx:node_idx + 1]
    explainer = shap.TreeExplainer(model.estimator)
    shap_values = explainer.shap_values(X_row, check_additivity=False)
    probs = model.predict_proba(X, adj_norm)
    pred_class = int(probs[node_idx].argmax())

    raw_feature_count = len(feature_names)
    sv = _normalize_shap_values(shap_values, len(model.estimator.classes_), X_aug.shape[1])
    sv_aligned = np.zeros((5, sv.shape[1]), dtype=np.float32)
    for i, cls in enumerate(model.estimator.classes_):
        sv_aligned[int(cls)] = sv[i]
    if sv_aligned.shape[1] >= raw_feature_count * 3:
        sv_collapsed = sv_aligned[:, :raw_feature_count] + sv_aligned[:, raw_feature_count:2 * raw_feature_count] + sv_aligned[:, 2 * raw_feature_count:3 * raw_feature_count]
    else:
        sv_collapsed = sv_aligned[:, :raw_feature_count]

    return {
        "shap_values": sv_collapsed,
        "pred_class": pred_class,
        "feature_names": feature_names,
        "node_idx": node_idx,
        "probs": probs[node_idx],
    }



def plot_shap_bar(shap_result: dict, top_n: int = 12) -> bytes:
    pred_cls = shap_result["pred_class"]
    feat_names = shap_result["feature_names"]
    sv = shap_result["shap_values"][pred_cls]
    idx = np.argsort(np.abs(sv))[::-1][:top_n]
    names = [feat_names[i] for i in idx][::-1]
    vals = sv[idx][::-1]
    colors = ["#F97316" if v > 0 else "#38BDF8" for v in vals]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor("#08131f")
    ax.set_facecolor("#08131f")
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#E2E8F0", linewidth=1)
    ax.set_title(f"SHAP explanation — predicted {LABEL_NAMES.get(pred_cls, pred_cls)}", color="white", pad=12)
    ax.set_xlabel("Impact on predicted risk score", color="#D1FAE5")
    ax.tick_params(colors="#E5F7EE")
    for spine in ax.spines.values():
        spine.set_color("#1F2937")
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()



def plot_global_shap(model: GraphRiskModel, X: np.ndarray, adj_norm: np.ndarray, feature_names: list[str], top_n: int = 12) -> bytes:
    X_aug = GraphRiskModel.augment_features(X, adj_norm)
    explainer = shap.TreeExplainer(model.estimator)
    shap_values = explainer.shap_values(X_aug, check_additivity=False)
    probs = model.predict_proba(X, adj_norm)
    sv = _normalize_shap_values(shap_values, len(model.estimator.classes_), X_aug.shape[1])
    sv_aligned = np.zeros((5, sv.shape[1]), dtype=np.float32)
    for i, cls in enumerate(model.estimator.classes_):
        sv_aligned[int(cls)] = sv[i]
    raw_feature_count = len(feature_names)
    if sv_aligned.shape[1] >= raw_feature_count * 3:
        sv = sv_aligned[:, :raw_feature_count] + sv_aligned[:, raw_feature_count:2 * raw_feature_count] + sv_aligned[:, 2 * raw_feature_count:3 * raw_feature_count]
    else:
        sv = sv_aligned[:, :raw_feature_count]
    imp = np.mean(np.abs(sv), axis=(0,))
    idx = np.argsort(imp)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("#08131f")
    ax.set_facecolor("#08131f")
    ax.barh([feature_names[i] for i in idx][::-1], imp[idx][::-1], color="#55D6BE")
    ax.set_title("Global SHAP importance", color="white", pad=12)
    ax.set_xlabel("Mean |SHAP|", color="#D1FAE5")
    ax.tick_params(colors="#E5F7EE")
    for spine in ax.spines.values():
        spine.set_color("#1F2937")
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
