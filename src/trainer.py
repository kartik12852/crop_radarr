"""
src/trainer.py
Training pipeline for the graph-aware ensemble model.
Handles warm-start incremental training for the learning-curve history.
"""

from __future__ import annotations

import json
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.gnn_model import GraphRiskModel
from src.paths import HISTORY_PATH, META_PATH, METRICS_PATH, MODEL_PATH

LABELS = ["Safe", "Drought", "Heat Stress", "Flood", "Soil Risk"]


class RiskTrainer:
    """Orchestrates training, validation, metric logging, and artefact saving."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model        = GraphRiskModel(random_state=random_state)

    def train(
        self,
        X: np.ndarray,
        adj_norm: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ):
        """
        Full training + evaluation pipeline.

        Steps:
        1. Stratified train/val split (75 / 25)
        2. Warm-start loop to build learning-curve history
        3. Full refit on complete dataset (best model for inference)
        4. Save model, metrics, history, and meta artefacts

        Returns:
            model, metrics, history, report, idx_train, idx_val, preds, probs
        """
        # ── 1. Split ──────────────────────────────────────────────────
        idx = np.arange(len(y))
        idx_train, idx_val = train_test_split(
            idx, test_size=0.25, stratify=y, random_state=self.random_state
        )
        X_train, X_val = X[idx_train], X[idx_val]
        y_train, y_val = y[idx_train], y[idx_val]
        A_train = adj_norm[np.ix_(idx_train, idx_train)] if adj_norm is not None else None
        A_val   = adj_norm[np.ix_(idx_val,   idx_val)]   if adj_norm is not None else None

        X_train_aug = GraphRiskModel.augment_features(X_train, A_train)
        X_val_aug   = GraphRiskModel.augment_features(X_val,   A_val)

        # ── 2. Learning-curve history ─────────────────────────────────
        base    = clone(self.model.estimator)
        base.set_params(warm_start=True, oob_score=False)
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "estimators": []}

        step = 20
        for n_trees in range(step, self.model.n_estimators + 1, step):
            base.set_params(n_estimators=n_trees)
            base.fit(X_train_aug, y_train)

            tr_prob = base.predict_proba(X_train_aug)
            vl_prob = base.predict_proba(X_val_aug)
            vl_pred = vl_prob.argmax(axis=1)

            history["estimators"].append(n_trees)
            history["train_loss"].append(
                float(log_loss(y_train, tr_prob, labels=base.classes_))
            )
            history["val_loss"].append(
                float(log_loss(y_val, vl_prob, labels=base.classes_))
            )
            history["val_acc"].append(float(accuracy_score(y_val, vl_pred)))

        # ── 3. Full refit ─────────────────────────────────────────────
        self.model.fit(X, y, adj_norm=adj_norm, feature_names=feature_names)

        # Evaluate on validation partition
        preds = self.model.predict(X_val, A_val)
        probs = self.model.predict_proba(X_val, A_val)

        # ── 4. Metrics ────────────────────────────────────────────────
        present = sorted(set(y_val.tolist()) | set(preds.tolist()))
        metrics = {
            "accuracy":     round(float(accuracy_score(y_val, preds)), 4),
            "precision":    round(float(precision_score(y_val, preds, average="weighted", zero_division=0)), 4),
            "recall":       round(float(recall_score(y_val, preds, average="weighted", zero_division=0)), 4),
            "f1_score":     round(float(f1_score(y_val, preds, average="weighted", zero_division=0)), 4),
            "roc_auc":      None,
            "model_family": "Graph-aware RandomForest (GCN-inspired)",
            "runtime":      "Pure scikit-learn — no PyTorch dependency",
            "n_zones":      int(len(y)),
            "n_features":   int(len(feature_names)),
        }
        try:
            metrics["roc_auc"] = round(
                float(roc_auc_score(y_val, probs, multi_class="ovr", average="weighted", labels=list(range(5)))),
                4,
            )
        except Exception:
            metrics["roc_auc"] = None

        report = classification_report(
            y_val, preds,
            labels=present,
            target_names=[LABELS[i] for i in present],
            zero_division=0,
        )

        # ── 5. Save artefacts ─────────────────────────────────────────
        joblib.dump(self.model, MODEL_PATH)

        with open(METRICS_PATH,  "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(HISTORY_PATH,  "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(META_PATH,     "w", encoding="utf-8") as f:
            json.dump({
                "feature_cols": feature_names,
                "n_features":   len(feature_names),
                "n_samples":    int(len(y)),
                "model_family": "Graph-aware RandomForest",
            }, f, indent=2)

        return self.model, metrics, history, report, idx_train, idx_val, preds, probs