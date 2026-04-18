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
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.gnn_model import GraphRiskModel
from src.paths import HISTORY_PATH, META_PATH, METRICS_PATH, MODEL_PATH

LABELS = ["Safe", "Drought", "Heat Stress", "Flood", "Soil Risk"]


class RiskTrainer:
    """Orchestrates training, validation, metric logging, and artefact saving."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model        = GraphRiskModel(random_state=random_state)

    # ------------------------------------------------------------------
    # Validation strategy
    # ------------------------------------------------------------------

    def _choose_eval_strategy(self, n: int, y: np.ndarray):
        """
        Choose evaluation strategy based on dataset size.
        - n < 60  → 5-fold cross-validation (too few samples for a clean split)
        - n >= 60 → 75/25 stratified train/val split
        Returns: ("cv", n_folds) or ("split", None)
        """
        classes, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        # Need at least 2 samples per class per fold
        if n < 60 or min_count < 10:
            n_folds = min(5, int(min_count))
            n_folds = max(2, n_folds)
            return "cv", n_folds
        return "split", None

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
        1. Choose evaluation strategy (CV for small datasets, split for large)
        2. Warm-start loop to build learning-curve history
        3. Full refit on complete dataset (best model for inference)
        4. Save model, metrics, history, and meta artefacts

        Returns:
            model, metrics, history, report, idx_train, idx_val, preds, probs
        """
        n = len(y)
        strategy, n_folds = self._choose_eval_strategy(n, y)
        print(f"  Evaluation strategy: {'%d-fold CV' % n_folds if strategy == 'cv' else 'train/val split'} (n={n})")

        # ── Split / CV setup ──────────────────────────────────────────
        if strategy == "split":
            idx = np.arange(n)
            idx_train, idx_val = train_test_split(
                idx, test_size=0.25, stratify=y, random_state=self.random_state
            )
            X_train, X_val = X[idx_train], X[idx_val]
            y_train, y_val = y[idx_train], y[idx_val]
            A_train = adj_norm[np.ix_(idx_train, idx_train)] if adj_norm is not None else None
            A_val   = adj_norm[np.ix_(idx_val,   idx_val)]   if adj_norm is not None else None
            X_train_aug = GraphRiskModel.augment_features(X_train, A_train)
            X_val_aug   = GraphRiskModel.augment_features(X_val,   A_val)
        else:
            # For small datasets: use CV for metrics, but train on ALL data for final model
            idx_train = np.arange(n)
            idx_val   = np.arange(n)      # will be replaced below
            # Full-augmented for CV
            X_aug_full  = GraphRiskModel.augment_features(X, adj_norm)
            cv_preds_all= np.zeros(n, dtype=int)
            cv_probs_all= np.zeros((n, 5), dtype=float)
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            for fold_train_idx, fold_val_idx in skf.split(X_aug_full, y):
                tmp = GraphRiskModel(random_state=self.random_state)
                tmp.fit(
                    X[fold_train_idx], y[fold_train_idx],
                    adj_norm=adj_norm[np.ix_(fold_train_idx, fold_train_idx)],
                    feature_names=feature_names,
                )
                fold_preds = tmp.predict(X[fold_val_idx], adj_norm[np.ix_(fold_val_idx, fold_val_idx)])
                fold_probs = tmp.predict_proba(X[fold_val_idx], adj_norm[np.ix_(fold_val_idx, fold_val_idx)])
                cv_preds_all[fold_val_idx] = fold_preds
                cv_probs_all[fold_val_idx] = fold_probs
            y_val         = y
            preds_for_eval= cv_preds_all
            probs_for_eval= cv_probs_all
            # For history, use a 80/20 proxy split (just for learning curve)
            idx_sp_train, idx_val = train_test_split(
                np.arange(n), test_size=0.20, stratify=y, random_state=self.random_state
            )
            idx_train     = idx_sp_train
            X_train_aug   = GraphRiskModel.augment_features(X[idx_train], adj_norm[np.ix_(idx_train, idx_train)])
            X_val_aug     = GraphRiskModel.augment_features(X[idx_val],   adj_norm[np.ix_(idx_val,   idx_val)])
            y_train, y_val_h = y[idx_train], y[idx_val]

        # ── Learning-curve history ─────────────────────────────────────
        base    = clone(self.model.estimator)
        base.set_params(warm_start=True, oob_score=False)
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "estimators": []}

        step = 20
        for n_trees in range(step, self.model.n_estimators + 1, step):
            base.set_params(n_estimators=n_trees)
            base.fit(X_train_aug, y_train if strategy == "split" else y[idx_train])

            tr_prob = base.predict_proba(X_train_aug)
            vl_prob = base.predict_proba(X_val_aug)
            vl_pred = vl_prob.argmax(axis=1)
            y_val_h2 = y_val if strategy == "split" else y[idx_val]

            history["estimators"].append(n_trees)
            history["train_loss"].append(
                float(log_loss(y_train if strategy == "split" else y[idx_train],
                               tr_prob, labels=base.classes_))
            )
            history["val_loss"].append(
                float(log_loss(y_val_h2, vl_prob, labels=base.classes_))
            )
            history["val_acc"].append(float(accuracy_score(y_val_h2, vl_pred)))

        # ── Full refit on ALL data ────────────────────────────────────
        self.model.fit(X, y, adj_norm=adj_norm, feature_names=feature_names)

        # For split strategy: evaluate on val partition
        if strategy == "split":
            preds_for_eval = self.model.predict(X_val, A_val)
            probs_for_eval = self.model.predict_proba(X_val, A_val)
            y_val          = y[idx_val]

        # ── Metrics ────────────────────────────────────────────────────
        present = sorted(set(y_val.tolist()) | set(preds_for_eval.tolist()))
        acc     = round(float(accuracy_score(y_val, preds_for_eval)), 4)

        # Warn clearly if performance looks suspiciously perfect
        overfit_warning = None
        if acc >= 0.99 and n <= 60:
            overfit_warning = (
                f"⚠️  Metrics are {acc:.0%} — this is likely due to the small training dataset "
                f"({n} zones). Fetch real multi-day weather data for more realistic evaluation."
            )
            print(f"\n  {overfit_warning}\n")

        metrics = {
            "accuracy":        acc,
            "precision":       round(float(precision_score(y_val, preds_for_eval, average="weighted", zero_division=0)), 4),
            "recall":          round(float(recall_score(y_val, preds_for_eval, average="weighted", zero_division=0)), 4),
            "f1_score":        round(float(f1_score(y_val, preds_for_eval, average="weighted", zero_division=0)), 4),
            "roc_auc":         None,
            "n_samples":       int(n),
            "eval_strategy":   f"{n_folds}-fold CV" if strategy == "cv" else "train/val split (75/25)",
            "overfit_warning": overfit_warning,
            "model_family":    "Graph-aware RandomForest (GCN-inspired)",
            "runtime":         "Pure scikit-learn — no PyTorch dependency",
            "n_zones":         int(len(y)),
            "n_features":      int(len(feature_names)),
        }
        try:
            metrics["roc_auc"] = round(
                float(roc_auc_score(
                    y_val, probs_for_eval,
                    multi_class="ovr", average="weighted",
                    labels=list(range(5))
                )), 4
            )
        except Exception:
            metrics["roc_auc"] = None

        report = classification_report(
            y_val, preds_for_eval,
            labels=present,
            target_names=[LABELS[i] for i in present],
            zero_division=0,
        )

        # ── Save artefacts ─────────────────────────────────────────────
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
                "eval_strategy": metrics["eval_strategy"],
            }, f, indent=2)

        return self.model, metrics, history, report, idx_train, idx_val, preds_for_eval, probs_for_eval
