"""
src/gnn_model.py
Runtime-safe graph-aware risk model.

Why this file keeps the old name:
- Existing app imports already point to src.gnn_model
- Renaming the module would create more churn for the Streamlit app

Implementation detail:
- Uses a graph-enhanced RandomForest classifier instead of PyTorch
- No torch dependency, so it avoids the Windows fbgemm.dll crash
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


DEFAULT_RANDOM_STATE = 42


def build_norm_adjacency(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    a = adj.copy()
    np.fill_diagonal(a, 1.0)
    deg = np.clip(a.sum(axis=1), 1.0, None)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return (d_inv_sqrt @ a @ d_inv_sqrt).astype(np.float32)


class GraphRiskModel:
    def __init__(
        self,
        n_estimators: int = 320,
        max_depth: int = 12,
        min_samples_leaf: int = 1,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=None,
            random_state=random_state,
            n_jobs=-1,
        )
        self.n_classes = 5
        self.embedding_pca: PCA | None = None
        self.feature_names: list[str] = []
        self.feature_importances_: np.ndarray | None = None
        self.is_fitted_: bool = False

    @staticmethod
    def augment_features(X: np.ndarray, adj_norm: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if adj_norm is None:
            return X
        smoothed = np.asarray(adj_norm, dtype=np.float32) @ X
        diff = X - smoothed
        return np.hstack([X, smoothed, diff]).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, adj_norm: np.ndarray | None = None, feature_names: list[str] | None = None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        X_aug = self.augment_features(X, adj_norm)
        self.estimator.fit(X_aug, y)
        self.is_fitted_ = True
        self.feature_names = list(feature_names or [])
        raw_n = X.shape[1]
        full_imp = np.asarray(self.estimator.feature_importances_, dtype=np.float32)
        if full_imp.size >= raw_n * 3:
            self.feature_importances_ = full_imp[:raw_n] + full_imp[raw_n:2 * raw_n] + full_imp[2 * raw_n:3 * raw_n]
        else:
            self.feature_importances_ = full_imp[:raw_n]
        n_components = max(2, min(8, X_aug.shape[0] - 1, X_aug.shape[1]))
        self.embedding_pca = PCA(n_components=n_components, random_state=self.random_state)
        self.embedding_pca.fit(X_aug)
        return self

    def predict_proba(self, X: np.ndarray, adj_norm: np.ndarray | None = None) -> np.ndarray:
        X_aug = self.augment_features(X, adj_norm)
        raw = self.estimator.predict_proba(X_aug)
        aligned = np.zeros((len(X_aug), self.n_classes), dtype=np.float32)
        for i, cls in enumerate(self.estimator.classes_):
            aligned[:, int(cls)] = raw[:, i]
        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return aligned / row_sums

    def predict(self, X: np.ndarray, adj_norm: np.ndarray | None = None) -> np.ndarray:
        return self.predict_proba(X, adj_norm).argmax(axis=1)

    def get_embeddings(self, X: np.ndarray, adj_norm: np.ndarray | None = None) -> np.ndarray:
        X_aug = self.augment_features(X, adj_norm)
        if self.embedding_pca is None:
            n_components = max(2, min(8, X_aug.shape[0] - 1, X_aug.shape[1]))
            self.embedding_pca = PCA(n_components=n_components, random_state=self.random_state)
            self.embedding_pca.fit(X_aug)
        return self.embedding_pca.transform(X_aug)
