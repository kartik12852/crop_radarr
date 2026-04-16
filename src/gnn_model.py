"""
src/gnn_model.py
Graph-aware risk model using a RandomForest ensemble with adjacency-smoothed features.

Design choice: We use scikit-learn's RandomForest (not PyTorch) to avoid
Windows DLL crashes with torch/fbgemm, while still capturing spatial
graph structure via neighbourhood feature propagation.

How it works (graph-awareness explained):
1. For each node (zone), we compute:
   - Raw features (X)
   - Smoothed features: A_norm @ X  (each zone absorbs neighbour info)
   - Difference features: X - (A_norm @ X)  (how different is this zone?)
2. These three views are concatenated → 3× feature space
3. A RandomForest learns from all three views simultaneously
4. This mimics one layer of a Graph Convolutional Network (GCN)
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


DEFAULT_RANDOM_STATE = 42


def build_norm_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Compute the symmetrically normalised adjacency matrix: D^{-1/2} A D^{-1/2}
    with self-loops added (standard GCN normalisation).

    Args:
        adj: Binary adjacency matrix (n x n), no self-loops needed.

    Returns:
        Normalised adjacency as float32 array.
    """
    adj = np.asarray(adj, dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square 2-D array.")
    a = adj.copy()
    np.fill_diagonal(a, 1.0)                           # self-loops
    deg = np.clip(a.sum(axis=1), 1.0, None)            # degree vector
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))           # D^{-1/2}
    return (d_inv_sqrt @ a @ d_inv_sqrt).astype(np.float32)


class GraphRiskModel:
    """
    Graph-enhanced RandomForest for multi-class climate risk prediction.

    Risk classes:
        0 = Safe
        1 = Drought
        2 = Heat Stress
        3 = Flood
        4 = Soil Risk

    The model accepts an optional normalised adjacency matrix and augments
    node features with one-hop neighbourhood aggregation, enabling it to
    learn spatially-aware risk patterns without a deep learning framework.
    """

    def __init__(
        self,
        n_estimators: int = 320,
        max_depth: int = 12,
        min_samples_leaf: int = 1,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.random_state      = random_state
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_leaf  = min_samples_leaf
        self.estimator = RandomForestClassifier(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            min_samples_leaf = min_samples_leaf,
            random_state     = random_state,
            n_jobs           = -1,
        )
        self.n_classes: int               = 5
        self.embedding_pca: PCA | None    = None
        self.feature_names: list[str]     = []
        self.feature_importances_: np.ndarray | None = None
        self.is_fitted_: bool             = False

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def augment_features(X: np.ndarray, adj_norm: np.ndarray | None = None) -> np.ndarray:
        """
        Concatenate raw features, neighbour-smoothed features, and residuals.

        If adj_norm is None (no graph), returns X unchanged.
        """
        X = np.asarray(X, dtype=np.float32)
        if adj_norm is None:
            return X
        smoothed = np.asarray(adj_norm, dtype=np.float32) @ X  # 1-hop aggregation
        diff     = X - smoothed                                  # zone vs neighbourhood
        return np.hstack([X, smoothed, diff]).astype(np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        adj_norm: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "GraphRiskModel":
        X     = np.asarray(X, dtype=np.float32)
        y     = np.asarray(y, dtype=int)
        X_aug = self.augment_features(X, adj_norm)

        self.estimator.fit(X_aug, y)
        self.is_fitted_    = True
        self.feature_names = list(feature_names or [])

        # Collapse augmented importances back to original feature space
        raw_n    = X.shape[1]
        full_imp = np.asarray(self.estimator.feature_importances_, dtype=np.float32)
        if full_imp.size >= raw_n * 3:
            self.feature_importances_ = (
                full_imp[:raw_n]
                + full_imp[raw_n : 2 * raw_n]
                + full_imp[2 * raw_n : 3 * raw_n]
            )
        else:
            self.feature_importances_ = full_imp[:raw_n]

        # PCA embedding for visualisation
        n_comp = max(2, min(8, X_aug.shape[0] - 1, X_aug.shape[1]))
        self.embedding_pca = PCA(n_components=n_comp, random_state=self.random_state)
        self.embedding_pca.fit(X_aug)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray, adj_norm: np.ndarray | None = None) -> np.ndarray:
        """Returns (n_samples, 5) probability matrix, always with 5 columns."""
        X_aug = self.augment_features(X, adj_norm)
        raw   = self.estimator.predict_proba(X_aug)

        # Align to full 5-class output even if some classes absent in training
        aligned = np.zeros((len(X_aug), self.n_classes), dtype=np.float32)
        for i, cls in enumerate(self.estimator.classes_):
            aligned[:, int(cls)] = raw[:, i]

        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return aligned / row_sums

    def predict(self, X: np.ndarray, adj_norm: np.ndarray | None = None) -> np.ndarray:
        return self.predict_proba(X, adj_norm).argmax(axis=1)

    def get_embeddings(self, X: np.ndarray, adj_norm: np.ndarray | None = None) -> np.ndarray:
        """Return low-dimensional embedding for each zone (used in graph explorer)."""
        X_aug = self.augment_features(X, adj_norm)
        if self.embedding_pca is None:
            n_comp = max(2, min(8, X_aug.shape[0] - 1, X_aug.shape[1]))
            self.embedding_pca = PCA(n_components=n_comp, random_state=self.random_state)
            self.embedding_pca.fit(X_aug)
        return self.embedding_pca.transform(X_aug)