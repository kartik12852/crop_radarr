"""
train_model.py
Train the runtime-safe graph-aware model.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.graph_builder import build_graph
from src.paths import HISTORY_PATH, META_PATH, METRICS_PATH, MODEL_PATH
from src.trainer import RiskTrainer


def main():
    print("=" * 62)
    print("  Climate Radar — graph-aware model training")
    print("=" * 62)
    X, adj_norm, adj_mask, y, df, feat_cols = build_graph(fit_scaler=True, fallback=True)
    print(f"Loaded {len(df)} zones | {len(feat_cols)} features | classes: {sorted(set(y.tolist()))}")

    trainer = RiskTrainer(random_state=42)
    model, metrics, history, report, idx_train, idx_val, preds, probs = trainer.train(X, adj_norm, y, feat_cols)

    print("\nValidation metrics")
    for k, v in metrics.items():
        print(f"  {k:14s}: {v}")
    print("\nClassification report\n")
    print(report)
    print("Saved artefacts:")
    print(f"  model   -> {MODEL_PATH}")
    print(f"  metrics -> {METRICS_PATH}")
    print(f"  history -> {HISTORY_PATH}")
    print(f"  meta    -> {META_PATH}")
    print("\nNext: streamlit run app.py")


if __name__ == "__main__":
    main()
