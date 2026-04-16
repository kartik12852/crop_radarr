"""
doctor.py — Climate Radar health check.
Designed to fail gracefully and never crash on broken environments.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PASS = "✅ PASS"
WARN = "⚠️ WARN"
FAIL = "❌ FAIL"


def check_import(name, pip_name=None, optional=False):
    try:
        __import__(name)
        print(f"  {name:22s} {PASS}")
        return True
    except Exception as e:
        tag = WARN if optional else FAIL
        print(f"  {name:22s} {tag}  -> install {pip_name or name} | {type(e).__name__}: {e}")
        return optional


def check_file(path: Path, label: str, optional=False):
    ok = path.exists()
    tag = PASS if ok else (WARN if optional else FAIL)
    print(f"  {label:28s} {tag}  -> {path}")
    return ok or optional


def main():
    print("=" * 68)
    print("  Climate Radar Doctor — environment and project health check")
    print("=" * 68)

    print("\n[1] Python")
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro} {' ' + PASS if (v.major, v.minor) >= (3, 10) else ' ' + WARN}")

    print("\n[2] Required packages")
    required = [
        ("streamlit", "streamlit", False),
        ("pandas", "pandas", False),
        ("numpy", "numpy", False),
        ("sklearn", "scikit-learn", False),
        ("matplotlib", "matplotlib", False),
        ("seaborn", "seaborn", False),
        ("plotly", "plotly", False),
        ("networkx", "networkx", False),
        ("joblib", "joblib", False),
        ("requests", "requests", False),
        ("dateutil", "python-dateutil", False),
        ("shap", "shap", True),
        ("fpdf", "fpdf2", True),
        ("scipy", "scipy", True),
        ("tqdm", "tqdm", True),
    ]
    all_ok = True
    for mod, pip_name, optional in required:
        all_ok = check_import(mod, pip_name, optional) and all_ok

    print("\n[3] Core files")
    for rel in [
        "app.py", "train_model.py", "doctor.py",
        "src/paths.py", "src/gnn_model.py", "src/graph_builder.py", "src/trainer.py", "src/model_utils.py", "src/alert_engine.py", "src/xai_explainer.py", "src/recommendation.py",
        "utils/theme.py", "utils/visualizer.py", "utils/pdf_exporter.py",
        "pages/1_Risk_Map.py", "pages/2_GNN_Explorer.py", "pages/3_XAI_Panel.py", "pages/4_Recommendations.py", "pages/5_Reports.py",
        "synthetic/synthetic_data_generator.py", "data/fetch_data.py",
    ]:
        all_ok = check_file(Path(ROOT) / rel, rel) and all_ok

    from src.paths import MERGED_CSV, SYNTHETIC_CSV, MODEL_PATH, SCALER_PATH, METRICS_PATH

    print("\n[4] Data files")
    check_file(MERGED_CSV, "merged_zones.csv", optional=True)
    check_file(SYNTHETIC_CSV, "synthetic_zones.csv", optional=True)

    print("\n[5] Model artefacts")
    check_file(MODEL_PATH, "risk_model.joblib", optional=True)
    check_file(SCALER_PATH, "feature_scaler.pkl", optional=True)
    check_file(METRICS_PATH, "metrics.json", optional=True)

    print("\n[6] Runtime smoke test")
    try:
        import numpy as np
        from src.gnn_model import GraphRiskModel
        model = GraphRiskModel(n_estimators=20)
        X = np.random.randn(12, 6).astype("float32")
        A = np.random.rand(12, 12).astype("float32")
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        y = np.random.randint(0, 5, size=12)
        model.fit(X, y, adj_norm=A, feature_names=[f"f{i}" for i in range(6)])
        p = model.predict_proba(X, A)
        assert p.shape == (12, 5)
        print(f"  GraphRiskModel smoke test   {PASS}")
    except Exception as e:
        all_ok = False
        print(f"  GraphRiskModel smoke test   {FAIL}  {type(e).__name__}: {e}")

    print("\n" + "=" * 68)
    if all_ok:
        print("  Doctor summary: project is ready or very close to ready.")
    else:
        print("  Doctor summary: fix the FAIL lines, then run again.")
    print("\n  Recommended run order:")
    print("    1. conda activate crop_radar")
    print("    2. python doctor.py")
    print("    3. python synthetic/synthetic_data_generator.py   OR   python data/fetch_data.py")
    print("    4. python train_model.py")
    print("    5. streamlit run app.py")
    print("=" * 68)


if __name__ == "__main__":
    main()
