"""
doctor.py — Climate Radar health check.
Checks environment, packages, files, and runs a smoke test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PASS  = "✅ PASS"
WARN  = "⚠️ WARN"
FAIL  = "❌ FAIL"
INFO  = "ℹ️ INFO"


def check_import(name, pip_name=None, optional=False):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  {name:24s} {PASS}  v{ver}")
        return True
    except Exception as e:
        tag = WARN if optional else FAIL
        hint = f"pip install {pip_name or name}"
        print(f"  {name:24s} {tag}  -> {hint}")
        return optional


def check_file(path: Path, label: str, optional=False):
    ok  = path.exists()
    tag = PASS if ok else (WARN if optional else FAIL)
    print(f"  {label:32s} {tag}  {path}")
    return ok or optional


def main():
    all_ok = True
    print("=" * 72)
    print("  Climate Crop Radar — Environment & Project Health Check")
    print("=" * 72)

    # ── Python version ────────────────────────────────────────────────────────
    print("\n[1/6] Python version")
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    tag = PASS if ok else WARN
    print(f"  Python {v.major}.{v.minor}.{v.micro}  {tag}")
    if not ok:
        print("        Recommend Python 3.10+ for full compatibility.")
    all_ok = all_ok and ok

    # ── Required packages ─────────────────────────────────────────────────────
    print("\n[2/6] Python packages")
    packages = [
        # (import_name,   pip_install_name,   optional)
        ("streamlit",     "streamlit",          False),
        ("pandas",        "pandas",             False),
        ("numpy",         "numpy",              False),
        ("sklearn",       "scikit-learn",       False),
        ("matplotlib",    "matplotlib",         False),
        ("seaborn",       "seaborn",            False),
        ("plotly",        "plotly",             False),
        ("networkx",      "networkx",           False),
        ("joblib",        "joblib",             False),
        ("requests",      "requests",           False),
        ("dateutil",      "python-dateutil",    False),
        ("tqdm",          "tqdm",               False),
        ("shap",          "shap",               True),   # optional — fallback exists
        ("fpdf",          "fpdf2",              True),   # optional — for PDF export
        ("scipy",         "scipy",              True),
    ]
    for mod, pip, opt in packages:
        r = check_import(mod, pip, opt)
        if not opt:
            all_ok = all_ok and r

    # ── Core source files ─────────────────────────────────────────────────────
    print("\n[3/6] Core source files")
    core_files = [
        "app.py",
        "train_model.py",
        "run_update.py",
        "doctor.py",
        "src/__init__.py",
        "src/paths.py",
        "src/gnn_model.py",
        "src/graph_builder.py",
        "src/trainer.py",
        "src/model_utils.py",
        "src/alert_engine.py",
        "src/xai_explainer.py",
        "src/recommendation.py",
        "src/auto_updater.py",
        "utils/theme.py",
        "utils/visualizer.py",
        "utils/pdf_exporter.py",
        "pages/1_Risk_Map.py",
        "pages/2_GNN_Explorer.py",
        "pages/3_XAI_Panel.py",
        "pages/4_Recommendations.py",
        "pages/5_Reports.py",
        "synthetic/synthetic_data_generator.py",
        "data/fetch_data.py",
    ]
    for rel in core_files:
        r = check_file(Path(ROOT) / rel, rel)
        all_ok = all_ok and r

    # ── Data files ────────────────────────────────────────────────────────────
    print("\n[4/6] Data & model files (optional but needed to run app)")
    from src.paths import MERGED_CSV, SYNTHETIC_CSV, MODEL_PATH, SCALER_PATH, METRICS_PATH, SOIL_CSV, WEATHER_CSV
    check_file(SYNTHETIC_CSV,  "synthetic/synthetic_zones.csv", optional=True)
    check_file(WEATHER_CSV,    "data/raw/weather_data.csv",     optional=True)
    check_file(SOIL_CSV,       "data/raw/soil_data.csv",        optional=True)
    check_file(MERGED_CSV,     "data/processed/merged_zones.csv", optional=True)
    check_file(MODEL_PATH,     "models/risk_model.joblib",      optional=True)
    check_file(SCALER_PATH,    "models/feature_scaler.pkl",     optional=True)
    check_file(METRICS_PATH,   "models/metrics.json",           optional=True)

    # ── Data freshness ────────────────────────────────────────────────────────
    print("\n[5/6] Data freshness")
    import datetime
    if MERGED_CSV.exists():
        mtime = datetime.datetime.fromtimestamp(MERGED_CSV.stat().st_mtime)
        age_h = (datetime.datetime.now() - mtime).total_seconds() / 3600
        tag   = PASS if age_h < 3 else (WARN if age_h < 24 else INFO)
        src   = "real API" if WEATHER_CSV.exists() else "synthetic"
        print(f"  merged_zones.csv ({src})       {tag}  last updated {age_h:.1f}h ago")
    else:
        print(f"  merged_zones.csv               {WARN}  not found — run setup first")

    # ── Smoke test ────────────────────────────────────────────────────────────
    print("\n[6/6] Runtime smoke test")
    try:
        import numpy as np
        from src.gnn_model import GraphRiskModel
        model = GraphRiskModel(n_estimators=20)
        X = np.random.randn(12, 6).astype("float32")
        A = np.random.rand(12, 12).astype("float32")
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        y = np.array([0,0,1,1,2,2,3,3,4,4,0,1], dtype=int)
        model.fit(X, y, adj_norm=A, feature_names=[f"f{i}" for i in range(6)])
        p = model.predict_proba(X, A)
        assert p.shape == (12, 5), f"Expected (12,5), got {p.shape}"
        print(f"  GraphRiskModel                 {PASS}  predict_proba shape: {p.shape}")
    except Exception as e:
        all_ok = False
        print(f"  GraphRiskModel                 {FAIL}  {type(e).__name__}: {e}")

    try:
        from src.xai_explainer import explain_node, SHAP_AVAILABLE
        import numpy as np
        # Quick fallback test
        if not SHAP_AVAILABLE:
            print(f"  xai_explainer (shap)           {WARN}  SHAP not installed — fallback mode active")
        else:
            print(f"  xai_explainer (shap)           {PASS}  SHAP available")
    except Exception as e:
        print(f"  xai_explainer                  {FAIL}  {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    if all_ok:
        print("  ✅  All checks passed — project is ready.")
    else:
        print("  ⚠️   Some checks failed — see FAIL lines above.")

    print("""
  ─────────────────────────────────────────────────────────────────
  Run order:

  A) QUICKSTART (synthetic data — offline):
     python synthetic/synthetic_data_generator.py
     python train_model.py
     streamlit run app.py

  B) REAL DATA (fetches from Open-Meteo + SoilGrids — free APIs):
     python run_update.py              ← full fetch + retrain
     streamlit run app.py              ← dashboard auto-refreshes every 2h

  C) INSTALL OPTIONAL EXTRAS:
     pip install shap                  ← exact SHAP explanations in XAI panel
     pip install fpdf2                 ← PDF report generation

  ─────────────────────────────────────────────────────────────────
""")
    print("=" * 72)


if __name__ == "__main__":
    main()
