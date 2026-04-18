"""
run_update.py — Manual data fetch + model retrain
Run this whenever you want fresh data without waiting for the 2-hour auto-cycle.

Usage:
    python run_update.py              # fetch weather + soil + retrain
    python run_update.py --weather    # weather only (no soil re-fetch, no retrain)
    python run_update.py --train      # retrain only on existing data
    python run_update.py --synthetic  # regenerate synthetic data + retrain
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_full_pipeline():
    print("\n🔄  Starting full data fetch + model retrain...\n")
    from src.auto_updater import run_pipeline
    success = run_pipeline()
    if success:
        print("\n✅  Pipeline completed successfully!")
        print("    Reload the Streamlit app or press 'Reload Dashboard' in the sidebar.")
    else:
        print("\n❌  Pipeline failed — see error messages above.")
    return success


def run_train_only():
    print("\n🔧  Retraining model on existing data...\n")
    from src.graph_builder import build_graph
    from src.trainer import RiskTrainer
    try:
        X, adj_norm, adj_mask, y, df, feat_cols = build_graph(fit_scaler=True, fallback=True)
        print(f"  Loaded: {len(df)} zones | {len(feat_cols)} features | classes: {sorted(set(y.tolist()))}")
        trainer = RiskTrainer(random_state=42)
        model, metrics, history, report, idx_train, idx_val, preds, probs = trainer.train(
            X, adj_norm, y, feat_cols
        )
        print("\n  Validation metrics:")
        for k, v in metrics.items():
            if k not in ("overfit_warning",):
                print(f"    {k:20s}: {v}")
        if metrics.get("overfit_warning"):
            print(f"\n  ⚠️  {metrics['overfit_warning']}")
        print("\n✅  Model saved. Reload Streamlit to use updated model.")
    except Exception as e:
        print(f"\n❌  Training failed: {e}")
        import traceback; traceback.print_exc()


def run_synthetic():
    print("\n🧬  Regenerating synthetic data + retraining...\n")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "synthetic" / "synthetic_data_generator.py")],
        capture_output=False,
    )
    if result.returncode != 0:
        print("❌  Synthetic data generation failed.")
        return
    print()
    run_train_only()


def weather_only():
    print("\n⛅  Fetching weather data only (no retrain)...\n")
    from src.auto_updater import (
        ZONES, _fetch_weather, _weather_to_agg, _engineer_features,
        _assign_risk_labels, CROP_INFO
    )
    import time, numpy as np, pandas as pd
    from src.paths import WEATHER_CSV, PROCESSED_DATA_DIR, MERGED_CSV, SOIL_CSV, RAW_DATA_DIR

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for i, z in enumerate(ZONES):
        print(f"  [{i+1:02d}/30] {z['zone_name']}", end="... ", flush=True)
        raw = _fetch_weather(z["lat"], z["lon"], days_back=30)
        agg = _weather_to_agg(raw)
        row = {"zone_id": z["zone_id"], "zone_name": z["zone_name"],
               "lat": z["lat"], "lon": z["lon"], "crop": z["crop"]}
        row.update(agg)
        records.append(row)
        print("✓" if agg else "⚠ fallback")
        time.sleep(0.25)

    weather_df = pd.DataFrame(records)
    weather_df.to_csv(WEATHER_CSV, index=False)
    print(f"\n  Weather saved → {WEATHER_CSV}")

    if SOIL_CSV.exists():
        soil_df = pd.read_csv(SOIL_CSV)
        crop_df = pd.DataFrame(CROP_INFO)
        merged  = weather_df.merge(soil_df, on="zone_id", how="left")
        num_cols = merged.select_dtypes(include=[np.number]).columns
        merged[num_cols] = merged[num_cols].fillna(merged[num_cols].median())
        merged = merged.fillna(0)
        merged = _engineer_features(merged)
        merged = _assign_risk_labels(merged, crop_df)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_csv(MERGED_CSV, index=False)
        print(f"  Merged data saved → {MERGED_CSV}")
        print(f"\n  Risk distribution:")
        for label, count in merged["risk_label"].value_counts().sort_index().items():
            names = {0: "Safe", 1: "Drought", 2: "Heat Stress", 3: "Flood", 4: "Soil Risk"}
            print(f"    {names.get(int(label), label)}: {count} zones")
        print("\n✅  Weather refresh done. Run 'python run_update.py --train' to retrain.")
    else:
        print("  ⚠️  No soil cache found — run full pipeline first to fetch soil data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Climate Crop Radar — manual update tool")
    parser.add_argument("--weather",   action="store_true", help="Fetch weather only, no retrain")
    parser.add_argument("--train",     action="store_true", help="Retrain on existing data only")
    parser.add_argument("--synthetic", action="store_true", help="Regenerate synthetic data + retrain")
    args = parser.parse_args()

    print("=" * 62)
    print("  Climate Crop Radar — Manual Update Tool")
    print("=" * 62)

    if args.weather:
        weather_only()
    elif args.train:
        run_train_only()
    elif args.synthetic:
        run_synthetic()
    else:
        run_full_pipeline()
