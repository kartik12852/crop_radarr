# 🌾 Climate Crop Radar Alert System

> **Graph Neural Networks + Explainable AI for Real-Time Agricultural Risk Intelligence**

GNN-powered climate risk monitoring for **30 Indian agricultural zones** using a pure-PyTorch LightGCN, SHAP explanations, and a live Streamlit dashboard.

---

## 🚀 Quick Start (Windows / Anaconda)

### Step 1 — Clone and enter the project

```powershell
git clone https://github.com/VibhorJain1974/crop_radar.git
cd crop_radar
```

### Step 2 — Create & activate the Conda environment

```powershell
conda env create -f environment.yml
conda activate crop_radar
```

### Step 3 — Health check (run this first, every time)

```powershell
python doctor.py
```

All items should show ✅ PASS (data + model files will show ❌ until you run Step 4 & 5).

### Step 4 — Generate data

**Option A: Real data from APIs (needs internet)**
```powershell
python data/fetch_data.py
```

**Option B: Synthetic data (always works, offline)**
```powershell
python synthetic/synthetic_data_generator.py
```

### Step 5 — Train the model

```powershell
python train_model.py
```

### Step 6 — Launch the dashboard

```powershell
streamlit run app.py
```

---

## 📁 Project Structure

```
crop_radar/
├── app.py                          ← Main Streamlit dashboard
├── train_model.py                  ← GCN training script
├── doctor.py                       ← Health check script
├── requirements.txt                ← pip dependencies
├── environment.yml                 ← conda environment (recommended)
│
├── data/
│   ├── fetch_data.py               ← Live API data pipeline
│   ├── raw/                        ← Downloaded CSVs
│   └── processed/
│       └── merged_zones.csv        ← Final training data
│
├── src/
│   ├── paths.py                    ← All file paths (centralised)
│   ├── gnn_model.py                ← LightGCN / LightGAT (pure PyTorch)
│   ├── graph_builder.py            ← Zone adjacency graph builder
│   ├── trainer.py                  ← Training loop + metrics
│   ├── model_utils.py              ← Inference helpers
│   ├── alert_engine.py             ← Alert generation
│   ├── xai_explainer.py            ← SHAP explainability
│   └── recommendation.py          ← Mitigation recommendations
│
├── pages/
│   ├── 1_Risk_Map.py               ← PyDeck interactive risk map
│   ├── 2_GNN_Explorer.py          ← Graph + embedding visualiser
│   ├── 3_XAI_Panel.py             ← SHAP XAI panel
│   ├── 4_Recommendations.py       ← Farmer recommendations
│   └── 5_Reports.py               ← Metrics + PDF export
│
├── utils/
│   ├── visualizer.py              ← Plotly/matplotlib chart helpers
│   └── pdf_exporter.py            ← PDF report generator
│
├── synthetic/
│   └── synthetic_data_generator.py ← Offline data fallback
│
└── models/                         ← Saved checkpoints (auto-created)
    ├── best_gcn.pt
    ├── feature_scaler.pkl
    ├── metrics.json
    └── training_history.json
```

---

## ⚠️ Troubleshooting

### "No module named 'src.xxx'" or import errors
```powershell
# Always run from INSIDE the project folder:
cd E:\path\to\crop_radar
python doctor.py
```

### "dateutil not found" / broken environment
```powershell
conda deactivate
conda env remove -n crop_radar -y
conda env create -f environment.yml
conda activate crop_radar
python doctor.py
```

### Stale model checkpoint after code changes
```powershell
Remove-Item models\* -Force -ErrorAction SilentlyContinue
python train_model.py
```

### Clean run sequence (always works)
```powershell
cd E:\path\to\crop_radar
conda activate crop_radar
python doctor.py
python synthetic/synthetic_data_generator.py
python train_model.py
streamlit run app.py
```

---

## 🏗️ Architecture

| Component | Technology |
|-----------|-----------|
| Graph model | Pure PyTorch LightGCN (no torch_geometric) |
| Explainability | SHAP KernelExplainer |
| Dashboard | Streamlit multi-page app |
| Map | PyDeck (Mapbox) |
| Weather data | Open-Meteo Archive API |
| Soil data | SoilGrids REST API v2 |
| Risk classes | Safe, Drought, Heat Stress, Flood, Soil Risk |

---

## 📖 Risk Categories

| Label | Risk | Trigger Condition |
|-------|------|-------------------|
| 0 | ✅ Safe | Normal conditions |
| 1 | 🌵 Drought | Low rainfall, high ET0 demand |
| 2 | 🔥 Heat Stress | Temperature > crop tolerance + 3°C |
| 3 | 🌊 Flood | Precipitation > 1.5× water need |
| 4 | 🪨 Soil Risk | High bulk density + low organic carbon |
