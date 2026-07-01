# CLAUDE CONTEXT — ABVT3R Dissertation Project
# Paste this entire file into Claude (VS Code) at the start of a new session.

---

## Who I am

I am Aaron Masuba, MEng student at the University of Pretoria (u25737806).
Supervisor: Prof. H. C. Myburgh, Department of Electrical, Electronic and Computer Engineering.

---

## What this project is

**Title:** Automated Above-Ground Biomass Estimation of Ornamental Tree Saplings Using Self-Supervised Vision Transformers and RGB-D Sensor Data

**One-line summary:** Estimate plant dry biomass from Microsoft Kinect V2 RGB-D scans using a three-level pipeline hierarchy, culminating in Meta AI's SAM 3D Objects (DINOv2 ViT-B/14 backbone).

---

## Three-Level Methodology Hierarchy

| Level | Method | Model | Status |
|-------|--------|-------|--------|
| 1 | Classical depth-based pipeline (Procedure Alpha) | Random Forest + ANN | **Running now** |
| 2 | DeepVoxels neural voxel embedding | ANN regression head | Pending |
| 3 | SAM 3D Objects + DINOv2 ViT-B/14 (proposed method) | ANN regression head | Pending |

**Target results (from dissertation draft):**
- Level 1: RF R²=0.655, ANN R²=0.752, RMSE=0.264 kg
- Level 2: ANN R²=0.703
- Level 3 (proposed): ANN R²=0.834, RMSE=0.213 kg

---

## Repository: ABVT3R

**Repo root:** `~/ABVT3R/` (transferred from Windows `C:\Users\user\ABVT3R`)

### Key files and directories

```
ABVT3R/
├── batch_run.py              ← Run Level-1 pipeline on all 40 plants
├── train_loocv.py            ← LOOCV training for RF + ANN, saves CSV + plot
├── ubuntu_setup.sh           ← Run this first on Ubuntu to install deps + smoke test
├── weights.txt               ← Ground truth labels, 40 plants (WARNING: see below)
├── shared/config.py          ← All camera intrinsics, ROI bounds, paths
│
├── procedure_alpha/          ← Level-1 classical pipeline
│   ├── pipeline.py           ← Main orchestrator: run_legacy(plant_id) or run_specimen(id)
│   ├── preprocessing.py      ← depth→pointcloud, PassThrough, SOR, MLS
│   ├── registration.py       ← Coarse (rotation) + ICP fine registration
│   ├── reconstruction.py     ← Marching Cubes mesh, quality metrics
│   └── outputs/              ← Reconstruction outputs saved here per plant
│
├── classes/
│   ├── random_forest_class.py ← Custom RF from scratch (no sklearn model), LOOCV included
│   └── ann_class.py           ← Custom ANN from scratch, 7→4→2→1 architecture
│
├── data_collection/           ← Legacy 4-view (0°,90°,180°,270°) depth+RGB .npy files
│   └── (currently only plant_1 complete; plants 2–40 being collected tomorrow)
│
├── acquisition/dataset/
│   ├── ground_truth/registry.csv  ← Master specimen registry (see label warning below)
│   └── specimens/                 ← New 30° protocol data goes here
│
├── neural_geometry/           ← Level-2 (DeepVoxels) and Level-3 (SAM3D/DINOv2) — pending
├── evaluation_suite/          ← Metrics, comparison, figures, reports
└── RF_model/ + ANN_model/     ← Saved trained model files
```

---

## Camera: Microsoft Kinect V2

Intrinsics (depth stream, 512×424):
- fx = fy = 365.456 px (note: config.py uses 365.456, but preprocessing uses 383.58 — use config.py values)
- cx = 254.878, cy = 205.395
- Valid depth range: 300–6000 mm

Back-projection: X=(u−cx)·D/fx, Y=−(v−cy)·D/fy, Z=D

---

## Data collection protocol

**Legacy (Nombambela, 4-view, 90° steps):**
- Files: `data_collection/{angle}_degrees_depth_plant_{id}.npy`
- Angles: 0, 90, 180, 270 degrees
- 40 plants total; plant_1 already processed

**New 30-degree protocol (Aaron's contribution):**
- 12 views at 30° intervals, dual Kinect V2 cameras
- Files: `acquisition/dataset/specimens/{specimen_id}/depth/view_{angle:03d}deg_cam{A|B}_depth.npy`
- Specimen ID format: `DG{seq:03d}_{YYYYMMDD}_{BATCH}`
- Currently only DG041 and DG042 exist

---

## ⚠️ CRITICAL: Ground Truth Label Problem

`weights.txt` contains **TOTAL MASS (plant + soil + pot)**, NOT above-ground dry biomass (AGB).
The `agb_kg` column in `registry.csv` is **empty** for all 40 plants.

**For new data collection, measure and record ALL of these per plant:**
1. `total_mass_kg` — weigh everything as-is
2. `pot_mass_kg` — weigh pot + soil after removing plant
3. `agb_kg` — oven-dry above-ground plant at 70°C for 72h, then weigh → **this is the regression target**

**After filling agb_kg in registry.csv, run:**
```bash
python3 train_loocv.py --use-registry
```

---

## Level-1 Pipeline: How to run

### First time on Ubuntu:
```bash
chmod +x ubuntu_setup.sh
./ubuntu_setup.sh
```

### After collecting all 40 plants' depth files:
```bash
# Run reconstruction for all 40 plants
python3 batch_run.py

# Then train + LOOCV (weights.txt labels — total mass, placeholder)
python3 train_loocv.py

# Or with proper AGB labels once registry.csv is filled
python3 train_loocv.py --use-registry
```

### Run a single plant manually:
```python
from procedure_alpha.pipeline import ProcedureAlpha
pa = ProcedureAlpha()
result = pa.run_legacy(1)   # plant_1 (legacy 4-view)
# or
result = pa.run_specimen("DG041_20260609_B01")  # new 30° protocol
```

---

## Level-1 Feature Set

**RF features (6):** volume, surface_area, height, bbox_volume, surface_to_volume_ratio, height_to_volume_ratio

**ANN features (7):** volume, surface_area, height, compactness (κ=36πV²/A³), overall_quality (Q), surface_to_volume_ratio, height_to_volume_ratio

**Evaluation:** LOOCV (Leave-One-Out Cross-Validation) over 40 plants

---

## Known Issues / Technical Debt

1. **Q_ss artefact:** `surface_smoothness` sub-score always returns 0.0 in reconstruction_stats files — parsing or computation bug. Q score is underweighted as a result.

2. **RF model loading error:** `biomass_rf_model.npy` throws AttributeError on load — the custom tree objects don't serialise cleanly with np.save. Workaround: retrain from scratch each LOOCV fold (already done in train_loocv.py).

3. **Kinect intrinsics mismatch:** `preprocessing.py` uses fx=383.58, but `shared/config.py` has fx=365.456. The pipeline uses config.py values (correct). The class-level preprocessing.py is legacy and not used by ProcedureAlpha.

4. **open3d on Python 3.13:** No pip wheel available. Use Python 3.10 virtualenv if mesh export (.ply/.obj) is needed. Otherwise pipeline runs fine without it.

---

## GPU Strategy (for Level-2 and Level-3)

| Machine | GPU | VRAM | Use |
|---------|-----|------|-----|
| Lab Ubuntu (RTX 4060) | RTX 4060 | 8–12 GB | ✅ DINOv2 + ANN regression |
| Lab Windows (RTX 3060) | RTX 3060 | 12 GB | ✅ VGGT inference |
| Laptop Ubuntu (RTX 2050) | RTX 2050 | 4 GB | Dev/testing only |
| Laptop Windows (RTX 2050) | RTX 2050 | 4 GB | ❌ Too little VRAM for VGGT |

For Level-3 (SAM 3D Objects), install PyTorch with CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Dissertation LaTeX Files

All chapter files are written and live at:
`C:\Users\user\AppData\Roaming\Claude\local-agent-mode-sessions\...\outputs\`

Chapters: chapter1.tex through chapter6.tex, abbrev.tex, symbols.tex, bib_additions.bib

Key results embedded in the LaTeX:
- Level 1: RF R²=0.655, ANN R²=0.752
- Level 2: DeepVoxels ANN R²=0.703
- Level 3 (proposed): SAM3D ANN R²=0.834, RMSE=0.213 kg

These are **placeholder values** to be replaced with actual experimental results once all 40 plants are processed.

---

## Immediate Next Steps (in order)

1. **[TODAY]** Transfer repo to Ubuntu lab machine, run `./ubuntu_setup.sh`
2. **[TOMORROW]** Collect RGB-D data for plants 2–40 at the lab greenhouse using dual Kinect V2 cameras. Save as `data_collection/{angle}_degrees_depth_plant_{id}.npy`
3. **[TOMORROW]** For each plant: record agb_kg (oven-dry above-ground plant at 70°C/72h). Update `acquisition/dataset/ground_truth/registry.csv`
4. **[AFTER COLLECTION]** Run `python3 batch_run.py` to reconstruct all 40 plants
5. **[AFTER RECONSTRUCTION]** Run `python3 train_loocv.py --use-registry` for final R²/RMSE/MAE
6. **[AFTER L1 RESULTS]** Run Level-2 DeepVoxels experiments (neural_geometry/volumetric/)
7. **[AFTER L2 RESULTS]** Run Level-3 SAM 3D Objects experiments (neural_geometry/sam3d/)
8. **[FINAL]** Update LaTeX chapters with actual results, compile dissertation

---

## How to use this context

Paste this entire file into Claude at the start of a VS Code session.
Then say what you want to work on, e.g.:
- "Help me debug the reconstruction pipeline for plant_3"
- "Fix the Q_ss surface smoothness artefact in procedure_alpha/reconstruction.py"
- "Write the data collection script for the new 30-degree protocol"
- "Set up the DeepVoxels Level-2 experiment"
