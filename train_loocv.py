"""
train_loocv.py
==============
Leave-One-Out Cross-Validation for Level-1 biomass estimation.

Loads reconstruction outputs for all plants that have both:
  - procedure_alpha/outputs/reconstruction_stats_plant_{id}.txt
  - procedure_alpha/outputs/final_vertices_plant_{id}.npy
  - a ground-truth entry in weights.txt

Runs LOOCV for both Random Forest and ANN, prints R²/RMSE/MAE,
saves a predictions CSV and a scatter plot.

Usage
-----
    python train_loocv.py

    # Use AGB column from registry instead of weights.txt
    python train_loocv.py --use-registry

Output
------
    evaluation_suite/reports/loocv_results.csv
    evaluation_suite/figures/loocv_scatter.png
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── repository root on sys.path ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from classes.random_forest_class import BiomassRandomForest
from classes.ann_class import BiomassANN
from shared.config import RECON_OUTPUTS_DIR, EVAL_REPORTS_DIR, EVAL_FIGURES_DIR

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ── Feature set (matches thesis Chapter 3) ───────────────────────────────────
RF_FEATURES  = ['volume', 'surface_area', 'height',
                'bbox_volume', 'surface_to_volume_ratio', 'height_to_volume_ratio']

ANN_FEATURES = ['volume', 'surface_area', 'height',
                'compactness', 'overall_quality',
                'surface_to_volume_ratio', 'height_to_volume_ratio']


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_weights_txt(path: Path) -> dict[int, float]:
    """Load plant_id → biomass (kg) from weights.txt."""
    weights = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, val = line.split(':', 1)
                pid = int(key.strip().split('_')[1])
                weights[pid] = float(val.strip())
    return weights


def load_registry_agb(path: Path) -> dict[int, float]:
    """Load legacy_id → agb_kg from registry.csv (skips rows where agb_kg is empty)."""
    weights = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            agb = row.get('agb_kg', '').strip()
            legacy = row.get('legacy_id', '').strip()
            if agb and legacy.startswith('plant_'):
                pid = int(legacy.split('_')[1])
                weights[pid] = float(agb)
    return weights


def extract_features(plant_id: int, recon_dir: Path) -> dict | None:
    """Extract feature dict for one plant from reconstruction outputs."""
    brf = BiomassRandomForest()
    return brf.extract_features_from_reconstruction(str(recon_dir), plant_id)


def metrics(actuals: np.ndarray, preds: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    mae  = float(np.mean(np.abs(actuals - preds)))
    ss_r = float(np.sum((actuals - preds) ** 2))
    ss_t = float(np.sum((actuals - np.mean(actuals)) ** 2))
    r2   = float(1 - ss_r / ss_t) if ss_t > 0 else float('nan')
    return dict(r2=r2, rmse=rmse, mae=mae)


def print_metrics(name: str, m: dict):
    print(f"  {name:20s}  R²={m['r2']:.4f}  RMSE={m['rmse']:.4f} kg  MAE={m['mae']:.4f} kg")


def save_scatter(actuals, rf_preds, ann_preds, out_path: Path):
    if not HAS_PLT:
        print("[plot] matplotlib not available — skipping scatter plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, preds, title in zip(axes,
                                 [rf_preds, ann_preds],
                                 ['Random Forest (LOOCV)', 'ANN (LOOCV)']):
        lo = min(actuals.min(), preds.min()) * 0.95
        hi = max(actuals.max(), preds.max()) * 1.05
        ax.scatter(actuals, preds, alpha=0.7, edgecolors='k', linewidths=0.5)
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
        m = metrics(actuals, preds)
        ax.set_title(f"{title}\n$R^2$={m['r2']:.3f}  RMSE={m['rmse']:.3f} kg")
        ax.set_xlabel("Actual AGB (kg)")
        ax.set_ylabel("Predicted AGB (kg)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    print(f"[plot] Scatter saved → {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Level-1 LOOCV training")
    parser.add_argument("--use-registry", action="store_true",
                        help="Use agb_kg from registry.csv instead of weights.txt")
    args = parser.parse_args()

    # ── Load ground truth ─────────────────────────────────────────────────────
    if args.use_registry:
        registry_path = REPO_ROOT / "acquisition" / "dataset" / "ground_truth" / "registry.csv"
        gt = load_registry_agb(registry_path)
        if not gt:
            print("ERROR: No agb_kg values found in registry.csv.")
            print("       Fill in the agb_kg column or run without --use-registry.")
            sys.exit(1)
        print(f"[GT] Loaded {len(gt)} AGB labels from registry.csv")
    else:
        weights_path = REPO_ROOT / "weights.txt"
        gt = load_weights_txt(weights_path)
        print(f"[GT] Loaded {len(gt)} labels from weights.txt")
        print("     NOTE: weights.txt contains total_mass_kg, not agb_kg.")
        print("           Use --use-registry once agb_kg column is filled.\n")

    # ── Extract features for all available plants ─────────────────────────────
    plant_ids   = sorted(gt.keys())
    feat_dicts  = []
    y_list      = []
    valid_ids   = []

    for pid in plant_ids:
        fd = extract_features(pid, RECON_OUTPUTS_DIR)
        if fd is None:
            print(f"[skip] plant_{pid}: no reconstruction output")
            continue
        # Check all required features exist
        missing_rf  = [f for f in RF_FEATURES  if f not in fd or fd[f] == 0]
        missing_ann = [f for f in ANN_FEATURES if f not in fd or fd[f] == 0]
        if missing_rf:
            print(f"[skip] plant_{pid}: missing RF features {missing_rf}")
            continue
        feat_dicts.append(fd)
        y_list.append(gt[pid])
        valid_ids.append(pid)

    n = len(valid_ids)
    if n < 3:
        print(f"ERROR: Only {n} plants with complete data. Run batch_run.py first.")
        sys.exit(1)

    print(f"\n[data] {n} plants ready for LOOCV  "
          f"(AGB range: {min(y_list):.2f}–{max(y_list):.2f} kg)\n")

    y = np.array(y_list)

    # ── RF feature matrix ─────────────────────────────────────────────────────
    X_rf = np.array([[fd[f] for f in RF_FEATURES] for fd in feat_dicts])

    # ── ANN feature matrix ────────────────────────────────────────────────────
    X_ann = np.array([[fd[f] for f in ANN_FEATURES] for fd in feat_dicts])

    # ── LOOCV — Random Forest ─────────────────────────────────────────────────
    print("Running LOOCV — Random Forest ...")
    rf_preds = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        X_tr = np.delete(X_rf, i, axis=0)
        y_tr = np.delete(y,    i)
        X_te = X_rf[i:i+1]

        model = BiomassRandomForest()
        model.feature_names = RF_FEATURES
        model.train(X_tr, y_tr, n_trees=100, max_depth=5, min_samples_split=2)
        rf_preds[i] = model.predict(X_te)[0]

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] plant_{valid_ids[i]:2d}: "
                  f"actual={y[i]:.3f}  pred={rf_preds[i]:.3f}  "
                  f"err={abs(y[i]-rf_preds[i]):.3f} kg")

    rf_m = metrics(y, rf_preds)
    print(f"\n  RF LOOCV complete in {time.time()-t0:.0f}s")
    print_metrics("Random Forest", rf_m)

    # ── LOOCV — ANN ──────────────────────────────────────────────────────────
    print("\nRunning LOOCV — ANN ...")
    ann_preds = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        X_tr = np.delete(X_ann, i, axis=0)
        y_tr = np.delete(y,     i)
        X_te = X_ann[i:i+1]

        ann = BiomassANN()
        ann.train(X_tr, y_tr, hidden_layers=[4, 2], epochs=2000, lr=1e-3)
        ann_preds[i] = ann.predict(X_te)[0]

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] plant_{valid_ids[i]:2d}: "
                  f"actual={y[i]:.3f}  pred={ann_preds[i]:.3f}  "
                  f"err={abs(y[i]-ann_preds[i]):.3f} kg")

    ann_m = metrics(y, ann_preds)
    print(f"\n  ANN LOOCV complete in {time.time()-t0:.0f}s")
    print_metrics("ANN", ann_m)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("LEVEL-1 LOOCV RESULTS")
    print(f"{'='*60}")
    print_metrics("Random Forest", rf_m)
    print_metrics("ANN",           ann_m)
    print(f"{'='*60}\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_REPORTS_DIR / "loocv_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['plant_id', 'actual_kg', 'rf_pred_kg', 'ann_pred_kg',
                         'rf_error_kg', 'ann_error_kg'])
        for i, pid in enumerate(valid_ids):
            writer.writerow([f"plant_{pid}", f"{y[i]:.4f}",
                             f"{rf_preds[i]:.4f}", f"{ann_preds[i]:.4f}",
                             f"{abs(y[i]-rf_preds[i]):.4f}",
                             f"{abs(y[i]-ann_preds[i]):.4f}"])
        writer.writerow([])
        writer.writerow(['metric', 'RF', 'ANN'])
        for k in ['r2', 'rmse', 'mae']:
            writer.writerow([k, f"{rf_m[k]:.4f}", f"{ann_m[k]:.4f}"])
    print(f"[csv] Results saved → {csv_path}")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    save_scatter(y, rf_preds, ann_preds,
                 EVAL_FIGURES_DIR / "loocv_scatter.png")


if __name__ == "__main__":
    main()
