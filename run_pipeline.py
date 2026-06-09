#!/usr/bin/env python3
"""
Full biomass estimation pipeline runner.

Runs for each plant ID that has complete depth data in data_collection/:
  1. Preprocessing  — depth maps → filtered point clouds
  2. Registration   — coarse circular arrangement + ICP fine alignment
  3. 3D Reconstruction — voxel-based mesh generation
  4. RF prediction  — Random Forest biomass estimate (saved model)
  5. ANN prediction — Neural Network biomass estimate (saved model)

Usage:
    python run_pipeline.py              # process all plants with complete data
    python run_pipeline.py 1            # process only plant 1
    python run_pipeline.py 1 3 5        # process plants 1, 3, and 5
"""

import sys
import os
import time
import traceback
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(ROOT, "data_collection")
RECONSTRUCTION_DIR = os.path.join(ROOT, "reconstruction_output")
RF_MODEL_PATH     = os.path.join(ROOT, "RF_model",  "biomass_rf_model")
ANN_MODEL_PATH    = os.path.join(ROOT, "ANN_model", "biomass_ann_model")
WEIGHTS_FILE      = os.path.join(ROOT, "weights.txt")

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from integration import Integration
except Exception as e:
    print(f"ERROR: Could not import Integration: {e}")
    sys.exit(1)

try:
    from classes.ann_class import BiomassANN
except Exception as e:
    print(f"ERROR: Could not import BiomassANN: {e}")
    sys.exit(1)

try:
    from classes.random_forest_class import (
        BiomassRandomForest, RandomForestRegressor, DecisionTreeRegressor
    )
    # Needed so pickle inside np.load can resolve the classes
    sys.modules['__main__'].DecisionTreeRegressor = DecisionTreeRegressor
    sys.modules['__main__'].RandomForestRegressor = RandomForestRegressor
except Exception as e:
    print(f"ERROR: Could not import BiomassRandomForest: {e}")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ground_truth(weights_file):
    """Return {plant_id: weight_kg} from weights.txt."""
    weights = {}
    if not os.path.exists(weights_file):
        return weights
    with open(weights_file) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                plant, w = line.split(':', 1)
                pid = int(plant.strip().split('_')[1])
                weights[pid] = float(w.strip())
    return weights


def find_complete_plants(data_dir):
    """Return sorted list of plant IDs that have all 4 depth .npy files."""
    views = ["0_degrees", "90_degrees", "180_degrees", "270_degrees"]
    found = set()
    for f in os.listdir(data_dir):
        if f.endswith(".npy") and "depth" in f:
            # e.g. 0_degrees_depth_plant_1.npy
            parts = f.replace(".npy", "").split("_plant_")
            if len(parts) == 2:
                try:
                    found.add((parts[0].replace("_depth", ""), int(parts[1])))
                except ValueError:
                    pass

    complete = []
    all_pids = sorted({pid for _, pid in found})
    for pid in all_pids:
        if all((view, pid) in found for view in views):
            complete.append(pid)
    return complete


def progress_cb(message, percentage=None):
    pct = f"{int(percentage):3d}%" if percentage is not None else "   "
    print(f"  [{pct}] {message}")


def predict_rf(plant_id):
    """Load saved RF model and predict biomass for plant_id. Returns float or None."""
    if not os.path.exists(RF_MODEL_PATH + ".npy"):
        print(f"  RF model not found at {RF_MODEL_PATH}.npy")
        return None
    try:
        rf = BiomassRandomForest()
        rf.load_model(RF_MODEL_PATH)

        SELECTED_FEATURES = [
            'volume', 'surface_area', 'height',
            'bbox_volume', 'surface_to_volume_ratio', 'height_to_volume_ratio'
        ]
        features = rf.extract_features_from_reconstruction(RECONSTRUCTION_DIR, plant_id)
        if not features:
            print(f"  Could not extract features for plant {plant_id}")
            return None

        missing = [f for f in SELECTED_FEATURES if f not in features]
        if missing:
            print(f"  Missing RF features: {missing}")
            return None

        X = np.array([[features[f] for f in SELECTED_FEATURES]])
        return float(rf.predict(X)[0])
    except Exception as e:
        print(f"  RF prediction failed: {e}")
        return None


def predict_ann(plant_id):
    """Load saved ANN model and predict biomass for plant_id. Returns float or None."""
    if not os.path.exists(ANN_MODEL_PATH + ".npy"):
        print(f"  ANN model not found at {ANN_MODEL_PATH}.npy")
        return None
    try:
        ann = BiomassANN()
        ann.load_model(ANN_MODEL_PATH)

        SELECTED_FEATURES = [
            'volume', 'surface_area', 'height', 'compactness', 'overall_quality'
        ]
        features = ann.extract_features_from_reconstruction(RECONSTRUCTION_DIR, plant_id)
        if not features:
            print(f"  Could not extract features for plant {plant_id}")
            return None

        missing = [f for f in SELECTED_FEATURES if f not in features]
        if missing:
            print(f"  Missing ANN features: {missing}")
            return None

        X = np.array([[features[f] for f in SELECTED_FEATURES]])
        result = ann.predict(X)
        # predict() returns shape (1,1) or (1,) depending on network
        return float(np.array(result).flat[0])
    except Exception as e:
        print(f"  ANN prediction failed: {e}")
        return None


def run_plant(plant_id, ground_truth):
    """Run the full pipeline for one plant. Returns a result dict."""
    print(f"\n{'='*70}")
    print(f"  PLANT {plant_id}")
    print(f"{'='*70}")

    t0 = time.time()

    # ── Step 1–4: reconstruction + RF (Integration.start handles all of this) ──
    print("\n>>> Reconstruction pipeline (preprocessing → registration → mesh → RF)")
    integ = Integration(progress_callback=progress_cb)
    try:
        integ.start(plant_id)
        rf_from_integration = integ.prediction_rf
    except Exception:
        print("  Reconstruction pipeline FAILED:")
        traceback.print_exc()
        return None

    # ── Step 5: ANN prediction ────────────────────────────────────────────────
    print("\n>>> ANN biomass prediction")
    ann_pred = predict_ann(plant_id)

    # ── Step 4b: RF re-prediction from saved model (cross-check) ─────────────
    print("\n>>> RF biomass prediction (saved model)")
    rf_pred = predict_rf(plant_id)

    elapsed = time.time() - t0

    # ── Report ────────────────────────────────────────────────────────────────
    actual = ground_truth.get(plant_id)

    print(f"\n{'─'*70}")
    print(f"  RESULTS — Plant {plant_id}   (total time: {elapsed:.1f}s)")
    print(f"{'─'*70}")

    def fmt_pred(label, pred):
        if pred is None:
            print(f"  {label:<20}: N/A")
            return
        if actual is not None:
            err = pred - actual
            pct = 100 * abs(err) / actual
            flag = "✓" if pct <= 10 else ("~" if pct <= 20 else "✗")
            print(f"  {label:<20}: {pred:.3f} kg   "
                  f"(actual {actual:.3f} kg, err {err:+.3f} kg / {pct:.1f}% {flag})")
        else:
            print(f"  {label:<20}: {pred:.3f} kg")

    fmt_pred("RF prediction",  rf_pred)
    fmt_pred("ANN prediction", ann_pred)
    if actual is not None:
        print(f"  {'Ground truth':<20}: {actual:.3f} kg")
    print(f"{'─'*70}")

    return {
        'plant_id':   plant_id,
        'rf_pred':    rf_pred,
        'ann_pred':   ann_pred,
        'actual':     actual,
        'elapsed_s':  elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RECONSTRUCTION_DIR, exist_ok=True)

    # Determine which plants to process
    if len(sys.argv) > 1:
        try:
            plant_ids = [int(a) for a in sys.argv[1:]]
        except ValueError:
            print("Usage: python run_pipeline.py [plant_id ...]")
            sys.exit(1)
    else:
        plant_ids = find_complete_plants(DATA_DIR)
        if not plant_ids:
            print(f"No plants with complete depth data found in {DATA_DIR}/")
            print("Need: 0_degrees_depth_plant_N.npy, 90_degrees_depth_plant_N.npy, "
                  "180_degrees_depth_plant_N.npy, 270_degrees_depth_plant_N.npy")
            sys.exit(1)
        print(f"Auto-detected plants with complete data: {plant_ids}")

    ground_truth = load_ground_truth(WEIGHTS_FILE)

    results = []
    for pid in plant_ids:
        result = run_plant(pid, ground_truth)
        if result:
            results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Plant':<8} {'RF (kg)':<12} {'ANN (kg)':<12} {'Actual (kg)':<12} {'Time (s)'}")
        print(f"  {'─'*8} {'─'*11} {'─'*11} {'─'*11} {'─'*8}")
        for r in results:
            rf  = f"{r['rf_pred']:.3f}"  if r['rf_pred']  is not None else "N/A"
            ann = f"{r['ann_pred']:.3f}" if r['ann_pred'] is not None else "N/A"
            act = f"{r['actual']:.3f}"   if r['actual']   is not None else "N/A"
            print(f"  {r['plant_id']:<8} {rf:<12} {ann:<12} {act:<12} {r['elapsed_s']:.1f}")

    print(f"\nDone. Results saved to {RECONSTRUCTION_DIR}/")


if __name__ == "__main__":
    main()
