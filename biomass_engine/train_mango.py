#!/usr/bin/env python3
"""
biomass_engine/train_mango.py
================================
Train RF and ANN biomass models on the Mango dataset (M001-M010), with
biomass in GRAMS (matching how it's actually measured — the field scale
readings and dataset/ground_truth.csv's net_weight_g column — rather than
the kg convention used for the old Duranta dataset in weights.txt).

Only 10 samples. Leave-one-out CV is used for both models (matching the
original 40-sample training approach) to get an honest, if noisy, sense of
generalisation before training the final model on all 10. Treat the ANN
result with real skepticism — the reference thesis flagged neural nets as
needing 1000+ samples even for the *original* 40-sample dataset; 10 is far
below the point an MLP can be expected to generalise at all. RF, with its
LOOCV + shallow trees, is far more sample-efficient and the more trustworthy
of the two here.

Usage
-----
    python biomass_engine/train_mango.py
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from biomass_engine.models.random_forest import (
    DecisionTreeRegressor, RandomForestRegressor, BiomassRandomForest,
)
from biomass_engine.models.ann import BiomassANN
from biomass_engine.predict_batch import extract_features, RF_FEATURES, ANN_FEATURES
from biomass_engine.evaluation_metrics import (
    extended_metrics, print_metrics_table, save_bland_altman_figure,
)
from shared.config import REPO_ROOT, RECON_OUTPUTS_DIR, TRAINED_MODELS_DIR, EVAL_FIGURES_DIR

GT_CSV = REPO_ROOT / "dataset" / "ground_truth.csv"


def load_mango_dataset():
    """Return {specimen_id: net_weight_g} for every Mango plant with both
    ground truth and a completed reconstruction."""
    weights = {}
    with GT_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            weights[row["plant_id"]] = float(row["net_weight_g"])

    X, y, ids = [], [], []
    for specimen_id, weight_g in sorted(weights.items()):
        stats_path = RECON_OUTPUTS_DIR / f"reconstruction_stats_specimen_{specimen_id}.txt"
        vertices_path = RECON_OUTPUTS_DIR / f"final_vertices_specimen_{specimen_id}.npy"
        if not stats_path.exists() or not vertices_path.exists():
            print(f"[Train] {specimen_id}: no reconstruction yet, skipping")
            continue
        features = extract_features(stats_path, vertices_path)
        X.append(features)
        y.append(weight_g)
        ids.append(specimen_id)

    return X, np.array(y), ids


def train_rf(X_feat_dicts, y, ids):
    print("\n" + "=" * 70)
    print("RANDOM FOREST — Mango, grams")
    print("=" * 70)
    X = np.array([[f[name] for name in RF_FEATURES] for f in X_feat_dicts])

    model = BiomassRandomForest()
    model.feature_names = RF_FEATURES
    loo = model.leave_one_out_cv(X, y, n_trees=50, max_depth=3)
    print(f"\nLOOCV (n={len(y)}): R²={loo['r2']:.3f}  MAE={loo['mae']:.1f}g  RMSE={loo['rmse']:.1f}g")

    model.train(X, y, n_trees=50, max_depth=3, min_samples_split=2)
    out_dir = TRAINED_MODELS_DIR / "RF_model_mango"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "biomass_rf_model"))
    return loo


def train_ann(X_feat_dicts, y, ids):
    print("\n" + "=" * 70)
    print("ANN — Mango, grams (n=10 — expect this to generalise poorly)")
    print("=" * 70)
    X = np.array([[f[name] for name in ANN_FEATURES] for f in X_feat_dicts])
    y_col = y.reshape(-1, 1)

    architecture = [X.shape[1], 4, 2, 1]

    # LOOCV: train fresh each fold, no internal val split (n too small) —
    # just run a fixed epoch count rather than early-stopping.
    preds = []
    for i in range(len(y)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y_col, i, axis=0)
        X_test  = X[i:i + 1]

        fold_model = BiomassANN()
        fold_model.feature_names = ANN_FEATURES
        fold_model.initialize_network(architecture)
        fold_model.train(X_train, y_train, epochs=200, learning_rate=0.001, verbose=False)
        pred = fold_model.predict(X_test)[0, 0]
        preds.append(pred)
        print(f"  {ids[i]}: actual={y[i]:.1f}g  predicted={pred:.1f}g")

    preds = np.array(preds)
    err = preds - y
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    print(f"\nLOOCV (n={len(y)}): R²={r2:.3f}  MAE={mae:.1f}g  RMSE={rmse:.1f}g")

    final_model = BiomassANN()
    final_model.feature_names = ANN_FEATURES
    final_model.initialize_network(architecture)
    final_model.train(X, y_col, epochs=200, learning_rate=0.001, verbose=False)
    out_dir = TRAINED_MODELS_DIR / "ANN_model_mango"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(out_dir / "biomass_ann_model"))
    return dict(r2=r2, mae=mae, rmse=rmse, predictions=preds, actuals=y)


def main():
    X_feat_dicts, y, ids = load_mango_dataset()
    print(f"[Train] Loaded {len(ids)} Mango specimens: {ids}")
    print(f"[Train] Weight range: {y.min():.0f}g - {y.max():.0f}g")

    rf_result  = train_rf(X_feat_dicts, y, ids)
    ann_result = train_ann(X_feat_dicts, y, ids)

    print("\n" + "=" * 70)
    print("SUMMARY (leave-one-out CV, n=10)")
    print("=" * 70)
    print(f"  RF : R²={rf_result['r2']:.3f}  MAE={rf_result['mae']:.1f}g")
    print(f"  ANN: R²={ann_result['r2']:.3f}  MAE={ann_result['mae']:.1f}g")

    # ------------------------------------------------------------------
    # Extended metrics: Bias, nRMSE, Lin's CCC, Bland-Altman
    # ------------------------------------------------------------------
    rf_metrics  = extended_metrics(rf_result["actuals"],  rf_result["predictions"])
    ann_metrics = extended_metrics(ann_result["actuals"], ann_result["predictions"])
    print_metrics_table({"RF": rf_metrics, "ANN": ann_metrics})

    EVAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_bland_altman_figure(
        rf_result["actuals"],  rf_result["predictions"],
        ann_result["actuals"], ann_result["predictions"],
        EVAL_FIGURES_DIR / "mango_bland_altman.png",
    )


if __name__ == "__main__":
    main()
