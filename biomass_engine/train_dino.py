#!/usr/bin/env python3
"""
biomass_engine/train_dino.py
===============================
Stage A: does a frozen, pre-trained DINOv2 ViT-B/14 carry biomass signal
beyond our classical reconstruction-derived geometric features?

Loads the cached per-specimen embeddings from neural_geometry/dino_features/
(run neural_geometry/dino_features.py first) and compares three LOOCV RF
configurations on the same primary n=38 specimen set train_all.py uses
(excludes V009-V011 -- estimated pot mass, see train_all.py's docstring):

  1. geometric only   -- the existing RF_FEATURES baseline (R2=0.451)
  2. DINOv2 only       -- 768-dim frozen CLS embedding, PCA-reduced per fold
  3. geometric + DINOv2 -- both concatenated

PCA is refit inside each LOOCV fold on the training embeddings only (never
on the held-out specimen) to avoid leaking test-set structure into the
reduction -- with n=37 training points and a 768-dim embedding, skipping
this would make the comparison meaningless.

This is inference-only (no fine-tuning): the DINOv2 backbone stays frozen
throughout, so overfitting risk here is bounded by RF's own hyperparameters
(n_trees, max_depth), not by the ViT's ~86M parameters. See sslpipeline.md
Stage B for what fine-tuning would require instead.

Usage
-----
    python neural_geometry/dino_features.py   # once, to populate the cache
    python biomass_engine/train_dino.py
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from biomass_engine.predict_batch import extract_features, RF_FEATURES
from biomass_engine.models.random_forest import BiomassRandomForest
from biomass_engine.evaluation_metrics import extended_metrics, print_metrics_table
from shared.config import REPO_ROOT, RECON_OUTPUTS_DIR, NEURAL_GEOMETRY_DIR, EVAL_REPORTS_DIR
from biomass_engine.train_all import ESTIMATED_POT_MASS_SPECIMENS

GT_CSV        = REPO_ROOT / "dataset" / "ground_truth.csv"
FEATURES_DIR  = NEURAL_GEOMETRY_DIR / "dino_features"
PCA_COMPONENT_OPTIONS = [3, 5, 8, 12]


def load_dataset(exclude_ids=ESTIMATED_POT_MASS_SPECIMENS):
    rows = {}
    with GT_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["plant_id"]] = row

    geo_feats, dino_embeds, y, ids = [], [], [], []
    for specimen_id, row in sorted(rows.items()):
        if specimen_id in exclude_ids:
            continue
        stats_path = RECON_OUTPUTS_DIR / f"reconstruction_stats_specimen_{specimen_id}.txt"
        vertices_path = RECON_OUTPUTS_DIR / f"final_vertices_specimen_{specimen_id}.npy"
        dino_path = FEATURES_DIR / f"{specimen_id}.npz"
        if not (stats_path.exists() and vertices_path.exists() and dino_path.exists()):
            print(f"[DINO-RF] {specimen_id}: missing reconstruction or DINOv2 cache, skipping")
            continue

        geo_feats.append(extract_features(stats_path, vertices_path))
        dino_embeds.append(np.load(dino_path)["aggregated"])
        y.append(float(row["net_weight_g"]))
        ids.append(specimen_id)

    return geo_feats, np.array(dino_embeds), np.array(y), ids


def pca_fit_transform(X_train, X_test, n_components):
    """SVD-based PCA, fit on X_train only. X_train: (n-1, D), X_test: (1, D)."""
    mean = X_train.mean(axis=0, keepdims=True)
    Xc = X_train - mean
    n_components = min(n_components, Xc.shape[0] - 1, Xc.shape[1])
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    return Xc @ components.T, (X_test - mean) @ components.T


def loocv_rf(build_fold_X, y, n_trees=50, max_depth=2):
    """build_fold_X(train_idx, test_idx) -> (X_train, X_test) for one LOOCV fold."""
    preds = []
    for i in range(len(y)):
        train_idx = [j for j in range(len(y)) if j != i]
        X_train, X_test = build_fold_X(train_idx, [i])
        model = BiomassRandomForest()
        model.feature_names = [f"f{k}" for k in range(X_train.shape[1])]
        model.train(X_train, y[train_idx], n_trees=n_trees, max_depth=max_depth, min_samples_split=2)
        preds.append(model.predict(X_test)[0])
    preds = np.array(preds)
    err = preds - y
    r2 = 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    return preds, r2, mae, rmse


def main():
    geo_feats, dino_embeds, y, ids = load_dataset()
    print(f"[DINO-RF] Loaded {len(ids)} specimens (primary set, matches train_all.py)")

    X_geo_full = np.array([[f[name] for name in RF_FEATURES] for f in geo_feats])

    # --- 1. Geometric only (reference baseline) ---
    _, r2_geo, mae_geo, rmse_geo = loocv_rf(
        lambda tr, te: (X_geo_full[tr], X_geo_full[te]), y)
    print(f"\n1) Geometric only (baseline)    R2={r2_geo:+.3f}  MAE={mae_geo:.1f}g  RMSE={rmse_geo:.1f}g")

    results = []
    for k in PCA_COMPONENT_OPTIONS:
        # --- 2. DINOv2 only, k PCA components ---
        def build_dino(tr, te, k=k):
            return pca_fit_transform(dino_embeds[tr], dino_embeds[te], k)
        preds_d, r2_d, mae_d, rmse_d = loocv_rf(build_dino, y)

        # --- 3. Geometric + DINOv2, k PCA components ---
        def build_combined(tr, te, k=k):
            dtr, dte = pca_fit_transform(dino_embeds[tr], dino_embeds[te], k)
            return np.hstack([X_geo_full[tr], dtr]), np.hstack([X_geo_full[te], dte])
        preds_c, r2_c, mae_c, rmse_c = loocv_rf(build_combined, y)

        print(f"2) DINOv2 only  (k={k:2d} PCs)      R2={r2_d:+.3f}  MAE={mae_d:.1f}g  RMSE={rmse_d:.1f}g")
        print(f"3) Geometric + DINOv2 (k={k:2d})    R2={r2_c:+.3f}  MAE={mae_c:.1f}g  RMSE={rmse_c:.1f}g")
        results.append((k, r2_d, mae_d, rmse_d, r2_c, mae_c, rmse_c))

    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EVAL_REPORTS_DIR / "dino_stage_a_metrics.txt"
    with report_path.open("w") as f:
        f.write(f"Stage A: frozen DINOv2 ViT-B/14 vs. classical geometric features\n")
        f.write(f"n={len(ids)} (primary set, excludes {sorted(ESTIMATED_POT_MASS_SPECIMENS)})\n\n")
        f.write(f"1) Geometric only (baseline): R2={r2_geo:.3f}  MAE={mae_geo:.1f}g  RMSE={rmse_geo:.1f}g\n\n")
        f.write(f"{'k (PCA)':>8}{'DINO R2':>10}{'DINO MAE':>10}{'Comb R2':>10}{'Comb MAE':>10}\n")
        for k, r2_d, mae_d, rmse_d, r2_c, mae_c, rmse_c in results:
            f.write(f"{k:>8}{r2_d:>10.3f}{mae_d:>10.1f}{r2_c:>10.3f}{mae_c:>10.1f}\n")
        f.write("\nPer-fold PCA (fit on training embeddings only, per LOOCV fold) -- "
                "no leakage from the held-out specimen into the reduction.\n")
        f.write("DINOv2 backbone: frozen dinov2_vitb14, mean-pooled CLS token across all "
                "captured views per specimen. No fine-tuning (see sslpipeline.md Stage B).\n")
    print(f"\n[DINO-RF] Report -> {report_path}")


if __name__ == "__main__":
    main()
