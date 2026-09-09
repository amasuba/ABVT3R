#!/usr/bin/env python3
"""
biomass_engine/train_all.py
==============================
Train RF and ANN biomass models on species currently in
dataset/ground_truth.csv (Mango + Eucalyptus, as of the 2026-09 batch that
added E001-E020/V001-V011). Mirrors train_mango.py's structure and metrics
but is species-agnostic: the RF/ANN feature set is purely geometric
(volume, surface area, height, ...), so nothing here treats species as a
model input -- it's reported for context only.

The PRIMARY/deployed model (what gets saved to RF_model_all/ANN_model_all,
and the number to cite) excludes V009-V011: they have *estimated*, not
measured, pot mass (dataset/README.md flags pot-mass provenance as the
least reliable ground-truth source), and V010 alone carried 25%+ of RF's
squared LOOCV error when included. This was a deliberate, reviewed decision
(RF LOOCV R² 0.363 -> 0.451 on n=38 vs n=41) -- not the default behaviour of
this script historically, and not something to re-derive silently if this
file is copied elsewhere. The full n=41 number is still computed and
reported alongside it, labeled as a reference point, so the exclusion stays
visible rather than disappearing into the training set.

Biomass is in GRAMS throughout, matching dataset/ground_truth.csv's
net_weight_g column (as-collected/fresh mass, not oven-dry -- see
dataset/README.md).

n is still small for an MLP (currently ~38, vs. the 1000+ the reference
thesis flags as needed for a neural net to generalise reliably) -- read
the ANN LOOCV result with real skepticism, same caveat as train_mango.py.

Usage
-----
    python biomass_engine/train_all.py
"""

import sys
import csv
import io
import contextlib
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from biomass_engine.models.random_forest import (
    DecisionTreeRegressor, RandomForestRegressor, BiomassRandomForest,
)
from biomass_engine.models.ann import BiomassANN
from biomass_engine.predict_batch import extract_features, RF_FEATURES, ANN_FEATURES
from biomass_engine.evaluation_metrics import (
    extended_metrics, print_metrics_table, save_bland_altman_figure,
    leverage_report, bootstrap_interval,
)
from shared.config import REPO_ROOT, RECON_OUTPUTS_DIR, TRAINED_MODELS_DIR, EVAL_FIGURES_DIR, EVAL_REPORTS_DIR

GT_CSV = REPO_ROOT / "dataset" / "ground_truth.csv"

# V009-V011 have *estimated*, not measured, pot mass (dataset/README.md flags
# pot-mass provenance as the least reliable ground-truth source) -- and V010
# alone already carried 25%+ of RF's squared LOOCV error when included.
# Excluded from the PRIMARY training set below (see module docstring); the
# full n=41 sample is still computed and reported as a labeled reference
# point in main(), so this exclusion stays visible rather than silent.
ESTIMATED_POT_MASS_SPECIMENS = {"V009", "V010", "V011"}

# V010: estimated pot mass, dominant LOOCV leverage (see above). V011: only
# 9 of 12 views captured (see Pipeline.md), reduced-view reconstruction --
# a geometry-quality issue, distinct from V010's target-quality one. X001
# was never a plant (the pedestal specimens were staged on) and was never
# in dataset/ground_truth.csv or dataset/plants/ to begin with -- nothing to
# drop there, it's listed for completeness.
USER_REQUESTED_DROP = {"V010", "V011"}

# Shoot-only feature set: every whole-mesh (pot+soil-inclusive) feature
# removed, keeping only quantities computed from the above-split-height
# region plus on_pedestal. This is the pure version of the "shoot volume
# matches the target better" hypothesis, without the whole-mesh features
# the kitchen-sink approach above found necessary to keep RF stable.
SHOOT_ONLY_RF_FEATURES = ['shoot_volume', 'shoot_surface_area', 'shoot_height', 'shoot_bbox_volume',
                          'shoot_surface_to_volume_ratio', 'shoot_height_to_volume_ratio', 'on_pedestal']
SHOOT_ONLY_ANN_FEATURES = ['shoot_volume', 'shoot_surface_area', 'shoot_height', 'shoot_compactness',
                           'overall_quality', 'on_pedestal']


def load_dataset(exclude_ids=frozenset()):
    """Return features/weights/ids/species for every specimen with both
    ground truth and a completed reconstruction, minus exclude_ids."""
    rows = {}
    with GT_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["plant_id"]] = row

    X, y, ids, species = [], [], [], []
    for specimen_id, row in sorted(rows.items()):
        if specimen_id in exclude_ids:
            continue
        stats_path = RECON_OUTPUTS_DIR / f"reconstruction_stats_specimen_{specimen_id}.txt"
        vertices_path = RECON_OUTPUTS_DIR / f"final_vertices_specimen_{specimen_id}.npy"
        if not stats_path.exists() or not vertices_path.exists():
            print(f"[Train] {specimen_id}: no reconstruction yet, skipping")
            continue
        features = extract_features(stats_path, vertices_path)
        X.append(features)
        y.append(float(row["net_weight_g"]))
        ids.append(specimen_id)
        species.append(row["species_breed"])

    return X, np.array(y), ids, species


def train_rf(X_feat_dicts, y, ids):
    print("\n" + "=" * 70)
    print("RANDOM FOREST — all species, grams")
    print("=" * 70)
    X = np.array([[f[name] for name in RF_FEATURES] for f in X_feat_dicts])

    model = BiomassRandomForest()
    model.feature_names = RF_FEATURES
    # max_depth=2, down from 3: with 9 features (up from 6) at n=41, shallower
    # trees generalise better in LOOCV -- see the ablation note in
    # predict_batch.py's RF_FEATURES comment.
    loo = model.leave_one_out_cv(X, y, n_trees=50, max_depth=2)
    print(f"\nLOOCV (n={len(y)}): R²={loo['r2']:.3f}  MAE={loo['mae']:.1f}g  RMSE={loo['rmse']:.1f}g")

    model.train(X, y, n_trees=50, max_depth=2, min_samples_split=2)
    out_dir = TRAINED_MODELS_DIR / "RF_model_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "biomass_rf_model"))
    return loo


def _ann_loocv(X, y, feature_names, verbose=True):
    """Leave-one-out CV for the ANN: fresh network each fold, fixed epoch
    count (n too small for an internal val split / early stopping).
    Returns (predictions, r2, mae, rmse)."""
    y_col = y.reshape(-1, 1)
    architecture = [X.shape[1], 4, 2, 1]

    preds = []
    for i in range(len(y)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y_col, i, axis=0)
        X_test  = X[i:i + 1]

        fold_model = BiomassANN()
        fold_model.feature_names = feature_names
        fold_model.initialize_network(architecture)
        fold_model.train(X_train, y_train, epochs=200, learning_rate=0.001, verbose=False)
        pred = fold_model.predict(X_test)[0, 0]
        preds.append(pred)
        if verbose:
            print(f"  actual={y[i]:.1f}g  predicted={pred:.1f}g")

    preds = np.array(preds)
    err = preds - y
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    return preds, r2, mae, rmse


def train_ann(X_feat_dicts, y, ids):
    print("\n" + "=" * 70)
    print(f"ANN — all species, grams (n={len(y)})")
    print("=" * 70)
    X = np.array([[f[name] for name in ANN_FEATURES] for f in X_feat_dicts])

    preds, r2, mae, rmse = _ann_loocv(X, y, ANN_FEATURES, verbose=False)
    for sid, actual, pred in zip(ids, y, preds):
        print(f"  {sid}: actual={actual:.1f}g  predicted={pred:.1f}g")
    print(f"\nLOOCV (n={len(y)}): R²={r2:.3f}  MAE={mae:.1f}g  RMSE={rmse:.1f}g")

    # Final model, trained on all data — this is the one whose training
    # loss curve gets plotted (LOOCV folds are 200-epoch fits on n-1 points
    # each; the curve worth reporting is the deployed model's).
    architecture = [X.shape[1], 4, 2, 1]
    final_model = BiomassANN()
    final_model.feature_names = ANN_FEATURES
    final_model.initialize_network(architecture)
    final_model.train(X, y.reshape(-1, 1), epochs=200, learning_rate=0.001, verbose=False)
    out_dir = TRAINED_MODELS_DIR / "ANN_model_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(out_dir / "biomass_ann_model"))
    final_model.plot_training_hist(save_path=str(EVAL_FIGURES_DIR / "all_species_ann_training_loss.png"))
    return dict(r2=r2, mae=mae, rmse=rmse, predictions=preds, actuals=y)


def sensitivity_check(X_feat_dicts, y, ids, exclude_ids, label,
                       rf_features=None, ann_features=None, max_depth=2):
    """LOOCV restricted to specimens outside exclude_ids, optionally with a
    different feature set (rf_features/ann_features default to the module's
    RF_FEATURES/ANN_FEATURES). Same hyperparameters as the headline run
    otherwise (n_trees=50, ANN 200 epochs), so this isolates the effect of
    the sample/feature change being tested, not "the best model we can find
    on fewer points." Prints and returns a report block; does not save any
    model -- these are reporting-only comparisons against the headline."""
    rf_features = rf_features or RF_FEATURES
    ann_features = ann_features or ANN_FEATURES

    mask = np.array([sid not in exclude_ids for sid in ids])
    n = int(mask.sum())
    sub_feat = [f for f, keep in zip(X_feat_dicts, mask) if keep]
    sub_y = y[mask]

    X_rf = np.array([[f[name] for name in rf_features] for f in sub_feat])
    rf_model = BiomassRandomForest()
    rf_model.feature_names = rf_features
    with contextlib.redirect_stdout(io.StringIO()):
        rf_loo = rf_model.leave_one_out_cv(X_rf, sub_y, n_trees=50, max_depth=max_depth)

    X_ann = np.array([[f[name] for name in ann_features] for f in sub_feat])
    _, ann_r2, ann_mae, ann_rmse = _ann_loocv(X_ann, sub_y, ann_features, verbose=False)

    lines = [
        f"Sensitivity check — {label} (excludes {sorted(exclude_ids) or 'none'}, n={n} of {len(ids)}):",
        f"  RF features : {rf_features}",
        f"  ANN features: {ann_features}",
        f"  RF : R²={rf_loo['r2']:.3f}  MAE={rf_loo['mae']:.1f}g  RMSE={rf_loo['rmse']:.1f}g",
        f"  ANN: R²={ann_r2:.3f}  MAE={ann_mae:.1f}g  RMSE={ann_rmse:.1f}g",
        f"  (full n={len(ids)} headline for comparison, not replaced by this)",
    ]
    print("\n" + "\n".join(lines))
    return lines


def main():
    # Full n=41 set is still loaded first -- needed for the reference check
    # and the shoot-only exploratory check below, both of which report
    # against the complete specimen list regardless of what's excluded from
    # the primary training set.
    X_feat_dicts_full, y_full, ids_full, species_full = load_dataset()

    # PRIMARY / deployed training set: excludes V009-V011 (estimated pot
    # mass -- see module docstring). This is what gets saved as
    # RF_model_all/ANN_model_all and the number to cite.
    X_feat_dicts, y, ids, species = load_dataset(exclude_ids=ESTIMATED_POT_MASS_SPECIMENS)
    print(f"[Train] Loaded {len(ids)} specimens (excludes {sorted(ESTIMATED_POT_MASS_SPECIMENS)}): {ids}")
    print(f"[Train] Species breakdown: {dict(Counter(species))}")
    print(f"[Train] Weight range: {y.min():.0f}g - {y.max():.0f}g")

    rf_result  = train_rf(X_feat_dicts, y, ids)
    ann_result = train_ann(X_feat_dicts, y, ids)

    print("\n" + "=" * 70)
    print(f"SUMMARY (leave-one-out CV, n={len(y)}, excludes estimated pot mass)")
    print("=" * 70)
    print(f"  RF : R²={rf_result['r2']:.3f}  MAE={rf_result['mae']:.1f}g")
    print(f"  ANN: R²={ann_result['r2']:.3f}  MAE={ann_result['mae']:.1f}g")

    # ------------------------------------------------------------------
    # Extended metrics: Bias, nRMSE, MARE, Lin's CCC, Bland-Altman
    # ------------------------------------------------------------------
    rf_metrics  = extended_metrics(rf_result["actuals"],  rf_result["predictions"])
    ann_metrics = extended_metrics(ann_result["actuals"], ann_result["predictions"])
    print_metrics_table({"RF": rf_metrics, "ANN": ann_metrics})

    # ------------------------------------------------------------------
    # Small-n diagnostics (CropCraft ggssvt eval suite): does one specimen
    # dominate the error, and how wide is the CI on RMSE/R²?
    # ------------------------------------------------------------------
    report_lines = []
    for name, result in (("RF", rf_result), ("ANN", ann_result)):
        lev = leverage_report(result["actuals"], result["predictions"])
        rmse_lo, rmse_hi = bootstrap_interval(result["actuals"], result["predictions"], metric="rmse")
        r2_lo, r2_hi = bootstrap_interval(result["actuals"], result["predictions"], metric="r2")
        worst_id = ids[lev["worst"]] if lev["worst"] is not None else None
        line = (f"{name}: worst specimen={worst_id} carries {lev['share']*100:.1f}% of "
                f"squared error (even share would be {lev['even_share']*100:.1f}%){'  <-- DOMINATED' if lev['dominated'] else ''}\n"
                f"{name}: 95% bootstrap CI  RMSE=[{rmse_lo:.1f}, {rmse_hi:.1f}]g  R²=[{r2_lo:.3f}, {r2_hi:.3f}]")
        print(f"\n{line}")
        report_lines.append(line)

    # Reference point: the full n=41 sample, no exclusion, same kitchen-sink
    # features as the primary run above. Kept visible so the V009-V011
    # exclusion stays a documented, reviewable decision rather than
    # disappearing into "the" training set.
    reference_lines = sensitivity_check(
        X_feat_dicts_full, y_full, ids_full, set(),
        "reference — full sample, no exclusion",
    )

    # Explored and not adopted: restricting to shoot-only features (dropping
    # V010/V011 for the same estimated-pot-mass reason, on the full set)
    # underperformed the kitchen-sink approach used above by ~0.03-0.05 R²
    # at every sample size tested. Kept as a documented negative result.
    shoot_only_lines = sensitivity_check(
        X_feat_dicts_full, y_full, ids_full, USER_REQUESTED_DROP,
        "explored, not adopted — drop V010/V011, shoot-only features",
        rf_features=SHOOT_ONLY_RF_FEATURES, ann_features=SHOOT_ONLY_ANN_FEATURES,
    )

    EVAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_bland_altman_figure(
        rf_result["actuals"],  rf_result["predictions"],
        ann_result["actuals"], ann_result["predictions"],
        EVAL_FIGURES_DIR / "all_species_bland_altman.png",
    )

    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EVAL_REPORTS_DIR / "train_all_metrics.txt"
    with report_path.open("w") as f:
        f.write(f"PRIMARY (deployed model): n={len(ids)}, excludes "
                f"{sorted(ESTIMATED_POT_MASS_SPECIMENS)} (estimated pot mass)\n")
        f.write(f"Specimens: {ids}\n")
        f.write(f"Species breakdown: {dict(Counter(species))}\n")
        f.write(f"Weight range: {y.min():.0f}g - {y.max():.0f}g\n\n")
        f.write(f"{'Model':<8}{'n':>4}{'MAE (g)':>10}{'RMSE (g)':>10}"
                f"{'Bias (g)':>10}{'nRMSE':>9}{'MARE':>8}{'R2':>8}{'CCC':>8}\n")
        for name, m in (("RF", rf_metrics), ("ANN", ann_metrics)):
            f.write(f"{name:<8}{m['n']:>4}{m['mae']:>10.1f}{m['rmse']:>10.1f}"
                    f"{m['bias']:>+10.1f}{m['nrmse']:>9.3f}{m['mare']:>8.3f}"
                    f"{m['r2']:>8.3f}{m['ccc']:>8.3f}\n")
        f.write("\n" + "\n".join(report_lines) + "\n")
        f.write("\n" + "\n".join(reference_lines) + "\n")
        f.write("\n" + "\n".join(shoot_only_lines) + "\n")
    print(f"\n[Train] Metrics report -> {report_path}")


if __name__ == "__main__":
    main()
