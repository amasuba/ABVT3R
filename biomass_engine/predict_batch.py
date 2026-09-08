#!/usr/bin/env python3
"""
biomass_engine/predict_batch.py
==================================
Run the trained RF and ANN biomass models against every specimen in
procedure_alpha/outputs/, appending "Biomass (RF)"/"Biomass (ANN)" lines
to each specimen's reconstruction_stats.txt. Nothing in the codebase did
this before — biomass_engine/predict_rf.py is corrupted (binary pickle
data saved under a .py filename), and predict_ann.py is an unmodified copy
of the legacy single-plant script pointing at legacy paths (reconstruction
files named plant_{N}, not specimen_{ID}). This is what
biomass_engine/visualisation/results_dashboard.py actually reads.

Usage
-----
    python biomass_engine/predict_batch.py
    python biomass_engine/predict_batch.py --specimen M001
"""

import sys
import re
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# Imported at module scope (this script's __main__) so that unpickling the
# RF model — which was itself saved from a __main__ context — can resolve
# DecisionTreeRegressor/RandomForestRegressor by name.
from biomass_engine.models.random_forest import (
    DecisionTreeRegressor, RandomForestRegressor, BiomassRandomForest,
)
from biomass_engine.models.ann import BiomassANN
from shared.config import RECON_OUTPUTS_DIR, TRAINED_MODELS_DIR

# Shoot-only features: the ground truth (net_weight_g) is above-ground mass
# only, but the whole-mesh 'volume'/'surface_area'/'height' below include the
# pot and soil -- and the pot's share of total volume swings from ~12% to
# ~88% across specimens (pot size varies independently of plant size), which
# swamps any volume-to-mass relationship the model could otherwise learn.
# The pot/shoot split (ProcedureAlpha.segment_pot_shoot) already isolates the
# above-split-height region.
#
# LOOCV ablation on the 41-specimen set found that *replacing* the whole-mesh
# features with their shoot-only equivalents makes RF worse (R2 0.34 -> 0.22-
# 0.31 depending on which subset) -- shoot_surface_area/shoot_bbox_volume in
# particular are noisier than their whole-mesh counterparts, since they come
# from cutting the mesh at an approximate, heuristic split plane rather than
# using its actual outer surface. Keeping both whole-mesh and shoot-only
# features together (letting the tree ensemble pick whichever split is more
# informative) beat the original 6-feature baseline on R2, MAE, and RMSE
# simultaneously once max_depth was reduced to compensate for the added
# dimensionality (see train_all.py's train_rf). Note the R2 differences
# between all of these variants are within the ~0.5-wide 95% bootstrap CI on
# the baseline itself -- treat this as "no worse, more physically grounded,
# better point-estimate MAE," not a statistically confirmed win.
RF_FEATURES  = ['volume', 'surface_area', 'height', 'bbox_volume',
                'surface_to_volume_ratio', 'height_to_volume_ratio',
                'shoot_volume', 'shoot_height', 'on_pedestal']
ANN_FEATURES = ['volume', 'surface_area', 'height', 'bbox_volume', 'compactness',
                 'overall_quality', 'shoot_volume', 'shoot_height', 'on_pedestal']

# M001-M010 and E001-E010 were staged on the X001 inverted-pot pedestal for
# capture (confirmed 2026-09-07); E011-E020, the V-batch, and X001 itself
# were not. The pedestal sits below the true pot rim, so it inflates
# reconstructed "pot" volume for these 30 specimens (CropCraft's
# ggssvt/eval/pot_mass.py found the same confound independently, from
# implied pot density). Rather than guessing a pedestal volume to subtract --
# CropCraft's own writeup explicitly warns that reverse-estimating a volume
# that includes unweighed furniture "would replace one error with a larger
# one" -- this is surfaced as a feature so the model can account for the
# systematic difference itself.
PEDESTAL_SPECIMENS = {f"M{i:03d}" for i in range(1, 11)} | {f"E{i:03d}" for i in range(1, 11)}


def extract_features(stats_path: Path, vertices_path: Path) -> dict:
    """Same feature set as BiomassRandomForest.extract_features_from_reconstruction,
    adapted for procedure_alpha's specimen_{ID} file naming."""
    features = {}
    split_y = None
    for line in stats_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = line.strip()
        if s.startswith('Merged points'):
            features['merged_points'] = float(s.split(':')[1].replace(',', '').strip())
        elif s.startswith('Final vertices'):
            features['vertices'] = float(s.split(':')[1].replace(',', '').strip())
        elif s.startswith('Final triangles'):
            features['triangles'] = float(s.split(':')[1].replace(',', '').strip())
        elif s.startswith('Surface area'):
            features['surface_area'] = float(s.split(':')[1].strip().split()[0])
        elif s.startswith('Volume'):
            features['volume'] = float(s.split(':')[1].strip().split()[0])
        elif s.startswith('Overall quality Q'):
            features['overall_quality'] = float(s.split(':')[1].strip())
        elif s.startswith('Geometric fidelity'):
            features['geometric_fidelity'] = float(s.split(':')[1].strip())
        elif s.startswith('Surface smoothness'):
            features['smoothness'] = float(s.split(':')[1].strip())
        elif s.startswith('Split height (Y)'):
            split_y = float(s.split(':')[1].strip().split()[0])
        elif s.startswith('Shoot volume (approx)'):
            features['shoot_volume'] = float(s.split(':')[1].strip().split()[0])

    vertices = np.load(vertices_path)
    features['height']   = float(vertices[:, 1].max() - vertices[:, 1].min())
    features['width_x']  = float(vertices[:, 0].max() - vertices[:, 0].min())
    features['width_z']  = float(vertices[:, 2].max() - vertices[:, 2].min())
    features['bbox_volume'] = features['height'] * features['width_x'] * features['width_z']
    features['compactness'] = (features['volume'] / features['bbox_volume']
                                if features['bbox_volume'] > 0 else 0.0)
    features['surface_to_volume_ratio'] = (features['surface_area'] / features['volume']
                                            if features.get('volume', 0) > 0 else 0.0)
    features['height_to_volume_ratio'] = (features['height'] / features['volume']
                                           if features.get('volume', 0) > 0 else 0.0)

    # Shoot-only geometry: same split height the pipeline already computed,
    # applied to the mesh (points above it = above-ground plant).
    triangles_path = vertices_path.parent / vertices_path.name.replace(
        "final_vertices", "final_triangles")
    if split_y is not None and triangles_path.exists():
        triangles = np.load(triangles_path)
        tri_verts = vertices[triangles]                      # (T, 3, 3)
        tri_centroid_y = tri_verts[:, :, 1].mean(axis=1)      # (T,)
        shoot_tri = triangles[tri_centroid_y > split_y]
        if len(shoot_tri):
            v0, v1, v2 = vertices[shoot_tri[:, 0]], vertices[shoot_tri[:, 1]], vertices[shoot_tri[:, 2]]
            areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2
            features['shoot_surface_area'] = float(areas.sum())
        else:
            features['shoot_surface_area'] = 0.0

        shoot_verts = vertices[vertices[:, 1] > split_y]
        if len(shoot_verts):
            features['shoot_height']  = float(shoot_verts[:, 1].max() - split_y)
            features['shoot_width_x'] = float(shoot_verts[:, 0].max() - shoot_verts[:, 0].min())
            features['shoot_width_z'] = float(shoot_verts[:, 2].max() - shoot_verts[:, 2].min())
        else:
            features['shoot_height'] = features['shoot_width_x'] = features['shoot_width_z'] = 0.0
    else:
        features['shoot_surface_area'] = features.get('surface_area', 0.0)
        features['shoot_height'] = features.get('height', 0.0)
        features['shoot_width_x'] = features.get('width_x', 0.0)
        features['shoot_width_z'] = features.get('width_z', 0.0)

    features.setdefault('shoot_volume', features.get('volume', 0.0))
    features['shoot_bbox_volume'] = (features['shoot_height'] * features['shoot_width_x']
                                      * features['shoot_width_z'])
    features['shoot_compactness'] = (features['shoot_volume'] / features['shoot_bbox_volume']
                                      if features['shoot_bbox_volume'] > 0 else 0.0)
    features['shoot_surface_to_volume_ratio'] = (
        features['shoot_surface_area'] / features['shoot_volume']
        if features['shoot_volume'] > 0 else 0.0)
    features['shoot_height_to_volume_ratio'] = (
        features['shoot_height'] / features['shoot_volume']
        if features['shoot_volume'] > 0 else 0.0)

    specimen_id = stats_path.stem.replace("reconstruction_stats_specimen_", "")
    features['on_pedestal'] = 1.0 if specimen_id in PEDESTAL_SPECIMENS else 0.0

    return features


def predict_all(specimen_ids=None):
    out_dir = RECON_OUTPUTS_DIR
    stats_files = sorted(out_dir.glob("reconstruction_stats_specimen_*.txt"))
    if specimen_ids:
        wanted = set(specimen_ids)
        stats_files = [f for f in stats_files
                       if f.stem.replace("reconstruction_stats_specimen_", "") in wanted]

    rf_model = BiomassRandomForest()
    rf_model.load_model(str(TRAINED_MODELS_DIR / "RF_model_all" / "biomass_rf_model"))

    ann_model = BiomassANN()
    ann_model.load_model(str(TRAINED_MODELS_DIR / "ANN_model_all" / "biomass_ann_model"))

    for stats_path in stats_files:
        specimen_id = stats_path.stem.replace("reconstruction_stats_specimen_", "")
        vertices_path = out_dir / f"final_vertices_specimen_{specimen_id}.npy"
        if not vertices_path.exists():
            print(f"[Predict] {specimen_id}: missing {vertices_path.name}, skipping")
            continue

        features = extract_features(stats_path, vertices_path)

        X_rf = np.array([[features[f] for f in RF_FEATURES]])
        rf_g = float(rf_model.predict(X_rf)[0])

        X_ann = np.array([[features[f] for f in ANN_FEATURES]])
        ann_g = float(ann_model.predict(X_ann)[0, 0])

        text = stats_path.read_text()
        # Replace any previous prediction block (header included) from an
        # earlier run of this script, so re-running stays idempotent instead
        # of appending dupes. Strips from the first "--- Biomass Prediction"
        # header through to the end of the file.
        text = re.sub(r"\n---\s*Biomass Prediction.*\Z", "", text, flags=re.DOTALL)
        text = text.rstrip("\n") + (
            f"\n\n--- Biomass Prediction (all-species models, grams) ---\n"
            f"Biomass (RF)  : {rf_g:.1f} g\n"
            f"Biomass (ANN) : {ann_g:.1f} g\n"
        )
        stats_path.write_text(text)
        print(f"[Predict] {specimen_id}: RF={rf_g:.1f}g  ANN={ann_g:.1f}g")


def main():
    p = argparse.ArgumentParser(description="Batch RF/ANN biomass prediction")
    p.add_argument("--specimen", action="append",
                    help="Specific specimen ID(s); default: all in procedure_alpha/outputs/")
    args = p.parse_args()
    predict_all(args.specimen)


if __name__ == "__main__":
    main()
