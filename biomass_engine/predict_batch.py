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

RF_FEATURES  = ['volume', 'surface_area', 'height', 'bbox_volume',
                'surface_to_volume_ratio', 'height_to_volume_ratio']
ANN_FEATURES = ['volume', 'surface_area', 'height', 'compactness', 'overall_quality']


def extract_features(stats_path: Path, vertices_path: Path) -> dict:
    """Same feature set as BiomassRandomForest.extract_features_from_reconstruction,
    adapted for procedure_alpha's specimen_{ID} file naming."""
    features = {}
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
    return features


def predict_all(specimen_ids=None):
    out_dir = RECON_OUTPUTS_DIR
    stats_files = sorted(out_dir.glob("reconstruction_stats_specimen_*.txt"))
    if specimen_ids:
        wanted = set(specimen_ids)
        stats_files = [f for f in stats_files
                       if f.stem.replace("reconstruction_stats_specimen_", "") in wanted]

    rf_model = BiomassRandomForest()
    rf_model.load_model(str(TRAINED_MODELS_DIR / "RF_model_mango" / "biomass_rf_model"))

    ann_model = BiomassANN()
    ann_model.load_model(str(TRAINED_MODELS_DIR / "ANN_model_mango" / "biomass_ann_model"))

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
        # Replace any previous prediction lines from an earlier run of this
        # script, so re-running stays idempotent instead of appending dupes.
        text = re.sub(r"\nBiomass \(RF\).*", "", text)
        text = re.sub(r"\nBiomass \(ANN\).*", "", text)
        text = text.rstrip("\n") + (
            f"\n\n--- Biomass Prediction (Mango models, grams) ---\n"
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
