#!/usr/bin/env python3
"""
acquisition/dataset/import_farm_dataset.py
=============================================
Bridges the field-collected dual-camera dataset (dataset/plants/{plant_id}/
{images,depth}/cam{A,B}_{angle:03d}.png, dataset/ground_truth.csv) into the
acquisition/dataset/specimens/ layout that
procedure_alpha.pipeline.ProcedureAlpha.run_specimen_dual() reads.

Usage
-----
    python acquisition/dataset/import_farm_dataset.py
    python acquisition/dataset/import_farm_dataset.py --plant M001
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import json
import numpy as np
from PIL import Image

from shared.config import (
    REPO_ROOT, SPECIMENS_DIR, GROUND_TRUTH_CSV,
    HALF_SWEEP_ANGLES_DEG, view_filename,
)

EXTERNAL_DATASET_DIR = REPO_ROOT / "dataset"
EXTERNAL_PLANTS_DIR  = EXTERNAL_DATASET_DIR / "plants"
EXTERNAL_GT_CSV       = EXTERNAL_DATASET_DIR / "ground_truth.csv"

GT_FIELDNAMES = [
    "specimen_id", "species", "variety", "collection_date",
    "collector", "location", "agb_kg", "total_mass_kg", "pot_mass_kg",
    "height_mm", "dbh_mm", "canopy_width_mm", "notes", "legacy_id",
]


def import_plant(plant_dir: Path, angles_deg=HALF_SWEEP_ANGLES_DEG) -> str:
    """Convert one field-collected plant folder into a specimens/ entry."""
    specimen_id = plant_dir.name
    spec_dir  = SPECIMENS_DIR / specimen_id
    rgb_dir   = spec_dir / "rgb"
    depth_dir = spec_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    for angle in angles_deg:
        for cam in ("A", "B"):
            src_img = plant_dir / "images" / f"cam{cam}_{angle:03d}.png"
            src_dep = plant_dir / "depth"  / f"cam{cam}_{angle:03d}.png"
            if not src_img.exists() or not src_dep.exists():
                raise FileNotFoundError(
                    f"Missing source file for {specimen_id} cam{cam} {angle}deg"
                )

            rgb   = np.array(Image.open(src_img).convert("RGB"))
            depth = np.array(Image.open(src_dep)).astype(np.uint16)

            Image.fromarray(rgb).save(rgb_dir / view_filename(angle, cam, "rgb", "jpg"))
            np.save(str(depth_dir / view_filename(angle, cam, "depth", "npy")), depth)

    meta = {
        "specimen_id": specimen_id,
        "angles_deg":  list(angles_deg),
        "protocol":    "dual_camera_6step",
        "source":      "imported from dataset/plants/ (field rig)",
    }
    (spec_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[Import] {specimen_id}: {len(angles_deg) * 2} views -> {spec_dir}")
    return specimen_id


def import_ground_truth(specimen_ids: set[str]):
    """Append/update rows in the project ground-truth registry from the field CSV."""
    if not EXTERNAL_GT_CSV.exists():
        print(f"[Import] No ground-truth CSV found at {EXTERNAL_GT_CSV}, skipping")
        return

    with EXTERNAL_GT_CSV.open(newline="") as f:
        field_rows = list(csv.DictReader(f))

    existing: list[dict] = []
    if GROUND_TRUTH_CSV.exists():
        with GROUND_TRUTH_CSV.open(newline="") as f:
            existing = list(csv.DictReader(f))
    by_id = {r["specimen_id"]: r for r in existing}

    for row in field_rows:
        if row["plant_id"] not in specimen_ids:
            continue
        by_id[row["plant_id"]] = {
            "specimen_id":     row["plant_id"],
            "species":         row.get("species_breed", ""),
            "variety":         "",
            "collection_date": row.get("date", ""),
            "collector":       "",
            "location":        "",
            "agb_kg":          f"{float(row['net_weight_g']) / 1000:.3f}" if row.get("net_weight_g") else "",
            "total_mass_kg":   f"{float(row['total_fresh_weight_with_pot_g']) / 1000:.3f}" if row.get("total_fresh_weight_with_pot_g") else "",
            "pot_mass_kg":     f"{float(row['pot_weight_g']) / 1000:.3f}" if row.get("pot_weight_g") else "",
            "height_mm": "", "dbh_mm": "", "canopy_width_mm": "",
            "notes":           f"pot_weight_source={row.get('pot_weight_source','')} {row.get('notes','')}".strip(),
            "legacy_id":       "",
        }

    GROUND_TRUTH_CSV.parent.mkdir(parents=True, exist_ok=True)
    with GROUND_TRUTH_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GT_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(by_id.values())
    print(f"[Import] Ground truth -> {GROUND_TRUTH_CSV} ({len(field_rows)} field rows merged)")


def main():
    p = argparse.ArgumentParser(description="Import field dual-camera dataset into specimens/")
    p.add_argument("--plant", help="Import a single plant ID (e.g. M001); default: all")
    args = p.parse_args()

    if not EXTERNAL_PLANTS_DIR.exists():
        print(f"[Import] {EXTERNAL_PLANTS_DIR} not found")
        sys.exit(1)

    plant_dirs = (
        [EXTERNAL_PLANTS_DIR / args.plant] if args.plant
        else sorted(d for d in EXTERNAL_PLANTS_DIR.iterdir() if d.is_dir())
    )

    imported = set()
    for d in plant_dirs:
        if not d.is_dir():
            continue
        imported.add(import_plant(d))

    import_ground_truth(imported)
    print(f"[Import] Done — {len(imported)} specimen(s) imported: {sorted(imported)}")


if __name__ == "__main__":
    main()
