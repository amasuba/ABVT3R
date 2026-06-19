"""
batch_run.py
============
Runs the Level-1 classical pipeline (Procedure Alpha) on all 40 legacy plants.

Usage
-----
    # Run all plants
    python batch_run.py

    # Run a subset (e.g. plants 1–10)
    python batch_run.py --start 1 --end 10

    # Re-run even if outputs already exist
    python batch_run.py --force

Prerequisites
-------------
    pip install numpy scikit-learn open3d --break-system-packages

Each plant requires 4 depth files in data_collection/:
    {angle}_degrees_depth_plant_{id}.npy   for angle in [0, 90, 180, 270]

Results are saved to procedure_alpha/outputs/ as:
    reconstruction_stats_plant_{id}.txt
    final_vertices_plant_{id}.npy
    final_triangles_plant_{id}.npy
    merged_points_plant_{id}.npy
    mesh_plant_{id}.ply   (if open3d is available)
"""

import argparse
import time
from pathlib import Path

import numpy as np

from procedure_alpha.pipeline import ProcedureAlpha
from shared.config import LEGACY_ANGLES_DEG, LEGACY_DATA_COLLECTION, RECON_OUTPUTS_DIR


def plant_data_complete(plant_id: int) -> bool:
    """Return True if all 4 depth files exist for this plant."""
    for angle in LEGACY_ANGLES_DEG:
        p = LEGACY_DATA_COLLECTION / f"{angle}_degrees_depth_plant_{plant_id}.npy"
        if not p.exists():
            return False
    return True


def output_exists(plant_id: int) -> bool:
    """Return True if reconstruction stats file already exists."""
    return (RECON_OUTPUTS_DIR / f"reconstruction_stats_plant_{plant_id}.txt").exists()


def main():
    parser = argparse.ArgumentParser(description="Batch Level-1 pipeline runner")
    parser.add_argument("--start", type=int, default=1,  help="First plant ID (default: 1)")
    parser.add_argument("--end",   type=int, default=40, help="Last plant ID  (default: 40)")
    parser.add_argument("--force", action="store_true",  help="Re-run even if output exists")
    args = parser.parse_args()

    pa = ProcedureAlpha()

    plant_ids   = list(range(args.start, args.end + 1))
    completed   = []
    skipped_no_data = []
    skipped_exists  = []
    failed      = []

    t_batch_start = time.time()

    print(f"\n{'='*60}")
    print(f"Batch Level-1 Pipeline  |  Plants {args.start}–{args.end}")
    print(f"{'='*60}\n")

    for plant_id in plant_ids:
        print(f"\n--- Plant {plant_id}/{args.end} ---")

        # Skip if data missing
        if not plant_data_complete(plant_id):
            missing = [
                f"{a}_degrees_depth_plant_{plant_id}.npy"
                for a in LEGACY_ANGLES_DEG
                if not (LEGACY_DATA_COLLECTION / f"{a}_degrees_depth_plant_{plant_id}.npy").exists()
            ]
            print(f"  [SKIP] Missing depth files: {missing}")
            skipped_no_data.append(plant_id)
            continue

        # Skip if already done (unless --force)
        if output_exists(plant_id) and not args.force:
            print(f"  [SKIP] Output already exists (use --force to re-run)")
            skipped_exists.append(plant_id)
            continue

        try:
            result = pa.run_legacy(plant_id)
            completed.append(plant_id)
            q = result['reconstruction'].get('surface_quality', {})
            mq = result['reconstruction'].get('mesh_quality', {})
            print(f"  [OK]  Q={q.get('overall_quality', 0):.3f}  "
                  f"V={mq.get('volume', 0)*1e6:.0f}cm³  "
                  f"t={result['elapsed_s']:.1f}s")
        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {e}")
            failed.append(plant_id)

    elapsed = time.time() - t_batch_start

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE  ({elapsed:.0f}s total)")
    print(f"  Processed   : {len(completed)}  {completed}")
    print(f"  No data     : {len(skipped_no_data)}  {skipped_no_data}")
    print(f"  Already done: {len(skipped_exists)}  {skipped_exists}")
    print(f"  Failed      : {len(failed)}  {failed}")
    print(f"{'='*60}\n")

    if skipped_no_data:
        print("NOTE: Add depth files to data_collection/ for skipped plants and re-run.")
        print("      Required naming: {angle}_degrees_depth_plant_{id}.npy")
        print("      Required angles: 0, 90, 180, 270\n")


if __name__ == "__main__":
    main()
