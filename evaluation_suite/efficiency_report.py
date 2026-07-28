#!/usr/bin/env python3
"""
evaluation_suite/efficiency_report.py
========================================
Compile efficiency/practicality metrics (reference doc §7) across the
pipeline: reconstruction wall-clock, model complexity, and NeRF training
throughput. Required for the methods paper since architectural claims
invite cost questions; also the natural home for the views-vs-accuracy
question the reference doc calls out as directly relevant to the two-camera
hardware constraint.

Usage
-----
    python evaluation_suite/efficiency_report.py
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from shared.config import RECON_OUTPUTS_DIR, TRAINED_MODELS_DIR, EVAL_REPORTS_DIR, REPO_ROOT

MANGO_IDS = [f"M{i:03d}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# procedure_alpha reconstruction timing (measured, from stats.txt)
# ---------------------------------------------------------------------------

def reconstruction_timing():
    times, verts, tris, merged_pts = [], [], [], []
    for sid in MANGO_IDS:
        stats_path = RECON_OUTPUTS_DIR / f"reconstruction_stats_specimen_{sid}.txt"
        if not stats_path.exists():
            continue
        text = stats_path.read_text()
        m = re.search(r"Processing time\s*:\s*([\d.]+)\s*s", text)
        if m:
            times.append(float(m.group(1)))
        m = re.search(r"Final vertices\s*:\s*([\d,]+)", text)
        if m:
            verts.append(int(m.group(1).replace(",", "")))
        m = re.search(r"Final triangles\s*:\s*([\d,]+)", text)
        if m:
            tris.append(int(m.group(1).replace(",", "")))
        m = re.search(r"Merged points\s*:\s*([\d,]+)", text)
        if m:
            merged_pts.append(int(m.group(1).replace(",", "")))
    return dict(times=times, verts=verts, tris=tris, merged_pts=merged_pts)


# ---------------------------------------------------------------------------
# Model complexity
# ---------------------------------------------------------------------------

def count_tree_nodes(node) -> int:
    if node.get("leaf"):
        return 1
    return 1 + count_tree_nodes(node["left"]) + count_tree_nodes(node["right"])


def rf_complexity():
    from biomass_engine.models.random_forest import (  # noqa: F401  (needed for unpickling)
        DecisionTreeRegressor, RandomForestRegressor, BiomassRandomForest,
    )
    model = BiomassRandomForest()
    model.load_model(str(TRAINED_MODELS_DIR / "RF_model_mango" / "biomass_rf_model"))
    total_nodes = sum(count_tree_nodes(t.tree) for t in model.model.trees)
    return dict(n_trees=model.model.n_trees, max_depth=model.model.max_depth,
                total_nodes=total_nodes, n_features=len(model.feature_names))


def ann_complexity():
    from biomass_engine.models.ann import BiomassANN
    model = BiomassANN()
    model.load_model(str(TRAINED_MODELS_DIR / "ANN_model_mango" / "biomass_ann_model"))
    n_params = sum(w.size for w in model.weights) + sum(b.size for b in model.biases)
    architecture = [model.weights[0].shape[0]] + [w.shape[1] for w in model.weights]
    return dict(architecture=architecture, n_params=n_params)


# ---------------------------------------------------------------------------
# NeRF training throughput (from the M001 training runs this session)
# ---------------------------------------------------------------------------

NERF_OBSERVED = dict(
    gpu="NVIDIA GeForce RTX 2050 (4GB VRAM)",
    iter_time_range_s=(0.345, 0.475),
    rays_per_batch=2048,
    rays_per_sec_range=(4600, 6150),
    vram_used_gb_range=(1.9, 3.5),
    runs=[
        dict(name="pilot", iterations=1000, note="plumbing validation only, floater-dominated"),
        dict(name="full",  iterations=50000, note="user-run; ~4.8-6.6h estimated from per-iter timing"),
    ],
)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 72)
    out("EFFICIENCY REPORT — ABVT3R")
    out("=" * 72)

    out("\n--- procedure_alpha reconstruction (classical, 12-view dual-camera) ---")
    rt = reconstruction_timing()
    if rt["times"]:
        t = np.array(rt["times"])
        out(f"  Specimens measured   : {len(t)}")
        out(f"  Wall-clock time      : mean={t.mean():.1f}s  std={t.std():.1f}s  "
            f"min={t.min():.1f}s  max={t.max():.1f}s")
        out(f"  Time per view        : {t.mean()/12:.2f}s  (12 views/specimen)")
        out(f"  Throughput           : {3600/t.mean():.1f} specimens/hour  "
            f"(single-threaded, no batching)")
    if rt["merged_pts"]:
        mp, v, tr = np.array(rt["merged_pts"]), np.array(rt["verts"]), np.array(rt["tris"])
        out(f"  Merged points        : mean={mp.mean():,.0f}  (range {mp.min():,}-{mp.max():,})")
        out(f"  Final mesh           : mean {v.mean():,.0f} vertices, {tr.mean():,.0f} triangles")

    out("\n--- Biomass model complexity ---")
    try:
        rf = rf_complexity()
        out(f"  RF  : {rf['n_trees']} trees, max_depth={rf['max_depth']}, "
            f"{rf['total_nodes']} total decision nodes, {rf['n_features']} input features")
    except Exception as e:
        out(f"  RF  : could not load ({e})")
    try:
        ann = ann_complexity()
        out(f"  ANN : architecture {ann['architecture']}, {ann['n_params']} parameters")
    except Exception as e:
        out(f"  ANN : could not load ({e})")

    out("\n--- NeRF (Nerfstudio nerfacto, assumed-geometry poses) ---")
    out(f"  GPU                  : {NERF_OBSERVED['gpu']}")
    out(f"  Rays per batch       : {NERF_OBSERVED['rays_per_batch']}")
    lo, hi = NERF_OBSERVED["iter_time_range_s"]
    out(f"  Iteration time       : {lo:.3f}-{hi:.3f} s/iter (observed range)")
    lo, hi = NERF_OBSERVED["rays_per_sec_range"]
    out(f"  Throughput           : {lo:,}-{hi:,} rays/sec")
    lo, hi = NERF_OBSERVED["vram_used_gb_range"]
    out(f"  VRAM used            : {lo:.1f}-{hi:.1f} GB (of 4.0 GB total)")
    for run in NERF_OBSERVED["runs"]:
        out(f"  Run '{run['name']}'         : {run['iterations']:,} iterations — {run['note']}")

    out("\n--- Views-vs-accuracy ---")
    out("  Not yet measured — would need reconstructing the same specimens with")
    out("  a reduced view subset (e.g. 6 or 4 of the 12) and comparing mesh")
    out("  quality/biomass error against the full 12-view result. The codebase")
    out("  already supports this: ProcedureAlpha.run_specimen_dual(specimen_id,")
    out("  half_angles_deg=<subset>) takes an explicit angle subset. Flagged as")
    out("  a scoped follow-up experiment rather than run speculatively here —")
    out("  see Pipeline.md.")

    out("\n" + "=" * 72)

    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EVAL_REPORTS_DIR / "efficiency_report.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\n[Efficiency] Report saved -> {report_path}")


if __name__ == "__main__":
    main()
