#!/usr/bin/env python3
"""
evaluation_suite/comparison.py
================================
Cross-method comparison engine.

Compares all pipeline levels:
  Level 1 — Classical baseline (Procedure Alpha: RF + ANN)
  Level 2 — Neural Geometry   (DINOv2 + Volumetric Transformer)

Produces:
  - Per-method regression metrics table (MAE, RMSE, MARE, R²)
  - Paired statistical significance tests (Wilcoxon signed-rank)
  - Chamfer Distance comparison for 3D quality (if GT meshes available)
  - LaTeX-ready results table for the thesis

Usage
-----
    python comparison.py                           # all specimens
    python comparison.py --export eval_report.pdf  # save figure + CSV
"""

import sys
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import csv
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats_lib

from evaluation_suite.metrics import regression_report, mae, rmse, mare, r_squared
from shared.config import (
    RECON_OUTPUTS_DIR, GROUND_TRUTH_CSV,
    EVAL_REPORTS_DIR, EVAL_FIGURES_DIR,
)


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_level1_results() -> dict[str, dict]:
    """
    Parse all procedure_alpha/outputs/reconstruction_stats_*.txt files.
    Returns {label: {gt_kg, rf_kg, ann_kg, volume_m3, …}}
    """
    gt = _load_gt()
    results = {}
    for stats_file in sorted(RECON_OUTPUTS_DIR.glob("reconstruction_stats_*.txt")):
        label  = stats_file.stem.replace("reconstruction_stats_", "")
        rec    = {"label": label}
        with stats_file.open() as f:
            for line in f:
                line = line.strip()
                if "Volume" in line and ":" in line:
                    rec["volume_m3"]       = _parse_float(line)
                elif "Surface area" in line and ":" in line:
                    rec["surface_area_m2"] = _parse_float(line)
                elif "Height" in line and ":" in line:
                    rec["height_cm"]       = _parse_float(line)
                elif "Biomass (RF)" in line and ":" in line:
                    rec["rf_kg"]           = _parse_float(line)
                elif "Biomass (ANN)" in line and ":" in line:
                    rec["ann_kg"]          = _parse_float(line)
        for key in [label, label.replace("specimen_", ""),
                    label.replace("plant_", "plant_")]:
            if key in gt:
                rec["gt_kg"] = gt[key]
                break
        results[label] = rec
    return results


def load_level2_results(results_dir: Path = None) -> dict[str, dict]:
    """
    Parse neural_geometry prediction results (JSON per specimen).
    Expected: neural_geometry/outputs/{specimen_id}_ng_results.json
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "neural_geometry" / "outputs"
    gt = _load_gt()
    results = {}
    if results_dir.exists():
        for jf in sorted(results_dir.glob("*_ng_results.json")):
            label = jf.stem.replace("_ng_results", "")
            rec   = json.loads(jf.read_text())
            rec["label"] = label
            for key in [label, label.replace("specimen_", "")]:
                if key in gt:
                    rec["gt_kg"] = gt[key]
                    break
            results[label] = rec
    return results


def _load_gt() -> dict:
    """Return {label: total_mass_g}. Registry stores kg; comparisons here
    are in grams (the unit the field scale actually reports), so convert."""
    gt = {}
    if not GROUND_TRUTH_CSV.exists():
        return gt
    with GROUND_TRUTH_CSV.open() as f:
        for row in csv.DictReader(f):
            sid = row["specimen_id"]
            val = (row.get("agb_kg", "").strip() or
                   row.get("total_mass_kg", "").strip())
            if val:
                gt[sid] = float(val) * 1000
            lid = row.get("legacy_id", "").strip()
            if lid and val:
                gt[lid] = float(val) * 1000
    return gt


def _parse_float(line: str) -> float:
    try:
        return float(line.split(":")[-1].strip().split()[0])
    except (ValueError, IndexError):
        return float("nan")


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------

def run_comparison(export_path: Path = None) -> dict:
    """
    Execute the full cross-method comparison and return a summary dict.
    """
    l1 = load_level1_results()
    l2 = load_level2_results()

    # Build paired arrays for specimens with GT
    methods = {}

    for label, rec in l1.items():
        if "gt_kg" not in rec:
            continue
        methods.setdefault("RF",   []).append((rec["gt_kg"], rec.get("rf_kg",  float("nan"))))
        methods.setdefault("ANN",  []).append((rec["gt_kg"], rec.get("ann_kg", float("nan"))))

    for label, rec in l2.items():
        if "gt_kg" not in rec:
            continue
        methods.setdefault("NG-DINOv2", []).append(
            (rec["gt_kg"], rec.get("ng_biomass_kg", float("nan"))))

    # Compute metrics per method
    all_metrics = {}
    for method, pairs in methods.items():
        pairs = [(yt, yp) for yt, yp in pairs
                 if not (np.isnan(yt) or np.isnan(yp))]
        if not pairs:
            continue
        yt = np.array([p[0] for p in pairs])
        yp = np.array([p[1] for p in pairs])
        all_metrics[method] = regression_report(yt, yp, method)

    # Wilcoxon significance between RF and ANN (if both present)
    wilcoxon_result = None
    if "RF" in methods and "ANN" in methods:
        rf_pairs  = {yt: yp for yt, yp in methods["RF"]  if not np.isnan(yp)}
        ann_pairs = {yt: yp for yt, yp in methods["ANN"] if not np.isnan(yp)}
        shared = sorted(set(rf_pairs) & set(ann_pairs))
        if len(shared) >= 6:
            rf_err  = np.abs(np.array([rf_pairs[k]  - k for k in shared]))
            ann_err = np.abs(np.array([ann_pairs[k] - k for k in shared]))
            stat, p = stats_lib.wilcoxon(rf_err, ann_err)
            wilcoxon_result = dict(statistic=stat, p_value=p,
                                   significant=(p < 0.05))
            print(f"\n[Comparison] Wilcoxon RF vs ANN  stat={stat:.2f}  "
                  f"p={p:.4f}  sig={wilcoxon_result['significant']}")

    # Build figure
    fig = _build_comparison_figure(all_metrics, methods, wilcoxon_result)

    # Save report
    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_latex_table(all_metrics)
    _write_csv_report(all_metrics)

    if export_path:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(export_path), dpi=200, bbox_inches="tight")
        print(f"[Comparison] Figure saved → {export_path}")
    else:
        plt.show()

    return dict(metrics=all_metrics, wilcoxon=wilcoxon_result)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _build_comparison_figure(all_metrics: dict, methods: dict,
                               wilcoxon_result: dict = None) -> plt.Figure:
    """Build the cross-method comparison figure."""
    n_methods = len(all_metrics)
    if n_methods == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No evaluation data found",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    colours = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    method_names = list(all_metrics.keys())

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("ABVT3R — Cross-Method Evaluation",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # ── Bar charts: MAE, RMSE, MARE, R² ──────────────────────────────────
    metric_specs = [
        ("mae",  "MAE (g)",     "lower = better",  gs[0, 0]),
        ("rmse", "RMSE (g)",    "lower = better",  gs[0, 1]),
        ("mare", "MARE (%)",    "lower = better",  gs[0, 2]),
        ("r2",   "R²",          "higher = better", gs[1, 0]),
    ]
    for key, ylabel, note, gspec in metric_specs:
        ax = fig.add_subplot(gspec)
        vals  = [all_metrics[m][key] for m in method_names]
        bars  = ax.bar(method_names, vals,
                       color=colours[:len(method_names)], alpha=0.85, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n({note})", fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis='x', rotation=15, labelsize=8)

    # ── Radar / spider chart ──────────────────────────────────────────────
    ax_rad = fig.add_subplot(gs[1, 1], polar=True)
    _radar_chart(ax_rad, all_metrics, method_names, colours)

    # ── Significance annotation ───────────────────────────────────────────
    ax_sig = fig.add_subplot(gs[1, 2])
    ax_sig.axis("off")
    if wilcoxon_result:
        sig_txt = ("Wilcoxon signed-rank test\n"
                   "RF errors  vs  ANN errors\n\n"
                   f"  statistic : {wilcoxon_result['statistic']:.2f}\n"
                   f"  p-value   : {wilcoxon_result['p_value']:.4f}\n"
                   f"  α = 0.05  : {'Significant ✓' if wilcoxon_result['significant'] else 'Not significant ✗'}")
        ax_sig.text(0.5, 0.5, sig_txt, ha="center", va="center",
                    fontsize=10, transform=ax_sig.transAxes,
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.6",
                              facecolor="#E8F5E9" if wilcoxon_result["significant"] else "#FFF3E0",
                              edgecolor="#388E3C" if wilcoxon_result["significant"] else "#F57C00"))
    else:
        ax_sig.text(0.5, 0.5, "Statistical test\nnot computed\n(need ≥6 paired\nGT specimens)",
                    ha="center", va="center", fontsize=10, color="grey",
                    transform=ax_sig.transAxes)

    return fig


def _radar_chart(ax, all_metrics: dict, method_names: list, colours: list):
    """Normalised radar chart for multi-metric visual comparison."""
    categories   = ["MAE↓", "RMSE↓", "MARE↓", "R²↑"]
    keys         = ["mae", "rmse", "mare", "r2"]
    N            = len(categories)
    angles       = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles      += angles[:1]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_yticks([])
    ax.set_title("Normalised Performance\n(radar)", fontsize=9)

    # Normalise each metric to [0, 1] where 1 = best
    vals_matrix = np.array([[all_metrics[m][k] for k in keys]
                             for m in method_names])  # (M, 4)
    normed = vals_matrix.copy()
    for j, k in enumerate(keys):
        col = vals_matrix[:, j]
        if k == "r2":
            # Higher is better; normalise to [0,1]
            mn, mx = col.min(), col.max()
            normed[:, j] = (col - mn) / (mx - mn + 1e-9)
        else:
            # Lower is better; invert
            mn, mx = col.min(), col.max()
            normed[:, j] = 1 - (col - mn) / (mx - mn + 1e-9)

    for i, method in enumerate(method_names):
        vals = normed[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, color=colours[i], label=method)
        ax.fill(angles, vals, alpha=0.1, color=colours[i])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _write_latex_table(all_metrics: dict):
    """Write a LaTeX results table to evaluation_suite/reports/."""
    path = EVAL_REPORTS_DIR / "comparison_table.tex"
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Biomass prediction evaluation across all pipeline levels}",
        r"  \label{tab:comparison}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Method & MAE (g) & RMSE (g) & MARE (\%) & $R^2$ \\",
        r"    \midrule",
    ]
    for m, met in all_metrics.items():
        lines.append(
            f"    {m} & {met['mae']:.3f} & {met['rmse']:.3f} & "
            f"{met['mare']:.1f} & {met['r2']:.3f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))
    print(f"[Comparison] LaTeX table → {path}")


def _write_csv_report(all_metrics: dict):
    path = EVAL_REPORTS_DIR / "comparison_metrics.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n", "mae", "rmse", "mare", "r2", "r"])
        w.writeheader()
        for m, met in all_metrics.items():
            w.writerow({k: met.get(k, "") for k in w.fieldnames} | {"method": m})
    print(f"[Comparison] CSV report → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ABVT3R Cross-method Evaluation")
    p.add_argument("--export", default=None, metavar="FILE",
                   help="Export comparison figure to PNG/PDF")
    args = p.parse_args()
    run_comparison(Path(args.export) if args.export else None)
