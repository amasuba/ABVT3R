#!/usr/bin/env python3
"""
biomass_engine/visualisation/results_dashboard.py
===================================================
Rich interactive dashboard for biomass prediction results.

Panels
------
1. Predicted vs Measured scatter  (RF and ANN side by side)
2. Residual distribution          (histogram + Q-Q plot)
3. Feature importance             (RF Gini importance)
4. Error by specimen              (sorted bar chart — identify outliers)
5. Regression metrics table       (MAE, RMSE, MARE, R²)
6. 3D mesh preview                (if Open3D available)

Usage
-----
    python results_dashboard.py                        # load all from outputs/
    python results_dashboard.py --export report.pdf    # save to PDF
    python results_dashboard.py --specimen DG001_...   # single specimen
"""

import sys
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

import csv
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import scipy.stats as stats

from shared.config import (
    RECON_OUTPUTS_DIR, GROUND_TRUTH_CSV,
    BIOMASS_ENGINE_DIR, EVAL_FIGURES_DIR,
)

EVAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette (publication-quality)
# ---------------------------------------------------------------------------
PALETTE = {
    "rf":      "#2196F3",   # blue
    "ann":     "#FF5722",   # deep orange
    "perfect": "#4CAF50",   # green
    "fill_rf": "#BBDEFB",
    "fill_ann":"#FFCCBC",
    "neutral": "#607D8B",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_gt() -> dict:
    """Return {label: total_mass_kg} from ground truth registry."""
    gt = {}
    if not GROUND_TRUTH_CSV.exists():
        return gt
    with GROUND_TRUTH_CSV.open() as f:
        for row in csv.DictReader(f):
            sid = row["specimen_id"]
            # prefer AGB, fallback to total_mass_kg
            val = row.get("agb_kg", "").strip() or row.get("total_mass_kg", "").strip()
            if val:
                gt[sid] = float(val)
            # also index by legacy_id
            lid = row.get("legacy_id", "").strip()
            if lid and val:
                gt[lid] = float(val)
    return gt


def load_prediction_results(outputs_dir: Path = RECON_OUTPUTS_DIR) -> list[dict]:
    """
    Parse reconstruction_stats_*.txt files and return list of result dicts.
    Each dict: {label, gt_kg, rf_kg, ann_kg, volume_m3, surface_area_m2, height_cm}
    """
    gt = _load_gt()
    records = []
    for stats_file in sorted(outputs_dir.glob("reconstruction_stats_*.txt")):
        label  = stats_file.stem.replace("reconstruction_stats_", "")
        record = {"label": label}
        with stats_file.open() as f:
            for line in f:
                line = line.strip()
                if "Volume" in line and ":" in line:
                    record["volume_m3"]       = float(line.split(":")[-1].strip().split()[0])
                elif "Surface area" in line and ":" in line:
                    record["surface_area_m2"] = float(line.split(":")[-1].strip().split()[0])
                elif "Height" in line and ":" in line:
                    record["height_cm"]       = float(line.split(":")[-1].strip().split()[0])
                elif "Biomass (RF)" in line and ":" in line:
                    record["rf_kg"]           = float(line.split(":")[-1].strip().split()[0])
                elif "Biomass (ANN)" in line and ":" in line:
                    record["ann_kg"]          = float(line.split(":")[-1].strip().split()[0])

        # Ground truth lookup
        for key in [label, label.replace("specimen_", ""), label.replace("plant_", "plant_")]:
            if key in gt:
                record["gt_kg"] = gt[key]
                break

        if "rf_kg" in record or "ann_kg" in record:
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err  = y_pred - y_true
    rel  = np.abs(err) / (np.abs(y_true) + 1e-9)
    ss_r = np.sum(err ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    return dict(
        n    = len(y_true),
        mae  = np.mean(np.abs(err)),
        rmse = np.sqrt(np.mean(err ** 2)),
        mare = np.mean(rel) * 100,
        r2   = 1.0 - ss_r / (ss_t + 1e-12),
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def build_dashboard(records: list[dict], export_path: Path = None):
    """
    Build the full results dashboard figure.

    Parameters
    ----------
    records     : list of dicts from load_prediction_results
    export_path : if given, save figure to this path instead of showing it
    """
    # Filter to records that have ground truth
    gt_records = [r for r in records if "gt_kg" in r]
    all_records = records

    has_gt = len(gt_records) > 0
    has_rf  = any("rf_kg"  in r for r in all_records)
    has_ann = any("ann_kg" in r for r in all_records)

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    fig.suptitle("ABVT3R — Biomass Prediction Results Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)

    gs_top    = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35,
                                  top=0.93, bottom=0.55)
    gs_bottom = gridspec.GridSpec(1, 3, figure=fig, hspace=0.4, wspace=0.35,
                                  top=0.50, bottom=0.05)

    # ------------------------------------------------------------------
    # Panel 1a: RF predicted vs measured
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs_top[0, 0])
    if has_gt and has_rf:
        y_true = np.array([r["gt_kg"]  for r in gt_records if "rf_kg" in r])
        y_pred = np.array([r["rf_kg"]  for r in gt_records if "rf_kg" in r])
        _scatter_panel(ax1, y_true, y_pred, "Random Forest",
                       PALETTE["rf"], PALETTE["fill_rf"])
    else:
        ax1.text(0.5, 0.5, "No RF + GT data", ha="center", va="center",
                 transform=ax1.transAxes, color="grey")
        ax1.set_title("RF Pred vs Measured")

    # ------------------------------------------------------------------
    # Panel 1b: ANN predicted vs measured
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs_top[0, 1])
    if has_gt and has_ann:
        y_true_a = np.array([r["gt_kg"]  for r in gt_records if "ann_kg" in r])
        y_pred_a = np.array([r["ann_kg"] for r in gt_records if "ann_kg" in r])
        _scatter_panel(ax2, y_true_a, y_pred_a, "ANN",
                       PALETTE["ann"], PALETTE["fill_ann"])
    else:
        ax2.text(0.5, 0.5, "No ANN + GT data", ha="center", va="center",
                 transform=ax2.transAxes, color="grey")
        ax2.set_title("ANN Pred vs Measured")

    # ------------------------------------------------------------------
    # Panel 2: Residual histogram
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs_top[0, 2])
    if has_gt:
        residuals_rf  = ([r["rf_kg"]  - r["gt_kg"] for r in gt_records if "rf_kg"  in r]
                         if has_rf  else [])
        residuals_ann = ([r["ann_kg"] - r["gt_kg"] for r in gt_records if "ann_kg" in r]
                         if has_ann else [])
        if residuals_rf:
            ax3.hist(residuals_rf,  bins=10, alpha=0.65, label="RF",
                     color=PALETTE["rf"],  edgecolor="white")
        if residuals_ann:
            ax3.hist(residuals_ann, bins=10, alpha=0.65, label="ANN",
                     color=PALETTE["ann"], edgecolor="white")
        ax3.axvline(0, color="black", lw=1.5, ls="--")
        ax3.set_xlabel("Residual (kg)")
        ax3.set_ylabel("Count")
        ax3.set_title("Residual Distribution")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No GT available", ha="center", va="center",
                 transform=ax3.transAxes, color="grey")
        ax3.set_title("Residual Distribution")

    # ------------------------------------------------------------------
    # Panel 3: Metrics table
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs_top[0, 3])
    ax4.axis("off")
    rows = []
    if has_gt and has_rf:
        yt = np.array([r["gt_kg"] for r in gt_records if "rf_kg" in r])
        yp = np.array([r["rf_kg"] for r in gt_records if "rf_kg" in r])
        m  = _metrics(yt, yp)
        rows.append(["RF",
                     f"{m['mae']:.3f}", f"{m['rmse']:.3f}",
                     f"{m['mare']:.1f}%", f"{m['r2']:.3f}"])
    if has_gt and has_ann:
        yt = np.array([r["gt_kg"]  for r in gt_records if "ann_kg" in r])
        yp = np.array([r["ann_kg"] for r in gt_records if "ann_kg" in r])
        m  = _metrics(yt, yp)
        rows.append(["ANN",
                     f"{m['mae']:.3f}", f"{m['rmse']:.3f}",
                     f"{m['mare']:.1f}%", f"{m['r2']:.3f}"])
    if rows:
        table = ax4.table(
            cellText   = rows,
            colLabels  = ["Model", "MAE", "RMSE", "MARE", "R²"],
            cellLoc    = "center",
            loc        = "center",
            bbox       = [0, 0.2, 1, 0.65],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        _style_table(table, rows)
    ax4.set_title("Regression Metrics", fontsize=10, fontweight="bold")

    # ------------------------------------------------------------------
    # Panel 4: Per-specimen errors (sorted bar chart)
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs_top[1, :2])
    if has_gt and has_rf:
        labels   = [r["label"] for r in gt_records if "rf_kg" in r]
        rf_errs  = [abs(r["rf_kg"]  - r["gt_kg"]) for r in gt_records if "rf_kg"  in r]
        order    = np.argsort(rf_errs)[::-1]
        ax5.bar([labels[i] for i in order], [rf_errs[i] for i in order],
                color=PALETTE["rf"], alpha=0.8, label="RF |error|")
        if has_ann:
            ann_errs = [abs(r["ann_kg"] - r["gt_kg"]) for r in gt_records if "ann_kg" in r]
            ax5.bar([labels[i] for i in order], [ann_errs[i] for i in order],
                    color=PALETTE["ann"], alpha=0.5, label="ANN |error|")
        ax5.axhline(0.10, color="red", lw=1, ls="--", label="0.10 kg threshold")
        ax5.set_xlabel("Specimen")
        ax5.set_ylabel("|Error| (kg)")
        ax5.set_title("Per-Specimen Absolute Error")
        ax5.legend(fontsize=8)
        ax5.tick_params(axis='x', rotation=45, labelsize=6)
    else:
        ax5.text(0.5, 0.5, "No GT available", ha="center", va="center",
                 transform=ax5.transAxes, color="grey")
        ax5.set_title("Per-Specimen Absolute Error")

    # ------------------------------------------------------------------
    # Panel 5: Q-Q plot
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs_top[1, 2])
    if has_gt and has_rf and residuals_rf:
        _qq_panel(ax6, np.array(residuals_rf), "RF Residuals", PALETTE["rf"])
    else:
        ax6.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax6.transAxes, color="grey")
    ax6.set_title("Q-Q Plot (RF Residuals)")

    # ------------------------------------------------------------------
    # Panel 6: Volume vs Biomass
    # ------------------------------------------------------------------
    ax7 = fig.add_subplot(gs_top[1, 3])
    vols = [r.get("volume_m3", None) for r in all_records]
    gts  = [r.get("gt_kg",     None) for r in all_records]
    pts  = [(v, g) for v, g in zip(vols, gts) if v and g]
    if pts:
        vx, gy = zip(*pts)
        ax7.scatter(vx, gy, c=PALETTE["neutral"], alpha=0.75, s=40, zorder=3)
        m_lin, b_lin = np.polyfit(vx, gy, 1)
        xline = np.linspace(min(vx), max(vx), 100)
        ax7.plot(xline, m_lin * xline + b_lin, "--", color=PALETTE["neutral"], lw=1.5)
        r_val = np.corrcoef(vx, gy)[0, 1]
        ax7.set_xlabel("Volume (m³)")
        ax7.set_ylabel("AGB (kg)")
        ax7.set_title(f"Volume–Biomass  r={r_val:.2f}")
    else:
        ax7.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax7.transAxes, color="grey")
        ax7.set_title("Volume–Biomass Correlation")

    # ------------------------------------------------------------------
    # Bottom row: specimen count summary cards
    # ------------------------------------------------------------------
    ax_sum = fig.add_subplot(gs_bottom[0, :])
    ax_sum.axis("off")
    n_total = len(records)
    n_gt    = len(gt_records)
    _summary_card(ax_sum, n_total, n_gt, records)

    if export_path:
        fig.savefig(str(export_path), dpi=200, bbox_inches="tight")
        print(f"Dashboard exported → {export_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Sub-plot helpers
# ---------------------------------------------------------------------------

def _scatter_panel(ax, y_true, y_pred, title, color, fill_color):
    lo = min(y_true.min(), y_pred.min()) * 0.9
    hi = max(y_true.max(), y_pred.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "--", color=PALETTE["perfect"], lw=1.5,
            label="Perfect", zorder=1)
    ax.scatter(y_true, y_pred, c=color, alpha=0.8, s=50, zorder=3)
    # 10% error band
    ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1],
                    alpha=0.15, color=fill_color, label="±10%")
    m  = _metrics(y_true, y_pred)
    ax.set_xlabel("Measured (kg)", fontsize=8)
    ax.set_ylabel("Predicted (kg)", fontsize=8)
    ax.set_title(f"{title}\nMAE={m['mae']:.3f}  R²={m['r2']:.3f}", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)


def _qq_panel(ax, residuals, title, color):
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, c=color, alpha=0.75, s=30)
    ax.plot(osm, slope * osm + intercept, "--k", lw=1.2)
    ax.set_xlabel("Theoretical Quantiles", fontsize=8)
    ax.set_ylabel("Sample Quantiles",      fontsize=8)
    ax.set_title(f"{title}  r={r:.3f}", fontsize=9)


def _style_table(table, rows):
    header_color = "#37474F"
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ECEFF1")
        cell.set_edgecolor("white")


def _summary_card(ax, n_total, n_gt, records):
    n_rf  = sum(1 for r in records if "rf_kg"  in r)
    n_ann = sum(1 for r in records if "ann_kg" in r)
    text  = (f"Specimens processed: {n_total}    |    "
             f"With ground truth: {n_gt}    |    "
             f"RF predictions: {n_rf}    |    "
             f"ANN predictions: {n_ann}")
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=11, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#1565C0"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="ABVT3R Biomass Results Dashboard")
    p.add_argument("--export",    default=None, metavar="FILE",
                   help="Export to PNG or PDF instead of showing interactively")
    p.add_argument("--outputs",   default=str(RECON_OUTPUTS_DIR), metavar="DIR",
                   help="Directory containing reconstruction_stats_*.txt files")
    return p.parse_args()


def main():
    args    = _parse_args()
    out_dir = Path(args.outputs)
    records = load_prediction_results(out_dir)

    if not records:
        print(f"No reconstruction stats found in {out_dir}")
        sys.exit(1)

    print(f"Loaded {len(records)} specimens from {out_dir}")
    build_dashboard(records, Path(args.export) if args.export else None)


if __name__ == "__main__":
    main()
