#!/usr/bin/env python3
"""
acquisition/dataset/dataset_viewer.py
========================================
Interactive viewer for the ABVT3R specimen dataset.

Features
--------
- Browse specimens by ID in a sorted list
- Display RGB and depth views as a contact sheet
- Show ground-truth biomass from registry.csv
- Export a publication-quality figure (PNG/PDF)

Usage
-----
    python dataset_viewer.py                   # GUI — all specimens
    python dataset_viewer.py DG001_20260609_B01
    python dataset_viewer.py --list            # print specimen table
    python dataset_viewer.py DG001_20260609_B01 --export fig.png
"""

import sys
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

import json
import argparse
import csv
from pathlib import Path

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1 import ImageGrid

from shared.config import (
    SPECIMENS_DIR, GROUND_TRUTH_CSV, CAPTURE_ANGLES_DEG, DEPTH_MAX_MM
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_ground_truth() -> dict:
    """Return {specimen_id: row_dict} from ground_truth/registry.csv."""
    if not GROUND_TRUTH_CSV.exists():
        return {}
    gt = {}
    with GROUND_TRUTH_CSV.open() as f:
        for row in csv.DictReader(f):
            gt[row["specimen_id"]] = row
    return gt


def list_specimens() -> list[Path]:
    """Return sorted list of specimen directories."""
    if not SPECIMENS_DIR.exists():
        return []
    return sorted(p for p in SPECIMENS_DIR.iterdir() if p.is_dir())


def load_specimen(spec_dir: Path) -> dict:
    """
    Load all views for one specimen.

    Returns
    -------
    dict with keys:
        specimen_id : str
        metadata    : dict or None
        views       : {angle_deg: {"rgb_A": arr, "depth_A": arr, "rgb_B": arr, "depth_B": arr}}
    """
    specimen_id = spec_dir.name
    meta_path   = spec_dir / "metadata.json"
    metadata    = json.loads(meta_path.read_text()) if meta_path.exists() else None

    angles = metadata["angles_deg"] if metadata else CAPTURE_ANGLES_DEG
    views  = {}
    for angle in angles:
        entry = {}
        for cam in ("A", "B"):
            rgb_jpg   = spec_dir / "rgb"   / f"view_{angle:03d}deg_cam{cam}_rgb.jpg"
            depth_npy = spec_dir / "depth" / f"view_{angle:03d}deg_cam{cam}_depth.npy"
            if rgb_jpg.exists():
                entry[f"rgb_{cam}"]   = cv2.cvtColor(cv2.imread(str(rgb_jpg)), cv2.COLOR_BGR2RGB)
            if depth_npy.exists():
                d = np.load(str(depth_npy))
                entry[f"depth_{cam}"] = (d.astype(np.float32) / DEPTH_MAX_MM * 255).astype(np.uint8)
        if entry:
            views[angle] = entry

    return {"specimen_id": specimen_id, "metadata": metadata, "views": views}


# ---------------------------------------------------------------------------
# Contact-sheet rendering
# ---------------------------------------------------------------------------

def render_contact_sheet(specimen: dict, gt_row: dict | None = None,
                          max_angles: int = 12,
                          show_cams: tuple = ("A", "B"),
                          modalities: tuple = ("rgb", "depth")) -> plt.Figure:
    """
    Render a contact sheet for all angles × cameras × modalities.

    Layout  : rows = angles,  cols = camera×modality  (e.g. RGB-A | Depth-A | RGB-B | Depth-B)
    """
    angles  = sorted(specimen["views"].keys())[:max_angles]
    n_rows  = len(angles)
    streams = [(cam, mod) for cam in show_cams for mod in modalities]
    n_cols  = len(streams)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.2),
                             squeeze=False)

    # Column headers
    for col_idx, (cam, mod) in enumerate(streams):
        axes[0, col_idx].set_title(f"Cam {cam} · {mod.upper()}", fontsize=9, pad=3)

    for row_idx, angle in enumerate(angles):
        view    = specimen["views"][angle]
        axes[row_idx, 0].set_ylabel(f"{angle}°", fontsize=8, rotation=0,
                                    labelpad=25, va='center')
        for col_idx, (cam, mod) in enumerate(streams):
            ax  = axes[row_idx, col_idx]
            key = f"{mod}_{cam}"
            if key in view:
                img = view[key]
                if mod == "depth":
                    ax.imshow(img, cmap="jet", vmin=0, vmax=255)
                else:
                    ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                        transform=ax.transAxes, color='grey', fontsize=7)
            ax.axis("off")

    # Title
    sid   = specimen["specimen_id"]
    n_ok  = len(specimen["views"])
    title = f"Specimen: {sid}   ({n_ok} angles)"
    if gt_row:
        agb  = gt_row.get("agb_kg", "?")
        tot  = gt_row.get("total_mass_kg", "?")
        pot  = gt_row.get("pot_mass_kg", "?")
        title += f"\nAGB: {agb} kg | Total: {tot} kg | Pot: {pot} kg"

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0.04, 0, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Interactive browser
# ---------------------------------------------------------------------------

class SpecimenBrowser:
    """
    Matplotlib-based interactive specimen browser with prev / next buttons.
    """

    def __init__(self, gt: dict):
        self.specimens = list_specimens()
        self.gt        = gt
        self.idx       = 0

        if not self.specimens:
            print("No specimens found in", SPECIMENS_DIR)
            return

        self.fig, self.main_ax = plt.subplots(figsize=(14, 9))
        plt.subplots_adjust(bottom=0.1)

        ax_prev = plt.axes([0.35, 0.02, 0.12, 0.05])
        ax_next = plt.axes([0.53, 0.02, 0.12, 0.05])
        self._btn_prev = Button(ax_prev, "← Previous")
        self._btn_next = Button(ax_next,  "Next →")
        self._btn_prev.on_clicked(self._prev)
        self._btn_next.on_clicked(self._next)

        self._render()
        plt.show()

    def _render(self):
        spec_dir = self.specimens[self.idx]
        specimen = load_specimen(spec_dir)
        gt_row   = self.gt.get(specimen["specimen_id"])
        sheet    = render_contact_sheet(specimen, gt_row)

        # Embed sheet in browser figure
        plt.close(sheet)   # prevent double window
        self.main_ax.clear()
        self.main_ax.axis("off")
        sheet.canvas.draw()
        buf    = np.frombuffer(sheet.canvas.tostring_rgb(), dtype=np.uint8)
        w, h   = sheet.canvas.get_width_height()
        img    = buf.reshape(h, w, 3)
        self.main_ax.imshow(img)
        n      = len(self.specimens)
        self.fig.suptitle(f"Specimen {self.idx + 1}/{n}", fontsize=10)
        self.fig.canvas.draw_idle()

    def _prev(self, _event):
        self.idx = (self.idx - 1) % len(self.specimens)
        self._render()

    def _next(self, _event):
        self.idx = (self.idx + 1) % len(self.specimens)
        self._render()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="ABVT3R Dataset Viewer")
    p.add_argument("specimen_id", nargs="?", default=None,
                   help="Specific specimen ID; omit to browse all")
    p.add_argument("--list",   action="store_true", help="Print specimen table and exit")
    p.add_argument("--export", default=None, metavar="FILE",
                   help="Export contact sheet to PNG or PDF")
    return p.parse_args()


def main():
    args = _parse_args()
    gt   = load_ground_truth()

    if args.list:
        specimens = list_specimens()
        print(f"\n{'Specimen ID':35s}  {'AGB (kg)':>10}  {'Views':>6}")
        print("-" * 58)
        for sd in specimens:
            row   = gt.get(sd.name, {})
            meta  = json.loads((sd / "metadata.json").read_text()) if (sd / "metadata.json").exists() else {}
            n_views = len(meta.get("angles_deg", []))
            print(f"{sd.name:35s}  {row.get('agb_kg', '—'):>10}  {n_views:>6}")
        print(f"\nTotal: {len(specimens)} specimens\n")
        return

    if args.specimen_id:
        spec_dir = SPECIMENS_DIR / args.specimen_id
        if not spec_dir.exists():
            print(f"Specimen not found: {spec_dir}")
            sys.exit(1)
        specimen = load_specimen(spec_dir)
        gt_row   = gt.get(args.specimen_id)
        fig      = render_contact_sheet(specimen, gt_row)

        if args.export:
            fig.savefig(args.export, dpi=200, bbox_inches="tight")
            print(f"Exported → {args.export}")
        else:
            plt.show()
    else:
        # Interactive browser
        matplotlib.use("TkAgg") if sys.platform != "darwin" else None
        SpecimenBrowser(gt)


if __name__ == "__main__":
    main()
