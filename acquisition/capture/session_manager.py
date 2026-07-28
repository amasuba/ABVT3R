#!/usr/bin/env python3
"""
acquisition/capture/session_manager.py
========================================
Orchestrates a full 30-degree angular-protocol capture session.

Workflow
--------
For each angle  θ ∈ {0°, 30°, 60°, …, 330°}:
  1. Signal turntable to rotate to θ  (via serial → Arduino stepper)
  2. Wait for settle_delay seconds
  3. Trigger Camera A (Green) and Camera B (Red) simultaneously
  4. Save both frames to  acquisition/dataset/specimens/{specimen_id}/

At the end of the session:
  5. Write  acquisition/dataset/specimens/{specimen_id}/metadata.json

Usage
-----
    python session_manager.py DG041_20260609_B02 --port /dev/ttyUSB0
    python session_manager.py DG041_20260609_B02 --simulate   # no hardware
"""

import sys
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

import json
import time
import argparse
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from shared.config import (
    CAPTURE_ANGLES_DEG, SPECIMENS_DIR,
    LEGACY_ANGLES_DEG, HALF_SWEEP_ANGLES_DEG,
)

# ---------------------------------------------------------------------------
# Optional serial (Arduino / Jetson → stepper motor)
# ---------------------------------------------------------------------------
try:
    import serial as _serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

class CaptureSession:
    """
    Full 12-view capture session using dual Kinect v2 cameras.

    Parameters
    ----------
    specimen_id   : canonical ID, e.g. ``DG041_20260609_B02``
    serial_port   : Arduino serial port for turntable control, or None
    baud_rate     : Arduino baud rate (default 9600)
    settle_delay  : seconds to wait after turntable moves before capturing
    simulate      : if True, skip hardware; generate synthetic noise frames
    trigger       : if True, pause before each capture and wait for Enter
    record_gt     : if True, prompt for ground-truth measurements after session
    angles_deg    : override capture angles (default: 30° protocol)
    """

    STEPS_PER_DEGREE = 10   # ← tune to match your stepper/gearbox

    def __init__(self,
                 specimen_id: str,
                 serial_port: str  = None,
                 baud_rate: int    = 9600,
                 settle_delay: float = 2.0,
                 simulate: bool    = False,
                 trigger: bool     = False,
                 record_gt: bool   = False,
                 angles_deg: list  = None):

        self.specimen_id  = specimen_id
        self.settle_delay = settle_delay
        self.simulate     = simulate
        self.trigger      = trigger
        self.record_gt    = record_gt
        self.angles_deg   = angles_deg if angles_deg is not None else CAPTURE_ANGLES_DEG

        self.arduino      = None
        self.cam_a        = None
        self.cam_b        = None

        self.spec_dir     = SPECIMENS_DIR / specimen_id
        self.spec_dir.mkdir(parents=True, exist_ok=True)

        # Track results per angle: {angle: {"camA": bool, "camB": bool}}
        self.results = {a: {"camA": False, "camB": False} for a in self.angles_deg}

        # Connect to Arduino if port given
        if serial_port and SERIAL_AVAILABLE and not simulate:
            try:
                self.arduino = _serial.Serial(serial_port, baud_rate, timeout=5)
                time.sleep(2)   # Arduino reset
                print(f"[Session] Arduino connected on {serial_port}")
            except Exception as exc:
                print(f"[Session] Warning: Arduino serial failed ({exc}). "
                      "Proceeding without motor control.")

    # -----------------------------------------------------------------------
    # Camera init
    # -----------------------------------------------------------------------

    def _init_cameras(self) -> bool:
        """Import and initialise both cameras.  Returns False if neither works."""
        from acquisition.capture.camera_green import KinectCameraA
        from acquisition.capture.camera_red   import KinectCameraB

        self.cam_a = KinectCameraA(self.specimen_id)
        self.cam_b = KinectCameraB(self.specimen_id)

        ok_a = self.cam_a.init()
        ok_b = self.cam_b.init()

        if not ok_a and not ok_b:
            print("[Session] FATAL: Neither camera initialised")
            return False
        if not ok_a:
            print("[Session] Warning: Camera A (Green) not available")
        if not ok_b:
            print("[Session] Warning: Camera B (Red) not available")
        return True

    # -----------------------------------------------------------------------
    # Turntable control
    # -----------------------------------------------------------------------

    def _move_to(self, angle_deg: int):
        """Send step command to Arduino for absolute position."""
        if self.arduino is None:
            return
        steps = angle_deg * self.STEPS_PER_DEGREE
        cmd   = f"GOTO {steps}\n".encode()
        self.arduino.write(cmd)
        # Wait for ACK
        ack = self.arduino.readline().decode().strip()
        print(f"[Turntable] → {angle_deg}°  Arduino ACK: {ack}")

    # -----------------------------------------------------------------------
    # Synthetic frames (simulate mode)
    # -----------------------------------------------------------------------

    @staticmethod
    def _synthetic_frames(angle_deg: int = 0):
        """
        Return plausible-looking synthetic frames for offline testing.

        Depth: radial gradient centred in frame, 500-900 mm range so the
        points survive preprocessing.py's ROI filter (ROI_Z_MIN/MAX =
        0.2-1.5 m in shared/config.py) -- the pipeline's real working
        volume, not an arbitrary display-friendly range.
        RGB: green-dominant disc on dark background, rotated per angle to
        make each view visually distinct.
        """
        H, W = 424, 512
        cy, cx = H // 2, W // 2
        ys = np.arange(H)[:, None] - cy
        xs = np.arange(W)[None, :] - cx
        r  = np.hypot(ys, xs)

        # Depth: plant-like sphere centred at ~700 mm, background at 900 mm
        max_r = min(cy, cx) * 0.65
        depth_f = np.where(r < max_r,
                           700.0 - (max_r - r) * (200.0 / max_r),
                           900.0)
        depth = depth_f.astype(np.uint16)

        # RGB: green disc, hue rotates with angle for visual distinction
        shift  = int(angle_deg / 360 * 255)
        green  = np.clip(200 - (r / max_r * 180).astype(np.uint8), 40, 200)
        mask   = r < max_r
        rgb    = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.where(mask, (shift // 2) % 80, 20)          # B
        rgb[:, :, 1] = np.where(mask, green, 20)                       # G
        rgb[:, :, 2] = np.where(mask, (255 - shift) % 120 + 40, 20)   # R

        return rgb, depth

    # -----------------------------------------------------------------------
    # Single-angle capture
    # -----------------------------------------------------------------------

    def _capture_at_angle(self, angle_deg: int):
        """Move turntable, settle, optionally wait for trigger, then fire both cameras."""
        print(f"\n[Session] ── Angle {angle_deg:3d}° ──────────────────────────────")

        self._move_to(angle_deg)
        time.sleep(self.settle_delay)

        if self.trigger:
            try:
                input(f"  >> Plant at {angle_deg}° — press Enter to capture "
                      f"(or Ctrl+C to stop) ... ")
            except EOFError:
                # Non-interactive stdin (e.g. piped); proceed automatically
                pass

        if self.simulate:
            rgb_a, dep_a = self._synthetic_frames(angle_deg)
            rgb_b, dep_b = self._synthetic_frames(angle_deg)
        else:
            # Trigger simultaneously via threads
            results_a, results_b = [None], [None]

            def _grab_a():
                results_a[0] = self.cam_a.capture() if self.cam_a else (None, None)

            def _grab_b():
                results_b[0] = self.cam_b.capture() if self.cam_b else (None, None)

            t_a = threading.Thread(target=_grab_a)
            t_b = threading.Thread(target=_grab_b)
            t_a.start(); t_b.start()
            t_a.join();  t_b.join()

            rgb_a, dep_a = results_a[0]
            rgb_b, dep_b = results_b[0]

        # Save
        if rgb_a is not None and (self.cam_a or self.simulate):
            cam_obj = self.cam_a if not self.simulate else _SimCamSaver(self.specimen_id, "A")
            self.results[angle_deg]["camA"] = cam_obj.save(angle_deg, rgb_a, dep_a)

        if rgb_b is not None and (self.cam_b or self.simulate):
            cam_obj = self.cam_b if not self.simulate else _SimCamSaver(self.specimen_id, "B")
            self.results[angle_deg]["camB"] = cam_obj.save(angle_deg, rgb_b, dep_b)

    # -----------------------------------------------------------------------
    # Full session run
    # -----------------------------------------------------------------------

    def run(self) -> bool:
        """
        Execute the complete capture sequence.
        Returns True if all views captured successfully.
        """
        print(f"\n{'='*60}")
        print(f"  ABVT3R Capture Session")
        print(f"  Specimen   : {self.specimen_id}")
        print(f"  Angles     : {self.angles_deg}")
        print(f"  Simulate   : {self.simulate}")
        print(f"  Trigger    : {self.trigger}")
        print(f"  Record GT  : {self.record_gt}")
        print(f"{'='*60}\n")

        started = datetime.now(timezone.utc).isoformat()

        if not self.simulate:
            if not self._init_cameras():
                return False

        try:
            for angle in self.angles_deg:
                self._capture_at_angle(angle)
        except KeyboardInterrupt:
            print("\n[Session] Interrupted by user")
        finally:
            self._cleanup()

        finished = datetime.now(timezone.utc).isoformat()
        self._write_metadata(started, finished)
        self._print_summary()

        if self.record_gt:
            self._collect_ground_truth()

        n_ok = sum(v["camA"] and v["camB"] for v in self.results.values())
        return n_ok == len(self.angles_deg)

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    def _write_metadata(self, started: str, finished: str):
        """Write session metadata JSON alongside specimen frames."""
        meta = {
            "specimen_id":   self.specimen_id,
            "session_start": started,
            "session_end":   finished,
            "angles_deg":    self.angles_deg,
            "angular_step":  self.angles_deg[1] - self.angles_deg[0] if len(self.angles_deg) > 1 else None,
            "simulated":     self.simulate,
            "capture_results": self.results,
        }
        out = self.spec_dir / "metadata.json"
        out.write_text(json.dumps(meta, indent=2))
        print(f"\n[Session] Metadata → {out}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def _print_summary(self):
        print(f"\n{'='*60}")
        print(f"  Capture summary — {self.specimen_id}")
        print(f"{'='*60}")
        print(f"  {'Angle':>6}  {'CamA':>6}  {'CamB':>6}")
        print(f"  {'------':>6}  {'----':>6}  {'----':>6}")
        for angle, ok in self.results.items():
            a_ok = "OK" if ok["camA"] else "FAIL"
            b_ok = "OK" if ok["camB"] else "FAIL"
            print(f"  {angle:>6}°  {a_ok:>6}  {b_ok:>6}")
        n_ok = sum(v["camA"] and v["camB"] for v in self.results.values())
        print(f"\n  Complete views : {n_ok}/{len(self.results)}")
        print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Ground truth collection
    # -----------------------------------------------------------------------

    def _collect_ground_truth(self):
        """
        Interactively prompt for specimen ground-truth measurements and
        append (or update) a row in acquisition/dataset/ground_truth/registry.csv.

        All fields are optional — press Enter to leave blank.
        Existing registry entry for this specimen_id is overwritten.
        """
        from shared.config import GROUND_TRUTH_CSV

        print(f"\n{'─'*60}")
        print(f"  Ground-truth entry — {self.specimen_id}")
        print(f"  (Press Enter to skip any field)")
        print(f"{'─'*60}")

        def _ask(label: str, unit: str = "", default: str = "") -> str:
            hint = f" [{default}]" if default else ""
            suffix = f" ({unit})" if unit else ""
            try:
                val = input(f"  {label}{suffix}{hint}: ").strip()
            except EOFError:
                val = ""
            return val if val else default

        # Parse specimen_id to guess defaults
        parts = self.specimen_id.split("_")
        date_str = parts[1] if len(parts) > 1 else datetime.now(timezone.utc).strftime("%Y%m%d")
        collection_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else date_str

        row = {
            "specimen_id":     self.specimen_id,
            "species":         _ask("Species",       default="Duranta"),
            "variety":         _ask("Variety",       default="Gold mini"),
            "collection_date": _ask("Collection date (YYYY-MM-DD)", default=collection_date),
            "collector":       _ask("Collector name"),
            "location":        _ask("Location",      default="Greenhouse A"),
            "agb_kg":          _ask("AGB mass",      "kg"),
            "total_mass_kg":   _ask("Total mass (plant+pot)", "kg"),
            "pot_mass_kg":     _ask("Pot mass",      "kg"),
            "height_mm":       _ask("Plant height",  "mm"),
            "dbh_mm":          _ask("Stem DBH",      "mm"),
            "canopy_width_mm": _ask("Canopy width",  "mm"),
            "notes":           _ask("Notes"),
            "legacy_id":       "",
        }

        self._append_registry(row, GROUND_TRUTH_CSV)

    def _append_registry(self, row: dict, csv_path: "Path"):
        """Write row to registry CSV; overwrites existing entry for specimen_id."""
        import csv as _csv

        FIELDNAMES = [
            "specimen_id", "species", "variety", "collection_date",
            "collector", "location", "agb_kg", "total_mass_kg", "pot_mass_kg",
            "height_mm", "dbh_mm", "canopy_width_mm", "notes", "legacy_id",
        ]

        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing rows (if any)
        existing: list[dict] = []
        if csv_path.exists():
            with csv_path.open(newline="") as f:
                existing = list(_csv.DictReader(f))

        # Replace or append
        updated = False
        for i, r in enumerate(existing):
            if r.get("specimen_id") == row["specimen_id"]:
                existing[i] = row
                updated = True
                break
        if not updated:
            existing.append(row)

        with csv_path.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            w.writerows(existing)

        action = "Updated" if updated else "Added"
        print(f"\n[GT] {action} registry entry → {csv_path}")
        print(f"[GT] agb={row['agb_kg']} kg  total={row['total_mass_kg']} kg  "
              f"height={row['height_mm']} mm")

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def _cleanup(self):
        if self.cam_a:
            self.cam_a.cleanup()
        if self.cam_b:
            self.cam_b.cleanup()
        if self.arduino:
            self.arduino.close()


# ---------------------------------------------------------------------------
# Thin wrapper that mimics camera .save() interface in simulate mode
# ---------------------------------------------------------------------------

class _SimCamSaver:
    """Minimal save shim used when simulate=True."""

    def __init__(self, specimen_id: str, label: str):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from shared.config import SPECIMENS_DIR, DEPTH_MAX_MM, view_filename
        import cv2
        self.rgb_dir   = SPECIMENS_DIR / specimen_id / "rgb"
        self.depth_dir = SPECIMENS_DIR / specimen_id / "depth"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.label      = label
        self.DEPTH_MAX  = DEPTH_MAX_MM
        self._vfn       = view_filename
        self._cv2       = cv2

    def save(self, angle_deg: int, rgb: np.ndarray, depth: np.ndarray) -> bool:
        import cv2
        rgb_jpg = self.rgb_dir   / self._vfn(angle_deg, self.label, "rgb",   "jpg")
        rgb_npy = self.rgb_dir   / self._vfn(angle_deg, self.label, "rgb",   "npy")
        dep_npy = self.depth_dir / self._vfn(angle_deg, self.label, "depth", "npy")
        dep_jpg = self.depth_dir / self._vfn(angle_deg, self.label, "depth", "jpg")
        try:
            cv2.imwrite(str(rgb_jpg), rgb)
            np.save(str(rgb_npy), rgb)
            np.save(str(dep_npy), depth)
            vis = ((depth.astype(np.float32) / self.DEPTH_MAX) * 255).astype(np.uint8)
            cv2.imwrite(str(dep_jpg), cv2.applyColorMap(vis, cv2.COLORMAP_JET))
            print(f"[Sim{self.label}]  {angle_deg:3d}° → {rgb_jpg.name}")
            return True
        except Exception as exc:
            print(f"[Sim{self.label}] Save error: {exc}")
            return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="ABVT3R 30-degree capture session manager")
    p.add_argument("specimen_id",     help="e.g. DG041_20260609_B02")
    p.add_argument("--port",          default=None,  help="Arduino serial port, e.g. /dev/ttyUSB0")
    p.add_argument("--baud",          default=9600,  type=int)
    p.add_argument("--settle",        default=2.0,   type=float, help="Settle delay after motor move (s)")
    p.add_argument("--simulate",      action="store_true", help="Simulate cameras (no hardware required)")
    p.add_argument("--trigger",       action="store_true", help="Pause at each angle and wait for Enter before capturing")
    p.add_argument("--gt",            action="store_true", help="Prompt for ground-truth measurements after capture")
    p.add_argument("--legacy",        action="store_true", help="Use legacy 4-view 90° protocol")
    p.add_argument("--half-sweep",    action="store_true",
                   help="Manual dual-camera protocol: 6 physical repositioning steps "
                        "(0-150 deg); Camera B (180 deg behind Camera A) covers the "
                        "other 6 angles simultaneously. Use this for a hand-repositioned "
                        "rig with no motorised turntable.")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.legacy:
        angles = LEGACY_ANGLES_DEG
    elif args.half_sweep:
        angles = HALF_SWEEP_ANGLES_DEG
    else:
        angles = CAPTURE_ANGLES_DEG
    session = CaptureSession(
        specimen_id  = args.specimen_id,
        serial_port  = args.port,
        baud_rate    = args.baud,
        settle_delay = args.settle,
        simulate     = args.simulate,
        trigger      = args.trigger,
        record_gt    = args.gt,
        angles_deg   = angles,
    )
    success = session.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
