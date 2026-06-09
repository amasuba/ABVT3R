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
    LEGACY_ANGLES_DEG,
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
    angles_deg    : override capture angles (default: 30° protocol)
    """

    STEPS_PER_DEGREE = 10   # ← tune to match your stepper/gearbox

    def __init__(self,
                 specimen_id: str,
                 serial_port: str  = None,
                 baud_rate: int    = 9600,
                 settle_delay: float = 2.0,
                 simulate: bool    = False,
                 angles_deg: list  = None):

        self.specimen_id  = specimen_id
        self.settle_delay = settle_delay
        self.simulate     = simulate
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
    def _synthetic_frames():
        """Return plausible-looking noise frames for offline testing."""
        rgb   = np.random.randint(50, 200, (424, 512, 3), dtype=np.uint8)
        depth = np.random.randint(400, 1200, (424, 512), dtype=np.uint16)
        return rgb, depth

    # -----------------------------------------------------------------------
    # Single-angle capture
    # -----------------------------------------------------------------------

    def _capture_at_angle(self, angle_deg: int):
        """Move turntable, settle, then fire both cameras simultaneously."""
        print(f"\n[Session] ── Angle {angle_deg:3d}° ──────────────────────────────")

        self._move_to(angle_deg)
        time.sleep(self.settle_delay)

        if self.simulate:
            rgb_a, dep_a = self._synthetic_frames()
            rgb_b, dep_b = self._synthetic_frames()
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
    p.add_argument("--legacy",        action="store_true", help="Use legacy 4-view 90° protocol")
    return p.parse_args()


def main():
    args    = _parse_args()
    angles  = LEGACY_ANGLES_DEG if args.legacy else CAPTURE_ANGLES_DEG
    session = CaptureSession(
        specimen_id  = args.specimen_id,
        serial_port  = args.port,
        baud_rate    = args.baud,
        settle_delay = args.settle,
        simulate     = args.simulate,
        angles_deg   = angles,
    )
    success = session.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
