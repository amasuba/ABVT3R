#!/usr/bin/env python3
"""
acquisition/capture/camera_green.py
=====================================
Kinect v2 Camera A (Green) — RGB-D acquisition with 30-degree angular protocol.

Saves to:  acquisition/dataset/specimens/{specimen_id}/{rgb|depth}/
           view_{angle:03d}deg_camA_{modality}.{ext}

Platform : Linux (NVIDIA Jetson Nano or desktop)
Hardware : Microsoft Kinect v2 (pylibfreenect2)
"""

import sys
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

import time
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

from shared.config import (
    DEPTH_WIDTH, DEPTH_HEIGHT, COLOR_WIDTH, COLOR_HEIGHT,
    DEPTH_MIN_MM, DEPTH_MAX_MM,
    CAPTURE_ANGLES_DEG, SPECIMENS_DIR,
    view_filename,
)

# ---------------------------------------------------------------------------
# Kinect v2 driver
# ---------------------------------------------------------------------------
KINECT_AVAILABLE = False
try:
    import pylibfreenect2
    from pylibfreenect2 import (
        Freenect2, SyncMultiFrameListener, FrameType,
        Registration, Frame, OpenGLPacketPipeline,
    )
    KINECT_AVAILABLE = True
    print("pylibfreenect2 loaded — Kinect v2 available")
except ImportError:
    print("Warning: pylibfreenect2 not available.  "
          "Install with: pip install pylibfreenect2")


# ---------------------------------------------------------------------------
# Camera class
# ---------------------------------------------------------------------------

class KinectCameraA:
    """
    Kinect v2 RGB-D host — Camera A (Green).

    Streams
    -------
    Colour  : 1920 × 1080  BGR  uint8   @ 30 fps
    Depth   :  512 ×  424  uint16  mm   (undistorted, aligned to IR frame)

    Saving convention
    -----------------
    All frames for one specimen are written under::

        acquisition/dataset/specimens/{specimen_id}/
            rgb/    view_{angle:03d}deg_camA_rgb.jpg
                    view_{angle:03d}deg_camA_rgb.npy
            depth/  view_{angle:03d}deg_camA_depth.npy
                    view_{angle:03d}deg_camA_depth.jpg
    """

    LABEL = "A"                     # camera identifier in filenames
    DEFAULT_SERIAL = None           # None → first enumerated Kinect v2

    def __init__(self,
                 specimen_id: str,
                 target_serial: str = DEFAULT_SERIAL,
                 fps: int = 30):
        """
        Parameters
        ----------
        specimen_id   : canonical specimen identifier, e.g. ``DG001_20260609_B01``
        target_serial : Kinect v2 serial number; None selects device index 0
        fps           : warm-up frame count multiplier
        """
        self.specimen_id   = specimen_id
        self.target_serial = target_serial
        self.fps           = fps

        self.device        = None
        self.listener      = None
        self.registration  = None
        self.camera_ok     = False
        self.capture_count = 0

        # Build output directories
        self.spec_dir   = SPECIMENS_DIR / specimen_id
        self.rgb_dir    = self.spec_dir / "rgb"
        self.depth_dir  = self.spec_dir / "depth"
        for d in (self.rgb_dir, self.depth_dir):
            d.mkdir(parents=True, exist_ok=True)

        print(f"[CamA] Specimen  : {specimen_id}")
        print(f"[CamA] RGB dir   : {self.rgb_dir}")
        print(f"[CamA] Depth dir : {self.depth_dir}")

    # -----------------------------------------------------------------------
    # Device discovery
    # -----------------------------------------------------------------------

    def list_devices(self) -> list[str]:
        """Return list of serial strings for all connected Kinect v2 devices."""
        if not KINECT_AVAILABLE:
            return []
        fn = Freenect2()
        n  = fn.enumerateDevices()
        serials = [fn.getDeviceSerialNumber(i) for i in range(n)]
        print(f"[CamA] Found {n} Kinect v2 device(s): {serials}")
        return serials

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def init(self) -> bool:
        """Initialise hardware and warm-up auto-exposure.  Returns True on success."""
        if not KINECT_AVAILABLE:
            print("[CamA] pylibfreenect2 not available — cannot initialise")
            return False

        serials = self.list_devices()
        if not serials:
            print("[CamA] No Kinect v2 devices found")
            return False

        serial = self.target_serial if self.target_serial in serials else serials[0]
        print(f"[CamA] Opening device serial={serial}")

        try:
            try:
                pipeline = OpenGLPacketPipeline()
            except Exception:
                from pylibfreenect2 import CpuPacketPipeline
                pipeline = CpuPacketPipeline()
                print("[CamA] Falling back to CPU packet pipeline")

            self.device   = Freenect2().openDevice(serial, pipeline=pipeline)
            self.listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
            self.device.setColorFrameListener(self.listener)
            self.device.setIrAndDepthFrameListener(self.listener)
            self.device.start()
            self.registration = Registration(
                self.device.getIrCameraParams(),
                self.device.getColorCameraParams()
            )
            self.camera_ok = True

            print(f"[CamA] Warming up ({self.fps * 2} frames) …")
            for _ in range(self.fps * 2):
                frames = self.listener.waitForNewFrame()
                self.listener.release(frames)

            print("[CamA] Ready")
            return True

        except Exception as exc:
            print(f"[CamA] Initialisation failed: {exc}")
            self.cleanup()
            return False

    # -----------------------------------------------------------------------
    # Frame capture
    # -----------------------------------------------------------------------

    def capture(self):
        """
        Capture one aligned (RGB, depth) frame pair.

        Returns
        -------
        rgb   : np.ndarray  (424, 512, 3)  uint8   BGR   or None
        depth : np.ndarray  (424, 512)     uint16  mm    or None
        """
        if not self.camera_ok:
            print("[CamA] Camera not initialised")
            return None, None
        try:
            frames      = self.listener.waitForNewFrame()
            color_frame = frames[FrameType.Color]
            depth_frame = frames[FrameType.Depth]

            undistorted = Frame(DEPTH_WIDTH, DEPTH_HEIGHT, 4)
            registered  = Frame(DEPTH_WIDTH, DEPTH_HEIGHT, 4)
            self.registration.apply(
                color_frame, depth_frame,
                undistorted, registered,
                bigdepth=None, color_depth_map=None
            )
            rgb   = registered.asarray()[..., :3].copy()
            depth = undistorted.asarray().astype(np.uint16)
            self.listener.release(frames)
            return rgb, depth

        except Exception as exc:
            print(f"[CamA] Capture error: {exc}")
            return None, None

    # -----------------------------------------------------------------------
    # Depth filtering
    # -----------------------------------------------------------------------

    def _filter_depth(self, depth: np.ndarray) -> np.ndarray:
        """Zero pixels outside the valid range and apply a 3×3 median blur."""
        d = depth.copy()
        d[(d < DEPTH_MIN_MM) | (d > DEPTH_MAX_MM)] = 0
        return cv2.medianBlur(d.astype(np.uint16), 3)

    # -----------------------------------------------------------------------
    # Saving
    # -----------------------------------------------------------------------

    def save(self, angle_deg: int, rgb: np.ndarray, depth: np.ndarray) -> bool:
        """
        Persist one view to disk using the canonical naming convention.

        Parameters
        ----------
        angle_deg : turntable angle at time of capture (degrees)
        rgb       : BGR image array
        depth     : raw depth array (mm, uint16)
        """
        self.capture_count += 1
        depth_filtered = self._filter_depth(depth)

        # Build paths
        rgb_jpg   = self.rgb_dir   / view_filename(angle_deg, self.LABEL, "rgb",   "jpg")
        rgb_npy   = self.rgb_dir   / view_filename(angle_deg, self.LABEL, "rgb",   "npy")
        dep_npy   = self.depth_dir / view_filename(angle_deg, self.LABEL, "depth", "npy")
        dep_jpg   = self.depth_dir / view_filename(angle_deg, self.LABEL, "depth", "jpg")

        try:
            cv2.imwrite(str(rgb_jpg), rgb)
            np.save(str(rgb_npy), rgb)

            np.save(str(dep_npy), depth_filtered)
            depth_vis = ((depth_filtered.astype(np.float32) / DEPTH_MAX_MM) * 255).astype(np.uint8)
            cv2.imwrite(str(dep_jpg), cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET))

            print(f"[CamA]  {angle_deg:3d}° → {rgb_jpg.name}  |  {dep_npy.name}")
            return True

        except Exception as exc:
            print(f"[CamA] Save error at {angle_deg}°: {exc}")
            return False

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def cleanup(self):
        if self.device:
            try:
                self.device.stop()
                self.device.close()
            except Exception:
                pass
            self.device = None
        self.camera_ok = False


# ---------------------------------------------------------------------------
# Standalone CLI  — single-angle capture
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Kinect v2 Camera A — single view capture")
    p.add_argument("specimen_id", help="Specimen ID, e.g. DG001_20260609_B01")
    p.add_argument("angle_deg",   type=int,
                   help=f"Turntable angle in degrees (suggested: {CAPTURE_ANGLES_DEG})")
    p.add_argument("--serial",    default=None, help="Kinect v2 device serial")
    return p.parse_args()


def main():
    args   = _parse_args()
    camera = KinectCameraA(specimen_id=args.specimen_id, target_serial=args.serial)

    if not camera.init():
        print("[CamA] Aborting — could not initialise camera")
        camera.cleanup()
        sys.exit(1)

    rgb, depth = camera.capture()
    if rgb is None:
        print("[CamA] Capture returned no data")
        camera.cleanup()
        sys.exit(1)

    camera.save(args.angle_deg, rgb, depth)
    camera.cleanup()


if __name__ == "__main__":
    main()
