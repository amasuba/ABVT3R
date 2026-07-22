#!/usr/bin/env python3
"""
acquisition/capture/camera_red.py
====================================
Kinect v2 Camera B (Red) — RGB-D acquisition with 30-degree angular protocol.

Identical interface to camera_green.py; differs only in:
  - LABEL   = "B"
  - device  selection prefers index 1 (second Kinect) then falls back to index 0

Platform : Linux (NVIDIA Jetson Nano or desktop)
Hardware : Microsoft Kinect v2 (pylibfreenect2)
"""

import sys
import os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
import cv2
from pathlib import Path

from shared.config import (
    DEPTH_WIDTH, DEPTH_HEIGHT,
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
        Registration, Frame, CpuPacketPipeline,
    )
    KINECT_AVAILABLE = True
    print("pylibfreenect2 loaded — Kinect v2 available")
except ImportError:
    print("Warning: pylibfreenect2 not available.  "
          "Install with: pip install pylibfreenect2")


# ---------------------------------------------------------------------------
# Camera class
# ---------------------------------------------------------------------------

class KinectCameraB:
    """
    Kinect v2 RGB-D host — Camera B (Red).

    Prefers device index 1 (second Kinect); falls back to index 0.
    See KinectCameraA for full documentation — interfaces are identical.
    """

    LABEL = "B"
    DEFAULT_SERIAL = "006158144547"

    def __init__(self,
                 specimen_id: str,
                 target_serial: str = DEFAULT_SERIAL,
                 fps: int = 30):
        self.specimen_id   = specimen_id
        self.target_serial = target_serial
        self.fps           = fps

        self.fn            = None
        self.device        = None
        self.listener      = None
        self.registration  = None
        self.camera_ok     = False
        self.capture_count = 0

        self.spec_dir   = SPECIMENS_DIR / specimen_id
        self.rgb_dir    = self.spec_dir / "rgb"
        self.depth_dir  = self.spec_dir / "depth"
        for d in (self.rgb_dir, self.depth_dir):
            d.mkdir(parents=True, exist_ok=True)

        print(f"[CamB] Specimen  : {specimen_id}")
        print(f"[CamB] RGB dir   : {self.rgb_dir}")
        print(f"[CamB] Depth dir : {self.depth_dir}")

    def list_devices(self) -> list[str]:
        if not KINECT_AVAILABLE:
            return []
        fn = Freenect2()
        n  = fn.enumerateDevices()
        serials = [fn.getDeviceSerialNumber(i).decode() for i in range(n)]
        print(f"[CamB] Found {n} Kinect v2 device(s): {serials}")
        return serials

    def init(self) -> bool:
        if not KINECT_AVAILABLE:
            print("[CamB] pylibfreenect2 not available")
            return False

        serials = self.list_devices()
        if not serials:
            print("[CamB] No Kinect v2 devices found")
            return False

        # Camera B prefers the second device
        if self.target_serial in serials:
            serial = self.target_serial
        elif len(serials) > 1:
            serial = serials[1]
            print(f"[CamB] Using device index 1: serial={serial}")
        else:
            serial = serials[0]
            print(f"[CamB] Only one device present — sharing serial={serial}")

        try:
            pipeline = CpuPacketPipeline()

            self.fn       = Freenect2()
            self.device   = self.fn.openDevice(serial.encode(), pipeline=pipeline)
            self.listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
            self.device.setColorFrameListener(self.listener)
            self.device.setIrAndDepthFrameListener(self.listener)
            self.device.start()
            self.registration = Registration(
                self.device.getIrCameraParams(),
                self.device.getColorCameraParams()
            )
            self.camera_ok = True

            print(f"[CamB] Warming up ({self.fps * 2} frames) …")
            for _ in range(self.fps * 2):
                frames = self.listener.waitForNewFrame()
                self.listener.release(frames)

            print("[CamB] Ready")
            return True

        except Exception as exc:
            print(f"[CamB] Initialisation failed: {exc}")
            self.cleanup()
            return False

    def capture(self):
        if not self.camera_ok:
            print("[CamB] Camera not initialised")
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
            rgb   = registered.asarray(dtype=np.uint8)[..., :3].copy()
            depth = undistorted.asarray(dtype=np.float32).astype(np.uint16)
            self.listener.release(frames)
            return rgb, depth

        except Exception as exc:
            print(f"[CamB] Capture error: {exc}")
            return None, None

    def _filter_depth(self, depth: np.ndarray) -> np.ndarray:
        d = depth.copy()
        d[(d < DEPTH_MIN_MM) | (d > DEPTH_MAX_MM)] = 0
        return cv2.medianBlur(d.astype(np.uint16), 3)

    def save(self, angle_deg: int, rgb: np.ndarray, depth: np.ndarray) -> bool:
        self.capture_count += 1
        depth_filtered = self._filter_depth(depth)

        rgb_jpg = self.rgb_dir   / view_filename(angle_deg, self.LABEL, "rgb",   "jpg")
        rgb_npy = self.rgb_dir   / view_filename(angle_deg, self.LABEL, "rgb",   "npy")
        dep_npy = self.depth_dir / view_filename(angle_deg, self.LABEL, "depth", "npy")
        dep_jpg = self.depth_dir / view_filename(angle_deg, self.LABEL, "depth", "jpg")

        try:
            cv2.imwrite(str(rgb_jpg), rgb)
            np.save(str(rgb_npy), rgb)
            np.save(str(dep_npy), depth_filtered)
            depth_vis = ((depth_filtered.astype(np.float32) / DEPTH_MAX_MM) * 255).astype(np.uint8)
            cv2.imwrite(str(dep_jpg), cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET))

            print(f"[CamB]  {angle_deg:3d}° → {rgb_jpg.name}  |  {dep_npy.name}")
            return True

        except Exception as exc:
            print(f"[CamB] Save error at {angle_deg}°: {exc}")
            return False

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
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Kinect v2 Camera B — single view capture")
    p.add_argument("specimen_id", help="Specimen ID, e.g. DG001_20260609_B01")
    p.add_argument("angle_deg",   type=int,
                   help=f"Turntable angle in degrees (suggested: {CAPTURE_ANGLES_DEG})")
    p.add_argument("--serial",    default=KinectCameraB.DEFAULT_SERIAL, help="Kinect v2 device serial")
    return p.parse_args()


def main():
    args   = _parse_args()
    camera = KinectCameraB(specimen_id=args.specimen_id, target_serial=args.serial)

    if not camera.init():
        print("[CamB] Aborting")
        camera.cleanup()
        sys.exit(1)

    rgb, depth = camera.capture()
    if rgb is None:
        print("[CamB] Capture returned no data")
        camera.cleanup()
        sys.exit(1)

    camera.save(args.angle_deg, rgb, depth)
    camera.cleanup()


if __name__ == "__main__":
    main()
