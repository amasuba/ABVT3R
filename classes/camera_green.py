#!/usr/bin/env python3
"""
Kinect v2 Camera Host - Camera A (Green)
Captures aligned RGB and Depth data from a Microsoft Kinect v2 camera and
saves to the data_collection/ directory.  Device selection is by serial number.

Platform : Linux (NVIDIA Jetson Nano or desktop)
Hardware : Microsoft Kinect v2
"""

import time
import numpy as np
import cv2
from datetime import datetime
import sys
import os
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Kinect v2 (libfreenect2) import
# ---------------------------------------------------------------------------
KINECT_AVAILABLE = False
try:
    import pylibfreenect2
    from pylibfreenect2 import (
        Freenect2, Freenect2Device, SyncMultiFrameListener, FrameType,
        Registration, Frame, FrameMap, OpenGLPacketPipeline
    )
    KINECT_AVAILABLE = True
    print("Using pylibfreenect2 library for Kinect v2")
except ImportError:
    KINECT_AVAILABLE = False
    print("Warning: pylibfreenect2 not available. "
          "Install with: pip install pylibfreenect2")


class KinectV2CameraHost:
    """
    Kinect v2 RGB-D camera host — Camera A (Green).

    Streams:
      - Colour : 1920 x 1080  BGRA  @ 30 fps
      - Depth  :  512 x  424  float32 mm  (undistorted, aligned to IR frame)

    The depth map is expressed in millimetres (uint16) so downstream
    processing is hardware-agnostic.
    """

    # Default serial for Camera A (Green).
    # Set to the serial printed on the camera label for a fixed assignment.
    # None → first enumerated Kinect v2 device (index 0).
    DEFAULT_SERIAL = None

    def __init__(self,
                 target_serial: str = DEFAULT_SERIAL,
                 filename: str = "default",
                 count: int = 1,
                 color_width: int = 1920,
                 color_height: int = 1080,
                 depth_width: int = 512,
                 depth_height: int = 424,
                 fps: int = 30):

        self.device         = None
        self.listener       = None
        self.registration   = None
        self.camera_working = False
        self.capture_count  = 0
        self.target_serial  = target_serial

        # Stream configuration
        self.color_width  = color_width
        self.color_height = color_height
        self.depth_width  = depth_width
        self.depth_height = depth_height
        self.fps          = fps

        # File-naming
        self.filename = filename
        self.count    = count

        # Output directory
        self.save_directory = os.path.join(os.getcwd(), "data_collection")
        os.makedirs(self.save_directory, exist_ok=True)
        print(f"Save directory: {self.save_directory}")

    # -----------------------------------------------------------------------
    # Device discovery
    # -----------------------------------------------------------------------

    def list_kinect_devices(self):
        """Return a list of serial-number strings for all connected Kinect v2 devices."""
        if not KINECT_AVAILABLE:
            print("pylibfreenect2 not available")
            return []
        fn = Freenect2()
        num_devices = fn.enumerateDevices()
        serials = []
        print(f"Found {num_devices} Kinect v2 device(s):")
        for i in range(num_devices):
            serial = fn.getDeviceSerialNumber(i)
            serials.append(serial)
            print(f"  Serial: {serial}")
        return serials

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def init_camera(self) -> bool:
        """
        Initialise the Kinect v2 device for acquisition.
        Returns True on success, False otherwise.
        """
        if not KINECT_AVAILABLE:
            print("pylibfreenect2 not available")
            return False

        fn = Freenect2()
        num_devices = fn.enumerateDevices()
        if num_devices == 0:
            print("No Kinect v2 devices found")
            return False

        serials = self.list_kinect_devices()
        serial = self.target_serial
        if serial is None or serial not in serials:
            serial = serials[0]  # Use first device if not specified
            print(f"Using Kinect serial: {serial}")

        try:
            try:
                pipeline = OpenGLPacketPipeline()
            except Exception:
                from pylibfreenect2 import CpuPacketPipeline
                pipeline = CpuPacketPipeline()
                print("OpenGL pipeline unavailable, using CPU pipeline")
            self.device = fn.openDevice(serial, pipeline=pipeline)
            self.listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
            self.device.setColorFrameListener(self.listener)
            self.device.setIrAndDepthFrameListener(self.listener)
            self.device.start()
            self.registration = Registration(self.device.getIrCameraParams(), self.device.getColorCameraParams())
            self.camera_working = True
            print(f"Kinect v2 device {serial} initialised")
            # Warm-up: discard initial frames so auto-exposure settles
            warmup_frames = self.fps * 2
            for _ in range(warmup_frames):
                frames = self.listener.waitForNewFrame()
                self.listener.release(frames)
            return True
        except Exception as e:
            print(f"Failed to initialise Kinect v2: {e}")
            self.cleanup()
            return False

    # -----------------------------------------------------------------------
    # Frame capture
    # -----------------------------------------------------------------------

    def capture_frames(self):
        """
        Wait for and return one aligned (colour, depth) frame pair.
        Returns
        -------
        rgb_array   : np.ndarray  shape (H, W, 3)  uint8   BGR
        depth_array : np.ndarray  shape (H, W)     uint16  depth in mm
        Both arrays are at the colour stream resolution after alignment.
        Returns (None, None) on failure.
        """
        if not self.camera_working:
            print("Camera not initialised")
            return None, None
        try:
            frames = self.listener.waitForNewFrame()
            color = frames[FrameType.Color]
            depth = frames[FrameType.Depth]
            # Registration output frames (both at depth resolution 512×424)
            undistorted = Frame(self.depth_width, self.depth_height, 4)
            registered  = Frame(self.depth_width, self.depth_height, 4)
            self.registration.apply(
                color, depth, undistorted, registered, bigdepth=None, color_depth_map=None
            )
            # registered: BGRA colour at each depth pixel (512×424)
            # undistorted: depth in mm, float32 (512×424)
            rgb_array   = registered.asarray()[..., :3].copy()   # drop alpha → BGR uint8
            depth_array = undistorted.asarray().astype(np.uint16)  # mm → uint16
            self.listener.release(frames)
            return rgb_array, depth_array
        except Exception as e:
            print(f"Error capturing frames: {e}")
            return None, None

    # -----------------------------------------------------------------------
    # Depth filtering
    # -----------------------------------------------------------------------

    def filter_depth_data(self, depth_data: np.ndarray) -> np.ndarray:
        """
        Clamp and median-filter raw depth data (values in mm).
        Pixels outside [300 mm, 6000 mm] are zeroed as noise / out-of-range.
        """
        filtered = depth_data.copy()
        filtered[filtered < 300]  = 0   # < 0.3 m  -> sensor noise
        filtered[filtered > 6000] = 0   # > 6.0 m  -> outside working range
        filtered = cv2.medianBlur(filtered.astype(np.uint16), 3)
        return filtered

    # -----------------------------------------------------------------------
    # Saving
    # -----------------------------------------------------------------------

    def save_capture(self, rgb_data: np.ndarray,
                     depth_data: np.ndarray) -> bool:
        """Save RGB and depth arrays (.npy + JPEG) to data_collection/."""
        self.capture_count += 1
        save_path = Path(self.save_directory)

        filtered_depth = self.filter_depth_data(depth_data)

        rgb_jpg        = save_path / f"{self.filename}_rgb_plant_{self.count}.jpg"
        rgb_npy        = save_path / f"{self.filename}_rgb_plant_{self.count}.npy"
        depth_orig_npy = save_path / f"{self.filename}_depth_plant_{self.count}.npy"
        depth_orig_jpg = save_path / f"{self.filename}_depth_plant_{self.count}.jpg"

        try:
            # RGB
            cv2.imwrite(str(rgb_jpg), rgb_data)
            np.save(str(rgb_npy),     rgb_data)

            # Depth (raw mm values)
            np.save(str(depth_orig_npy), depth_data)
            depth_display = (
                (depth_data.astype(np.float32) / 6000.0) * 255
            ).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imwrite(str(depth_orig_jpg), depth_colored)

            print(f"Saved capture {self.capture_count}:")
            print(f"  RGB  : {rgb_jpg.name}  /  {rgb_npy.name}")
            print(f"  Depth: {depth_orig_jpg.name}  /  {depth_orig_npy.name}")
            return True

        except Exception as e:
            print(f"Failed to save files: {e}")
            return False

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def cleanup(self):
        """Stop the Kinect v2 device and release all resources."""
        if self.device:
            try:
                self.device.stop()
                self.device.close()
            except Exception:
                pass
            self.device = None
        self.camera_working = False


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Kinect v2 Camera Host (Camera A - Green)")
    parser.add_argument(
        "filename", nargs="?", default="default",
        help="Filename prefix for saved files")
    parser.add_argument(
        "count", nargs="?", type=int, default=1,
        help="Plant number used in output file names")
    parser.add_argument(
        "--serial", default=KinectV2CameraHost.DEFAULT_SERIAL,
        help="Target Kinect v2 device serial number")
    return parser.parse_args()



def main():
    args = parse_arguments()

    print("=== Kinect v2 Camera Host (Camera A - Green) ===")
    print(f"Serial   : {args.serial}")
    print(f"Filename : {args.filename}")
    print(f"Plant No : {args.count}")

    camera = KinectV2CameraHost(
        target_serial=args.serial,
        filename=args.filename,
        count=args.count
    )

    print("Initialising Kinect v2 ...")
    if not camera.init_camera():
        print("Failed to initialise camera")
        camera.cleanup()
        return

    print("Capturing frames ...")
    rgb_data, depth_data = camera.capture_frames()

    if rgb_data is not None and depth_data is not None:
        if camera.save_capture(rgb_data, depth_data):
            print("Capture completed successfully")
        else:
            print("Failed to save capture files")
    else:
        print("Capture failed - no data received from camera")

    camera.cleanup()


if __name__ == "__main__":
    main()
