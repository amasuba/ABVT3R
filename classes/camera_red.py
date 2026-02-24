#!/usr/bin/env python3
"""
Intel RealSense Camera Host - Camera B (Red)
Captures aligned RGB and Depth data from an Intel RealSense camera and saves
to the data_collection/ directory.  Device selection is by serial number.

Replaces : Kinect V2 (pylibfreenect2)
Platform : NVIDIA Jetson Nano
Hardware : Intel RealSense D415 / D435 / D435i / D455
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
# Optional RealSense import
# ---------------------------------------------------------------------------
REALSENSE_AVAILABLE = False
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("Using pyrealsense2 library")
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available. "
          "Install with: pip install pyrealsense2")


class RealSenseCameraHost:
    """
    Intel RealSense RGB-D camera host.

    Streams are configured as:
      - Colour  : 1280 x 720  BGR8  @ 30 fps
      - Depth   :  640 x 480  Z16   @ 30 fps  (aligned to colour resolution)

    The depth map returned is aligned to the colour frame and expressed in
    millimetres (uint16), matching the data format previously produced by the
    Kinect V2 pipeline so that downstream processing is unchanged.
    """

    # Default serial for Camera B (Red).
    # Replace with the actual serial printed on the camera label.
    DEFAULT_SERIAL = "815412070002"

    def __init__(self,
                 target_serial: str = DEFAULT_SERIAL,
                 filename: str = "default",
                 count: int = 1,
                 color_width: int = 1280,
                 color_height: int = 720,
                 depth_width: int = 640,
                 depth_height: int = 480,
                 fps: int = 30):

        self.pipeline       = None
        self.config         = None
        self.align          = None
        self.profile        = None
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

    def list_realsense_devices(self):
        """Return a list of serial-number strings for all connected devices."""
        if not REALSENSE_AVAILABLE:
            print("pyrealsense2 not available")
            return []

        ctx     = rs.context()
        devices = ctx.query_devices()
        serials = []

        print(f"Found {len(devices)} RealSense device(s):")
        for dev in devices:
            serial = dev.get_info(rs.camera_info.serial_number)
            name   = dev.get_info(rs.camera_info.name)
            serials.append(serial)
            print(f"  Serial: {serial}  Model: {name}")

        return serials

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def init_camera(self) -> bool:
        """
        Initialise the RealSense pipeline for the target device serial.
        Returns True on success, False otherwise.
        """
        if not REALSENSE_AVAILABLE:
            print("pyrealsense2 not available")
            return False

        available = self.list_realsense_devices()
        if not available:
            print("No RealSense devices found")
            return False

        if self.target_serial not in available:
            print(f"Device {self.target_serial} not found. "
                  f"Available: {available}")
            return False

        try:
            self.pipeline = rs.pipeline()
            self.config   = rs.config()

            # Bind pipeline to the specific device
            self.config.enable_device(self.target_serial)

            # Configure colour and depth streams
            self.config.enable_stream(
                rs.stream.color,
                self.color_width, self.color_height,
                rs.format.bgr8, self.fps
            )
            self.config.enable_stream(
                rs.stream.depth,
                self.depth_width, self.depth_height,
                rs.format.z16, self.fps
            )

            # Start the pipeline
            self.profile = self.pipeline.start(self.config)

            # Depth frames will be spatially aligned to the colour frame
            self.align = rs.align(rs.stream.color)

            # Warm-up: discard frames until auto-exposure and AWB stabilise
            print("Warming up (auto-exposure / AWB) for 2 s ...")
            warmup_frames = self.fps * 2
            for _ in range(warmup_frames):
                self.pipeline.wait_for_frames()

            self.camera_working = True
            print(f"RealSense camera {self.target_serial} initialised")
            return True

        except Exception as e:
            print(f"Failed to initialise RealSense camera: {e}")
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
            frames         = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            colour_frame = aligned_frames.get_color_frame()
            depth_frame  = aligned_frames.get_depth_frame()

            if not colour_frame or not depth_frame:
                print("Empty frame received")
                return None, None

            # Convert to NumPy -- depth is in mm (Z16 format)
            rgb_array   = np.asanyarray(colour_frame.get_data())
            depth_array = np.asanyarray(depth_frame.get_data())

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
        """Stop the RealSense pipeline and release all resources."""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        self.camera_working = False


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Intel RealSense Camera Host (Camera B - Red)")
    parser.add_argument(
        "filename", nargs="?", default="default",
        help="Filename prefix for saved files")
    parser.add_argument(
        "count", nargs="?", type=int, default=1,
        help="Plant number used in output file names")
    parser.add_argument(
        "--serial", default=RealSenseCameraHost.DEFAULT_SERIAL,
        help="Target RealSense device serial number")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=== RealSense Camera Host (Camera B - Red) ===")
    print(f"Serial   : {args.serial}")
    print(f"Filename : {args.filename}")
    print(f"Plant No : {args.count}")

    camera = RealSenseCameraHost(
        target_serial=args.serial,
        filename=args.filename,
        count=args.count
    )

    print("Initialising RealSense camera ...")
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
