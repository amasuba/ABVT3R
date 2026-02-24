#!/usr/bin/env python3
"""
Intel RealSense Live Viewer - Camera A (Green)
Displays aligned colour and depth streams side-by-side in real-time.

Replaces : Kinect V2 viewer (pylibfreenect2)
Platform : NVIDIA Jetson Nano
Hardware : Intel RealSense D415 / D435 / D435i / D455

Usage
-----
    python viewer_green.py [--serial <DEVICE_SERIAL>]

Press 'q' or ESC to quit.
"""

import numpy as np
import cv2
import argparse

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 is not installed.")
    print("Install with: pip install pyrealsense2")
    raise SystemExit(1)

# Index of the target device when no serial is specified (0 = first device)
DEFAULT_DEVICE_INDEX = 1   # Green camera is typically the second enumerated device


def list_devices(ctx: "rs.context"):
    """Print and return info for all connected RealSense devices."""
    devices = ctx.query_devices()
    print(f"Found {len(devices)} RealSense device(s):")
    for i, dev in enumerate(devices):
        name   = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"  [{i}]  {name}   serial: {serial}")
    return devices


def main(target_serial: str = None):
    ctx     = rs.context()
    devices = list_devices(ctx)

    if len(devices) == 0:
        print("No RealSense device connected!")
        return

    # Resolve serial to use
    if target_serial:
        serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
        if target_serial not in serials:
            print(f"Device {target_serial} not found. Available: {serials}")
            return
        serial = target_serial
    else:
        idx    = min(DEFAULT_DEVICE_INDEX, len(devices) - 1)
        serial = devices[idx].get_info(rs.camera_info.serial_number)
        print(f"No serial specified - using device [{idx}]  serial: {serial}")

    # -----------------------------------------------------------------------
    # Configure and start the pipeline
    # -----------------------------------------------------------------------
    pipeline = rs.pipeline()
    config   = rs.config()

    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 1280, 720,  rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth,  640, 480,  rs.format.z16,  30)

    profile = pipeline.start(config)
    align   = rs.align(rs.stream.color)

    print(f"Stream started for device {serial}")
    print("Press 'q' or ESC to quit.")

    try:
        frame_count = 0

        while True:
            frames         = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            colour_frame = aligned_frames.get_color_frame()
            depth_frame  = aligned_frames.get_depth_frame()

            if not colour_frame or not depth_frame:
                continue

            colour_image = np.asanyarray(colour_frame.get_data())   # BGR uint8
            depth_image  = np.asanyarray(depth_frame.get_data())    # mm  uint16

            # Colour-map depth for display (scale 0-6000 mm -> 0-255)
            depth_display = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=255.0 / 6000.0),
                cv2.COLORMAP_JET
            )

            # Resize both images to the same height for horizontal stacking
            colour_resized = cv2.resize(colour_image, (640, 360))
            depth_resized  = cv2.resize(depth_display, (640, 360))

            combined = np.hstack((colour_resized, depth_resized))
            cv2.imshow("RealSense - Green Camera  [ colour | depth ]", combined)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Frames received: {frame_count}")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):   # 'q' or ESC
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Intel RealSense Live Viewer (Camera A - Green)")
    parser.add_argument(
        "--serial", default=None,
        help="Target RealSense device serial number "
             "(default: second enumerated device)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(target_serial=args.serial)
