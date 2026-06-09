#!/usr/bin/env python3
"""
Kinect v2 Live Viewer - Camera A (Green)
Displays colour and depth streams side-by-side in real-time.

Hardware : Microsoft Kinect v2
Usage    : python viewer_green.py [--serial <DEVICE_SERIAL>]

Press 'q' or ESC to quit.
"""

import numpy as np
import cv2
import argparse

KINECT_AVAILABLE = False
try:
    from pylibfreenect2 import (
        Freenect2, SyncMultiFrameListener, FrameType,
        Registration, Frame, OpenGLPacketPipeline
    )
    KINECT_AVAILABLE = True
except ImportError:
    print("ERROR: pylibfreenect2 is not installed.")
    print("Activate the project venv: source abvt310/bin/activate")
    raise SystemExit(1)

# Camera A (Green) is the first enumerated device (index 0)
DEFAULT_DEVICE_INDEX = 0


def list_devices(fn: "Freenect2"):
    num = fn.enumerateDevices()
    print(f"Found {num} Kinect v2 device(s):")
    serials = []
    for i in range(num):
        s = fn.getDeviceSerialNumber(i)
        serials.append(s)
        print(f"  [{i}]  serial: {s}")
    return serials


def main(target_serial: str = None):
    fn = Freenect2()
    serials = list_devices(fn)

    if not serials:
        print("No Kinect v2 device connected!")
        return

    if target_serial and target_serial in serials:
        serial = target_serial
    else:
        idx = min(DEFAULT_DEVICE_INDEX, len(serials) - 1)
        serial = serials[idx]
        print(f"Using device [{idx}]  serial: {serial}")

    try:
        pipeline = OpenGLPacketPipeline()
    except Exception:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
        print("OpenGL pipeline unavailable, using CPU pipeline")

    device = fn.openDevice(serial, pipeline=pipeline)
    listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)
    device.start()

    registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())
    undistorted = Frame(512, 424, 4)
    registered  = Frame(512, 424, 4)

    print(f"Stream started — device {serial}")
    print("Press 'q' or ESC to quit.")

    frame_count = 0
    try:
        while True:
            frames = listener.waitForNewFrame()
            color = frames[FrameType.Color]
            depth = frames[FrameType.Depth]

            registration.apply(color, depth, undistorted, registered,
                               bigdepth=None, color_depth_map=None)

            colour_image = registered.asarray()[..., :3]      # BGR uint8, 512×424
            depth_image  = undistorted.asarray()              # float32 mm, 512×424

            listener.release(frames)

            # Colour-map depth (0–4500 mm → 0–255)
            depth_display = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=255.0 / 4500.0),
                cv2.COLORMAP_JET
            )

            colour_resized = cv2.resize(colour_image, (640, 360))
            depth_resized  = cv2.resize(depth_display, (640, 360))

            combined = np.hstack((colour_resized, depth_resized))
            cv2.imshow("Kinect v2 - Green Camera  [ colour | depth ]", combined)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Frames received: {frame_count}")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        device.stop()
        device.close()
        cv2.destroyAllWindows()
        print("Cleanup complete")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Kinect v2 Live Viewer (Camera A - Green)")
    parser.add_argument(
        "--serial", default=None,
        help="Target Kinect v2 device serial number "
             "(default: first enumerated device)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(target_serial=args.serial)
