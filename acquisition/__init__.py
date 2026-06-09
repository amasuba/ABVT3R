"""
acquisition — Subsystem 1: RGB-D Data Acquisition & Dataset Management
=======================================================================
Captures 12-view (30° angular protocol) RGB-D frames from dual Microsoft
Kinect v2 cameras and organises them under acquisition/dataset/specimens/.

Entry points
------------
capture/session_manager.py   Full 12-view capture session (CLI + Python API)
capture/camera_green.py      Single-view capture — Camera A (Green)
capture/camera_red.py        Single-view capture — Camera B (Red)
capture/viewer_green.py      Live depth/RGB preview — Camera A
capture/viewer_red.py        Live depth/RGB preview — Camera B
dataset/dataset_viewer.py    Browse & export specimen contact sheets
"""
