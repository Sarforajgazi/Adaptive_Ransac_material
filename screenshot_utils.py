"""
screenshot_utils.py

Open3D's interactive viewer has a built-in "P" (and PrtSc) key binding that
dumps a screenshot (ScreenCapture_<timestamp>.png) and its matching camera
parameters (ScreenCamera_<timestamp>.json) into the current working
directory -- there's no argument to redirect that, since it's handled
entirely inside Open3D's C++ key callback.

The wrappers below run the interactive window with the process cwd
temporarily switched to screenshots/ (restored immediately after), so those
files land in one place instead of scattered throughout the repo.
"""
import contextlib
import os
from pathlib import Path

import open3d as o3d

SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"


@contextlib.contextmanager
def _cwd_in_screenshot_dir():
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    previous = os.getcwd()
    os.chdir(SCREENSHOT_DIR)
    try:
        yield
    finally:
        os.chdir(previous)


def draw_geometries(*args, **kwargs):
    """Drop-in replacement for o3d.visualization.draw_geometries()."""
    with _cwd_in_screenshot_dir():
        o3d.visualization.draw_geometries(*args, **kwargs)


def run(vis):
    """Drop-in replacement for vis.run() on an interactive Visualizer."""
    with _cwd_in_screenshot_dir():
        vis.run()
