"""
convert_laz_to_ply.py

Converts the downloaded OpenTopography LAZ tiles (off_road_data/raw_laz/) to
.ply (off_road_data/ply/), keeping x/y/z plus the ASPRS `classification`
field. Classification 2 = Ground (real ground truth from the survey's own
processing, not something we're inferring) -- see SESSION_PROGRESS_LOG.md
for the field inspection. Coordinates are left as raw UTM (absolute,
hundreds-of-thousands scale) here; recentering happens downstream in
visualize_offroad_rl.py, same pattern as new_data/.

Usage:
    python convert_laz_to_ply.py
"""
import os
import glob
import numpy as np
import laspy
from plyfile import PlyData, PlyElement

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "off_road_data", "raw_laz")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "off_road_data", "ply")


def convert(laz_path, ply_path):
    las = laspy.read(laz_path)
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    dims = set(las.point_format.dimension_names)

    # Not every survey provides both -- e.g. Sawpit Wash/San Andreas Fault
    # (airborne) have real ASPRS classification but no RGB; Ridgecrest
    # (mobile scan) has RGB but classification is all 0 (never run).
    has_classification = "classification" in dims and np.any(np.asarray(las.classification) != 0)
    has_color = "red" in dims and np.any(np.asarray(las.red) != 0)

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if has_classification:
        fields.append(("classification", "u1"))
    if has_color:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    vertex = np.zeros(len(x), dtype=fields)
    vertex["x"], vertex["y"], vertex["z"] = x, y, z
    info = []
    if has_classification:
        classification = np.asarray(las.classification, dtype=np.uint8)
        vertex["classification"] = classification
        n_ground = int((classification == 2).sum())
        info.append(f"{n_ground} ground ({n_ground/len(x)*100:.1f}%)")
    if has_color:
        # LAS stores 16-bit channels; scale down to 8-bit for ply
        vertex["red"] = (np.asarray(las.red, dtype=np.uint32) >> 8).astype(np.uint8)
        vertex["green"] = (np.asarray(las.green, dtype=np.uint32) >> 8).astype(np.uint8)
        vertex["blue"] = (np.asarray(las.blue, dtype=np.uint32) >> 8).astype(np.uint8)
        info.append("has RGB")
    if not info:
        info.append("xyz only, no classification/RGB")

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(ply_path)
    print(f"  {os.path.basename(laz_path)} -> {os.path.basename(ply_path)}: "
          f"{len(x)} points, {', '.join(info)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    laz_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.laz")))
    print(f"Converting {len(laz_files)} tiles...")
    for laz_path in laz_files:
        name = os.path.splitext(os.path.basename(laz_path))[0]
        ply_path = os.path.join(OUT_DIR, f"{name}.ply")
        convert(laz_path, ply_path)


if __name__ == "__main__":
    main()
