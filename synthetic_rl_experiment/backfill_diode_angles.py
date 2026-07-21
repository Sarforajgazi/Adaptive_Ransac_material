"""
One-off backfill: results.csv had a blank angle_from_vertical_deg column
for the 18 DIODE rows logged before the dict-key bug in
visualize_diode_rl.py was fixed (key was "angle", CSV field was
"angle_from_vertical_deg" -> silently wrote ""). The underlying detection
was never wrong -- select_shape() computed the angle correctly and used
it for genuine/fallback classification; only the CSV write was affected.

This recomputes the angle for each blank row using the exact eps/
min_supp/norm_th already recorded in that row (no RL re-inference
needed), on the same crop's back-projected points, and fills in just
the missing column -- leaving found/pred_pct/etc. exactly as originally
logged.
"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schnabel_cython"))

import schnabel_ransac
from visualize_diode_rl import DIODE_ROOT, backproject, select_shape

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_screenshots", "rgbd", "diode", "results.csv")

with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

filled = 0
for row in rows:
    if row["angle_from_vertical_deg"]:
        continue
    crop_stem = os.path.join(DIODE_ROOT, row["crop"])
    points_world, _ = backproject(crop_stem)
    centroid = points_world.mean(axis=0)
    points = points_world - centroid

    eps = float(row["eps"])
    min_supp = int(row["min_supp"])
    norm_th = float(row["norm_th"])

    shapes, _ = schnabel_ransac.detect(
        points, shapes=["plane"], relative_epsilon=False,
        epsilon=eps, normal_thresh=norm_th, min_support=min_supp,
        probability=0.001, normal_knn=20, max_shapes=10,
    )
    if shapes:
        _, angle, _ = select_shape(shapes, points)
        row["angle_from_vertical_deg"] = angle
        filled += 1
        print(f"  {row['crop']}: angle={angle:.1f} deg")
    else:
        print(f"  {row['crop']}: RANSAC found no shapes on recompute (unexpected)")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["crop", "n_points", "found", "pred_pct", "angle_from_vertical_deg", "eps", "min_supp", "norm_th"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nBackfilled {filled} rows -> {CSV_PATH}")
