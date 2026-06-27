"""
batch_segment_lidar.py

Runs Schnabel C++ RANSAC ground segmentation on all per-timestep LiDAR
frames from TartanAir Office trajectory P0000.

Input  : data/Office/Data_omni/P0000/lidar/*.ply   (678 frames, ~48k pts each)
Output : data/Office/Data_omni/P0000/schnabel_segmented/
           000000_segmented.ply   (green=ground, grey=obstacles)
           000001_segmented.ply
           ...
           summary.csv            (per-frame stats for quality checking)

Coordinate system: TartanAir uses NED (Z-down), so ground = highest Z value.
"""

import sys
import os
import glob
import time
import csv

import numpy as np
from plyfile import PlyData

# Add schnabel_cython/ to path so we can import the compiled .pyd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))

try:
    import schnabel_ransac
    print("schnabel_ransac loaded successfully.")
except ImportError as e:
    print(f"ERROR: Could not import schnabel_ransac: {e}")
    print("Make sure you are inside the .venv and the .pyd file is in schnabel_cython/")
    sys.exit(1)


# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE   = os.path.dirname(os.path.abspath(__file__))
LIDAR_DIR   = os.path.join(WORKSPACE, "data", "Office", "Data_omni", "P0000", "lidar")
OUT_DIR     = os.path.join(WORKSPACE, "data", "Office", "Data_omni", "P0000", "schnabel_segmented")
SUMMARY_CSV = os.path.join(OUT_DIR, "summary.csv")

# ── RANSAC parameters ──────────────────────────────────────────────────────────
# Absolute distance threshold in metres.
# Frames are sparse (~48k pts over ~150m scene) so 0.3m is a reasonable fit tolerance.
EPSILON        = 0.3
NORMAL_THRESH  = 0.9      # cos(~25°) — normals must roughly align with ground
MIN_SUPPORT    = 300      # minimum inlier points to accept a plane
PROBABILITY    = 0.01     # miss-probability stopping criterion
MAX_SHAPES     = 20       # detect up to 20 planes per frame, then pick ground

# ── Ground identification ──────────────────────────────────────────────────────
# TartanAir NED convention: Z increases downward, so the ground (floor) has
# the HIGHEST (most positive) average Z among all horizontal planes.
Z_MODE = "z_down"

# Minimum |normal_z| to consider a plane "horizontal" (floor/ceiling candidate)
HORIZONTAL_THRESH = 0.80


def load_ply_xyz(filepath):
    """Read x,y,z from a PLY file, return float32 numpy array (N,3)."""
    ply = PlyData.read(filepath)
    v   = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    return pts


def save_segmented_ply(filepath, points, ground_mask):
    """
    Save a colour-coded binary PLY:
      green  [0, 200, 50]  → ground plane inliers
      grey   [150,150,150] → everything else (obstacles, walls, noise)
    """
    n = len(points)
    colors = np.full((n, 3), 150, dtype=np.uint8)
    colors[ground_mask] = [0, 200, 50]

    vertex = np.zeros(n, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"]     = points[:, 0]
    vertex["y"]     = points[:, 1]
    vertex["z"]     = points[:, 2]
    vertex["red"]   = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"]  = colors[:, 2]

    from plyfile import PlyElement
    el = PlyElement.describe(vertex, "vertex")
    from plyfile import PlyData as _PlyData
    _PlyData([el]).write(filepath)


def find_ground_plane(shapes, points, z_mode):
    """
    From detected RANSAC shapes, pick the ground plane.

    Strategy:
      1. Keep only horizontal planes (|normal_z| >= HORIZONTAL_THRESH).
      2. Among those, pick the one with the highest avg_z (z_down / NED)
         or lowest avg_z (z_up / standard).
      3. If no horizontal plane found, fall back to largest plane.

    Returns (ground_shape, avg_z, z_alignment) or (None, None, None).
    """
    if not shapes:
        return None, None, None

    candidates = []
    for shape in shapes:
        mask        = shape["inlier_mask"]
        plane_pts   = points[mask]
        if len(plane_pts) < 10:
            continue

        # Estimate plane normal via PCA on inlier points
        cov           = np.cov(plane_pts.T)
        evals, evecs  = np.linalg.eig(cov)
        normal        = evecs[:, np.argmin(evals)]
        z_align       = abs(float(normal[2]))
        avg_z         = float(np.mean(plane_pts[:, 2]))

        candidates.append({
            "shape":    shape,
            "avg_z":    avg_z,
            "z_align":  z_align,
            "n_points": shape["n_points"],
        })

    # Filter to horizontal planes
    horizontal = [c for c in candidates if c["z_align"] >= HORIZONTAL_THRESH]

    if not horizontal:
        # Fallback: use the plane with the most inliers
        horizontal = sorted(candidates, key=lambda c: c["n_points"], reverse=True)[:1]
        if not horizontal:
            return None, None, None

    # Sort by Z to find ground
    reverse = (z_mode == "z_down")   # NED: ground has highest Z
    horizontal.sort(key=lambda c: c["avg_z"], reverse=reverse)

    best = horizontal[0]
    return best["shape"], best["avg_z"], best["z_align"]


def process_frame(filepath, frame_name, out_dir):
    """
    Process one LiDAR frame. Returns a stats dict.
    """
    stats = {
        "frame":          frame_name,
        "n_points":       0,
        "n_planes":       0,
        "ground_found":   False,
        "ground_pts":     0,
        "ground_pct":     0.0,
        "ground_avg_z":   None,
        "z_alignment":    None,
        "time_s":         0.0,
        "error":          "",
    }

    t0 = time.perf_counter()

    try:
        points = load_ply_xyz(filepath)
        stats["n_points"] = len(points)

        if len(points) < MIN_SUPPORT:
            stats["error"] = "too_few_points"
            return stats

        # Run Schnabel RANSAC
        shapes, _ = schnabel_ransac.detect(
            points,
            shapes            = ["plane"],
            relative_epsilon  = False,
            epsilon           = EPSILON,
            normal_thresh     = NORMAL_THRESH,
            min_support       = MIN_SUPPORT,
            probability       = PROBABILITY,
            max_shapes        = MAX_SHAPES,
        )
        stats["n_planes"] = len(shapes)

        # Pick ground plane
        ground_shape, avg_z, z_align = find_ground_plane(shapes, points, Z_MODE)

        if ground_shape is None:
            stats["error"] = "no_ground_found"
            return stats

        ground_mask = ground_shape["inlier_mask"]
        stats["ground_found"] = True
        stats["ground_pts"]   = int(ground_mask.sum())
        stats["ground_pct"]   = round(100.0 * stats["ground_pts"] / len(points), 2)
        stats["ground_avg_z"] = round(avg_z, 3)
        stats["z_alignment"]  = round(z_align, 3)

        # Save segmented PLY
        out_path = os.path.join(out_dir, frame_name.replace(".ply", "_segmented.ply"))
        save_segmented_ply(out_path, points, ground_mask)

    except Exception as e:
        stats["error"] = str(e)

    stats["time_s"] = round(time.perf_counter() - t0, 3)
    return stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(LIDAR_DIR, "*.ply")))
    if not files:
        print(f"ERROR: No .ply files found in {LIDAR_DIR}")
        sys.exit(1)

    total = len(files)
    print(f"Found {total} LiDAR frames.")
    print(f"Output directory: {OUT_DIR}")
    print(f"Parameters: epsilon={EPSILON}m, min_support={MIN_SUPPORT}, "
          f"normal_thresh={NORMAL_THRESH}, z_mode={Z_MODE}\n")

    results     = []
    n_success   = 0
    n_failed    = 0
    t_total     = time.perf_counter()

    csv_fields = ["frame", "n_points", "n_planes", "ground_found",
                  "ground_pts", "ground_pct", "ground_avg_z", "z_alignment",
                  "time_s", "error"]

    with open(SUMMARY_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        for i, filepath in enumerate(files):
            frame_name = os.path.basename(filepath)
            stats      = process_frame(filepath, frame_name, OUT_DIR)
            writer.writerow(stats)
            csvfile.flush()   # write each row immediately in case of crash
            results.append(stats)

            status = "OK" if stats["ground_found"] else f"FAIL ({stats['error']})"
            if stats["ground_found"]:
                n_success += 1
                print(
                    f"[{i+1:3d}/{total}] {frame_name}  "
                    f"planes={stats['n_planes']}  "
                    f"ground={stats['ground_pts']} pts ({stats['ground_pct']}%)  "
                    f"avg_z={stats['ground_avg_z']}  "
                    f"time={stats['time_s']}s  [{status}]"
                )
            else:
                n_failed += 1
                print(f"[{i+1:3d}/{total}] {frame_name}  [{status}]")

    elapsed = time.perf_counter() - t_total

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("BATCH SEGMENTATION COMPLETE")
    print("="*70)
    print(f"  Total frames    : {total}")
    print(f"  Succeeded       : {n_success}  ({100*n_success/total:.1f}%)")
    print(f"  Failed          : {n_failed}  ({100*n_failed/total:.1f}%)")
    print(f"  Total time      : {elapsed:.1f}s  ({elapsed/total:.2f}s per frame)")

    if n_success > 0:
        ground_pcts = [r["ground_pct"] for r in results if r["ground_found"]]
        avg_z_vals  = [r["ground_avg_z"] for r in results if r["ground_found"]]
        print(f"\n  Ground coverage (% of frame points labelled as ground):")
        print(f"    Min  : {min(ground_pcts):.1f}%")
        print(f"    Max  : {max(ground_pcts):.1f}%")
        print(f"    Mean : {sum(ground_pcts)/len(ground_pcts):.1f}%")
        print(f"\n  Ground avg Z across frames: {sum(avg_z_vals)/len(avg_z_vals):.2f}m")

    print(f"\n  Segmented PLY files : {OUT_DIR}/")
    print(f"  Per-frame stats CSV : {SUMMARY_CSV}")
    print("\n  To visually check a frame, run:")
    print(f"    python check_frame.py 0")
    print(f"    python check_frame.py 100")


if __name__ == "__main__":
    main()
