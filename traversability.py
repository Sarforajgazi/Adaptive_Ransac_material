"""
traversability.py

Standalone grid-based ground/obstacle traversability classifier.

Does NOT depend on the RL pipeline (ransac_env.py, train_rl.py,
rl_evaluator.py) -- kept fully independent so it can be evaluated and
compared against the RANSAC/RL ground segmentation separately, without
sharing state or logic. It CAN optionally use the real Schnabel engine
(plane_fit_method="schnabel") as one of two interchangeable per-cell
plane-fitting strategies -- see below -- but never touches the RL side.

Pipeline:
    1. Load + voxel downsample the raw point cloud.
    2. Estimate per-point normals.
    3. Split points into horizontal-ish (floor/ceiling candidates) vs
       vertical-ish (walls/obstacles) using |normal_z|.
    4. Grid into XY cells. Per cell:
         - keep only the lowest height cluster of horizontal-ish points
           (the presumed floor, not a tabletop/shelf sitting above it)
         - fit a small local plane, via one of two interchangeable
           strategies (plane_fit_method):
             "lightweight" (default) -- a minimal custom RANSAC in plain
               numpy, robust to a handful of outliers, no dependencies.
             "schnabel" -- the real Schnabel C++ engine applied to just
               this cell's points instead of the whole frame. min_support/
               epsilon/normal_thresh are scaled way down from the
               whole-frame pipeline's values since a cell only has a
               handful of points (see SCHNABEL_CELL_* constants).
         - compute slope, roughness, inlier ratio
         - if the cell contains any vertical-surface points -> obstacle
         - else classify by slope/roughness thresholds
    5. Compare each traversable cell's floor height to its neighbors to
       catch steps/curbs/ledges.
    6. Output a grid of TRAVERSABLE / OBSTACLE / UNKNOWN labels.

Usage:
    python traversability.py --dataset Office
    python traversability.py --dataset OldScandinavia --frame 000032
    python traversability.py --dataset Office --plane-fit-method schnabel
"""

import os
import glob
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(WORKSPACE, "data")

# ---------------- Config ----------------
VOXEL_SIZE = 0.05
NORMAL_KNN = 20
HORIZONTAL_NORMAL_THRESH = 0.7      # |normal_z| >= this -> horizontal-ish
CELL_SIZE = 0.18                    # metres, roughly a robot-footprint scale
LOWEST_CLUSTER_BAND = 0.15          # metres above local min height still counted as floor
MIN_POINTS_PER_CELL = 5
LOCAL_RANSAC_ITERS = 30
LOCAL_RANSAC_EPS = 0.03             # metres
SLOPE_THRESH_DEG = 25.0
ROUGHNESS_THRESH = 0.05             # metres
STEP_THRESH = 0.12                  # metres, vs. neighbor cell
Z_MODE = "z_down"                   # TartanAir NED convention: ground = highest Z

# Real-Schnabel per-cell fitting (plane_fit_method="schnabel"). min_support is
# an absolute point count, tuned much lower than the whole-frame pipeline's
# 500-800 -- cells only have a handful of points. epsilon/normal_thresh mirror
# this module's own ROUGHNESS_THRESH/SLOPE_THRESH_DEG so a cell that "passes"
# Schnabel's own inlier search is consistent with what this module would
# otherwise call traversable.
SCHNABEL_CELL_EPSILON = ROUGHNESS_THRESH
SCHNABEL_CELL_MIN_SUPPORT = 10
SCHNABEL_CELL_NORMAL_THRESH = 0.7   # cos(~45deg) -- looser than SLOPE_THRESH_DEG=25deg
                                     # on purpose; the slope/roughness check afterward
                                     # still applies the real threshold
SCHNABEL_CELL_PROBABILITY = 0.05    # cells are tiny; an exhaustive search is cheap

TRAVERSABLE, OBSTACLE, UNKNOWN = "traversable", "obstacle", "unknown"
LABEL_COLORS = {
    TRAVERSABLE: [0.0, 1.0, 0.0],   # green
    OBSTACLE:    [1.0, 0.0, 0.0],   # red
    UNKNOWN:     [0.5, 0.5, 0.5],   # grey
}


def load_points(filepath, voxel_size=VOXEL_SIZE):
    ply = PlyData.read(filepath)
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    if voxel_size is not None and voxel_size > 0.0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pts = np.asarray(pcd.points, dtype=np.float32)
    return pts


def estimate_normals(points, knn=NORMAL_KNN):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.asarray(pcd.normals)


def local_ransac_plane(points, iters=LOCAL_RANSAC_ITERS, eps=LOCAL_RANSAC_EPS, rng=None):
    """
    Fit a plane to a small cluster of points with a lightweight RANSAC,
    robust to a handful of outliers within the cell (unlike a plain PCA
    normal, which every point influences equally).

    Returns (normal, mean_residual, inlier_ratio). Falls back to a plain
    PCA fit if there aren't enough points to sample distinct triplets.
    """
    n = len(points)
    if n < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0, 0.0

    rng = rng or np.random.default_rng()
    best_inliers = None
    best_count = -1

    idx = rng.integers(0, n, size=(iters, 3))
    p0, p1, p2 = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-8
    candidate_normals = np.divide(cross, norms, out=np.zeros_like(cross), where=norms > 1e-8)

    for i in range(iters):
        if not valid[i]:
            continue
        normal = candidate_normals[i]
        d = -np.dot(normal, p0[i])
        dist = np.abs(points @ normal + d)
        inliers = dist <= eps
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        centroid = points.mean(axis=0)
        cov = np.cov((points - centroid).T)
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]
        residual = float(np.mean(np.abs((points - centroid) @ normal)))
        return normal, residual, 1.0

    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    cov = np.cov((inlier_pts - centroid).T)
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0]
    residual = float(np.mean(np.abs((inlier_pts - centroid) @ normal)))
    inlier_ratio = best_count / n
    return normal, residual, inlier_ratio


def schnabel_local_plane(points, epsilon=SCHNABEL_CELL_EPSILON, min_support=SCHNABEL_CELL_MIN_SUPPORT,
                          normal_thresh=SCHNABEL_CELL_NORMAL_THRESH, probability=SCHNABEL_CELL_PROBABILITY):
    """
    Same interface/return shape as local_ransac_plane (normal, mean_residual,
    inlier_ratio), but the plane fit itself comes from the real Schnabel C++
    engine (schnabel_cython) applied to just this cell's points, instead of
    the lightweight custom RANSAC above. Imported lazily so the rest of this
    module still works without the compiled .pyd present.
    """
    import sys
    sys.path.insert(0, os.path.join(WORKSPACE, "schnabel_cython"))
    import schnabel_ransac

    n = len(points)
    if n < max(3, min_support):
        centroid = points.mean(axis=0)
        cov = np.cov((points - centroid).T)
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]
        residual = float(np.mean(np.abs((points - centroid) @ normal)))
        return normal, residual, 0.0

    try:
        shapes, _ = schnabel_ransac.detect(
            points.astype(np.float64), shapes=["plane"], relative_epsilon=False,
            epsilon=epsilon, normal_thresh=normal_thresh, min_support=min_support,
            probability=probability, normal_knn=min(NORMAL_KNN, n - 1), max_shapes=3,
        )
    except Exception:
        shapes = []

    if not shapes:
        centroid = points.mean(axis=0)
        cov = np.cov((points - centroid).T)
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]
        residual = float(np.mean(np.abs((points - centroid) @ normal)))
        return normal, residual, 0.0

    best = max(shapes, key=lambda s: s["n_points"])
    mask = best["inlier_mask"]
    inlier_pts = points[mask]
    centroid = inlier_pts.mean(axis=0)
    cov = np.cov((inlier_pts - centroid).T)
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0]
    residual = float(np.mean(np.abs((inlier_pts - centroid) @ normal)))
    inlier_ratio = int(mask.sum()) / n
    return normal, residual, inlier_ratio


def classify_frame(points, cell_size=CELL_SIZE, z_mode=Z_MODE, seed=0,
                    lowest_cluster_band=LOWEST_CLUSTER_BAND, plane_fit_method="lightweight"):
    """
    Runs the full pipeline on one frame's points.
    Returns (cell_results, cell_idx) where cell_results maps
    (cx, cy) -> {label, n_points, height, slope_deg, roughness}
    and cell_idx is a per-point (cx, cy) array for coloring/visualization.
    """
    rng = np.random.default_rng(seed)
    normals = estimate_normals(points)
    z_align = np.abs(normals[:, 2])
    horiz_mask = z_align >= HORIZONTAL_NORMAL_THRESH

    xy = points[:, :2]
    cell_idx = np.floor(xy / cell_size).astype(np.int64)

    cells = {}
    for i in range(len(points)):
        key = (int(cell_idx[i, 0]), int(cell_idx[i, 1]))
        cells.setdefault(key, []).append(i)

    cell_results = {}
    for key, idxs in cells.items():
        idxs = np.array(idxs)
        cell_pts = points[idxs]
        cell_horiz_mask = horiz_mask[idxs]
        has_vertical = bool(np.any(~cell_horiz_mask))

        horiz_pts = cell_pts[cell_horiz_mask]
        if len(horiz_pts) < MIN_POINTS_PER_CELL:
            cell_results[key] = {"label": UNKNOWN, "n_points": len(cell_pts),
                                  "height": None, "slope_deg": None, "roughness": None}
            continue

        z = horiz_pts[:, 2]
        ref_height = z.max() if z_mode == "z_down" else z.min()
        floor_mask = (z >= ref_height - lowest_cluster_band) if z_mode == "z_down" \
            else (z <= ref_height + lowest_cluster_band)
        floor_pts = horiz_pts[floor_mask]

        if len(floor_pts) < MIN_POINTS_PER_CELL:
            cell_results[key] = {"label": UNKNOWN, "n_points": len(cell_pts),
                                  "height": None, "slope_deg": None, "roughness": None}
            continue

        if plane_fit_method == "schnabel":
            normal, roughness, inlier_ratio = schnabel_local_plane(floor_pts)
            if inlier_ratio == 0.0:
                # schnabel_local_plane fell back to an unvalidated PCA fit --
                # either fewer points than SCHNABEL_CELL_MIN_SUPPORT, or Schnabel's
                # own search found no shape at all. A real successful fit's best
                # shape always has >= min_support inliers, so inlier_ratio==0.0
                # unambiguously means "don't trust this," not "a bad but real fit."
                cell_results[key] = {"label": UNKNOWN, "n_points": len(cell_pts),
                                      "height": None, "slope_deg": None, "roughness": None}
                continue
        else:
            normal, roughness, inlier_ratio = local_ransac_plane(floor_pts, rng=rng)
        slope_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))
        mean_height = float(floor_pts[:, 2].mean())

        if has_vertical:
            label = OBSTACLE
        elif slope_deg <= SLOPE_THRESH_DEG and roughness <= ROUGHNESS_THRESH:
            label = TRAVERSABLE
        else:
            label = OBSTACLE

        cell_results[key] = {
            "label": label, "n_points": len(cell_pts), "height": mean_height,
            "slope_deg": slope_deg, "roughness": roughness,
        }

    # Step/discontinuity pass: demote traversable cells next to a big height jump
    for key, info in cell_results.items():
        if info["label"] != TRAVERSABLE or info["height"] is None:
            continue
        cx, cy = key
        for nkey in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]:
            neighbor = cell_results.get(nkey)
            if neighbor and neighbor["height"] is not None:
                if abs(neighbor["height"] - info["height"]) > STEP_THRESH:
                    info["label"] = OBSTACLE
                    break

    return cell_results, cell_idx


def summarize(cell_results):
    counts = {TRAVERSABLE: 0, OBSTACLE: 0, UNKNOWN: 0}
    for info in cell_results.values():
        counts[info["label"]] += 1
    total = sum(counts.values())
    print(f"Total cells: {total}")
    for label, n in counts.items():
        pct = (n / total * 100) if total else 0.0
        print(f"  {label:12s}: {n:5d} ({pct:.1f}%)")
    return counts


def resolve_frame(dataset, frame):
    dataset_dir = os.path.join(DATA_ROOT, dataset, "Data_omni", "P0000", "lidar")
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.ply")))
    if not files:
        raise FileNotFoundError(f"No .ply files found under {dataset_dir}")
    if frame is None:
        return files[0]
    matches = [f for f in files if frame in os.path.basename(f)]
    if not matches:
        raise FileNotFoundError(f"No frame matching '{frame}' found in {dataset_dir}")
    return matches[0]


def visualize(points, cell_results, cell_idx):
    colors = np.zeros((len(points), 3), dtype=np.float64)
    for i in range(len(points)):
        key = (int(cell_idx[i, 0]), int(cell_idx[i, 1]))
        colors[i] = LABEL_COLORS[cell_results[key]["label"]]

    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points)
    pcd_original.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3)) * 0.7)

    pcd_labeled = o3d.geometry.PointCloud()
    pcd_labeled.points = o3d.utility.Vector3dVector(points)
    pcd_labeled.colors = o3d.utility.Vector3dVector(colors)

    bbox = pcd_original.get_axis_aligned_bounding_box()
    offset = bbox.get_extent()[0] * 1.2 + 2.0
    pcd_labeled.translate([offset, 0, 0])

    print("\nOpening Interactive 3D Viewer...")
    print("LEFT: Original | RIGHT: Traversable (Green) / Obstacle (Red) / Unknown (Grey)")
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    o3d.visualization.draw_geometries(
        [pcd_original, pcd_labeled, axes],
        window_name="Original (Left) vs Traversability (Right)",
        width=1600, height=800
    )


def main():
    parser = argparse.ArgumentParser(description="Standalone grid-based traversability classifier")
    parser.add_argument("--dataset", type=str, default="Office", help="Dataset folder under data/")
    parser.add_argument("--frame", type=str, default=None, help="Frame filename or substring")
    parser.add_argument("--plane-fit-method", type=str, default="lightweight", choices=["lightweight", "schnabel"],
                         help="Per-cell plane fitting strategy")
    parser.add_argument("--no-viz", action="store_true", help="Skip the interactive viewer, just print stats")
    args = parser.parse_args()

    filepath = resolve_frame(args.dataset, args.frame)
    print(f"Dataset: {args.dataset}  |  Frame: {os.path.basename(filepath)}")

    points = load_points(filepath)
    print(f"Points after voxel downsample: {len(points)}")

    cell_results, cell_idx = classify_frame(points, plane_fit_method=args.plane_fit_method)
    summarize(cell_results)

    if not args.no_viz:
        visualize(points, cell_results, cell_idx)


if __name__ == "__main__":
    main()
