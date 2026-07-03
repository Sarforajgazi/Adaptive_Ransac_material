"""
compare_on_complete_cloud.py

Runs BOTH ground-detection approaches on the "complete" multi-view
reconstructed point clouds in schnabel_cython/tartanair_data/<Env>/, and
compares their outputs directly against each other.

Note: <Env>_rgb_segmented.ply also exists alongside these point clouds,
but it was itself produced by an earlier Schnabel RANSAC run, not an
independent/official ground truth -- so it is NOT used here as a scoring
reference. Comparing against it would partly just measure "does Schnabel
agree with Schnabel," not real accuracy. This script only compares the
two live methods to each other.

  1. Schnabel RANSAC (schnabel_cython, the same engine ransac_env.py uses)
     -- single best plane, picked the same way find_ground_plane() does.
  2. The standalone grid-based traversability classifier (traversability.py)
     -- unchanged, imported read-only so it stays comparable later.

Usage:
    python compare_on_complete_cloud.py --env Downtown
    python compare_on_complete_cloud.py --env OldScandinavia --cell_size 2.5
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))
import schnabel_ransac

import traversability as trav

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
COMPLETE_DIR = os.path.join(WORKSPACE, "schnabel_cython", "tartanair_data")


def load_complete_cloud(env, voxel_size=0.05):
    orig_path = os.path.join(COMPLETE_DIR, env, f"{env}_rgb_original.ply")
    orig = PlyData.read(orig_path)
    ov = orig["vertex"]
    orig_pts = np.stack([ov["x"], ov["y"], ov["z"]], axis=-1).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(orig_pts)
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    down_pts = np.asarray(down.points, dtype=np.float32)
    return down_pts


def run_schnabel(points, epsilon=0.15, min_support=500, normal_thresh=0.85, z_mode="z_up"):
    """
    z_mode default is "z_up" here, NOT "z_down" like ransac_env.py's per-frame
    sensor data. These complete/reconstructed clouds are in world coordinates
    (height increases upward) rather than the per-frame NED sensor convention
    -- confirmed by simple geometric reasoning (building rooftops sit at a
    higher Z than street level in this data), independent of any label file.
    """
    shapes, _ = schnabel_ransac.detect(
        points,
        shapes=["plane"],
        relative_epsilon=False,
        epsilon=epsilon,
        normal_thresh=normal_thresh,
        min_support=min_support,
        probability=0.001,
        normal_knn=20,
        max_shapes=20,
    )

    candidates = []
    for shape in shapes:
        mask = shape["inlier_mask"]
        plane_pts = points[mask]
        if len(plane_pts) < 10:
            continue
        cov = np.cov(plane_pts.T)
        evals, evecs = np.linalg.eig(cov)
        normal = evecs[:, np.argmin(evals)]
        z_align = abs(float(normal[2]))
        avg_z = float(np.mean(plane_pts[:, 2]))
        candidates.append({"mask": mask, "avg_z": avg_z, "z_align": z_align, "n_points": shape["n_points"]})

    horizontal = [c for c in candidates if c["z_align"] >= 0.80]
    if not horizontal:
        horizontal = sorted(candidates, key=lambda c: c["n_points"], reverse=True)[:1]
    if not horizontal:
        return np.zeros(len(points), dtype=bool)

    reverse = (z_mode == "z_down")
    horizontal.sort(key=lambda c: c["avg_z"], reverse=reverse)
    return horizontal[0]["mask"]


def run_traversability(points, cell_size, lowest_cluster_band=trav.LOWEST_CLUSTER_BAND, plane_fit_method="lightweight"):
    cell_results, cell_idx = trav.classify_frame(points, cell_size=cell_size, lowest_cluster_band=lowest_cluster_band,
                                                  plane_fit_method=plane_fit_method)
    pred_ground = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        key = (int(cell_idx[i, 0]), int(cell_idx[i, 1]))
        pred_ground[i] = cell_results[key]["label"] == trav.TRAVERSABLE
    return pred_ground, cell_results, cell_idx


def visualize_comparison(points, schnabel_ground, cell_results, cell_idx):
    """
    Three panels side by side:
      LEFT:   original point cloud (grey)
      MIDDLE: Schnabel result       (green = ground, grey = rest)
      RIGHT:  grid-based result     (green = traversable, red = obstacle, grey = unknown)
    """
    n = len(points)

    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points)
    pcd_original.colors = o3d.utility.Vector3dVector(np.ones((n, 3)) * 0.7)

    schnabel_colors = np.ones((n, 3)) * 0.5
    schnabel_colors[schnabel_ground] = [0.0, 1.0, 0.0]
    pcd_schnabel = o3d.geometry.PointCloud()
    pcd_schnabel.points = o3d.utility.Vector3dVector(points)
    pcd_schnabel.colors = o3d.utility.Vector3dVector(schnabel_colors)

    grid_colors = np.zeros((n, 3))
    for i in range(n):
        key = (int(cell_idx[i, 0]), int(cell_idx[i, 1]))
        grid_colors[i] = trav.LABEL_COLORS[cell_results[key]["label"]]
    pcd_grid = o3d.geometry.PointCloud()
    pcd_grid.points = o3d.utility.Vector3dVector(points)
    pcd_grid.colors = o3d.utility.Vector3dVector(grid_colors)

    bbox = pcd_original.get_axis_aligned_bounding_box()
    offset = bbox.get_extent()[0] * 1.15 + 5.0
    pcd_schnabel.translate([offset, 0, 0])
    pcd_grid.translate([offset * 2, 0, 0])

    print("\nOpening Interactive 3D Viewer...")
    print("LEFT: Original | MIDDLE: Schnabel (Green=ground) | RIGHT: Grid-based (Green=traversable, Red=obstacle, Grey=unknown)")
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    o3d.visualization.draw_geometries(
        [pcd_original, pcd_schnabel, pcd_grid, axes],
        window_name="Original | Schnabel | Grid-based",
        width=1900, height=800
    )


def compare_methods(pred_a, pred_b, name_a, name_b):
    """
    Direct agreement between two methods' predictions. Neither is treated
    as ground truth -- this just describes where they agree and disagree.
    """
    both = np.sum(pred_a & pred_b)
    only_a = np.sum(pred_a & ~pred_b)
    only_b = np.sum(~pred_a & pred_b)
    neither = np.sum(~pred_a & ~pred_b)
    union = both + only_a + only_b
    agreement_iou = both / union if union else 0.0

    print(f"\n  {name_a}: {pred_a.sum()}/{len(pred_a)} points called ground ({pred_a.mean()*100:.1f}%)")
    print(f"  {name_b}: {pred_b.sum()}/{len(pred_b)} points called ground ({pred_b.mean()*100:.1f}%)")
    print(f"  Agree on ground:      {both}")
    print(f"  Only {name_a}:  {only_a}")
    print(f"  Only {name_b}:  {only_b}")
    print(f"  Agree on not-ground:  {neither}")
    print(f"  Agreement IoU:        {agreement_iou:.3f}")
    return {"both": int(both), "only_a": int(only_a), "only_b": int(only_b),
            "neither": int(neither), "agreement_iou": agreement_iou}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Downtown")
    parser.add_argument("--cell_size", type=float, default=2.5, help="Grid cell size for the traversability classifier on this sparser complete cloud")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--min_support", type=int, default=500)
    parser.add_argument("--no-viz", action="store_true", help="Skip the interactive 3-panel viewer, just print stats")
    args = parser.parse_args()

    print(f"Loading complete point cloud for {args.env}...")
    points = load_complete_cloud(args.env, voxel_size=args.voxel_size)
    print(f"  {len(points)} points after downsample")

    print(f"\nRunning Schnabel RANSAC (epsilon={args.epsilon}, min_support={args.min_support})...")
    schnabel_ground = run_schnabel(points, epsilon=args.epsilon, min_support=args.min_support)

    print(f"Running grid-based traversability classifier (cell_size={args.cell_size}m)...")
    trav_ground, cell_results, cell_idx = run_traversability(points, cell_size=args.cell_size)
    trav.summarize(cell_results)

    print("\n=== Direct comparison (neither side treated as ground truth) ===")
    compare_methods(schnabel_ground, trav_ground, "Schnabel", "Grid-based")

    if not args.no_viz:
        visualize_comparison(points, schnabel_ground, cell_results, cell_idx)

    return points, schnabel_ground, trav_ground


if __name__ == "__main__":
    main()
