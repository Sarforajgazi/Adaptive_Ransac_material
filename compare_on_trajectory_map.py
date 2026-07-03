"""
compare_on_trajectory_map.py

Same Schnabel-vs-grid-based comparison as compare_on_complete_cloud.py, but
run on a map BUILT from the per-frame LiDAR scans in data/<Env>/.../lidar/
(accumulate_trajectory.py) instead of the pre-made "complete" reconstruction
in schnabel_cython/tartanair_data/.

Coordinate convention note: empirically, this accumulated map behaves as
Z-up (ground = lowest Z among horizontal candidates), NOT the Z-down/NED
convention the per-frame sensor data uses before transformation. Verified
directly via schnabel_ransac.detect() -- the single largest, cleanest
horizontal plane (matching the street) sits at the *lowest* Z among
horizontal candidates, not the highest. Likely the pose-transform (world
rotation applied to a NED-local frame) flips the net Z sense; not something
to assume next time either -- check empirically per data source.

Usage:
    python compare_on_trajectory_map.py --env Downtown
    python compare_on_trajectory_map.py --env OldScandinavia --stride 5 --epsilon 0.5
"""

import argparse

from accumulate_trajectory import accumulate_trajectory
from compare_on_complete_cloud import (
    run_schnabel, run_traversability, compare_methods, visualize_comparison,
)
import traversability as trav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Downtown")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--per_frame_voxel", type=float, default=0.05)
    parser.add_argument("--final_voxel", type=float, default=0.4)
    parser.add_argument("--cell_size", type=float, default=2.5)
    parser.add_argument("--lowest_cluster_band", type=float, default=0.15,
                         help="traversability.py's normal default. Previously had to be loosened to 0.5m "
                              "to compensate for registration noise in the naive pose-only merge -- with "
                              "ICP refinement now in accumulate_trajectory.py, test whether the default "
                              "works again.")
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--min_support", type=int, default=200)
    parser.add_argument("--icp_max_corr_dist", type=float, default=0.5)
    parser.add_argument("--no-icp", action="store_true", help="Skip ICP refinement, use raw pose transform only (old behavior)")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    print(f"Accumulating trajectory map for {args.env} (stride={args.stride}, icp={not args.no_icp})...")
    points = accumulate_trajectory(args.env, stride=args.stride,
                                    per_frame_voxel=args.per_frame_voxel,
                                    final_voxel=args.final_voxel,
                                    use_icp=not args.no_icp,
                                    icp_max_corr_dist=args.icp_max_corr_dist)

    print(f"\nRunning Schnabel RANSAC (epsilon={args.epsilon}, min_support={args.min_support}, z_mode=z_up)...")
    schnabel_ground = run_schnabel(points, epsilon=args.epsilon, min_support=args.min_support, z_mode="z_up")

    print(f"Running grid-based traversability classifier (cell_size={args.cell_size}m, lowest_cluster_band={args.lowest_cluster_band}m)...")
    trav_ground, cell_results, cell_idx = run_traversability(points, cell_size=args.cell_size, lowest_cluster_band=args.lowest_cluster_band)
    trav.summarize(cell_results)

    print("\n=== Direct comparison (neither side treated as ground truth) ===")
    compare_methods(schnabel_ground, trav_ground, "Schnabel", "Grid-based")

    if not args.no_viz:
        visualize_comparison(points, schnabel_ground, cell_results, cell_idx)

    return points, schnabel_ground, trav_ground


if __name__ == "__main__":
    main()
