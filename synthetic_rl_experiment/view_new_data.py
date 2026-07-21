"""
view_new_data.py

Simple interactive viewer for the new_data/ point clouds -- real, dense
(3-3.6M points), true-RGB, semantically-labeled tiles (see
SESSION_PROGRESS_LOG.md for the field inspection: x/y/z, red/green/blue,
semantic, instance, visible, confidence). Just shows true color; no model
inference here yet.

Voxel-downsamples by default since these are 3M+ points/tile -- too dense
to rotate smoothly otherwise. Pass --voxel 0 to view at full resolution.

Usage:
    python view_new_data.py --file 0                 # first tile, default voxel
    python view_new_data.py --file 1 --voxel 0.05
    python view_new_data.py --file 0 --voxel 0
"""
import os
import sys
import glob
import argparse
import open3d as o3d

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=int, default=0, help="index into sorted new_data/*.ply")
    parser.add_argument("--voxel", type=float, default=0.3, help="voxel size for downsampling (0 = full resolution)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.ply")))
    if not files:
        print(f"No .ply files found in {DATA_DIR}")
        sys.exit(1)
    if args.file >= len(files):
        print(f"--file {args.file} out of range (only {len(files)} files)")
        sys.exit(1)

    path = files[args.file]
    print(f"Loading {path} ...")
    pcd = o3d.io.read_point_cloud(path)
    print(f"  {len(pcd.points)} points, has_colors={pcd.has_colors()}")

    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f"  downsampled to {len(pcd.points)} points (voxel={args.voxel})")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"{os.path.basename(path)} (true RGB, voxel={args.voxel})",
        width=1600, height=900,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
