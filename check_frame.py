"""
check_frame.py  <frame_index>

Opens an Open3D window showing the original LiDAR frame side-by-side with
its segmented version (green = ground, grey = obstacles).

Usage:
    python check_frame.py 0       # check frame 000000
    python check_frame.py 100     # check frame 000100
    python check_frame.py 350     # check frame 000350
"""

import sys
import os
import glob
import open3d as o3d

WORKSPACE   = os.path.dirname(os.path.abspath(__file__))
LIDAR_DIR   = os.path.join(WORKSPACE, "data", "Office", "Data_omni", "P0000", "lidar")
SEG_DIR     = os.path.join(WORKSPACE, "data", "Office", "Data_omni", "P0000", "schnabel_segmented")

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_frame.py <frame_index>")
        print("Example: python check_frame.py 0")
        sys.exit(1)

    idx = int(sys.argv[1])

    orig_files = sorted(glob.glob(os.path.join(LIDAR_DIR, "*.ply")))
    seg_files  = sorted(glob.glob(os.path.join(SEG_DIR,   "*_segmented.ply")))

    if idx >= len(orig_files):
        print(f"ERROR: Frame index {idx} out of range (0–{len(orig_files)-1}).")
        sys.exit(1)

    orig_path = orig_files[idx]
    frame_name = os.path.basename(orig_path).replace(".ply", "")
    seg_path   = os.path.join(SEG_DIR, frame_name + "_segmented.ply")

    print(f"Frame {idx:04d}: {os.path.basename(orig_path)}")
    print(f"Original : {orig_path}")
    print(f"Segmented: {seg_path}")

    # Load original
    pcd_orig = o3d.io.read_point_cloud(orig_path)
    pcd_orig.paint_uniform_color([0.6, 0.6, 0.6])   # grey

    if not os.path.exists(seg_path):
        print("\nWARNING: Segmented file not found. Showing original only.")
        print("Run batch_segment_lidar.py first.")
        o3d.visualization.draw_geometries(
            [pcd_orig],
            window_name=f"Frame {idx} — Original only"
        )
        return

    # Load segmented (already has green/grey colours from batch script)
    pcd_seg = o3d.io.read_point_cloud(seg_path)

    # Place segmented copy to the right of the original
    bbox  = pcd_orig.get_axis_aligned_bounding_box()
    width = bbox.get_extent()[0]
    pcd_seg.translate([width * 1.15, 0, 0])

    print("\nLeft  = Original (grey)")
    print("Right = Segmented (green=ground, grey=obstacles)")
    print("Controls: Left-drag=rotate, Shift+drag=pan, Scroll=zoom, Q=quit")

    o3d.visualization.draw_geometries(
        [pcd_orig, pcd_seg],
        window_name=f"Frame {idx:04d} | Left: Original | Right: Segmented"
    )

if __name__ == "__main__":
    main()
