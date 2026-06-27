import open3d as o3d
import os
import sys

ground_file = os.path.join("data", "Office", "Data_omni", "P0000", "segmented", "ground.ply")
obstacles_file = os.path.join("data", "Office", "Data_omni", "P0000", "segmented", "obstacles.ply")

if not os.path.exists(ground_file) or not os.path.exists(obstacles_file):
    print("Could not find the segmented files. Make sure ground.ply and obstacles.ply exist in this directory.")
    sys.exit(1)

print("Loading point clouds...")
ground_pcd = o3d.io.read_point_cloud(ground_file)
obstacles_pcd = o3d.io.read_point_cloud(obstacles_file)

print(f"Ground points: {len(ground_pcd.points)}")
print(f"Obstacle points: {len(obstacles_pcd.points)}")

# Color ground green
ground_pcd.paint_uniform_color([0.0, 1.0, 0.0])

# Color obstacles red
obstacles_pcd.paint_uniform_color([1.0, 0.0, 0.0])

print("Opening visualizer window...")
print("The ground points are in GREEN, and obstacles are in RED.")
print("You can rotate (Left Mouse Button), pan (Shift + Left Mouse Button) and zoom (Scroll).")
print("Close the window to exit the script.")

o3d.visualization.draw_geometries([ground_pcd, obstacles_pcd], window_name="Ground (Green) vs Obstacles (Red)")
