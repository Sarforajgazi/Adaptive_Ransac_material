import open3d as o3d
import shutil
import os

print("Downloading Open3D Office Dataset (Contains floor/ground)...")
# This downloads a collection of point clouds of an office room
dataset = o3d.data.OfficePointClouds()

# Grab the first point cloud in the dataset
source_ply = dataset.paths[0]
target_ply = "office_with_ground.ply"

print(f"Copying to {target_ply}...")
shutil.copy(source_ply, target_ply)

print("\nDownload complete! ✅")
print(f"You can now run: python ground_segmentation.py {target_ply}")
