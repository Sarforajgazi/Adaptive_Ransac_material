import os
import glob
import numpy as np
from plyfile import PlyData
import pyransac3d as pyrsc

def save_ply_ascii(filename, points):
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    lidar_dir = os.path.join(workspace_dir, "data", "Office", "Data_omni", "P0000", "lidar")
    segmented_dir = os.path.join(workspace_dir, "data", "Office", "Data_omni", "P0000", "segmented")
    os.makedirs(segmented_dir, exist_ok=True)
    
    print(f"Reading LiDAR files from: {lidar_dir}")
    files = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))
    if len(files) == 0:
        print("[ERROR] No PLY files found in LiDAR folder.")
        return
        
    first_file = files[0]
    print(f"Loading point cloud: {os.path.basename(first_file)}")
    
    # Load point cloud
    plydata = PlyData.read(first_file)
    vertex = plydata['vertex']
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    points = np.stack([x, y, z], axis=-1)
    
    print(f"Total input points: {len(points)}")
    
    # Perform RANSAC plane fitting
    print("Fitting ground plane using RANSAC...")
    plane = pyrsc.Plane()
    # thresh=0.20 meters, maxIteration=1000
    equation, inlier_indices = plane.fit(points, thresh=0.20, maxIteration=1000)
    
    # Separate ground (inliers) and obstacles (outliers)
    ground_points = points[inlier_indices]
    outlier_mask = np.ones(len(points), dtype=bool)
    outlier_mask[inlier_indices] = False
    obstacle_points = points[outlier_mask]
    
    # Print results
    print("[SUCCESS] Segmentation completed!")
    print(f"  Ground plane equation: {equation[0]:.4f}x + {equation[1]:.4f}y + {equation[2]:.4f}z + {equation[3]:.4f} = 0")
    print(f"  Ground points: {len(ground_points)} ({len(ground_points)/len(points)*100:.2f}%)")
    print(f"  Obstacle points: {len(obstacle_points)} ({len(obstacle_points)/len(points)*100:.2f}%)")
    
    # Save segmented point clouds
    ground_output = os.path.join(segmented_dir, "ground.ply")
    obstacle_output = os.path.join(segmented_dir, "obstacles.ply")
    
    print(f"\nSaving ground points to: {os.path.basename(ground_output)}")
    save_ply_ascii(ground_output, ground_points)
    
    print(f"Saving obstacle points to: {os.path.basename(obstacle_output)}")
    save_ply_ascii(obstacle_output, obstacle_points)
    
    print("[SUCCESS] Segmented PLY files saved successfully!")

if __name__ == "__main__":
    main()
