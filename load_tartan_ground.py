import os
import glob
import numpy as np
from plyfile import PlyData

def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    lidar_dir = os.path.join(workspace_dir, "data", "Office", "Data_omni", "P0000", "lidar")
    
    print(f"Checking LiDAR data directory: {lidar_dir}")
    if not os.path.exists(lidar_dir):
        print(f"[ERROR] LiDAR directory does not exist yet. Please run download_tartan_ground.py first.")
        return
        
    # List files in the lidar directory
    files = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))
    print(f"Total PLY files found in LiDAR folder: {len(files)}")
    
    if len(files) == 0:
        print("[ERROR] No PLY files found in the LiDAR folder.")
        return
        
    # Show the first 5 files
    print("\nFirst 5 files:")
    for f in files[:5]:
        print(f"  - {os.path.basename(f)} (Size: {os.path.getsize(f)} bytes)")
        
    # Inspect and load the first file
    first_file = files[0]
    print(f"\nLoading point cloud from file: {os.path.basename(first_file)}")
    try:
        plydata = PlyData.read(first_file)
        vertex = plydata['vertex']
        
        # Stack coordinates
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        points = np.stack([x, y, z], axis=-1)
        
        print("[SUCCESS] Point cloud loaded successfully!")
        print(f"  Shape of point cloud array: {points.shape} (N points, 3 coords)")
        print(f"  Data Type: {points.dtype}")
        print(f"  Sample Points (first 5):\n{points[:5]}")
        print(f"  Bounding Box:")
        print(f"    Min (x, y, z): {points.min(axis=0)}")
        print(f"    Max (x, y, z): {points.max(axis=0)}")
    except Exception as e:
        print(f"[ERROR] Failed to parse point cloud file: {e}")

if __name__ == "__main__":
    main()
