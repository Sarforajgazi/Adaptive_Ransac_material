import os
import time
import numpy as np
from plyfile import PlyData

from features.scene_features import compute_scene_features

def load_ply_xyz(filepath):
    ply = PlyData.read(filepath)
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    return pts

def main():
    workspace = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(workspace, "data", "Office", "Data_omni", "P0000", "lidar", "000000_lcam_front_lidar.ply")
    
    if not os.path.exists(test_file):
        print("Test file not found.")
        return

    print("Loading points...")
    pts = load_ply_xyz(test_file)
    print(f"Loaded {len(pts)} points.")

    print("Computing 21 Scene Features...")
    start = time.time()
    feats = compute_scene_features(pts)
    elapsed = time.time() - start

    print("-" * 60)
    print(f"Time Taken: {elapsed:.4f} seconds")
    print(f"Feature Vector Shape: {feats.shape}")
    print("Features array:")
    
    names = [
        "bbox_dx", "bbox_dy", "bbox_dz", "bbox_volume", "point_density",
        "z_mean", "z_std", "z_min", "z_max",
        "scan_range_mean", "scan_range_std",
        "eig_0", "eig_1", "eig_2",
        "normal_x_std", "normal_y_std", "normal_z_std",
        "normal_consistency", "z_density_ground", "ground_slope_estimate", "mean_knn_dist"
    ]
    
    for name, val in zip(names, feats):
        print(f"  {name:25s}: {val:.4f}")
        
    print("-" * 60)
    
    if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
        print("ERROR: NaN or Inf found in features!")
    else:
        print("Success! Features are clean.")

if __name__ == "__main__":
    main()
