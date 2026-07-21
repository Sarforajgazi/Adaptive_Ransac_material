import os
import itertools
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

def get_permutations_and_signs():
    # 3! permutations
    perms = list(itertools.permutations([0, 1, 2]))
    # 2^3 sign combinations
    signs = list(itertools.product([1, -1], repeat=3))
    
    transforms = []
    for p in perms:
        for s in signs:
            transforms.append((p, s))
    return transforms

def run():
    env_name = "House"
    frame_idx = 0
    data_dir = "data"
    
    # Load LiDAR
    lidar_file = os.path.join(data_dir, env_name, "Data_omni", "P0000", "lidar", f"{frame_idx:06d}_lcam_front_lidar.ply")
    frame_pcd = o3d.io.read_point_cloud(lidar_file)
    pts_local = np.asarray(frame_pcd.points)
    
    # Load Pose
    pose_file = os.path.join(data_dir, env_name, "Data_omni", "P0000", "pose_lcam_front.txt")
    poses = np.loadtxt(pose_file)
    pose = poses[frame_idx]
    t = pose[:3]
    q = pose[3:] # qx, qy, qz, qw
    r = R.from_quat(q).as_matrix()
    
    # Load Environment
    pcd_path = os.path.join(data_dir, env_name, f"{env_name}_sem.pcd")
    global_pcd = o3d.io.read_point_cloud(pcd_path)
    global_points = np.asarray(global_pcd.points)
    
    # KDTree
    print("Building KDTree...")
    kdtree = cKDTree(global_points)
    
    transforms = get_permutations_and_signs()
    
    best_rate = 0
    best_transform = None
    
    print("Testing 48 combinations (Local Coordinate Systems)...")
    for p, s in transforms:
        # Transform local points
        pts_test = pts_local.copy()
        pts_test = pts_test[:, p]
        pts_test = pts_test * np.array(s)
        
        # Apply pose
        pts_global = (r @ pts_test.T).T + t
        
        distances, _ = kdtree.query(pts_global, k=1, distance_upper_bound=0.2)
        match_rate = np.mean(distances != np.inf)
        
        if match_rate > best_rate:
            best_rate = match_rate
            best_transform = (p, s)
            print(f"New best: {p}, {s} -> {match_rate*100:.2f}%")
            if match_rate > 0.90:
                print("Found match > 90%!")
                break
                
    if best_rate < 0.9:
        print("Testing 48 combinations (Inverse Pose + Local Coordinate Systems)...")
        # Maybe the pose is world-to-sensor? (Usually not, but we check)
        for p, s in transforms:
            pts_test = pts_local.copy()
            pts_test = pts_test[:, p]
            pts_test = pts_test * np.array(s)
            
            # pts_local = R * global + t => global = R_inv * (pts_local - t)
            pts_global = (r.T @ (pts_test - t).T).T
            
            distances, _ = kdtree.query(pts_global, k=1, distance_upper_bound=0.2)
            match_rate = np.mean(distances != np.inf)
            
            if match_rate > best_rate:
                best_rate = match_rate
                best_transform = ("INV", p, s)
                print(f"New best (INV): {p}, {s} -> {match_rate*100:.2f}%")
                if match_rate > 0.90:
                    break

if __name__ == '__main__':
    run()
