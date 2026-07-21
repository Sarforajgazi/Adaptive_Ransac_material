import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

def find_ground_color():
    env_name = "Gascola"
    frame_idx = 0
    data_dir = "data"
    
    # 1. Load LiDAR
    lidar_file = os.path.join(data_dir, env_name, "Data_omni", "P0000", "lidar", f"{frame_idx:06d}_lcam_front_lidar.ply")
    frame_pcd = o3d.io.read_point_cloud(lidar_file)
    pts_local = np.asarray(frame_pcd.points)
    
    # Extract lowest Z points (assuming Z is vertical in local frame, House ground is around Z = -1.8)
    z_coords = pts_local[:, 2]
    # In AirSim NED, Z is down, so ground has POSITIVE Z? 
    # Wait, in the debug script, min Z was -1.8, max Z was 0.6.
    # Ground is usually flat. Let's find a large horizontal plane.
    # We can just use the colors of the lowest points in the transformed frame.
    
    # 2. Load Pose
    pose_file = os.path.join(data_dir, env_name, "Data_omni", "P0000", "pose_lcam_front.txt")
    poses = np.loadtxt(pose_file)
    pose = poses[frame_idx]
    t = pose[:3]
    q = pose[3:] # qx, qy, qz, qw
    r = R.from_quat(q).as_matrix()
    
    # 3. Transform
    pts_local_transformed = pts_local[:, [1, 0, 2]] * np.array([1, 1, -1])
    pts_global = (r @ pts_local_transformed.T).T + t
    
    # Z in global frame is Up or Down. Let's find the flat plane.
    # Actually, we can just find points with Z close to min(global Z).
    
    # 4. Load Environment
    pcd_path = os.path.join(data_dir, env_name, f"{env_name}_sem.pcd")
    global_pcd = o3d.io.read_point_cloud(pcd_path)
    global_points = np.asarray(global_pcd.points)
    global_colors = np.asarray(global_pcd.colors)
    
    # Find Z bounds of the global map
    print(f"Global min Z: {global_points[:, 2].min()}, max Z: {global_points[:, 2].max()}")
    
    kdtree = cKDTree(global_points)
    distances, indices = kdtree.query(pts_global, k=1, distance_upper_bound=0.2)
    valid = distances != np.inf
    
    nn_colors = global_colors[indices[valid]]
    
    # Convert colors to integers for counting (R,G,B in 0-255)
    colors_int = (nn_colors * 255).astype(int)
    unique_colors, counts = np.unique(colors_int, axis=0, return_counts=True)
    
    # Print the top 5 most frequent colors
    sort_idx = np.argsort(counts)[::-1]
    print("\nTop 10 Colors mapped to LiDAR points:")
    for i in range(min(10, len(unique_colors))):
        c = unique_colors[sort_idx[i]]
        count = counts[sort_idx[i]]
        print(f"Color {c}: {count} points ({count/len(nn_colors)*100:.1f}%)")
        
    print("\nWe know ground makes up ~20-50% of points. We can visually inspect which color is the floor.")

if __name__ == '__main__':
    find_ground_color()
