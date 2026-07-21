import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualize(env_name="Gascola", frame_idx=0):
    data_dir = "data"
    
    # 1. Load LiDAR frame
    lidar_file = os.path.join(data_dir, env_name, "Data_omni", "P0000", "lidar", f"{frame_idx:06d}_lcam_front_lidar.ply")
    if not os.path.exists(lidar_file):
        print(f"Error: {lidar_file} not found")
        return
        
    frame_pcd = o3d.io.read_point_cloud(lidar_file)
    pts_local = np.asarray(frame_pcd.points)
    print(f"LiDAR points: {len(pts_local)}")
    
    # 2. Load Mask
    mask_file = lidar_file.replace(".ply", "_gt_mask.npy")
    if not os.path.exists(mask_file):
        print(f"Error: {mask_file} not found")
        return
        
    mask = np.load(mask_file)
    print(f"Ground mask fraction: {np.mean(mask)*100:.2f}%")
    
    # 3. Visualize
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot non-ground points in grey
    non_ground = pts_local[~mask]
    ax.scatter(non_ground[:, 0], non_ground[:, 1], non_ground[:, 2], c='grey', s=0.5, alpha=0.5, label='Non-ground')
    
    # Plot ground points in green
    ground = pts_local[mask]
    ax.scatter(ground[:, 0], ground[:, 1], ground[:, 2], c='green', s=1.0, alpha=0.8, label='Ground')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set view angle (often Z is up or down in local frame)
    # Since X=front, Y=right, Z=down in standard NED, let's look from top-down
    ax.view_init(elev=20, azim=45)
    
    plt.title(f"{env_name} Frame {frame_idx:06d} - Ground Truth Mask\nGround Fraction: {np.mean(mask)*100:.1f}%")
    plt.legend()
    
    out_path = fr"C:\Users\sarfo\.gemini\antigravity-ide\brain\a42672da-8d35-484c-9394-95a425a28c16\house_gt_mask_viz_{frame_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    env_name = "Gascola"
    frames_to_test = [441, 442]
    for frame in frames_to_test:
        visualize(env_name, frame)
