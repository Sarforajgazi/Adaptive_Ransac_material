import os
import sys
import numpy as np
import open3d as o3d
from stable_baselines3 import PPO

# Import our custom environment components
from ransac_env import RansacEnv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))
import schnabel_ransac

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(WORKSPACE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_ransac_final.zip")

def main():
    print("=" * 60)
    print("Visualizing PPO Inference")
    print("=" * 60)

    print("Initializing environment...")
    env = RansacEnv()
    obs, info = env.reset()

    filename = os.path.basename(info['file'])
    print(f"\nProcessing Frame: {filename}")

    # Temporarily using fixed parameters since the Day 2 PPO model (size 4) 
    # cannot read the Day 4 environment (size 21)
    eps = 0.15
    min_supp = 500
    norm_th = 0.85
    print(f"Using Baseline Parameters: Epsilon={eps}m, MinSupport={min_supp}, NormalThresh={norm_th}")

    # Run actual RANSAC on the points
    points = env.current_points
    try:
        shapes, _ = schnabel_ransac.detect(
            points,
            shapes=["plane"],
            relative_epsilon=False,
            epsilon=eps,
            normal_thresh=norm_th,
            min_support=min_supp,
            probability=0.01,
            max_shapes=20,
        )
    except Exception as e:
        print(f"RANSAC crashed with chosen parameters: {e}")
        return

    # Use the env's method to pick the correct ground plane
    from ransac_env import find_ground_plane
    ground_shape, _, _ = find_ground_plane(shapes, points, z_mode="z_down")

    if ground_shape is None:
        print("No ground plane was found!")
        return

    # Color the point cloud
    # Default: Grey background for noise/clutter
    colors = np.ones((len(points), 3)) * 0.5 
    
    # Color ALL detected shapes Blue first
    for shape in shapes:
        mask = shape["inlier_mask"]
        colors[mask] = [0.0, 0.4, 1.0]  # Blue for roofs/walls
        
    # Overwrite the specific GROUND plane to be Bright Green
    ground_mask = ground_shape["inlier_mask"]
    colors[ground_mask] = [0.0, 1.0, 0.0]  # Bright Green
    
    print(f"Total shapes detected: {len(shapes)}")
    print(f"Found ground with {np.sum(ground_mask)} points ({(np.sum(ground_mask)/len(points))*100:.1f}%)")

    # 1. Create Original Point Cloud
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points)
    # The original TartanAir lidar PLY files do not contain RGB data (only XYZ)
    # So we display it as a clean light-grey model
    pcd_original.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3)) * 0.7) 

    # 2. Create Colored Point Cloud (and shift it so they are side-by-side)
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(points)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)
    
    # Translate colored point cloud to the right (X-axis) so they don't overlap
    pcd_colored.translate([50, 0, 0])

    print("\n" + "=" * 60)
    print("Opening Interactive 3D Viewer...")
    print("LEFT: Original Point Cloud | RIGHT: Detected Ground (Green)")
    print("Controls:")
    print(" - Click & Drag to rotate")
    print(" - Scroll to zoom")
    print(" - Press 'P' to save a screenshot of the window")
    print(" - Close the window to exit")
    print("=" * 60)
    
    # Optional: Coordinate frame
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    
    o3d.visualization.draw_geometries([pcd_original, pcd_colored, axes], window_name="Original (Left) vs Colored (Right)", width=1600, height=800)

if __name__ == "__main__":
    main()
