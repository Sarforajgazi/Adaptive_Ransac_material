import numpy as np
import time
import sys
import copy

try:
    import open3d as o3d
except ImportError:
    print("ERROR: Open3D is not installed. Please run: pip install open3d")
    sys.exit(1)

try:
    import schnabel_ransac
except ImportError as e:
    print(f"ERROR: Could not import schnabel_ransac: {e}")
    sys.exit(1)

def main():
    # 1. Load the Point Cloud
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f"Loading custom point cloud: {filename} ...")
        pcd = o3d.io.read_point_cloud(filename)
    else:
        print("Downloading/Loading Default Open3D Dataset (~2 MB)...")
        dataset = o3d.data.PCDPointCloud()
        pcd = o3d.io.read_point_cloud(dataset.path)
        
    if len(pcd.points) == 0:
        print("ERROR: Failed to load point cloud or file is empty.")
        sys.exit(1)
        
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    points = np.asarray(pcd.points).astype(np.float32)
    print(f"Loaded point cloud with {len(points)} points.")
    
    # 2. Run Cython RANSAC (Planes Only)
    print("\nRunning C++ Cython Wrapper to find all planes...")
    t0 = time.perf_counter()
    shapes, n_remaining = schnabel_ransac.detect(
        points,
        shapes=["plane"],
        epsilon=0.02,
        normal_thresh=0.9,
        min_support=1500,
        probability=0.01
    )
    elapsed = time.perf_counter() - t0
    print(f"Found {len(shapes)} planes in {elapsed:.3f} seconds.")
    
    if len(shapes) == 0:
        print("No planes found! Cannot segment ground.")
        sys.exit(0)
        
    # 3. Identify the Ground Plane
    # The ground is typically the plane with the lowest average elevation.
    # In many 3D scans, the vertical axis is Z (index 2) or Y (index 1).
    # We will assume Z is up for this example.
    
    ground_plane = None
    min_height = float('inf')
    
    for shape in shapes:
        mask = shape["inlier_mask"]
        plane_points = points[mask]
        
        # Calculate the average Z coordinate of this plane
        avg_z = np.mean(plane_points[:, 2])
        
        if avg_z < min_height:
            min_height = avg_z
            ground_plane = shape

    print(f"\n✅ Ground Plane Successfully Segmented!")
    print(f"   -> It contains {ground_plane['n_points']} points at an average height of Z={min_height:.3f}")
    
    # 4. Visualization
    # Create a completely red point cloud (Obstacles)
    obstacle_colors = np.ones((len(points), 3)) * np.array([1.0, 0.0, 0.0])
    
    # Paint ONLY the ground plane green
    mask = ground_plane["inlier_mask"]
    obstacle_colors[mask] = [0.0, 1.0, 0.0] # Green
    
    # Create the segmented point cloud
    pcd_segmented = copy.deepcopy(pcd)
    pcd_segmented.colors = o3d.utility.Vector3dVector(obstacle_colors)
    
    # Place them side by side
    bbox = pcd.get_axis_aligned_bounding_box()
    width = bbox.get_extent()[0]
    pcd_segmented.translate([width * 1.1, 0, 0])
    
    print("\n[Visualizing] - Left: Original | Right: Green Ground & Red Obstacles")
    o3d.visualization.draw_geometries([pcd, pcd_segmented], window_name="Ground Segmentation")

if __name__ == "__main__":
    main()
