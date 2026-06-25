import numpy as np
import time
import sys
import copy

try:
    import open3d as o3d
except ImportError:
    print("ERROR: Open3D is not installed.")
    print("Please run:  pip install open3d")
    sys.exit(1)

try:
    import schnabel_ransac
except ImportError as e:
    print(f"ERROR: Could not import schnabel_ransac: {e}")
    print("Did you compile it with: python setup.py build_ext --inplace ?")
    sys.exit(1)

def main():
    if len(sys.argv) > 1:
        # Load the user's custom point cloud file
        filename = sys.argv[1]
        print(f"1. Loading Custom Point Cloud: {filename} ...")
        pcd = o3d.io.read_point_cloud(filename)
        if len(pcd.points) == 0:
            print("ERROR: Failed to load point cloud or file is empty.")
            sys.exit(1)
    else:
        # Use the built-in dataset as a fallback
        print("1. Downloading/Loading Default Open3D Dataset (~2 MB)...")
        print("   (Tip: You can pass your own file! Try: python real_data_demo.py my_cloud.ply)")
        dataset = o3d.data.PCDPointCloud()
        pcd = o3d.io.read_point_cloud(dataset.path)
    
    # Downsample slightly to clean up noisy scanner data and speed things up
    pcd = pcd.voxel_down_sample(voxel_size=0.02) # 2cm voxels
    points = np.asarray(pcd.points).astype(np.float32)
    
    print(f"   -> Loaded point cloud with {len(points)} points.")
    
    print("\n2. Running C++ Cython Wrapper (Planes Only)...")
    t0 = time.perf_counter()
    
    # Run the detector, restricted strictly to "plane" shapes
    shapes, n_remaining = schnabel_ransac.detect(
        points,
        shapes=["plane"],       # ONLY look for planes
        epsilon=0.02,           # 2cm distance tolerance for inliers
        normal_thresh=0.9,      # Normal angular tolerance
        min_support=1500,       # A plane must contain at least 1500 points
        probability=0.01        # 99% confidence stopping criteria
    )
    
    elapsed = time.perf_counter() - t0
    print(f"   -> Detection finished in {elapsed:.3f} seconds!")
    print(f"   -> Found {len(shapes)} distinct planes.")
    
    print("\n3. Opening 3D Visualization...")
    # Paint the whole point cloud light grey (unassigned points)
    colors = np.ones((len(points), 3)) * 0.7
    
    # Define some bright colors for the planes we found
    palette = [
        [1.0, 0.0, 0.0], # Red
        [0.0, 1.0, 0.0], # Green
        [0.0, 0.0, 1.0], # Blue
        [1.0, 1.0, 0.0], # Yellow
        [1.0, 0.0, 1.0], # Magenta
        [0.0, 1.0, 1.0], # Cyan
        [1.0, 0.5, 0.0], # Orange
    ]
    
    # Color each detected plane with a distinct color
    for i, shape in enumerate(shapes):
        mask = shape["inlier_mask"]
        c = palette[i % len(palette)]
        colors[mask] = c
        print(f"   - Plane {i}: {shape['n_points']} points")
        
    # Create a copy of the point cloud for the segmented version
    pcd_segmented = copy.deepcopy(pcd)
    pcd_segmented.colors = o3d.utility.Vector3dVector(colors)
    
    # Translate the segmented point cloud along the X-axis so they are side-by-side
    bbox = pcd.get_axis_aligned_bounding_box()
    width = bbox.get_extent()[0]
    pcd_segmented.translate([width * 1.1, 0, 0])
    
    # Show the interactive 3D window with BOTH clouds
    print("\n[Visualizing Side-by-Side] - Left: Original | Right: Segmented Planes")
    o3d.visualization.draw_geometries([pcd, pcd_segmented], window_name="Side-by-Side Comparison")

if __name__ == "__main__":
    main()
