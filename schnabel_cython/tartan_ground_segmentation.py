import numpy as np
import open3d as o3d
import time
import sys
import os

try:
    import schnabel_ransac
except ImportError:
    # If run from outside directory, add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import schnabel_ransac
    except ImportError as e:
        print(f"ERROR: Could not import schnabel_ransac: {e}")
        print("Please compile the wrapper first: python setup.py build_ext --inplace")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python tartan_ground_segmentation.py <point_cloud_file> [z_up/z_down] [voxel_size]")
        print("\nExample:")
        print("  python tartan_ground_segmentation.py tartanair_data/Supermarket/Supermarket_rgb.pcd z_down 0.3")
        sys.exit(1)

    filename = sys.argv[1]
    
    # Determine Z-axis orientation
    z_mode = "z_down" # Default to z_down for TartanAir/TartanGround
    if len(sys.argv) > 2:
        val = sys.argv[2].strip().lower()
        if val in ["z_up", "z_down"]:
            z_mode = val
    else:
        # Auto-detect based on filename
        if "tartan" not in filename.lower() and "office" in filename.lower():
            z_mode = "z_up"
            print("Auto-detected Z-up coordinate system (standard Open3D dataset).")
        else:
            print("Auto-detected Z-down (NED) coordinate system (TartanAir dataset default).")

    # 1. Load the Point Cloud
    print(f"\n[1/5] Loading point cloud from {filename}...")
    t_start = time.perf_counter()
    pcd = o3d.io.read_point_cloud(filename)
    n_points_orig = len(pcd.points)
    if n_points_orig == 0:
        print("ERROR: Point cloud is empty or failed to load.")
        sys.exit(1)
    print(f"      Loaded {n_points_orig:,} points in {time.perf_counter() - t_start:.2f} seconds.")

    # 2. Estimate & Apply Voxel Downsampling
    # Large datasets (like 7.8M points) need downsampling for speed and memory efficiency.
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_extent = max(extent)
    print(f"      Bounding Box Extents: X={extent[0]:.2f}m, Y={extent[1]:.2f}m, Z={extent[2]:.2f}m")
    
    # Custom voxel size or auto-tune
    if len(sys.argv) > 3:
        try:
            voxel_size = float(sys.argv[3])
        except ValueError:
            voxel_size = 0.2
    else:
        if n_points_orig > 1_000_000:
            voxel_size = max(0.1, round(max_extent / 100.0, 2))
            print(f"      Dataset is large. Auto-selected voxel_size={voxel_size}m for downsampling.")
        else:
            voxel_size = 0.02
            print(f"      Dataset is small. Using default voxel_size={voxel_size}m.")

    print(f"[2/5] Downsampling point cloud (voxel_size={voxel_size}m)...")
    t_ds = time.perf_counter()
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(pcd_ds.points).astype(np.float32)
    print(f"      Downsampled to {len(points):,} points in {time.perf_counter() - t_ds:.2f} seconds.")

    # 3. Detect Planes using C++ Schnabel RANSAC
    # We use absolute distance (relative_epsilon=False) in meters
    epsilon = max(0.05, voxel_size * 0.75)  # distance threshold (e.g. 15-20cm for 30cm voxel size)
    min_support = min(5000, max(500, int(len(points) * 0.02)))  # min points to form a plane (at least 2% of point cloud, capped at 5000)
    
    print(f"[3/5] Running C++ Cython RANSAC to detect planes...")
    print(f"      Parameters: epsilon={epsilon:.2f}m (absolute), min_support={min_support}, shape_types=['plane']")
    
    t_ransac = time.perf_counter()
    shapes, n_remaining = schnabel_ransac.detect(
        points,
        shapes=["plane"],
        relative_epsilon=False,
        epsilon=epsilon,
        normal_thresh=0.9, # normals within ~25 deg
        min_support=min_support,
        probability=0.01,
        max_shapes=30
    )
    print(f"      Detected {len(shapes)} planes in {time.perf_counter() - t_ransac:.2f} seconds.")

    if len(shapes) == 0:
        print("ERROR: No planes detected in the scene. Cannot segment ground.")
        sys.exit(1)

    # 4. Identify Ground Plane
    # Ground plane is a horizontal plane (normal aligned with vertical Z axis)
    # Z-up: floor is the horizontal plane with lowest average Z (most negative Z)
    # Z-down (NED): floor is the horizontal plane with highest average Z (most positive Z)
    print(f"[4/5] Filtering horizontal planes to identify Ground...")
    
    horizontal_planes = []
    for idx, shape in enumerate(shapes):
        mask = shape["inlier_mask"]
        plane_points = points[mask]
        
        # Estimate normal of this plane using covariance
        cov = np.cov(plane_points.T)
        evals, evecs = np.linalg.eig(cov)
        normal = evecs[:, np.argmin(evals)]
        
        # We want the normal to align with vertical Z axis: |normal_z| >= 0.85
        z_alignment = abs(normal[2])
        if z_alignment >= 0.85:
            avg_z = np.mean(plane_points[:, 2])
            horizontal_planes.append({
                "shape_index": idx,
                "shape": shape,
                "avg_z": avg_z,
                "z_alignment": z_alignment,
                "n_points": shape["n_points"],
                "normal": normal
            })

    if not horizontal_planes:
        print("WARNING: No strictly horizontal planes (|Nz| >= 0.85) were detected.")
        print("Falling back to the lowest/highest plane among all detected planes.")
        # Fallback to all planes
        for idx, shape in enumerate(shapes):
            mask = shape["inlier_mask"]
            plane_points = points[mask]
            avg_z = np.mean(plane_points[:, 2])
            horizontal_planes.append({
                "shape_index": idx,
                "shape": shape,
                "avg_z": avg_z,
                "z_alignment": 0.0,
                "n_points": shape["n_points"],
                "normal": np.array([0, 0, 1])
            })

    # Sort planes to find ground
    if z_mode == "z_up":
        # Lowest Z is ground
        horizontal_planes.sort(key=lambda x: x["avg_z"])
    else:
        # Highest Z is ground (Z-down NED)
        horizontal_planes.sort(key=lambda x: x["avg_z"], reverse=True)

    ground = horizontal_planes[0]
    print(f"      Selected plane {ground['shape_index']} as the Ground Plane:")
    print(f"        - Average Z elevation: {ground['avg_z']:.2f}m")
    print(f"        - Points: {ground['n_points']}")
    print(f"        - Normal alignment with vertical: {ground['z_alignment']:.3f}")

    # 5. Color and Save Segmented Point Cloud
    print(f"[5/5] Coloring and saving segmented point cloud...")
    
    # Default color: Grey for unclassified / obstacles
    colors = np.ones((len(points), 3)) * 0.6  # Grey
    
    # Paint other detected horizontal planes Blue (e.g. shelves, ceilings)
    for hp in horizontal_planes[1:]:
        mask = hp["shape"]["inlier_mask"]
        colors[mask] = [0.1, 0.4, 0.8]  # Blue
        
    # Paint Ground Plane Green
    ground_mask = ground["shape"]["inlier_mask"]
    colors[ground_mask] = [0.1, 0.8, 0.2]  # Green

    pcd_segmented = o3d.geometry.PointCloud()
    pcd_segmented.points = o3d.utility.Vector3dVector(points)
    pcd_segmented.colors = o3d.utility.Vector3dVector(colors)

    output_filename = os.path.splitext(filename)[0] + "_segmented.ply"
    o3d.io.write_point_cloud(output_filename, pcd_segmented)
    print(f"      Successfully saved segmented point cloud to: {output_filename}")
    print("      -> Green = Ground Plane")
    print("      -> Blue = Other Horizontal Planes (shelves, ceilings, structures)")
    print("      -> Grey = Rest of the environment (vertical walls, noise, obstacles)")

    # Also save the original downsampled point cloud to a PLY format so external viewers can load its RGB colors correctly
    original_output_filename = os.path.splitext(filename)[0] + "_original.ply"
    o3d.io.write_point_cloud(original_output_filename, pcd_ds)
    print(f"      Successfully saved original (downsampled) point cloud to: {original_output_filename}")

    # 6. Visualization (optional)
    try:
        print("\n[Visualizing] - Showing original (left) and segmented (right)")
        # Keep original colors for the left point cloud
        # (pcd_ds already retains the real RGB colors from the dataset)
        
        # Translate segmented to the side
        width = extent[0]
        pcd_segmented.translate([width * 1.1, 0, 0])
        
        o3d.visualization.draw_geometries([pcd_ds, pcd_segmented], window_name="TartanAir Ground Segmentation")
    except Exception as e:
        print(f"Visualization skipped: {e}")

if __name__ == "__main__":
    main()
