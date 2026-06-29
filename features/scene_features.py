import numpy as np
import open3d as o3d

def compute_scene_features(points):
    """
    Computes 21-dimensional geometric features from a LiDAR point cloud.
    Returns a 1D numpy array of shape (21,) in float32.
    """
    if len(points) == 0:
        return np.zeros(21, dtype=np.float32)

    # ---------------------------------------------------------
    # 1. Bounding Box & Density (5 features)
    # ---------------------------------------------------------
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    diff = maxs - mins
    
    bbox_dx = diff[0]
    bbox_dy = diff[1]
    bbox_dz = diff[2]
    
    bbox_volume = np.prod(diff)
    point_density = len(points) / (bbox_volume + 1e-6)

    # ---------------------------------------------------------
    # 2. Height Distribution (NED Z-Down) (4 features)
    # ---------------------------------------------------------
    z_coords = points[:, 2]
    z_mean = np.mean(z_coords)
    z_std = np.std(z_coords)
    z_min = np.min(z_coords)  # Top/Ceiling in Z-down
    z_max = np.max(z_coords)  # Bottom/Floor in Z-down

    # ---------------------------------------------------------
    # 3. Range/Distance Metrics (2 features)
    # ---------------------------------------------------------
    ranges = np.linalg.norm(points, axis=1)
    scan_range_mean = np.mean(ranges)
    scan_range_std = np.std(ranges)

    # ---------------------------------------------------------
    # 4. PCA & Eigenvalues (Full Cloud) (3 features)
    # ---------------------------------------------------------
    cov = np.cov(points.T)
    evals, _ = np.linalg.eig(cov)
    evals = np.sort(evals)[::-1]  # Sort descending
    
    eig_0 = evals[0] if len(evals) > 0 else 0.0
    eig_1 = evals[1] if len(evals) > 1 else 0.0
    eig_2 = evals[2] if len(evals) > 2 else 0.0

    # ---------------------------------------------------------
    # 5. Normals & Local Geometry via Open3D (7 features)
    # ---------------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals using 20 KNN
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    normals = np.asarray(pcd.normals)
    
    # Normal Variance (How chaotic is the room surface?)
    n_std = np.std(normals, axis=0)
    normal_x_std = n_std[0]
    normal_y_std = n_std[1]
    normal_z_std = n_std[2]
    
    # Normal Consistency (Subsample 1000 pairs for speed)
    # We take random points, find their normals, and dot them with their neighbor
    # Open3D KDTree allows fast neighbor lookup
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    num_samples = min(1000, len(points))
    sample_indices = np.random.choice(len(points), num_samples, replace=False)
    
    consistency_sum = 0.0
    mean_knn_dist_sum = 0.0
    
    for idx in sample_indices:
        # Get 5 nearest neighbors
        [k, idx_knn, dist_sq] = kdtree.search_knn_vector_3d(pcd.points[idx], 5)
        
        # Distance (skip the point itself which is dist 0)
        mean_knn_dist_sum += np.mean(np.sqrt(np.clip(dist_sq[1:], 0, None)))
        
        # Normal consistency with 1st nearest neighbor (not itself)
        if k > 1:
            n1 = normals[idx]
            n2 = normals[idx_knn[1]]
            consistency_sum += abs(np.dot(n1, n2))
            
    normal_consistency = consistency_sum / num_samples
    mean_knn_dist = mean_knn_dist_sum / num_samples

    # Z-density Ground (fraction of points in bottom 0.5 meters)
    # Z-down means ground has HIGHER Z. So we look for points with Z > (z_max - 0.5)
    ground_thresh = z_max - 0.5
    ground_pts = np.sum(z_coords > ground_thresh)
    z_density_ground = ground_pts / len(points)
    
    # Ground Slope Estimate (PCA on lowest 10% of points to find ground tilt)
    lowest_10_percent = int(len(points) * 0.1)
    if lowest_10_percent > 3:
        # Sort by Z descending (so highest Z = lowest points in NED)
        sorted_indices = np.argsort(z_coords)[::-1]
        lowest_pts = points[sorted_indices[:lowest_10_percent]]
        
        low_cov = np.cov(lowest_pts.T)
        low_evals, low_evecs = np.linalg.eig(low_cov)
        
        # The normal is the eigenvector of the smallest eigenvalue
        low_normal = low_evecs[:, np.argmin(low_evals)]
        # Slope angle from vertical (Z axis)
        z_align = abs(low_normal[2])
        ground_slope_estimate = np.arccos(z_align)  # Radians
    else:
        ground_slope_estimate = 0.0

    # ---------------------------------------------------------
    # Assemble final 21-dim vector
    # ---------------------------------------------------------
    features = np.array([
        bbox_dx, bbox_dy, bbox_dz, bbox_volume, point_density,
        z_mean, z_std, z_min, z_max,
        scan_range_mean, scan_range_std,
        eig_0, eig_1, eig_2,
        normal_x_std, normal_y_std, normal_z_std,
        normal_consistency, z_density_ground, ground_slope_estimate, mean_knn_dist
    ], dtype=np.float32)
    
    # Replace NaNs with 0
    features = np.nan_to_num(features)
    
    return features
