import numpy as np
import matplotlib.pyplot as plt
import schnabel_ransac

def generate_synthetic_data():
    """Generates a point cloud with a Plane, a Cylinder, and some random noise."""
    np.random.seed(42)
    
    # 1. Generate a Plane (z = 0)
    plane_x = np.random.uniform(-5, 5, 2000)
    plane_y = np.random.uniform(-5, 5, 2000)
    plane_z = np.random.normal(0, 0.01, 2000)
    plane_pts = np.column_stack((plane_x, plane_y, plane_z))
    
    # 2. Generate a Cylinder (Radius=2, Height=6, standing on x=0, y=0)
    angles = np.random.uniform(0, 2 * np.pi, 2000)
    z_cyl = np.random.uniform(0, 6, 2000)
    x_cyl = 2.0 * np.cos(angles) + np.random.normal(0, 0.01, 2000)
    y_cyl = 2.0 * np.sin(angles) + np.random.normal(0, 0.01, 2000)
    cylinder_pts = np.column_stack((x_cyl, y_cyl, z_cyl))
    
    # 3. Generate Random Noise
    noise_pts = np.random.uniform(-6, 6, (1000, 3))
    
    # Combine them all
    cloud = np.vstack((plane_pts, cylinder_pts, noise_pts)).astype(np.float32)
    np.random.shuffle(cloud) # Shuffle so they aren't perfectly ordered
    return cloud

def main():
    print("Generating 3D Point Cloud...")
    points = generate_synthetic_data()
    print(f"Total points: {len(points)}")
    
    print("\nRunning C++ Cython Wrapper...")
    # Run the extremely fast C++ algorithm!
    # We require shapes to have at least 1000 points to be valid
    shapes, n_remaining = schnabel_ransac.detect(
        points,
        epsilon=0.05,
        normal_thresh=0.9,
        min_support=1000,
        probability=0.01
    )
    
    print(f"\nDetection Complete! Found {len(shapes)} shapes in the noise.")
    
    # ---------- VISUALIZATION ----------
    print("Opening 3D Plot...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate unassigned (noise) mask
    unassigned_mask = np.ones(len(points), dtype=bool)
    for shape in shapes:
        unassigned_mask &= ~shape["inlier_mask"]
        
    # Plot unassigned points (Noise) as small grey dots
    ax.scatter(points[unassigned_mask, 0], 
               points[unassigned_mask, 1], 
               points[unassigned_mask, 2], 
               c='grey', s=5, alpha=0.3, label="Noise / Unassigned")
    
    # Colors for our detected shapes
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, shape in enumerate(shapes):
        shape_type = shape["description"]
        mask = shape["inlier_mask"]
        
        c = colors[i % len(colors)]
        # Plot the assigned points brightly
        ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], 
                   c=c, s=15, alpha=0.8, 
                   label=f"Shape {i}: {shape_type} ({shape['n_points']} pts)")
        
        print(f"  - Plotted Shape {i}: {shape_type} with {shape['n_points']} points.")

    ax.set_title("Schnabel RANSAC (C++ via Cython) Detection")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
