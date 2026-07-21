import sys
import os
import numpy as np

# Add parent dir to path so we can import schnabel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schnabel_cython"))

import schnabel_ransac
from data_generator import SyntheticPlaneGenerator
from synthetic_env import angle_between_normals

def main():
    gen = SyntheticPlaneGenerator(seed=42)
    
    # Refinement steps: Loose → Super Strict
    # All eps ≥ 0.08m (mean inlier NN distance in a 10000-pt, [-10,10]^3 cloud)
    # normal_thresh in [0.65, 0.85]: tight enough to discriminate planes, loose enough
    # for Schnabel's KNN normal estimation to succeed on noisy data.
    steps = [
        (0.40, 50,  0.65),   # Loose:        captures high-noise inliers broadly
        (0.25, 75,  0.70),   # Medium-Loose
        (0.15, 75,  0.75),   # Standard
        (0.10, 100, 0.80),   # Strict
        (0.08, 150, 0.85),   # Super Strict: tightest eps ≥ NN spacing
    ]
    
    num_scenes = 10
    print("Running Refinement Benefit Check on 10 Scenes...")
    print("-" * 60)
    
    for i in range(num_scenes):
        # Generate a scene with moderate noise
        pts, gt_mask, n_true, d_true = gen.generate_scene(
            num_points=5000, 
            inlier_ratio=np.random.uniform(0.1, 0.5), 
            noise_sigma=np.random.uniform(0.05, 0.15), 
            slope_angle_deg=np.random.uniform(0, 30)
        )
        
        true_inliers_count = np.sum(gt_mask)
        print(f"Scene {i+1}: True Inliers={true_inliers_count}, Noise={np.std(pts):.3f}")
        
        for step_idx, (eps, min_supp, norm_th) in enumerate(steps):
            shapes, _ = schnabel_ransac.detect(
                pts,
                shapes=["plane"],
                relative_epsilon=False,
                epsilon=eps,
                normal_thresh=norm_th,
                min_support=min_supp,
                probability=0.001,
                normal_knn=20,
                max_shapes=5,
            )
            
            best_angle = 180.0
            best_inlier_recovery = 0.0
            num_shapes_found = len(shapes) if shapes else 0
            failure_mode = "NO_SHAPES"  # default: sentinel failure

            if shapes:
                for shape in shapes:
                    mask = shape["inlier_mask"]
                    plane_pts = pts[mask]
                    if len(plane_pts) < 10:
                        continue
                    
                    cov = np.cov(plane_pts.T)
                    evals, evecs = np.linalg.eig(cov)
                    normal = evecs[:, np.argmin(evals)]
                    
                    if np.dot(normal, n_true) < 0:
                        normal = -normal
                        
                    angle = angle_between_normals(normal, n_true)
                    
                    # Compute recovery
                    true_positives = np.sum(mask & gt_mask)
                    inlier_recovery = true_positives / (true_inliers_count + 1e-8)
                    
                    if angle < best_angle:
                        best_angle = angle
                        best_inlier_recovery = inlier_recovery
                        failure_mode = "NONE"  # a valid plane was found

            # Classify the 180° outcome
            if best_angle >= 179.0:
                if num_shapes_found == 0:
                    failure_label = f"[SENTINEL: 0 shapes detected]"
                else:
                    failure_label = f"[GENUINE FAILURE: {num_shapes_found} shapes found but all wrong]"
            else:
                failure_label = f"[OK: {num_shapes_found} shapes]"
                    
            print(f"  Step {step_idx+1} (eps={eps:.3f}, supp={min_supp}, norm={norm_th:.2f}) -> "
                  f"Angle Err: {best_angle:.2f} deg | Inlier Recovery: {best_inlier_recovery:.3f} {failure_label}")
        print("-" * 60)

if __name__ == "__main__":
    main()
