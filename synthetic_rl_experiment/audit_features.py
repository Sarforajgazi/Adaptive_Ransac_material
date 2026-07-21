import sys
import os
import numpy as np

# Add parent dir to path so we can import features
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.scene_features import compute_scene_features
from data_generator import SyntheticPlaneGenerator

def audit_features():
    gen = SyntheticPlaneGenerator(seed=42)
    features_list = []
    
    # Generate 20 scenes sweeping across noise and inlier ratios
    noises = np.linspace(0.01, 0.20, 5)
    ratios = np.linspace(0.03, 0.9, 4)
    
    print("Generating scenes and computing features...")
    for n in noises:
        for r in ratios:
            slope = np.random.uniform(0, 30)
            pts, _, _, _ = gen.generate_scene(num_points=10000, inlier_ratio=r, noise_sigma=n, slope_angle_deg=slope)
            feat = compute_scene_features(pts)
            features_list.append(feat)
            
    feat_matrix = np.vstack(features_list)
    variances = np.var(feat_matrix, axis=0)
    
    print("\nFeature Variance Analysis (21 Geometric Features):")
    for i, var in enumerate(variances):
        if var < 1e-6:
            print(f"Feature {i}: FLAT (Variance = {var:.2e})")
        else:
            print(f"Feature {i}: Active (Variance = {var:.2e})")
            
    num_flat = np.sum(variances < 1e-6)
    print(f"\nTotal flat features: {num_flat}/21")
    
    if num_flat > 10:
        print("WARNING: Over half of your features are degenerate on synthetic planes. "
              "The RL agent will have a much narrower signal than expected.")
    else:
        print("Feature diversity looks acceptable.")

if __name__ == "__main__":
    audit_features()
