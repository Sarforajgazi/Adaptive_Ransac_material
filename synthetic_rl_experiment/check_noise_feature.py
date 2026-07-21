import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl_experiment.data_generator import SyntheticPlaneGenerator
from features.scene_features import compute_scene_features

def test_noise_features():
    gen = SyntheticPlaneGenerator(seed=42)
    noise_levels = [0.01, 0.05, 0.10, 0.15, 0.20]
    
    print("Testing local surface variation feature vs injected noise_sigma:")
    print(f"{'Noise':<10} | {'Mean Surface Var':<20} | {'P25 Surface Var':<20}")
    print("-" * 55)
    
    for noise in noise_levels:
        # Generate scene at fixed 50% inlier ratio
        pts, _, _, _ = gen.generate_scene(
            num_points=10000, 
            inlier_ratio=0.5, 
            noise_sigma=noise, 
            slope_angle_deg=10.0,
            orientation="ground"
        )
        
        # Compute features
        features = compute_scene_features(pts)
        
        # We appended the two new features at the end (indices 21 and 22 of a 23-dim array)
        mean_var = features[21]
        p25_var = features[22]
        
        print(f"{noise:<10.3f} | {mean_var:<20.6f} | {p25_var:<20.6f}")

if __name__ == "__main__":
    test_noise_features()
