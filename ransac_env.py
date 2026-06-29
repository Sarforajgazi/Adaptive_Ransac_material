import os
import glob
import sys
import time
import csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from plyfile import PlyData

# Add schnabel_cython/ to path so we can import the compiled .pyd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))
import schnabel_ransac

from features.scene_features import compute_scene_features

def load_ply_xyz(filepath):
    """Load raw XYZ point cloud from a .ply file. No downsampling for benchmarks."""
    ply = PlyData.read(filepath)
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    return pts

def find_ground_plane(shapes, points, z_mode="z_down", horizontal_thresh=0.80):
    if not shapes:
        return None, None, None

    candidates = []
    for shape in shapes:
        mask = shape["inlier_mask"]
        plane_pts = points[mask]
        if len(plane_pts) < 10:
            continue
        cov = np.cov(plane_pts.T)
        evals, evecs = np.linalg.eig(cov)
        normal = evecs[:, np.argmin(evals)]
        z_align = abs(float(normal[2]))
        mean_pt = np.mean(plane_pts, axis=0)
        avg_z = float(mean_pt[2])
        
        # Calculate exactly how thick/noisy the plane is (Mean Absolute Error)
        # Distance = | (Point - Mean) dot Normal |
        distances = np.abs(np.dot(plane_pts - mean_pt, normal))
        residual = float(np.mean(distances))
        
        candidates.append({
            "shape": shape,
            "avg_z": avg_z,
            "z_align": z_align,
            "residual": residual,
            "n_points": shape["n_points"],
        })

    horizontal = [c for c in candidates if c["z_align"] >= horizontal_thresh]
    if not horizontal:
        horizontal = sorted(candidates, key=lambda c: c["n_points"], reverse=True)[:1]
        if not horizontal:
            return None, None, None

    reverse = (z_mode == "z_down")
    horizontal.sort(key=lambda c: c["avg_z"], reverse=reverse)
    best = horizontal[0]
    return best["shape"], best["avg_z"], best["z_align"], best["residual"]

class RansacEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Controls the RANSAC parameters for point cloud ground segmentation.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, data_dir=None, log_name="evaluation_metrics.csv"):
        super(RansacEnv, self).__init__()
        
        # Action space: epsilon, min_support, normal_thresh
        # normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: 
        # 21 geometric features computed by Open3D and PCA
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        
        # Setup CSV Logging
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, log_name)
        
        # Create CSV header if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame_id", "epsilon", "min_support", "normal_threshold", 
                    "runtime", "reward", "steps_used", "inlier_ratio", 
                    "plane_normal", "residual"
                ])
        
        # Parameter mappings
        # epsilon: [0.05, 0.5]
        # min_support: [100, 1000]
        # normal_thresh: [0.7, 0.95]
        
        self.data_dir = data_dir
        if self.data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Office", "Data_omni", "P0000", "lidar")
        
        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.ply")))
        if not self.files:
            raise ValueError(f"No .ply files found in {self.data_dir}")
            
        self.current_points = None
        self.current_file = None
        self.current_features = None

    def _unnormalize_action(self, action):
        # action is in [-1, 1]
        a = (action + 1.0) / 2.0 # now in [0, 1]
        eps = 0.05 + a[0] * (0.5 - 0.05)
        min_supp = int(100 + a[1] * (1000 - 100))
        norm_th = 0.7 + a[2] * (0.95 - 0.7)
        return eps, min_supp, norm_th

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick a random frame
        idx = self.np_random.integers(0, len(self.files))
        self.current_file = self.files[idx]
        self.current_points = load_ply_xyz(self.current_file)
        
        # Precompute and cache the 21 geometric features so step() is fast
        self.current_features = compute_scene_features(self.current_points)
        
        # Compute observation
        obs = self._get_obs()
        info = {"file": self.current_file}
        return obs, info

    def _get_obs(self):
        # Return the precomputed 21-dim feature vector
        if self.current_features is None:
            return np.zeros(21, dtype=np.float32)
        return self.current_features

    def step(self, action):
        start_time = time.time()
        eps, min_supp, norm_th = self._unnormalize_action(action)
        
        frame_id = os.path.basename(self.current_file) if self.current_file else "unknown"
        reward = 0.0
        ground_pct = 0.0
        z_align = 0.0
        info = {}
        
        if len(self.current_points) < min_supp:
            # Cannot run RANSAC with these parameters
            return self._get_obs(), -1.0, True, False, {"error": "too_few_points"}
            
        try:
            shapes, _ = schnabel_ransac.detect(
                self.current_points,
                shapes=["plane"],
                relative_epsilon=False,
                epsilon=eps,
                normal_thresh=norm_th,
                min_support=min_supp,
                probability=0.001,
                normal_knn=20,
                max_shapes=20,
            )
            
            ground_shape, avg_z, z_align, residual = find_ground_plane(shapes, self.current_points, z_mode="z_down")
            
            if ground_shape is None:
                reward = -1.0
                info = {"error": "no_ground_found"}
            else:
                ground_pts = ground_shape["n_points"]
                ground_pct = ground_pts / len(self.current_points)
                
                # Reward: We want maximum ground points and good horizontal alignment
                # ground_pct is in [0, 1]
                # z_align is in [0, 1], we want it close to 1.0
                reward = (ground_pct * 10.0) + (z_align * 2.0)
                info = {
                    "ground_pct": ground_pct,
                    "z_align": z_align,
                    "residual": residual,
                    "avg_z": avg_z,
                    "epsilon": eps,
                    "min_support": min_supp,
                    "normal_thresh": norm_th
                }
                
        except Exception as e:
            reward = -2.0
            info = {"error": str(e)}
            
        runtime = time.time() - start_time
        
        # Log to CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            # Use info.get("residual", 0.0) in case RANSAC completely failed
            res = info.get("residual", 0.0)
            writer.writerow([
                frame_id, 
                round(float(eps), 5), 
                int(min_supp), 
                round(float(norm_th), 5), 
                round(runtime, 5), 
                round(float(reward), 5), 
                1,  # steps_used (1-shot for now)
                round(float(ground_pct), 5), 
                round(float(z_align), 5), 
                round(float(res), 5)
            ])
            
        # For this setup, 1 frame = 1 episode. So it's always terminated after 1 step.
        terminated = True
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, info
