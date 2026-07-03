import os
import glob
import sys
import time
import csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from plyfile import PlyData
import open3d as o3d

# Add schnabel_cython/ to path so we can import the compiled .pyd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))
import schnabel_ransac

from features.scene_features import compute_scene_features

def load_ply_xyz(filepath, voxel_size=0.05):
    """Load raw XYZ point cloud from a .ply file and apply voxel downsampling."""
    ply = PlyData.read(filepath)
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    
    # Phase 1: fixed voxel_size = 0.05m
    if voxel_size is not None and voxel_size > 0.0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pts = np.asarray(pcd.points, dtype=np.float32)
        
    return pts

def find_ground_plane(shapes, points, z_mode="z_down", horizontal_thresh=0.80):
    if not shapes:
        return None, None, None, None

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
            return None, None, None, None

    reverse = (z_mode == "z_down")
    horizontal.sort(key=lambda c: c["avg_z"], reverse=reverse)
    best = horizontal[0]
    return best["shape"], best["avg_z"], best["z_align"], best["residual"]

EPS_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
MIN_SUPPORT_LEVELS = [50, 100, 200, 300, 500, 800]


class RansacEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Controls the RANSAC parameters for point cloud ground segmentation.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, data_dir=None, log_name="evaluation_metrics.csv", fixed_normal_thresh=None):
        super(RansacEnv, self).__init__()

        # None (default) preserves the RL training/eval behavior exactly: normal_thresh
        # fixed at 0.90 for every step. Set this to override it -- used by
        # baseline_evaluator.py so Standard (0.85) and Loose (0.80) match
        # BASELINE_CONFIG.md instead of silently running at 0.90 too.
        self.fixed_normal_thresh = fixed_normal_thresh

        # Action space: epsilon (8 levels), min_support (6 levels), stop/continue (2 levels)
        self.action_space = spaces.MultiDiscrete([8, 6, 2])
        
        # Observation space: 31 dims
        # 21 geometric features + 10 feedback features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)
        
        # Feedback state variables
        self.step_count = 0
        self.inlier_ratio = 0.0
        self.prev_inlier_ratio = 0.0
        self.mean_residual = 0.0
        self.plane_normal = np.zeros(3, dtype=np.float32)
        self.prev_epsilon = 0.0
        self.prev_min_support = 0
        self.prev_normal_thresh = 0.0
        
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
            # Default to the root data directory to train on all environments
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        # Search recursively for all .ply files in any subfolder
        search_pattern = os.path.join(self.data_dir, "**", "*.ply")
        all_files = glob.glob(search_pattern, recursive=True)
        
        # Filter out bad files: cache folders, ground truth labels (_segmented), and outputs (ground.ply, obstacles.ply)
        valid_files = []
        for f in all_files:
            f_lower = f.lower()
            if "cache" in f_lower:
                continue
            if "segmented" in f_lower:
                continue
            if "ground.ply" in f_lower or "obstacles.ply" in f_lower:
                continue
            valid_files.append(f)
            
        self.files = sorted(valid_files)
        
        if not self.files:
            raise ValueError(f"No .ply files found recursively in {self.data_dir}")
            
        # Group files by folder for Adaptive Sampling
        self.folders_dict = {}
        for f in self.files:
            folder_name = os.path.basename(os.path.dirname(f))
            if folder_name not in self.folders_dict:
                self.folders_dict[folder_name] = []
            self.folders_dict[folder_name].append(f)
            
        self.folder_names = list(self.folders_dict.keys())
        self.folder_rewards = {folder: 0.0 for folder in self.folder_names}
        
        print(f"Loaded {len(self.files)} point cloud frames across {len(self.folder_names)} folders.")
            
        self.current_points = None
        self.current_file = None
        self.current_features = None

    def _decode_action(self, action):
        eps = EPS_LEVELS[int(action[0])]
        min_supp = MIN_SUPPORT_LEVELS[int(action[1])]
        stop = bool(action[2] == 0) # 0 is stop, 1 is continue

        return eps, min_supp, stop

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state variables
        self.step_count = 0
        self.inlier_ratio = 0.0
        self.prev_inlier_ratio = 0.0
        self.mean_residual = 0.0
        self.plane_normal = np.zeros(3, dtype=np.float32)
        self.prev_epsilon = 0.0
        self.prev_min_support = 0
        self.prev_normal_thresh = 0.0
        
        # Adaptive Sampling: Convert rewards to probabilities (lower reward = higher prob)
        rewards = np.array([self.folder_rewards[f] for f in self.folder_names])
        inverted = -rewards
        # Shift to prevent large negative numbers in exp
        inverted = inverted - np.max(inverted)
        exp_inv = np.exp(inverted)
        probs = exp_inv / np.sum(exp_inv)
        
        # Pick a folder based on probability
        chosen_folder = self.np_random.choice(self.folder_names, p=probs)
        # Pick a random frame inside that folder
        self.current_file = self.np_random.choice(self.folders_dict[chosen_folder])
        self.current_points = load_ply_xyz(self.current_file)
        
        # Precompute and cache the 21 geometric features so step() is fast
        self.current_features = compute_scene_features(self.current_points)
        
        # Compute observation
        obs = self._get_obs()
        info = {"file": self.current_file}
        return obs, info

    def _get_obs(self):
        if self.current_features is None:
            scene_feat = np.zeros(21, dtype=np.float32)
        else:
            scene_feat = self.current_features
            
        feedback_feat = np.array([
            self.inlier_ratio,
            self.mean_residual,
            self.plane_normal[0],
            self.plane_normal[1],
            self.plane_normal[2],
            float(self.step_count),
            self.prev_inlier_ratio,
            self.prev_epsilon,
            float(self.prev_min_support),
            self.prev_normal_thresh
        ], dtype=np.float32)
        
        return np.concatenate([scene_feat, feedback_feat])

    def step(self, action):
        start_time = time.time()
        self.step_count += 1
        eps, min_supp, stop = self._decode_action(action)
        norm_th = self.fixed_normal_thresh if self.fixed_normal_thresh is not None else 0.90  # Fixed for Phase 1
        
        frame_id = os.path.basename(self.current_file) if self.current_file else "unknown"
        info = {}
        
        # Save previous state before running Schnabel
        self.prev_epsilon = eps
        self.prev_min_support = min_supp
        self.prev_normal_thresh = norm_th
        self.prev_inlier_ratio = self.inlier_ratio
        
        if len(self.current_points) < min_supp:
            self.inlier_ratio = 0.0
            self.mean_residual = 0.0
            self.plane_normal = np.zeros(3, dtype=np.float32)
            z_align = 0.0
            reward = -1.0
            info = {"error": "too_few_points"}
            terminated = True
            return self._get_obs(), reward, terminated, False, info
            
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
                self.inlier_ratio = 0.0
                self.mean_residual = 0.0
                self.plane_normal = np.zeros(3, dtype=np.float32)
                z_align = 0.0
                info = {"error": "no_ground_found"}
            else:
                ground_pts = ground_shape["n_points"]
                self.inlier_ratio = ground_pts / len(self.current_points)
                self.mean_residual = residual
                
                # Get the actual normal from the inliers
                mask = ground_shape["inlier_mask"]
                plane_pts = self.current_points[mask]
                cov = np.cov(plane_pts.T)
                evals, evecs = np.linalg.eig(cov)
                self.plane_normal = evecs[:, np.argmin(evals)]
                
                info = {
                    "ground_pct": self.inlier_ratio,
                    "z_align": z_align,
                    "residual": residual,
                    "avg_z": avg_z,
                    "epsilon": eps,
                    "min_support": min_supp,
                    "normal_thresh": norm_th
                }
                
        except Exception as e:
            self.inlier_ratio = 0.0
            self.mean_residual = 0.0
            self.plane_normal = np.zeros(3, dtype=np.float32)
            z_align = 0.0
            info = {"error": str(e)}
            
        runtime = time.time() - start_time
        
        # Decide if we terminate and what the reward is
        terminated = stop or (self.step_count >= 5)
        
        if terminated:
            # Terminal reward
            normal_consistency = self.current_features[17] if self.current_features is not None else 0.0
            reward = (1.0 * self.inlier_ratio) - (0.1 * runtime) - (0.5 * self.mean_residual) + (0.3 * normal_consistency) - (0.05 * self.step_count)
            
            # Update folder moving average reward (EMA)
            folder_name = os.path.basename(os.path.dirname(self.current_file))
            self.folder_rewards[folder_name] = (0.9 * self.folder_rewards[folder_name]) + (0.1 * reward)
        else:
            reward = 0.0
        
        # Log to CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id, 
                round(float(eps), 5), 
                int(min_supp), 
                round(float(norm_th), 5), 
                round(runtime, 5), 
                round(float(reward), 5), 
                self.step_count,
                round(float(self.inlier_ratio), 5), 
                round(float(z_align), 5), 
                round(float(self.mean_residual), 5)
            ])
            
        return self._get_obs(), float(reward), terminated, False, info
