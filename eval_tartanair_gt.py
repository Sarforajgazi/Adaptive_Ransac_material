import os
import glob
import json
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ransac_env import RansacEnv, load_ply_xyz

def load_ground_colors(env_name):
    # Import the correct logic from the precompute script which handles RGB mapping properly
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from precompute_gt_masks import get_ground_colors
    colors = get_ground_colors(env_name, data_dir="data")
    if colors is None or len(colors) == 0:
        print(f"Warning: No ground colors found for {env_name}.")
        return []
    return colors

def evaluate_tartanair(env_name, num_frames=10):
    print(f"\n--- EVALUATING {env_name} ---")
    
    print(f"Loading global sem_pcd for {env_name} (this takes a moment)...")
    pcd = o3d.io.read_point_cloud(f"data/{env_name}/{env_name}_sem.pcd")
    global_points = np.asarray(pcd.points)
    global_colors = np.asarray(pcd.colors)
    
    ground_colors = load_ground_colors(env_name)
    if len(ground_colors) == 0:
        return 0, 0, 0
    
    print("Building KDTree for global map...")
    kdtree = cKDTree(global_points)
    
    poses_path = f"data/{env_name}/Data_omni/P0000/pose_lcam_front.txt"
    if os.path.exists(poses_path):
        poses = np.loadtxt(poses_path)
    else:
        print(f"Missing poses for {env_name}")
        return 0, 0, 0
    
    from debug_alignment import apply_transform
    from scipy.spatial.transform import Rotation as R
    
    # swapxy_negz is confirmed correct for all TartanAir ground robot scenes
    # (verified by check_alignment.py: 93-100% match rate vs ~10% for 'raw')
    transform_name = "swapxy_negz"
    
    model_path = "models/ppo_ransac_v4_model_final.zip"
    stats_path = "models/ppo_ransac_v4_model_final_vecnormalize.pkl"
    if not os.path.exists(model_path):
        model_path = "models/ppo_ransac_v4_model_model_100000_steps.zip"
        stats_path = "models/ppo_ransac_v4_model_model_vecnormalize_100000_steps.pkl"
        
    def make_env(): return RansacEnv(data_dir=f"data/{env_name}")
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_path, env)
    env.training, env.norm_reward = False, False
    model = PPO.load(model_path, env=env)
    
    lidar_files = sorted(glob.glob(f"data/{env_name}/Data_omni/P0000/lidar/*.ply"))[:num_frames]
    ious, precisions, recalls = [], [], []
    
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))
    import schnabel_ransac
    from ransac_env import find_ground_plane, EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS
    from features.scene_features import compute_scene_features
        
    for i, file_path in enumerate(lidar_files):
        frame_idx = int(os.path.basename(file_path).split('_')[0])
        pose = poses[frame_idx]
        rot = R.from_quat(pose[3:]).as_matrix()
        t = pose[:3]
        
        frame_points = load_ply_xyz(file_path, voxel_size=0.05)
        if len(frame_points) == 0: continue
        
        # --- Load precomputed GT mask if available, else compute on-the-fly ---
        mask_path = file_path.replace(".ply", "_gt_mask.npy")
        if os.path.exists(mask_path):
            gt_labels = np.load(mask_path)
            if len(gt_labels) != len(frame_points):
                # Shape mismatch: recompute
                gt_labels = None
        else:
            gt_labels = None
        
        if gt_labels is None:
            # Compute on-the-fly with the correct coordinate transform
            pts_sensor = apply_transform(frame_points, transform_name)
            pts_global = (rot @ pts_sensor.T).T + t
            
            distances, indices = kdtree.query(pts_global, k=1, distance_upper_bound=0.5)
            valid_mask = distances != np.inf
            nn_colors = global_colors[indices[valid_mask]]
            
            is_floor_gt = np.zeros(len(nn_colors), dtype=bool)
            for c in ground_colors:
                is_floor_gt = np.logical_or(is_floor_gt, np.linalg.norm(nn_colors - c, axis=1) < 0.05)
                
            gt_labels = np.zeros(len(frame_points), dtype=bool)
            gt_labels[valid_mask] = is_floor_gt
        
        if i < 3:  # Print debug stats for first 3 frames
            pct = 100.0 * gt_labels.sum() / max(len(gt_labels), 1)
            print(f"  [Frame {frame_idx}] pts={len(frame_points)} gt_ground={gt_labels.sum()} ({pct:.1f}%)")
        
        features = compute_scene_features(frame_points)
        obs = np.concatenate([features, np.zeros(10)]) 
        obs_normalized = env.normalize_obs(obs.reshape(1, -1))
        
        action, _ = model.predict(obs_normalized, deterministic=True)
        action = action[0]
        
        # Decode actions using the EXACT same tables as the training environment
        # (EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS from ransac_env.py)
        eps         = EPS_LEVELS[action[0]]
        min_pts     = MIN_SUPPORT_LEVELS[action[1]]
        norm_thresh = NORM_THRESH_LEVELS[action[2]]
        
        try:
            shapes, _ = schnabel_ransac.detect(
                frame_points,
                shapes=["plane"],
                relative_epsilon=False,
                epsilon=eps,
                normal_thresh=norm_thresh,
                min_support=min_pts,
                probability=0.001,
                normal_knn=20,
                max_shapes=20,
            )
            # z_up: raw sensor frame has z pointing up, so ground = lowest/most-negative z
            ground_shape, _, _, _ = find_ground_plane(shapes, frame_points, z_mode="z_up")
            if ground_shape is not None:
                best_inliers = ground_shape["inlier_mask"]
            else:
                best_inliers = []
        except Exception as e:
            best_inliers = []
            
        pred_labels = np.zeros(len(frame_points), dtype=bool)
        pred_labels[best_inliers] = True
        
        intersection = np.logical_and(gt_labels, pred_labels).sum()
        union = np.logical_or(gt_labels, pred_labels).sum()
        
        iou = intersection / union if union > 0 else 0
        precision = intersection / pred_labels.sum() if pred_labels.sum() > 0 else 0
        recall = intersection / gt_labels.sum() if gt_labels.sum() > 0 else 0
        
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)

    mean_iou = np.mean(ious)
    mean_prec = np.mean(precisions)
    mean_rec = np.mean(recalls)
    
    print(f"Results for {env_name}:")
    print(f"  Mean IoU:       {mean_iou:.4f}")
    print(f"  Mean Precision: {mean_prec:.4f}")
    print(f"  Mean Recall:    {mean_rec:.4f}")
    
    return mean_iou, mean_prec, mean_rec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', nargs='+', default=['House', 'CoalMine', 'AbandonedFactory', 'WesternDesertTown'])
    parser.add_argument('--frames', type=int, default=10)
    args = parser.parse_args()
    
    all_ious = []
    for env in args.envs:
        iou, prec, rec = evaluate_tartanair(env, num_frames=args.frames)
        all_ious.append(iou)
        
    print("\n=================================")
    print("FINAL SUMMARY ACROSS ALL SCENES:")
    for env, iou in zip(args.envs, all_ious):
        print(f"  {env}: IoU = {iou:.4f}")
    print(f"OVERALL MEAN IOU: {np.mean(all_ious):.4f}")
    print("=================================")
