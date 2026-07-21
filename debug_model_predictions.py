"""
Diagnostic: For the first few frames of each held-out scene, show:
1. Actual model actions (eps, min_pts, norm_thresh)
2. How many RANSAC shapes were found
3. Whether find_ground_plane succeeded
4. Baseline RANSAC result with sensible fixed params (IoU sanity check)
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))
import schnabel_ransac
from ransac_env import RansacEnv, load_ply_xyz, find_ground_plane, EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS
from features.scene_features import compute_scene_features
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---- action space (must match training) ----
# From eval_tartanair_gt.py
EPSILONS   = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MIN_POINTS = [500, 1000, 2000, 3000, 4000, 5000]
NORMALS    = [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]

# Check what the training env used
print("Training EPS_LEVELS:    ", EPS_LEVELS)
print("Training MIN_SUPPORT:   ", MIN_SUPPORT_LEVELS)
print("Training NORM_THRESH:   ", NORM_THRESH_LEVELS)

ENVS = ["Gascola", "House", "NordicHarbor", "WesternDesertTown"]
N_FRAMES = 5

model_path = "models/ppo_ransac_v4_model_final.zip"
stats_path = "models/ppo_ransac_v4_model_final_vecnormalize.pkl"

def make_env_fn(env_name):
    return lambda: RansacEnv(data_dir=f"data/{env_name}")

for env_name in ENVS:
    print(f"\n{'='*60}")
    print(f"SCENE: {env_name}")
    print(f"{'='*60}")
    
    vec_env = DummyVecEnv([make_env_fn(env_name)])
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training, vec_env.norm_reward = False, False
    model = PPO.load(model_path, env=vec_env)

    files = sorted(glob.glob(f"data/{env_name}/Data_omni/P0000/lidar/*.ply"))[:N_FRAMES]

    for fpath in files:
        pts = load_ply_xyz(fpath, voxel_size=0.05)
        fidx = int(os.path.basename(fpath).split("_")[0])
        
        # --- Raw point cloud statistics ---
        z = pts[:, 2]
        print(f"\n  Frame {fidx}: {len(pts)} pts | z=[{z.min():.2f}, {z.max():.2f}] mean_z={z.mean():.2f}")

        # --- Model action ---
        features = compute_scene_features(pts)
        obs = np.concatenate([features, np.zeros(10)])
        obs_norm = vec_env.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(obs_norm, deterministic=True)
        action = action[0]
        
        # Decode action (eval_tartanair_gt.py decoding)
        eps        = EPSILONS[action[0]]
        min_pts    = MIN_POINTS[action[1]]
        norm_thresh= NORMALS[action[2]]
        print(f"  Model action: eps={eps}, min_pts={min_pts}, norm_thresh={norm_thresh}")
        
        # Also decode using training env's levels
        eps_tr  = EPS_LEVELS[action[0]] if action[0] < len(EPS_LEVELS) else "OOB"
        mpts_tr = MIN_SUPPORT_LEVELS[action[1]] if action[1] < len(MIN_SUPPORT_LEVELS) else "OOB"
        nt_tr   = NORM_THRESH_LEVELS[action[2]] if action[2] < len(NORM_THRESH_LEVELS) else "OOB"
        print(f"  Training levels: eps={eps_tr}, min_pts={mpts_tr}, norm_thresh={nt_tr}")
        
        # --- Run RANSAC with model params ---
        try:
            shapes, _ = schnabel_ransac.detect(
                pts, shapes=["plane"], relative_epsilon=False,
                epsilon=eps, normal_thresh=norm_thresh,
                min_support=min_pts, probability=0.001,
                normal_knn=20, max_shapes=20,
            )
            ground_shape, avg_z, z_align, res = find_ground_plane(shapes, pts, z_mode="z_down")
            n_shapes = len(shapes) if shapes else 0
            print(f"  RANSAC (model params): {n_shapes} shapes found | "
                  f"ground={'YES' if ground_shape else 'NO'} "
                  f"(avg_z={avg_z}, z_align={z_align})")
        except Exception as e:
            print(f"  RANSAC (model params): ERROR - {e}")

        # --- Baseline: sensible fixed params ---
        try:
            shapes_b, _ = schnabel_ransac.detect(
                pts, shapes=["plane"], relative_epsilon=False,
                epsilon=0.2, normal_thresh=0.9,
                min_support=200, probability=0.001,
                normal_knn=20, max_shapes=20,
            )
            ground_b, avg_z_b, z_align_b, _ = find_ground_plane(shapes_b, pts, z_mode="z_down")
            nb = len(shapes_b) if shapes_b else 0
            if ground_b is not None:
                n_pred = ground_b["inlier_mask"].sum()
            else:
                n_pred = 0
            print(f"  RANSAC (baseline eps=0.2): {nb} shapes | ground={'YES' if ground_b else 'NO'} "
                  f"| pred_ground_pts={n_pred} ({100*n_pred/len(pts):.1f}%)")
        except Exception as e:
            print(f"  RANSAC (baseline): ERROR - {e}")

print("\nDone.")
