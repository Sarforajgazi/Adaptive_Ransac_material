import os
import time
import pickle
import argparse
import numpy as np
from stable_baselines3 import PPO
from ransac_env import RansacEnv, load_ply_xyz
from features.scene_features import compute_scene_features
from download_lidar_frames import ENVIRONMENTS, count_frames, DATA_ROOT

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(WORKSPACE, "models")
# The 2026-07-03 retrain (ent_coef + VecNormalize fix for the action-collapse
# bug -- see DAILY_LOG.md Day 8). Supersedes the earlier 50k-step run, which
# is archived under models/pre_entropy_fix/.
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_ransac_final.zip")
DEFAULT_VECNORMALIZE_PATH = os.path.join(MODEL_DIR, "ppo_ransac_final_vecnormalize.pkl")


def load_obs_normalizer(vecnormalize_path):
    """
    This model was trained inside VecNormalize(norm_obs=True) -- its policy
    only makes sense on observations scaled the same way. Loads the saved
    running statistics and returns a function replicating SB3's own
    VecNormalize._normalize_obs formula exactly, without needing to
    reconstruct a full VecEnv just to hold them.
    """
    if vecnormalize_path is None or not os.path.exists(vecnormalize_path):
        print("No VecNormalize stats found -- using raw observations "
              "(only correct if the model was NOT trained with VecNormalize).")
        return lambda obs: obs

    with open(vecnormalize_path, "rb") as f:
        vec_normalize = pickle.load(f)
    obs_rms = vec_normalize.obs_rms
    clip_obs = vec_normalize.clip_obs
    epsilon = vec_normalize.epsilon

    def normalize(obs):
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -clip_obs, clip_obs).astype(np.float32)

    return normalize


def main():
    parser = argparse.ArgumentParser(description="Run RL Agent Evaluator")
    parser.add_argument("--env", type=str, default="all", help="Specific dataset to run, or 'all'")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to a trained model .zip")
    parser.add_argument("--vecnormalize", type=str, default=DEFAULT_VECNORMALIZE_PATH,
                         help="Path to the matching VecNormalize .pkl (must correspond to --model)")
    args = parser.parse_args()

    print("=" * 60)
    print("Running RL Agent Evaluation")
    print("=" * 60)

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    print(f"Loading observation normalizer: {args.vecnormalize}")
    normalize_obs = load_obs_normalizer(args.vecnormalize)

    envs_to_run = ENVIRONMENTS if args.env == "all" else [args.env]
    overall_start_time = time.time()

    for i, env_name in enumerate(envs_to_run):
        print(f"\n--- [{i+1}/{len(envs_to_run)}] Dataset: {env_name} ---")
        
        dataset_dir = os.path.join(DATA_ROOT, env_name, "Data_omni", "P0000", "lidar")
        if not os.path.exists(dataset_dir):
            print(f"Skipping {env_name}, folder does not exist.")
            continue
            
        num_frames = count_frames(env_name)
        if num_frames == 0:
            print(f"Skipping {env_name}, no frames found.")
            continue
            
        print(f"Evaluating {num_frames} frames for {env_name} (RL mode)...")
        
        csv_name = f"{env_name}_rl.csv"
        env = RansacEnv(data_dir=dataset_dir, log_name=csv_name)
        
        total_reward = 0.0
        start_time = time.time()
        
        # Sort files to ensure deterministic ordering
        sorted_files = sorted(env.files)
        
        for j in range(len(sorted_files)):
            # Manually reset state to avoid adaptive sampling randomizing the files
            env.step_count = 0
            env.inlier_ratio = 0.0
            env.prev_inlier_ratio = 0.0
            env.mean_residual = 0.0
            env.plane_normal = np.zeros(3, dtype=np.float32)
            env.prev_epsilon = 0.0
            env.prev_min_support = 0
            env.prev_normal_thresh = 0.0
            
            env.current_file = sorted_files[j]
            filename = os.path.basename(env.current_file)
            
            try:
                env.current_points = load_ply_xyz(env.current_file)
                env.current_features = compute_scene_features(env.current_points)
                
                obs = env._get_obs()
                terminated = False

                while not terminated:
                    action, _states = model.predict(normalize_obs(obs), deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                
                if (j + 1) % 10 == 0 or j == len(sorted_files) - 1:
                    print(f"  [{j+1}/{num_frames}] {filename} -> Steps: {env.step_count}, Reward: {reward:.4f} | Ground: {info.get('ground_pct', 0)*100:.1f}%")
            except Exception as e:
                print(f"  Error on {env.current_file}: {e}")
                
        elapsed = time.time() - start_time
        avg_reward = total_reward / num_frames
        print(f"Completed {env_name} (RL) in {elapsed:.2f}s | Avg Reward: {avg_reward:.4f}")

    total_elapsed = time.time() - overall_start_time
    print("=" * 60)
    print(f"FINISHED RL EVALUATION in {total_elapsed / 60:.2f} minutes.")
    print("=" * 60)

if __name__ == "__main__":
    main()
