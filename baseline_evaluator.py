import os
import time
import argparse
import numpy as np
from ransac_env import RansacEnv, EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS
from download_lidar_frames import ENVIRONMENTS, count_frames, DATA_ROOT

# BASELINE_CONFIG.md's three fixed-parameter modes. Values must exist exactly
# in EPS_LEVELS/MIN_SUPPORT_LEVELS/NORM_THRESH_LEVELS (ransac_env.py's
# discrete action space) -- all three do.
BASELINE_MODES = {
    "strict":   {"epsilon": 0.10, "min_support": 800, "normal_thresh": 0.90},
    "standard": {"epsilon": 0.15, "min_support": 500, "normal_thresh": 0.85},
    "loose":    {"epsilon": 0.25, "min_support": 200, "normal_thresh": 0.80},
}


def build_action(epsilon, min_support, normal_thresh):
    """MultiDiscrete([epsilon_level, min_support_level, normal_thresh_level, stop]).
    stop=0 -- baseline evaluation is a single fixed-parameter attempt per frame,
    not multi-step refinement. RansacEnv's fixed_normal_thresh override is what
    actually forces the exact value during a baseline run -- this index is
    passed for correctness/clarity, not because the override depends on it."""
    eps_idx = EPS_LEVELS.index(epsilon)
    min_supp_idx = MIN_SUPPORT_LEVELS.index(min_support)
    norm_th_idx = NORM_THRESH_LEVELS.index(normal_thresh)
    return np.array([eps_idx, min_supp_idx, norm_th_idx, 0], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Run RANSAC Baseline Evaluator")
    parser.add_argument("mode", choices=["strict", "standard", "loose"],
                        help="Which baseline to run: strict, standard, or loose")
    parser.add_argument("--env", type=str, default="all",
                         help="Specific dataset to run, or 'all' (the training-set ENVIRONMENTS list). "
                              "Use this to evaluate held-out scenes (e.g. Gascola, House) that are "
                              "deliberately excluded from ENVIRONMENTS.")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Running Baseline Evaluation: {args.mode.upper()}")
    print("=" * 60)

    config = BASELINE_MODES[args.mode]
    action = build_action(config["epsilon"], config["min_support"], config["normal_thresh"])

    envs_to_run = ENVIRONMENTS if args.env == "all" else [args.env]
    total_datasets = len(envs_to_run)
    overall_start_time = time.time()

    for i, env_name in enumerate(envs_to_run):
        print(f"\n--- [{i+1}/{total_datasets}] Dataset: {env_name} ---")
        
        # 2. Check if dataset exists locally
        dataset_dir = os.path.join(DATA_ROOT, env_name, "Data_omni", "P0000", "lidar")
        if not os.path.exists(dataset_dir):
            print(f"Skipping {env_name}, folder does not exist.")
            continue
            
        num_frames = count_frames(env_name)
        if num_frames == 0:
            print(f"Skipping {env_name}, no frames found.")
            continue
            
        print(f"Evaluating {num_frames} frames for {env_name} ({args.mode.upper()} mode)...")
        
        # 3. Setup Environment for this specific dataset and mode
        # E.g., logs/Hospital_strict.csv
        csv_name = f"{env_name}_{args.mode}.csv"
        
        env = RansacEnv(data_dir=dataset_dir, log_name=csv_name, fixed_normal_thresh=config["normal_thresh"])

        print(f"Using: Epsilon={config['epsilon']:.4f}m, MinSupport={config['min_support']}, "
              f"NormalThresh={config['normal_thresh']:.3f}")
        
        total_reward = 0.0
        start_time = time.time()
        
        # 4. Evaluate every single frame
        for j in range(len(env.files)):
            env.current_file = env.files[j]
            from ransac_env import load_ply_xyz
            from features.scene_features import compute_scene_features
            
            try:
                env.current_points = load_ply_xyz(env.current_file)
                env.current_features = compute_scene_features(env.current_points)
                
                # Step triggers RANSAC and writes to the CSV automatically
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                
                if (j + 1) % 10 == 0 or j == len(env.files) - 1:
                    filename = os.path.basename(env.current_file)
                    print(f"  [{j+1}/{num_frames}] {filename} -> Reward: {reward:.4f} | Ground: {info.get('ground_pct', 0)*100:.1f}%")
            except Exception as e:
                print(f"  Error on {env.current_file}: {e}")
                
        elapsed = time.time() - start_time
        avg_reward = total_reward / num_frames
        print(f"Completed {env_name} ({args.mode.upper()}) in {elapsed:.2f}s | Avg Reward: {avg_reward:.4f}")

    total_elapsed = time.time() - overall_start_time
    print("=" * 60)
    print(f"FINISHED {args.mode.upper()} EVALUATION in {total_elapsed / 60:.2f} minutes.")
    print("=" * 60)

if __name__ == "__main__":
    main()
