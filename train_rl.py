import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from ransac_env import RansacEnv
from download_lidar_frames import NEVER_TRAIN_ENVIRONMENTS

# Define paths
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORKSPACE, "logs")
MODEL_DIR = os.path.join(WORKSPACE, "models")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for Adaptive RANSAC")
    parser.add_argument("--timesteps", type=int, default=5000, help="Total training timesteps")
    parser.add_argument("--env_path", type=str, default=None, help="Path to lidar .ply frames")
    parser.add_argument("--load", type=str, default=None, help="Path to a saved model .zip to resume training")
    parser.add_argument("--vecnormalize", type=str, default=None, help="Path to a saved VecNormalize .pkl to resume with matching normalization stats")
    parser.add_argument("--tag", type=str, default=None,
                         help="Distinguishes this run's checkpoints/final model/training log from other "
                              "tagged runs (e.g. 'v2_reward') -- without a tag, uses the original untagged "
                              "filenames (ppo_ransac_final.zip, evaluation_metrics.csv), so existing runs "
                              "are never overwritten by a new tagged run.")
    parser.add_argument("--exclude_envs", type=str, default=None,
                         help="Comma-separated environment names to ADDITIONALLY exclude from training, on "
                              "top of NEVER_TRAIN_ENVIRONMENTS (held-out generalization-test scenes + "
                              "RELLIS-3D's real-world ground-truth data), which is always excluded regardless "
                              "of this flag -- no need to type those out, and they can't be accidentally "
                              "included even if this is left unset.")
    parser.add_argument("--split", type=str, default=None, choices=["train", "test"],
                         help="Frame-level train/test split within each included environment (chronological "
                              "tail holdout, see RansacEnv). Omit to train on every frame of every included "
                              "environment (original behavior).")
    parser.add_argument("--test_frac", type=float, default=0.15,
                         help="Fraction of each environment's frames (chronological tail) reserved for "
                              "split='test', when --split is set.")
    parser.add_argument("--gap_frac", type=float, default=0.02,
                         help="Fraction of each environment's frames dropped as a buffer between train and "
                              "test (right before the test tail), so a near-duplicate frame pair can't "
                              "straddle the train/test boundary. Only applies when --split is set.")
    args = parser.parse_args()

    user_excludes = args.exclude_envs.split(",") if args.exclude_envs else []
    exclude_envs = sorted(set(NEVER_TRAIN_ENVIRONMENTS) | set(user_excludes))

    suffix = f"_{args.tag}" if args.tag else ""
    model_prefix = f"ppo_ransac{suffix}_model"
    final_name = f"ppo_ransac{suffix}_final"
    interrupted_name = f"ppo_ransac{suffix}_interrupted"
    training_log_name = f"evaluation_metrics{suffix}.csv"

    print("=" * 60)
    print("Starting Day 2: PPO Agent Training")
    print("=" * 60)

    # 1. Initialize and Wrap the Environment
    print(f"[1/4] Initializing RansacEnv...")
    
    # We create a function to instantiate the environment so it can be vectorized
    def make_env():
        # Pass env_path if provided, else it defaults to Office
        env = RansacEnv(data_dir=args.env_path, log_name=training_log_name,
                         exclude_envs=exclude_envs, split=args.split, test_frac=args.test_frac,
                         gap_frac=args.gap_frac)
        # Monitor logs episode rewards, lengths, and other stats
        env = Monitor(env)
        return env

    # Vectorize the environment
    vec_env = DummyVecEnv([make_env])

    # Normalize observations (the 31-dim state mixes wildly different scales,
    # e.g. bbox_volume vs normal_consistency) and rewards. Resuming a run must
    # load the same running stats the model was trained with, not start fresh.
    if args.vecnormalize:
        print(f"Loading VecNormalize stats from: {args.vecnormalize}")
        vec_env = VecNormalize.load(args.vecnormalize, vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 2. Setup the PPO Model
    print(f"[2/4] Setting up PPO Model (MlpPolicy)...")
    if args.load:
        print(f"Loading existing model from: {args.load}")
        model = PPO.load(
            args.load,
            env=vec_env,
            tensorboard_log=LOG_DIR,
            device="auto"
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=0.0003,
            ent_coef=0.01,
            tensorboard_log=LOG_DIR,
            device="auto" # Will use GPU if available, else CPU
        )

    # 3. Setup Checkpoint Callback
    print(f"[3/4] Configuring Callbacks...")
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=MODEL_DIR,
        name_prefix=model_prefix,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # 4. Start Training
    print(f"[4/4] Starting Training Loop for {args.timesteps} timesteps...")
    print(f"TensorBoard logs will be saved to: {LOG_DIR}")
    print("-" * 60)
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=(args.load is None),
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save the final model + its matching normalization stats
        final_model_path = os.path.join(MODEL_DIR, final_name)
        model.save(final_model_path)
        vec_env.save(os.path.join(MODEL_DIR, f"{final_name}_vecnormalize.pkl"))
        print("-" * 60)
        print(f"Training Complete! Final model saved to: {final_model_path}.zip")

    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving current model state...")
        model.save(os.path.join(MODEL_DIR, interrupted_name))
        vec_env.save(os.path.join(MODEL_DIR, f"{interrupted_name}_vecnormalize.pkl"))
        print("Interrupted model saved.")

if __name__ == "__main__":
    main()
