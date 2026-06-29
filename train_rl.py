import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from ransac_env import RansacEnv

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
    args = parser.parse_args()

    print("=" * 60)
    print("Starting Day 2: PPO Agent Training")
    print("=" * 60)

    # 1. Initialize and Wrap the Environment
    print(f"[1/4] Initializing RansacEnv...")
    
    # We create a function to instantiate the environment so it can be vectorized
    def make_env():
        # Pass env_path if provided, else it defaults to Office
        env = RansacEnv(data_dir=args.env_path)
        # Monitor logs episode rewards, lengths, and other stats
        env = Monitor(env)
        return env

    # Vectorize the environment
    vec_env = DummyVecEnv([make_env])
    
    # 2. Setup the PPO Model
    print(f"[2/4] Setting up PPO Model (MlpPolicy)...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        tensorboard_log=LOG_DIR,
        device="auto" # Will use GPU if available, else CPU
    )

    # 3. Setup Checkpoint Callback
    print(f"[3/4] Configuring Callbacks...")
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=MODEL_DIR,
        name_prefix="ppo_ransac_model",
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
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save the final model
        final_model_path = os.path.join(MODEL_DIR, "ppo_ransac_final")
        model.save(final_model_path)
        print("-" * 60)
        print(f"Training Complete! Final model saved to: {final_model_path}.zip")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving current model state...")
        model.save(os.path.join(MODEL_DIR, "ppo_ransac_interrupted"))
        print("Interrupted model saved.")

if __name__ == "__main__":
    main()
