import os
import argparse
from stable_baselines3 import PPO
from ransac_env import RansacEnv

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(WORKSPACE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_ransac_final.zip")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained PPO Agent")
    parser.add_argument("--env_path", type=str, default=None, help="Path to lidar .ply frames")
    parser.add_argument("--num_tests", type=int, default=5, help="Number of random frames to test")
    args = parser.parse_args()

    print("=" * 60)
    print("Starting Day 3: Evaluating Trained PPO Agent")
    print("=" * 60)

    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("You must complete the Day 2 training run first.")
        return

    # 2. Load the Agent
    print(f"[1/3] Loading trained model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    # 3. Initialize Environment
    print(f"[2/3] Initializing RansacEnv...")
    env = RansacEnv(data_dir=args.env_path)

    # 4. Evaluation Loop
    print(f"[3/3] Running Inference on {args.num_tests} random frames...")
    print("-" * 60)

    avg_reward = 0.0
    for i in range(args.num_tests):
        obs, info = env.reset()
        filename = os.path.basename(info['file'])
        
        print(f"\nTest {i+1}/{args.num_tests} | Frame: {filename}")
        print(f"Observation (Scene Features): {obs}")
        
        # Predict the best action (deterministic=True prevents random exploration)
        action, _states = model.predict(obs, deterministic=True)
        
        # Execute the action in the environment to get the results
        obs, reward, terminated, truncated, step_info = env.step(action)
        
        avg_reward += reward
        
        # Determine success
        if "error" in step_info:
            print(f"  [!] Failed: {step_info['error']}")
        else:
            pct = step_info.get('ground_pct', 0.0) * 100.0
            print(f"  -> Agent Choice: Epsilon={step_info['epsilon']:.4f}m, MinSupport={step_info['min_support']}, NormalThresh={step_info['normal_thresh']:.3f}")
            print(f"  -> Result: Ground identified = {pct:.2f}%, Horizontal Alignment = {step_info['z_align']:.4f}")
            print(f"  -> Generated Reward: {reward:.4f}")

    print("-" * 60)
    print(f"Evaluation Complete! Average Reward over {args.num_tests} frames: {avg_reward / args.num_tests:.4f}")

if __name__ == "__main__":
    main()
