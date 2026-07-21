import os
import subprocess
from download_lidar_frames import HELD_OUT_ENVIRONMENTS

print("Starting held-out environments evaluation...")

for env in HELD_OUT_ENVIRONMENTS:
    print(f"\n=======================================================")
    print(f"EVALUATING HELD-OUT ENVIRONMENT: {env}")
    print(f"=======================================================")
    
    # 1. Run baselines
    for mode in ["strict", "standard", "loose"]:
        print(f"Running {mode} baseline for {env}...")
        subprocess.run([".venv\\Scripts\\python.exe", "baseline_evaluator.py", mode, "--env", env], check=True)
    
    # 2. Run RL model
    print(f"Running RL model (v4_model) for {env}...")
    subprocess.run([
        ".venv\\Scripts\\python.exe", "rl_evaluator.py", 
        "--env", env, 
        "--model", "models/ppo_ransac_v4_model_final.zip", 
        "--vecnormalize", "models/ppo_ransac_v4_model_final_vecnormalize.pkl", 
        "--tag", "v4_model"
    ], check=True)

# Generate final comparison chart automatically at the end
print("\nEvaluation complete! Generating comparison charts...")
subprocess.run([".venv\\Scripts\\python.exe", "compare_results.py"])
subprocess.run([".venv\\Scripts\\python.exe", "per_frame_comparison.py"])
subprocess.run([".venv\\Scripts\\python.exe", "plot_comparison.py"])

print("Done! Check plots/ directory for the latest charts.")
