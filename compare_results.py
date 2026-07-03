import os
import glob
import pandas as pd
from download_lidar_frames import ENVIRONMENTS

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORKSPACE, "logs")
ARTIFACTS_DIR = r"C:\Users\sarfo\.gemini\antigravity-ide\brain\0fd74363-6560-41d1-9fa2-c8c74db59e8a"

def main():
    modes = ["strict", "standard", "loose", "rl"]
    results = []
    
    for env in ENVIRONMENTS:
        for mode in modes:
            csv_path = os.path.join(LOG_DIR, f"{env}_{mode}.csv")
            if not os.path.exists(csv_path):
                continue
                
            df = pd.read_csv(csv_path)
            
            # For RL, multiple steps per frame might exist. Take the terminal step per frame.
            if mode == "rl":
                # The terminal step has reward != 0.0 or is the last step for that frame
                # RansacEnv sets reward = 0 for intermediate steps
                # Alternatively, group by frame_id and take the row with max steps_used
                df = df.sort_values(['frame_id', 'steps_used']).groupby('frame_id').tail(1)
            
            num_frames = len(df)
            if num_frames == 0:
                continue
                
            mean_inlier = df['inlier_ratio'].mean()
            bad_frames = len(df[df['inlier_ratio'] < 0.01])
            mean_residual = df['residual'].mean()
            mean_steps = df['steps_used'].mean()
            total_runtime = df['runtime'].sum()
            
            results.append({
                "Dataset": env,
                "Mode": mode,
                "Frames": num_frames,
                "Mean Inlier": mean_inlier,
                "Bad Frames": bad_frames,
                "Mean Residual": mean_residual,
                "Mean Steps": mean_steps,
                "Total Runtime (s)": total_runtime
            })
            
    if not results:
        print("No evaluation CSVs found in logs/.")
        return
        
    df_results = pd.DataFrame(results)
    
    # Generate Markdown Table
    md_lines = ["# Agent vs Baseline Evaluation\n"]
    md_lines.append("| Dataset | Mode | Frames | Mean Inlier Ratio | Bad Frames (<1%) | Mean Residual | Mean Steps | Total Runtime (s) |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    
    for _, row in df_results.iterrows():
        md_lines.append(f"| {row['Dataset']} | {row['Mode']} | {row['Frames']} | {row['Mean Inlier']:.4f} | {row['Bad Frames']} | {row['Mean Residual']:.4f} | {row['Mean Steps']:.2f} | {row['Total Runtime (s)']:.2f} |")
    
    md_lines.append("\n## Overall Summary\n")
    md_lines.append("| Mode | Avg Inlier Ratio | Total Bad Frames | Avg Residual | Avg Steps |")
    md_lines.append("|---|---|---|---|---|")
    
    for mode in modes:
        mode_data = df_results[df_results["Mode"] == mode]
        if len(mode_data) > 0:
            avg_inlier = mode_data["Mean Inlier"].mean()
            tot_bad = mode_data["Bad Frames"].sum()
            avg_residual = mode_data["Mean Residual"].mean()
            avg_steps = mode_data["Mean Steps"].mean()
            md_lines.append(f"| {mode} | {avg_inlier:.4f} | {tot_bad} | {avg_residual:.4f} | {avg_steps:.2f} |")
    
    md_content = "\n".join(md_lines)
    print(md_content)
    
    # Save to artifact directory
    out_path = os.path.join(ARTIFACTS_DIR, "evaluation_summary.md")
    with open(out_path, "w") as f:
        f.write(md_content)
    print(f"\nSummary saved to {out_path}")

if __name__ == "__main__":
    main()
