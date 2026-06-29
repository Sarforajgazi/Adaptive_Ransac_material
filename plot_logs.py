import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_logs(log_dir, output_path):
    # Find the most recent event file
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        print("No event files found in", log_dir)
        return

    # Sort to get the latest
    event_file = sorted(event_files)[-1]
    print(f"Reading logs from: {event_file}")

    # Load the event file
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # Extract data
    tags = event_acc.Tags()['scalars']
    
    if 'rollout/ep_rew_mean' not in tags:
        print("Required tags not found in the logs.")
        return

    rew_events = event_acc.Scalars('rollout/ep_rew_mean')
    
    rew_steps = [x.step for x in rew_events]
    rew_values = [x.value for x in rew_events]

    # Plot
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot Reward
    ax1.plot(rew_steps, rew_values, color='#00ff99', linewidth=2.5, marker='o')
    ax1.set_xlabel('Timesteps', fontsize=12)
    ax1.set_ylabel('Mean Episode Reward', color='#00ff99', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#00ff99')
    ax1.set_title('PPO Training Progress (Day 2)', fontsize=14, pad=15)
    ax1.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    WORKSPACE = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(WORKSPACE, "logs")
    # Save directly to the Gemini artifacts directory so it can be embedded
    OUT_PATH = r"C:\Users\sarfo\.gemini\antigravity-ide\brain\719484d7-2378-4009-a9fc-3574afcee4d9\reward_graph.png"
    
    plot_tensorboard_logs(LOG_DIR, OUT_PATH)
