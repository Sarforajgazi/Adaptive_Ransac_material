import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/evaluation_metrics_v4_model.csv")
df['rolling_reward'] = df['reward'].rolling(1000).mean()
df['rolling_inlier'] = df['inlier_ratio'].rolling(1000).mean()
df['rolling_steps'] = df['steps_used'].rolling(1000).mean()

plt.style.use('dark_background')

plt.figure(figsize=(10, 6))
plt.plot(df['rolling_reward'], color='#00ff99', label='Rolling Mean Reward (1000 steps)')
plt.title('Training Progress (v4_model)')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.savefig(r'C:\Users\sarfo\.gemini\antigravity-ide\brain\a42672da-8d35-484c-9394-95a425a28c16\v4_reward.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(df['rolling_inlier'], color='#2a78d6', label='Rolling Mean Inlier Ratio (1000 steps)')
plt.title('Inlier Ratio Progress (v4_model)')
plt.xlabel('Steps')
plt.ylabel('Inlier Ratio')
plt.legend()
plt.savefig(r'C:\Users\sarfo\.gemini\antigravity-ide\brain\a42672da-8d35-484c-9394-95a425a28c16\v4_inlier.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(df['rolling_steps'], color='#d6722a', label='Rolling Mean Steps (1000 steps)')
plt.title('Average Steps Used (v4_model)')
plt.xlabel('Steps')
plt.ylabel('Steps Used')
plt.legend()
plt.savefig(r'C:\Users\sarfo\.gemini\antigravity-ide\brain\a42672da-8d35-484c-9394-95a425a28c16\v4_steps.png')
plt.close()

print("Plots generated successfully!")
