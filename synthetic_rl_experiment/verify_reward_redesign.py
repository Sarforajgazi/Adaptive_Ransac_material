"""
Verification-only pass for the reward redesign in synthetic_env.py's step()
(asymmetric step penalty -- see the comment there and SESSION_PROGRESS_LOG.md).
Checks whether mean steps/episode actually climbs off the one-shot-collapse
baseline (1.88 steps/episode, measured under the OLD flat-penalty reward in
the density throughput pilot) once real gradient updates happen under the
NEW reward.

Density is pinned at a fixed 10000 (fixed_num_points=10000) rather than the
new log-uniform 10k-100k default, per the agreed methodology: keep the
reward before/after comparison at a fixed density so that axis stays clean,
and treat density generalization as a separate, additive result measured
later. This also keeps the run's per-step cost at the original ~0.29s/step
rather than the density-mixed ~0.525s/step.

No checkpoint is saved -- this exists purely to check whether the reward
change moves episode length before committing to the full retrain.
"""
import os
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from synthetic_env import SyntheticRansacEnv
from train_synthetic import LOG_DIR

TOTAL_TIMESTEPS = 16384  # 8 PPO rollouts/updates at n_steps=2048
BLOCK_SIZE = 50  # episodes per reported block, to show the trend over training


class EpisodeLengthCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_lengths = []
        self._cur_ep_len = 0

    def _on_step(self) -> bool:
        self._cur_ep_len += 1
        if self.locals["dones"][0]:
            self.episode_lengths.append(self._cur_ep_len)
            self._cur_ep_len = 0
        return True


def main():
    def make_env():
        return Monitor(SyntheticRansacEnv(max_steps=5, fixed_num_points=10000))

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=2048, batch_size=64,
                learning_rate=3e-4, ent_coef=0.01, device="cpu")

    callback = EpisodeLengthCallback()
    print(f"Running {TOTAL_TIMESTEPS} timesteps at fixed N=10000 under the new reward...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    lengths = np.array(callback.episode_lengths)
    print(f"\nEpisodes completed: {len(lengths)}")
    print(f"Overall mean steps/episode: {lengths.mean():.3f} "
          f"(old-reward baseline, different run: 1.881)")
    print(f"Overall distribution (lengths 1-5): {np.bincount(lengths, minlength=6)[1:6].tolist()}")

    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = os.path.join(LOG_DIR, "reward_redesign_verification.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_index", "length"])
        w.writerows(enumerate(lengths))
    print(f"Raw per-episode lengths saved to {csv_path}")

    print(f"\n--- Trend over training (block size = {BLOCK_SIZE} episodes) ---")
    n_blocks = len(lengths) // BLOCK_SIZE
    for i in range(n_blocks):
        block = lengths[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
        dist = np.bincount(block, minlength=6)[1:6].tolist()
        print(f"  Episodes {i*BLOCK_SIZE:4d}-{(i+1)*BLOCK_SIZE:4d}: "
              f"mean={block.mean():.3f}  dist(1-5)={dist}")
    remainder = lengths[n_blocks * BLOCK_SIZE:]
    if len(remainder) > 0:
        dist = np.bincount(remainder, minlength=6)[1:6].tolist()
        print(f"  Episodes {n_blocks*BLOCK_SIZE:4d}-{len(lengths):4d}: "
              f"mean={remainder.mean():.3f}  dist(1-5)={dist}")

    if n_blocks >= 2:
        first_mean = lengths[:BLOCK_SIZE].mean()
        last_block_start = (n_blocks - 1) * BLOCK_SIZE
        last_mean = lengths[last_block_start:last_block_start + BLOCK_SIZE].mean()
        print(f"\nFirst block mean: {first_mean:.3f}  |  Last full block mean: {last_mean:.3f}"
              f"  |  Delta: {last_mean - first_mean:+.3f}")


if __name__ == "__main__":
    main()
