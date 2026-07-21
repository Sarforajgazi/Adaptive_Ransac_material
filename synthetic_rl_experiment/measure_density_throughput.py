"""
Timing-only pilot: measures training throughput (time/step) as a function of
true_num_points under the new log-uniform [10k, 100k] density randomization
(synthetic_env.py, reset()). The old 0.29s/step baseline (train_synthetic.py)
was measured under a fixed-10k distribution that no longer applies once
num_points varies per episode -- this collects (N, dt) pairs from one real
rollout and regresses time_per_step ~= a + b*N, instead of reporting a single
mean that would be biased by whatever mix of densities happened to land in a
short window.

Also records steps-per-episode under the current (pre-reward-redesign) pilot
training dynamics, as a "before" reference -- expected to rise once the
reward redesign restores iterative refinement, but useful to have measured
now rather than assumed.

No checkpoint is saved -- this run exists purely to produce timing numbers.
"""
import os
import csv
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from synthetic_env import SyntheticRansacEnv
from train_synthetic import LOG_DIR

N_STEPS = 2048  # one full PPO rollout -- the minimum unit of work model.learn() can do


class ThroughputCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.records = []  # (true_num_points, dt_seconds)
        self.episode_lengths = []
        self._cur_ep_len = 0
        self._last_t = None

    def _on_step(self) -> bool:
        now = time.time()
        if self._last_t is not None:
            dt = now - self._last_t
            n = self.training_env.venv.envs[0].unwrapped.true_num_points
            self.records.append((n, dt))
        self._last_t = now

        self._cur_ep_len += 1
        if self.locals["dones"][0]:
            self.episode_lengths.append(self._cur_ep_len)
            self._cur_ep_len = 0

        if len(self.records) % 200 == 0:
            print(f"  ...{len(self.records)}/{N_STEPS} steps timed", flush=True)
        return True


def main():
    def make_env():
        return Monitor(SyntheticRansacEnv(max_steps=5))

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=N_STEPS, batch_size=64,
                learning_rate=3e-4, ent_coef=0.01, device="cpu")

    callback = ThroughputCallback()
    print(f"Running {N_STEPS} timesteps under the new density distribution (log-uniform 10k-100k)...")
    t0 = time.time()
    model.learn(total_timesteps=N_STEPS, callback=callback)
    total_elapsed = time.time() - t0
    print(f"Done. Total wall-clock: {total_elapsed:.1f}s over {N_STEPS} steps "
          f"({total_elapsed / N_STEPS:.4f}s/step naive mean, includes 1 PPO update).")

    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = os.path.join(LOG_DIR, "density_throughput_pilot.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_num_points", "step_time_seconds"])
        w.writerows(callback.records)
    print(f"Raw per-step timing saved to {csv_path}")

    records = np.array(callback.records)
    n_vals = records[:, 0]
    dt_vals = records[:, 1]

    # time_per_step ~= a + b * N
    b, a = np.polyfit(n_vals, dt_vals, 1)
    pred = a + b * n_vals
    ss_res = np.sum((dt_vals - pred) ** 2)
    ss_tot = np.sum((dt_vals - dt_vals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("\n--- Cost model: time_per_step ~= a + b * N ---")
    print(f"a (intercept, s) = {a:.5f}")
    print(f"b (slope, s/point) = {b:.8f}")
    print(f"R^2 = {r2:.4f}  (n={len(records)} step observations)")

    for n_ref, label in [(10000, "10k"), (39086, "E[N] log-uniform 10k-100k"),
                          (31623, "geomean 10k-100k"), (100000, "100k")]:
        print(f"  predicted time/step at N={n_ref} ({label}): {a + b * n_ref:.4f}s")

    lengths = np.array(callback.episode_lengths)
    print(f"\n--- Episode length (current pilot-era dynamics, 'before' reference) ---")
    print(f"Episodes completed: {len(lengths)}")
    if len(lengths) > 0:
        print(f"Mean steps/episode: {lengths.mean():.3f}  (max_steps=5, min=1)")
        print(f"Distribution: {np.bincount(lengths, minlength=6)[1:6].tolist()} "
              f"(counts for length 1,2,3,4,5)")

    # Budget projection using the fitted cost model at the true expected N
    # under log-uniform sampling: E[N] = (H-L)/ln(H/L), NOT the geometric
    # mean (which is the median of a log-uniform, not the mean).
    L, H = 10000, 100000
    e_n = (H - L) / np.log(H / L)
    time_per_step_expected = a + b * e_n
    mean_ep_len = lengths.mean() if len(lengths) > 0 else float("nan")
    print(f"\nE[N] under log-uniform[{L},{H}] = {e_n:.0f} "
          f"(vs geomean {np.sqrt(L*H):.0f}, which is the median not the mean)")
    print(f"Projected time/step at E[N]: {time_per_step_expected:.4f}s")
    for target_steps in [50000, 100000]:
        est_hours = target_steps * time_per_step_expected / 3600
        print(f"Projected wall-clock for {target_steps} timesteps: {est_hours:.2f}h "
              f"(mean_ep_len={mean_ep_len:.2f} steps/ep, this pilot's dynamics -- "
              f"will shift if the reward redesign restores iterative refinement)")


if __name__ == "__main__":
    main()
