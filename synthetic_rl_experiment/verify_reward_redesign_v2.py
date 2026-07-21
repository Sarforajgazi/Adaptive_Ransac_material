"""
Paired verification of the best-of-episode reward redesign against a real
old-reward baseline, both at fixed N=10000 (density held constant so the
comparison isolates the reward change, per the agreed methodology).

Leads with realized-improvement-rate (in what fraction of episodes did a
step after step 1 raise best_score, and by how much) rather than raw
steps/episode, because episode length alone conflates "the agent is
refining" with "the agent is wasting steps" and length-when-unwarranted
would be a different memorized policy, not evidence of adaptation.

Stratifies by the EXOGENOUS scene-difficulty variables (true_noise_sigma,
true_inlier_ratio) rather than step-1's own score, since step-1 score is
endogenous -- a bad step-1 draw can reflect RANSAC's own call-to-call noise
rather than a genuinely hard scene, so conditioning on it would partly
measure regression-to-the-mean rather than learned adaptive behavior. A
step-1-score bucket is still reported as a secondary view, with that
caveat attached.

Everything needed is read from the `info` dict each step.step() returns,
never from live env attributes -- SB3's VecEnv auto-resets the underlying
env the instant a step returns done=True, so reading env.best_score /
env.true_noise_sigma etc. from inside the callback after a terminal step
would already reflect the NEXT episode, not the one that just ended. The
info dict itself predates that reset and is safe.

No checkpoints saved -- this exists purely to decide whether the reward
redesign should go into Phase 6.
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

TOTAL_TIMESTEPS = 16384  # 8 PPO rollouts/updates at n_steps=2048, matches the prior pass


class RealizedImprovementCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rows = []
        self._ep_step = 0
        self._step1_best = None
        self._noise_sigma = None
        self._inlier_ratio = None

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._ep_step += 1
        if self._ep_step == 1:
            self._step1_best = info["score"]
            self._noise_sigma = info["true_noise_sigma"]
            self._inlier_ratio = info["true_inlier_ratio"]
        if self.locals["dones"][0]:
            final_best = info["score"]
            self.rows.append({
                "true_noise_sigma": self._noise_sigma,
                "true_inlier_ratio": self._inlier_ratio,
                "episode_length": self._ep_step,
                "step1_best": self._step1_best,
                "final_best": final_best,
                "realized_improvement": final_best - self._step1_best,
            })
            self._ep_step = 0
        return True


def run_condition(reward_mode):
    def make_env():
        return Monitor(SyntheticRansacEnv(max_steps=5, fixed_num_points=10000, reward_mode=reward_mode))

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=2048, batch_size=64,
                learning_rate=3e-4, ent_coef=0.01, device="cpu")

    callback = RealizedImprovementCallback()
    print(f"\n=== Running reward_mode='{reward_mode}', {TOTAL_TIMESTEPS} steps, fixed N=10000 ===")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    print(f"  {len(callback.rows)} episodes completed.")
    return callback.rows


def analyze(rows, label):
    # Pre-registered before seeing full-run results (see SESSION_PROGRESS_LOG.md):
    # calibrated from repeated single-step calls at IDENTICAL scene+params, which
    # isolates pure RANSAC call-to-call jitter (no policy, no generator randomness).
    # Measured jitter std ranged 0.0 (easy/dominant scenes) to 0.053 (hardest
    # scenes: low inlier_ratio, full clutter -- well within what domain
    # randomization actually samples). 0.05 sits at that worst-case jitter level,
    # so a "new best" clearing it is very unlikely to be pure noise even in the
    # hardest scenes, and is comfortably above jitter everywhere else.
    eps = 0.05
    n = len(rows)
    improved = [r for r in rows if r["realized_improvement"] > eps]
    rate = len(improved) / n if n else float("nan")
    mean_gain_all = np.mean([r["realized_improvement"] for r in rows])
    mean_gain_when_improved = np.mean([r["realized_improvement"] for r in improved]) if improved else 0.0
    mean_len = np.mean([r["episode_length"] for r in rows])

    print(f"\n--- {label}: primary (realized improvement) ---")
    print(f"Episodes: {n}  Mean steps/episode: {mean_len:.3f}")
    print(f"Realized-improvement rate (fraction of episodes where a step "
          f"after step 1 raised best_score): {rate:.4f}")
    print(f"Mean improvement (all episodes, 0 if none): {mean_gain_all:.4f}")
    print(f"Mean improvement (episodes that improved only): {mean_gain_when_improved:.4f}")

    print(f"\n--- {label}: stratified by exogenous true_noise_sigma (terciles) ---")
    sigmas = np.array([r["true_noise_sigma"] for r in rows])
    edges = np.quantile(sigmas, [0, 1/3, 2/3, 1.0])
    for i in range(3):
        lo, hi = edges[i], edges[i+1]
        bucket = [r for r in rows if lo <= r["true_noise_sigma"] <= hi] if i == 2 else \
                 [r for r in rows if lo <= r["true_noise_sigma"] < hi]
        if not bucket:
            continue
        b_rate = sum(1 for r in bucket if r["realized_improvement"] > eps) / len(bucket)
        b_len = np.mean([r["episode_length"] for r in bucket])
        print(f"  noise_sigma [{lo:.3f}-{hi:.3f}): n={len(bucket):4d}  "
              f"improved-rate={b_rate:.4f}  mean_len={b_len:.3f}")

    print(f"\n--- {label}: stratified by exogenous true_inlier_ratio (terciles) ---")
    ratios = np.array([r["true_inlier_ratio"] for r in rows])
    edges = np.quantile(ratios, [0, 1/3, 2/3, 1.0])
    for i in range(3):
        lo, hi = edges[i], edges[i+1]
        bucket = [r for r in rows if lo <= r["true_inlier_ratio"] <= hi] if i == 2 else \
                 [r for r in rows if lo <= r["true_inlier_ratio"] < hi]
        if not bucket:
            continue
        b_rate = sum(1 for r in bucket if r["realized_improvement"] > eps) / len(bucket)
        b_len = np.mean([r["episode_length"] for r in bucket])
        print(f"  inlier_ratio [{lo:.3f}-{hi:.3f}): n={len(bucket):4d}  "
              f"improved-rate={b_rate:.4f}  mean_len={b_len:.3f}")

    print(f"\n--- {label}: SECONDARY view, bucketed by step-1 score (endogenous -- "
          f"regression-to-the-mean caveat applies, see script docstring) ---")
    s1 = np.array([r["step1_best"] for r in rows])
    median_s1 = np.median(s1)
    low_bucket = [r for r in rows if r["step1_best"] < median_s1]
    high_bucket = [r for r in rows if r["step1_best"] >= median_s1]
    for name, bucket in [("low step-1 score", low_bucket), ("high step-1 score", high_bucket)]:
        if not bucket:
            continue
        b_rate = sum(1 for r in bucket if r["realized_improvement"] > eps) / len(bucket)
        b_len = np.mean([r["episode_length"] for r in bucket])
        print(f"  {name} (n={len(bucket):4d}): improved-rate={b_rate:.4f}  mean_len={b_len:.3f}")

    return {"n": n, "rate": rate, "mean_gain_all": mean_gain_all, "mean_len": mean_len}


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    results = {}
    for reward_mode in ["old_flat", "best_of_episode"]:
        rows = run_condition(reward_mode)
        csv_path = os.path.join(LOG_DIR, f"reward_redesign_v2_{reward_mode}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["true_noise_sigma", "true_inlier_ratio",
                                               "episode_length", "step1_best", "final_best",
                                               "realized_improvement"])
            w.writeheader()
            w.writerows(rows)
        print(f"Raw per-episode data saved to {csv_path}")
        results[reward_mode] = analyze(rows, reward_mode)

    print("\n=== Head-to-head: old_flat vs best_of_episode ===")
    old, new = results["old_flat"], results["best_of_episode"]
    print(f"Realized-improvement rate: {old['rate']:.4f} -> {new['rate']:.4f}  "
          f"(delta {new['rate']-old['rate']:+.4f})")
    print(f"Mean improvement (all episodes): {old['mean_gain_all']:.4f} -> {new['mean_gain_all']:.4f}")
    print(f"Mean steps/episode: {old['mean_len']:.3f} -> {new['mean_len']:.3f}")


if __name__ == "__main__":
    main()
