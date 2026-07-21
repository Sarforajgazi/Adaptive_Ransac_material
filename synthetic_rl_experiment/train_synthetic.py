import os
import sys
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from synthetic_env import SyntheticRansacEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")


def run_name_for(tag):
    """Single naming rule shared by train/evaluate/plot: 'synthetic_ppo' or
    'synthetic_ppo_<tag>' (e.g. 'v2'). Every artifact for a run -- model,
    VecNormalize stats, training log, eval CSV, plot -- derives its filename
    from this one string, so nothing has to be kept in sync by hand."""
    return f"synthetic_ppo_{tag}" if tag else "synthetic_ppo"


class Tee:
    """Mirrors writes to multiple streams -- lets SB3's own verbose=1 output
    and this script's prints land in both the console and a run-named log
    file, without changing every individual print call."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_eps = []
        self.nan_reward_steps = 0

    def _on_step(self) -> bool:
        import numpy as np
        rewards = self.locals.get("rewards")
        if rewards is not None and np.any(np.isnan(rewards)):
            self.nan_reward_steps += 1

        if "dones" in self.locals and self.locals["dones"][0]:
            info = self.locals["infos"][0]
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
            if "score" in info:
                self.episode_scores.append(info["score"])
            if "epsilon" in info:
                self.episode_eps.append(info["epsilon"])

            if len(self.episode_rewards) % 10 == 0:
                mean_rew = sum(self.episode_rewards[-10:]) / 10
                mean_score = sum(self.episode_scores[-10:]) / 10
                recent_eps = self.episode_eps[-10:]
                print(f"Episodes: {len(self.episode_rewards)} | Mean Reward (last 10): {mean_rew:.3f} | Mean Score: {mean_score:.3f} | Last-10 eps: {recent_eps} | NaN reward steps so far: {self.nan_reward_steps}")
                import sys
                sys.stdout.flush()
        return True

def train(timesteps, run_name, force=False):
    model_path = os.path.join(MODEL_DIR, run_name)
    vecnormalize_path = model_path + "_vecnormalize.pkl"
    log_path = os.path.join(LOG_DIR, run_name + "_train.log")

    if os.path.exists(model_path + ".zip") and not force:
        raise SystemExit(f"Refusing to overwrite existing model at {model_path}.zip -- pass a different --tag, or --force to overwrite.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    try:
        # Optional: wrap with Monitor to get episode info for callback
        from stable_baselines3.common.monitor import Monitor

        def make_env():
            return Monitor(SyntheticRansacEnv(max_steps=5))

        # VecNormalize + ent_coef: without these, PPO previously collapsed to
        # a single constant action regardless of state (see syntheticRL.md
        # sec.4) -- the same failure mode ransac_env.py/train_rl.py hit and
        # fixed with this exact combination. ent_coef adds an entropy bonus
        # that discourages the policy from settling on one deterministic
        # output; VecNormalize is needed because the 33-dim obs vector mixes
        # wildly different scales (e.g. bbox_volume vs. a 0-1
        # normal_consistency ratio), which can otherwise swamp the small
        # state-dependent signal.
        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # CPU, not "auto": measured per-step cost is 100% CPU-bound (Open3D
        # feature computation + C++ RANSAC call, ~196ms/step combined), and
        # the MlpPolicy itself is far too small for GPU transfer overhead to
        # pay off.
        model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=2048, batch_size=64,
                    learning_rate=3e-4, ent_coef=0.01, device="cpu")

        print(f"Run name: {run_name}")
        print(f"Starting training for {timesteps} timesteps...")
        callback = LoggingCallback()

        import time
        t0 = time.time()
        model.learn(total_timesteps=timesteps, callback=callback)
        elapsed = time.time() - t0

        model.save(model_path)
        vec_env.save(vecnormalize_path)
        print(f"Model saved to {model_path}.zip")
        print(f"VecNormalize stats saved to {vecnormalize_path}")
        print(f"Wall-clock: {elapsed:.1f}s total, {elapsed/timesteps:.4f} s/step ({timesteps} steps)")
        print(f"Unique eps values chosen across all {len(callback.episode_eps)} episodes: {sorted(set(callback.episode_eps))}")
        print(f"Last 20 episodes' chosen eps: {callback.episode_eps[-20:]}")
        print(f"Total steps with a NaN reward: {callback.nan_reward_steps}")
    finally:
        sys.stdout = original_stdout
        log_file.close()
    print(f"Training log saved to {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--tag", type=str, default=None,
                         help="Names this run distinctly (e.g. 'v2') so its model, VecNormalize "
                              "stats, and training log never collide with or overwrite a previous "
                              "run's files. Without a tag, uses the plain 'synthetic_ppo' name.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing model with the same tag.")
    args = parser.parse_args()

    train(timesteps=args.timesteps, run_name=run_name_for(args.tag), force=args.force)
