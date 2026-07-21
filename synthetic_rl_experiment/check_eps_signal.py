import argparse
import numpy as np
import scipy.stats as stats
from synthetic_env import SyntheticRansacEnv, EPS_LEVELS, NORM_THRESH_LEVELS

def check_eps_signal(num_seeds=250, full_complexity=False):
    env = SyntheticRansacEnv(max_steps=5)

    noises = [0.01, 0.20]
    # Sweep the real action space, not a hand-copied list -- a previous local
    # copy here drifted out of sync with synthetic_env.py's EPS_LEVELS and
    # silently mislabeled every result (index 0 was printed as "0.05" but the
    # environment actually ran 0.08, etc).
    eps_indices = list(range(len(EPS_LEVELS)))
    eps_vals = EPS_LEVELS
    nth_idx = NORM_THRESH_LEVELS.index(0.75) # 0.75
    min_supp_idx = 2 # 75

    scene_options = {"inlier_ratio": 0.50}
    if full_complexity:
        # Force Phase 2/4/5 features ON (not the default reset() random
        # inclusion, which would dilute easy/hard scenes together) -- this is
        # a worst-case stress test of whether the eps-vs-noise gradient
        # survives the full generator, not a representative-average check.
        scene_options.update({"num_bumps": 2, "num_cylinders": 1, "num_boxes": 1, "add_intersecting_plane": True})

    label = "Full-complexity (bumps+clutter+intersecting plane forced ON)" if full_complexity else "Baseline"
    print(f'Starting EPS Signal Check ({label} Environment, {num_seeds} seeds/eps)...')
    for noise in noises:
        print(f'\n=== Noise: {noise:.2f} ===')

        scores_by_seed = {eps_i: [] for eps_i in eps_indices}
        fi_by_seed = {eps_i: [] for eps_i in eps_indices}

        for seed in range(num_seeds):
            for eps_i in eps_indices:
                # noise_sigma/inlier_ratio must be pinned via reset(options=...) --
                # env.reset() ignores plain instance attributes and always
                # re-samples both uniformly, which silently broke this sweep
                # before (both "noise" arms ended up drawing from the same full
                # [0.01, 0.20] random range instead of a fixed value).
                env.reset(options={
                    **scene_options,
                    "noise_sigma": noise,
                    "generator_seed": seed,
                })

                action = np.array([eps_i, min_supp_idx, nth_idx, 1])
                obs, reward, done, _, info = env.step(action)

                scores_by_seed[eps_i].append(info['score'])
                fi_by_seed[eps_i].append(info['false_inlier_rate'])

        # Find the best eps average score
        avgs = {eps_i: np.mean(scores_by_seed[eps_i]) for eps_i in eps_indices}
        best_eps_idx = max(avgs, key=avgs.get)

        print(f"Scores & False Inlier Rates:")
        for eps_i in eps_indices:
            avg_score = avgs[eps_i]
            se_score = np.std(scores_by_seed[eps_i]) / np.sqrt(num_seeds)
            avg_fi = np.mean(fi_by_seed[eps_i])
            print(f'Eps {eps_vals[eps_i]:.2f} | Score: {avg_score:.4f} +/- {se_score:.4f} | FI: {avg_fi:.4f}')

        print(f"\nSignal Magnitude (vs worst eps):")
        worst_eps_idx = min(avgs, key=avgs.get)
        signal = avgs[best_eps_idx] - avgs[worst_eps_idx]
        print(f"Max Gradient: {signal:.4f} (from Eps {eps_vals[worst_eps_idx]:.2f} to Eps {eps_vals[best_eps_idx]:.2f})")

        # Paired t-test between best and worst
        diffs = np.array(scores_by_seed[best_eps_idx]) - np.array(scores_by_seed[worst_eps_idx])
        se_diff = np.std(diffs, ddof=1) / np.sqrt(num_seeds)
        print(f"Paired Diff SE: {se_diff:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=250, help="Seeds per eps level (250 = original rigor, slower).")
    parser.add_argument("--full_complexity", action="store_true", help="Force bumps+clutter+intersecting plane ON (Phase 2/4/5 stress test).")
    args = parser.parse_args()
    check_eps_signal(num_seeds=args.seeds, full_complexity=args.full_complexity)
