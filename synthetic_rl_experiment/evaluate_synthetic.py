import os
import csv
import glob
import pickle
import argparse
import numpy as np
from stable_baselines3 import PPO
from synthetic_env import SyntheticRansacEnv, EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS
from train_synthetic import run_name_for, MODEL_DIR, LOG_DIR

def run_baseline(env, eps_val, min_supp_val, norm_th_val, reset_options):
    # eps_val is absolute meters (EPS_LEVELS is currently absolute -- the
    # scale-relative version is shelved until the real-point-cloud phase, see
    # synthetic_env.py). min_supp_val is an absolute point count but
    # MIN_SUPPORT_LEVELS is a fraction of point count, so that one still needs
    # reset() first to know the scene's actual size.
    obs, info = env.reset(options=reset_options)

    eps_idx = np.argmin(np.abs(np.array(EPS_LEVELS) - eps_val))
    min_supp_frac = min_supp_val / len(env.current_points)
    min_supp_idx = np.argmin(np.abs(np.array(MIN_SUPPORT_LEVELS) - min_supp_frac))
    norm_th_idx = np.argmin(np.abs(np.array(NORM_THRESH_LEVELS) - norm_th_val))

    action = [eps_idx, min_supp_idx, norm_th_idx, 0] # 0 = stop immediately
    _, _, done, _, info = env.step(action)
    return info

def run_rl(env, model, reset_options, normalize_obs):
    obs, info = env.reset(options=reset_options)
    done = False
    final_info = None
    while not done:
        action, _ = model.predict(normalize_obs(obs), deterministic=True)
        obs, reward, done, _, final_info = env.step(action)
    return final_info

def load_obs_normalizer(vecnormalize_path):
    """
    If the model was trained inside VecNormalize(norm_obs=True), its policy
    only makes sense on observations scaled the same way. Replicates SB3's
    VecNormalize._normalize_obs formula from the saved running stats.
    """
    if vecnormalize_path is None or not os.path.exists(vecnormalize_path):
        print("No VecNormalize stats found -- using raw observations "
              "(only correct if the model was NOT trained with VecNormalize).")
        return lambda obs: obs

    with open(vecnormalize_path, "rb") as f:
        vec_normalize = pickle.load(f)
    obs_rms = vec_normalize.obs_rms
    clip_obs = vec_normalize.clip_obs
    epsilon = vec_normalize.epsilon

    def normalize(obs):
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -clip_obs, clip_obs).astype(np.float32)

    return normalize

def resolve_model_path(tag):
    """
    --tag given: must match a trained run exactly (models/synthetic_ppo_<tag>.zip),
    errors out rather than silently falling back to a different model.
    --tag omitted: auto-detects the most recently modified models/synthetic_ppo*.zip,
    so this never needs its candidate list hand-edited as new tags are trained.
    """
    if tag:
        path = os.path.join(MODEL_DIR, run_name_for(tag) + ".zip")
        if not os.path.exists(path):
            raise SystemExit(f"No model found at {path} -- check --tag, or train it first with train_synthetic.py --tag {tag}.")
        return path

    candidates = sorted(glob.glob(os.path.join(MODEL_DIR, "synthetic_ppo*.zip")), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def evaluate(tag):
    model_path = resolve_model_path(tag)

    if model_path is not None:
        vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
        print(f"Using RL model: {model_path}")
        model = PPO.load(model_path, device="cpu")
        normalize_obs = load_obs_normalizer(vecnormalize_path)
        run_name = os.path.splitext(os.path.basename(model_path))[0]
    else:
        print(f"Warning: no model found in {MODEL_DIR}. RL evaluation will fail.")
        model = None
        normalize_obs = lambda obs: obs
        run_name = run_name_for(tag)

    env = SyntheticRansacEnv(max_steps=5)

    noise_levels = np.linspace(0.01, 0.20, 5)
    inlier_ratios = np.linspace(0.03, 0.9, 5)
    num_trials_per_bin = 20

    os.makedirs(LOG_DIR, exist_ok=True)
    out_csv = os.path.join(LOG_DIR, run_name + "_eval.csv")

    # Baselines calibrated to cloud geometry:
    # Mean inlier NN distance ≈ 0.08m, so eps must be ≥ 0.08 to accumulate support.
    # normal_thresh loosened to 0.60–0.85 to tolerate Schnabel KNN normal estimation noise.
    baselines = {
        "Super Strict": (0.08,  150, 0.85),   # tightest eps ≥ NN dist; high angle fidelity
        "Strict":       (0.10,  100, 0.80),
        "Standard":     (0.20,   75, 0.75),
        "Loose":        (0.40,   50, 0.65),
    }

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "noise_sigma", "inlier_ratio", "trial",
            "plane_type", "noise_type", "has_bumps", "has_clutter", "has_intersecting_plane",
            "eps", "min_supp", "norm_th",
            "angle_error", "offset_error", "inlier_recovery_rate", "false_inlier_rate",
            "achieved_inlier_ratio", "score"
        ])

        for noise in noise_levels:
            for ir in inlier_ratios:
                print(f"Evaluating bin: Noise={noise:.2f}, InlierRatio={ir:.2f}")
                for trial in range(num_trials_per_bin):
                    # For a fair comparison, RL and each baseline must see the
                    # IDENTICAL scene -- not just the same noise/inlier_ratio,
                    # but the same plane orientation, noise model, and
                    # bump/clutter/intersecting-plane composition. Previously
                    # these weren't in reset_options at all, so each method's
                    # own independent reset() call inside run_baseline/run_rl
                    # drew them from the env's continuously-advancing
                    # self.np_random stream -- silently giving baselines and
                    # RL DIFFERENT scene compositions even within "the same
                    # trial." Drawing them here, once, from a seeded RNG fixes
                    # that and makes the trial fully reproducible.
                    seed = int(noise * 1000 + ir * 100 + trial)
                    scene_rng = np.random.default_rng(seed)
                    orientation = str(scene_rng.choice(["ground", "ground", "vertical"]))
                    noise_type = str(scene_rng.choice(["gaussian", "gaussian", "laplacian", "uniform", "spatially_varying", "mixed"]))
                    num_bumps = int(scene_rng.choice([0, 0, 0, 1, 2, 3]))
                    num_cylinders = int(scene_rng.choice([0, 0, 0, 1, 1, 2]))
                    num_boxes = int(scene_rng.choice([0, 0, 0, 0, 1]))
                    add_intersecting_plane = bool(scene_rng.random() < 0.25)

                    has_bumps = num_bumps > 0
                    has_clutter = (num_cylinders > 0) or (num_boxes > 0)

                    reset_options = {
                        "inlier_ratio": ir,
                        "noise_sigma": noise,
                        "slope_angle_deg": 10.0,
                        "orientation": orientation,
                        "noise_type": noise_type,
                        "num_bumps": num_bumps,
                        "num_cylinders": num_cylinders,
                        "num_boxes": num_boxes,
                        "add_intersecting_plane": add_intersecting_plane,
                        "generator_seed": seed,
                    }
                    scene_cols = [orientation, noise_type, has_bumps, has_clutter, add_intersecting_plane]

                    # Evaluate Baselines
                    for b_name, (b_eps, b_supp, b_norm) in baselines.items():
                        info = run_baseline(env, b_eps, b_supp, b_norm, reset_options)

                        writer.writerow([
                            b_name, noise, ir, trial,
                            *scene_cols,
                            info.get("epsilon", b_eps),
                            info.get("min_support", b_supp),
                            info.get("normal_thresh", b_norm),
                            info.get("angle_error", 180.0),
                            info.get("offset_error", 10.0),
                            info.get("inlier_recovery_rate", 0.0),
                            info.get("false_inlier_rate", 1.0),
                            info.get("achieved_inlier_ratio", 0.0),
                            info.get("score", 0.0)
                        ])

                    # Evaluate RL
                    if model is not None:
                        info = run_rl(env, model, reset_options, normalize_obs)
                        writer.writerow([
                            "RL_Agent", noise, ir, trial,
                            *scene_cols,
                            info.get("epsilon", 0.0),
                            info.get("min_support", 0),
                            info.get("normal_thresh", 0.0),
                            info.get("angle_error", 180.0),
                            info.get("offset_error", 10.0),
                            info.get("inlier_recovery_rate", 0.0),
                            info.get("false_inlier_rate", 1.0),
                            info.get("achieved_inlier_ratio", 0.0),
                            info.get("score", 0.0)
                        ])

    print(f"Evaluation complete. Results saved to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None,
                         help="Evaluate the model trained with this tag (e.g. 'v2' -> "
                              "models/synthetic_ppo_v2.zip). Omit to auto-pick the most "
                              "recently trained synthetic_ppo*.zip.")
    args = parser.parse_args()
    evaluate(tag=args.tag)
