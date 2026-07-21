import os
import sys
import time
import numpy as np
import open3d as o3d
import gymnasium as gym
from gymnasium import spaces

# Add parent dir to path so we can import features and schnabel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schnabel_cython"))

import schnabel_ransac
from features.scene_features import compute_scene_features
from data_generator import SyntheticPlaneGenerator

# min_support is scale-relative (fraction of point count), not a fixed
# absolute count -- confirmed empirically on a real 102k-point scene:
# min_support=75 (0.75% of a 10k-pt training cloud) claimed only 0.07% of the
# real cloud, fragmenting it into 30+ spurious tiny "planes" instead of a few
# real structural ones. At the 10000-pt training cloud these fractions
# reproduce the exact old absolute values (0.0075 * 10000 = 75), so nothing
# about synthetic-domain training/eval changes -- num_points is fixed at
# 10000 everywhere in this environment. This is safe and needed no retrain.
MIN_SUPPORT_LEVELS = [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02]

# EPS_LEVELS: back to fixed absolute meters, matching what
# models/synthetic_ppo_pilot.zip was actually trained under. A scale-relative
# version (multiples of estimate_eps_scale_reference() below) was tried and
# works mechanically, but real point-cloud compatibility (why it existed) is
# explicitly deferred until Phases 7-8 of roadmap_synthetic_to_real.md --
# doing it now would invalidate the pilot's eps calibration and force a
# retrain before Phase 2 even starts. estimate_eps_scale_reference() is kept
# below, unused, so that later phase doesn't have to redo this from scratch.
EPS_LEVELS = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
NORM_THRESH_LEVELS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

DEFAULT_EPS_SCALE_REF = 0.15  # fallback before any scene has been generated


def estimate_eps_scale_reference(points, k=5, percentile=10, sample_size=500, seed=None):
    """
    Low-percentile k-NN distance over a random sample of `points` -- a scale
    reference for eps that resists being dominated by sparse outlier points.
    NOT currently used by _decode_action() (see EPS_LEVELS comment above --
    shelved until the real-point-cloud phase). Kept here, tested and working,
    for when that phase starts. Standalone from compute_scene_features() on
    purpose: the trained model's observation space is a fixed 33 dims, so
    this must never be added to it.
    """
    n = len(points)
    if n < k + 1:
        return DEFAULT_EPS_SCALE_REF

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, min(sample_size, n), replace=False)

    dists = []
    for i in sample_idx:
        found, idx, d2 = kdtree.search_knn_vector_3d(pcd.points[i], k + 1)
        if found > 1:
            dists.append(np.mean(np.sqrt(np.clip(d2[1:], 0, None))))

    if not dists:
        return DEFAULT_EPS_SCALE_REF
    return max(float(np.percentile(dists, percentile)), 1e-4)

def angle_between_normals(n1, n2):
    """Returns angle in degrees between two normals."""
    # Ensure unit vectors
    n1 = n1 / (np.linalg.norm(n1) + 1e-8)
    n2 = n2 / (np.linalg.norm(n2) + 1e-8)
    dot = np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0)
    return np.degrees(np.arccos(dot))

class SyntheticRansacEnv(gym.Env):
    """
    RL Environment for adapting RANSAC parameters on purely synthetic data
    with known geometric ground truth.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, seed=None, fixed_normal_thresh=None, max_steps=5, fixed_num_points=None,
                 reward_mode="best_of_episode", track_source_labels=False):
        super().__init__()

        self.generator = SyntheticPlaneGenerator(seed)
        self.fixed_normal_thresh = fixed_normal_thresh
        self.max_steps = max_steps
        # Off by default -- only visualize_synthetic_plane.py turns this on.
        # When True, reset() additionally asks the generator for per-point
        # source labels (which points are a bump, a cylinder, the
        # intersecting wall, etc. -- see data_generator.py's SOURCE_*
        # constants) and stores them as self.current_source_labels. Nothing
        # in step()/the reward/the observation reads this -- it exists
        # purely so a caller doing a SEPARATE generate_scene() call for
        # visualization doesn't have to (which would silently produce a
        # DIFFERENT random scene than the one the agent actually saw).
        self.track_source_labels = track_source_labels
        self.current_source_labels = None
        # "best_of_episode" (default): reward/report against the running
        # best score (see step()). "old_flat": the original reward, kept
        # only so verify_reward_redesign_v2.py can produce a real paired
        # baseline -- reward = (current_score - previous_step_score) -
        # step_penalty, unconditionally, every step. best_score/best_info
        # bookkeeping and the returned `info` still reflect the running
        # best in BOTH modes, so a fair comparison isolates exactly the
        # reward signal, not also a change in what gets reported.
        self.reward_mode = reward_mode
        # Lets a caller pin density for a controlled experiment (e.g. isolating
        # the reward-redesign effect on episode length at a fixed N) even
        # though PPO's automatic VecEnv resets never pass `options` -- without
        # this, only reset(options=...) could override the log-uniform
        # 10k-100k default, which the training loop never exercises.
        self.fixed_num_points = fixed_num_points
        
        # MultiDiscrete: [eps, min_supp, norm_th, stop/continue]
        self.action_space = spaces.MultiDiscrete([len(EPS_LEVELS), 6, 6, 2])  # [eps(11), min_supp(6), norm_th(6), stop(2)]
        # 33 dims: 23 geometric + 10 feedback
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32)
        
        # State
        self.step_count = 0
        self.current_points = None
        self.current_gt_mask = None
        self.n_true = None
        self.d_true = None
        self.true_inlier_ratio = 0.0
        self.true_noise_sigma = 0.0
        self.eps_scale_ref = DEFAULT_EPS_SCALE_REF
        # Scene-composition readback -- reset() decides these (randomly, unless
        # pinned via options) but previously never exposed the result, so no
        # caller could find out what was actually generated for a given
        # episode. Needed for evaluation scripts that want to log/stratify by
        # plane type, clutter presence, etc.
        self.true_orientation = "ground"
        self.true_noise_type = "gaussian"
        self.true_num_bumps = 0
        self.true_num_sine_waves = 0
        self.true_num_cylinders = 0
        self.true_num_boxes = 0
        self.true_add_intersecting_plane = False
        self.true_num_points = 10000

        # Feedback state
        self.inlier_ratio = 0.0
        self.mean_residual = 0.0
        self.plane_normal = np.zeros(3, dtype=np.float32)
        self.prev_inlier_ratio = 0.0
        self.prev_epsilon = 0.0
        self.prev_min_support_frac = 0.0
        self.prev_normal_thresh = 0.0
        
        # Metrics state. best_score/best_info track the running best across
        # the episode, not the latest step -- see step()'s reward comment.
        self.current_score = 0.0
        self.prev_current_score = 0.0  # only used by reward_mode="old_flat"
        self.best_score = 0.0
        self.best_info = {
            "error": None,
            "angle_error": 0.0,
            "inlier_recovery_rate": 0.0,
            "false_inlier_rate": 0.0,
            "achieved_inlier_ratio": 0.0,
            "epsilon": 0.0,
            "min_support": 0,
            "normal_thresh": 0.0,
        }

    def _decode_action(self, action):
        eps = EPS_LEVELS[int(action[0])]
        min_supp_frac = MIN_SUPPORT_LEVELS[int(action[1])]
        min_supp = max(1, int(round(min_supp_frac * len(self.current_points))))
        norm_th = NORM_THRESH_LEVELS[int(action[2])]
        stop = bool(action[3] == 0) # 0 is stop, 1 is continue
        return eps, min_supp, min_supp_frac, norm_th, stop

    def _get_obs(self):
        scene_feat = compute_scene_features(self.current_points)
        feedback_feat = np.array([
            self.inlier_ratio,
            self.mean_residual,
            self.plane_normal[0],
            self.plane_normal[1],
            self.plane_normal[2],
            float(self.step_count),
            self.prev_inlier_ratio,
            self.prev_epsilon,
            # Fraction, not the decoded absolute point count -- an absolute
            # count is density-dependent (scales with num_points), so it
            # would push this dimension out of the training distribution the
            # moment num_points varies. Feeding back the same fraction the
            # action space already uses keeps this feature density-invariant
            # like eps and the min_support action itself.
            self.prev_min_support_frac,
            self.prev_normal_thresh
        ], dtype=np.float32)
        return np.concatenate([scene_feat, feedback_feat])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        # Domain randomization: uniformly sample inlier ratio and noise sigma,
        # unless the caller pins specific values via `options` (used by eval /
        # signal-check scripts that need a controlled scene rather than a
        # random one -- see synthetic_rl_experiment/syntheticRL.md).
        if "generator_seed" in options:
            self.generator.rng = np.random.default_rng(options["generator_seed"])

        inlier_ratio = options["inlier_ratio"] if "inlier_ratio" in options else self.np_random.uniform(0.03, 0.9)
        noise_sigma = options["noise_sigma"] if "noise_sigma" in options else self.np_random.uniform(0.01, 0.20)
        slope = options["slope_angle_deg"] if "slope_angle_deg" in options else self.np_random.uniform(0.0, 45.0)
        orientation = options["orientation"] if "orientation" in options else self.np_random.choice(["ground", "ground", "vertical"])
        # Density randomization: log-uniform over [10k, 100k] so both orders
        # of magnitude get equal sampling weight (a plain uniform draw would
        # put ~90% of mass above 50k). This is what makes the policy actually
        # learn density-invariant behavior instead of just being handed
        # density-invariant action semantics (eps in meters, min_support as a
        # fraction) and hoping generalization follows for free -- see
        # SESSION_PROGRESS_LOG.md for the prev_min_support OOD analysis this
        # is paired with.
        if "num_points" in options:
            num_points = options["num_points"]
        elif self.fixed_num_points is not None:
            num_points = self.fixed_num_points
        else:
            num_points = int(10 ** self.np_random.uniform(np.log10(10000), np.log10(100000)))
        # Biased toward gaussian (most common / matches the noise-signal
        # verification already done); the other four simulate different
        # real-sensor error profiles -- see data_generator.py's _apply_noise().
        noise_type = options["noise_type"] if "noise_type" in options else self.np_random.choice(
            ["gaussian", "gaussian", "laplacian", "uniform", "spatially_varying", "mixed"]
        )
        # 50% chance of no bumps, otherwise 1-3 -- terrain deformation, see
        # data_generator.py's _add_surface_deformations().
        num_bumps = options["num_bumps"] if "num_bumps" in options else int(self.np_random.choice([0, 0, 0, 1, 2, 3]))
        # Sinusoidal (wavy/undulating) terrain -- rolling ground, washboard
        # gravel -- independent of and stackable with the localized bumps
        # above, see data_generator.py's _add_sinusoidal_deformation().
        num_sine_waves = options["num_sine_waves"] if "num_sine_waves" in options else int(self.np_random.choice([0, 0, 0, 1, 2]))
        # Structured clutter (false-positive traps) and an intersecting
        # plane -- see data_generator.py's _generate_cylinder/_generate_box/
        # _generate_intersecting_plane().
        num_cylinders = options["num_cylinders"] if "num_cylinders" in options else int(self.np_random.choice([0, 0, 0, 1, 1, 2]))
        num_boxes = options["num_boxes"] if "num_boxes" in options else int(self.np_random.choice([0, 0, 0, 0, 1]))
        add_intersecting_plane = options["add_intersecting_plane"] if "add_intersecting_plane" in options else bool(self.np_random.random() < 0.25)

        self.true_inlier_ratio = inlier_ratio
        self.true_noise_sigma = noise_sigma
        self.true_orientation = orientation
        self.true_noise_type = noise_type
        self.true_num_bumps = num_bumps
        self.true_num_sine_waves = num_sine_waves
        self.true_num_cylinders = num_cylinders
        self.true_num_boxes = num_boxes
        self.true_add_intersecting_plane = add_intersecting_plane
        self.true_num_points = num_points

        gen_kwargs = dict(
            num_points=num_points,
            inlier_ratio=inlier_ratio,
            noise_sigma=noise_sigma,
            num_bumps=num_bumps,
            num_sine_waves=num_sine_waves,
            num_cylinders=num_cylinders,
            num_boxes=num_boxes,
            add_intersecting_plane=add_intersecting_plane,
            slope_angle_deg=slope,
            orientation=orientation,
            noise_type=noise_type
        )
        if self.track_source_labels:
            pts, mask, n_true, d_true, source_labels = self.generator.generate_scene(
                return_labels=True, **gen_kwargs)
            self.current_source_labels = source_labels
        else:
            pts, mask, n_true, d_true = self.generator.generate_scene(**gen_kwargs)
        self.current_points = pts
        self.current_gt_mask = mask
        self.n_true = n_true
        self.d_true = d_true
        
        # Reset feedback
        self.step_count = 0
        self.inlier_ratio = 0.0
        self.mean_residual = 0.0
        self.plane_normal = np.zeros(3, dtype=np.float32)
        self.prev_inlier_ratio = 0.0
        self.prev_epsilon = 0.0
        self.prev_min_support_frac = 0.0
        self.prev_normal_thresh = 0.0

        self.current_score = 0.0
        self.prev_current_score = 0.0  # only used by reward_mode="old_flat"
        self.best_score = 0.0
        self.best_info = {
            "error": None,
            "angle_error": 0.0,
            "inlier_recovery_rate": 0.0,
            "false_inlier_rate": 0.0,
            "achieved_inlier_ratio": 0.0,
            "epsilon": 0.0,
            "min_support": 0,
            "normal_thresh": 0.0,
        }

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        eps, min_supp, min_supp_frac, decoded_norm_th, stop = self._decode_action(action)
        norm_th = self.fixed_normal_thresh if self.fixed_normal_thresh is not None else decoded_norm_th

        self.prev_epsilon = eps
        self.prev_min_support_frac = min_supp_frac
        self.prev_normal_thresh = norm_th
        self.prev_inlier_ratio = self.inlier_ratio

        info = {
            "error": None,
            "angle_error": 0.0,
            "inlier_recovery_rate": 0.0,
            "false_inlier_rate": 0.0,
            "achieved_inlier_ratio": 0.0
        }

        # Run RANSAC
        try:
            shapes, _ = schnabel_ransac.detect(
                self.current_points,
                shapes=["plane"],
                relative_epsilon=False,
                epsilon=eps,
                normal_thresh=norm_th,
                min_support=min_supp,
                probability=0.001,
                normal_knn=20,
                max_shapes=5,
            )
            
            # Find the plane closest to our true normal
            best_shape = None
            best_angle = 180.0
            
            if shapes:
                for shape in shapes:
                    mask = shape["inlier_mask"]
                    plane_pts = self.current_points[mask]
                    if len(plane_pts) < 10:
                        continue
                    
                    cov = np.cov(plane_pts.T)
                    evals, evecs = np.linalg.eig(cov)
                    normal = evecs[:, np.argmin(evals)]
                    
                    angle = angle_between_normals(normal, self.n_true)
                    if angle < best_angle:
                        best_angle = angle
                        best_shape = shape
                        if np.dot(normal, self.n_true) < 0:
                            normal = -normal
                        self.plane_normal = normal
            
            if best_shape is None:
                self.inlier_ratio = 0.0
                self.mean_residual = 0.0
                self.current_score = 0.0
                info["error"] = "no_valid_plane"
            else:
                pred_mask = best_shape["inlier_mask"]
                
                # Metrics
                true_inliers = np.sum(self.current_gt_mask)
                pred_inliers = np.sum(pred_mask)
                
                true_positives = np.sum(pred_mask & self.current_gt_mask)
                false_positives = np.sum(pred_mask & ~self.current_gt_mask)
                
                inlier_recovery_rate = true_positives / (true_inliers + 1e-8)
                false_inlier_rate = false_positives / (pred_inliers + 1e-8)
                achieved_ratio = pred_inliers / len(self.current_points)
                
                self.inlier_ratio = achieved_ratio
                
                # Mean residual of the points to the fitted plane
                mean_pt = np.mean(self.current_points[pred_mask], axis=0)
                distances = np.abs(np.dot(self.current_points[pred_mask] - mean_pt, self.plane_normal))
                self.mean_residual = float(np.mean(distances))
                
                # Calculate offset error
                d_fitted = np.dot(mean_pt, self.plane_normal)
                offset_error = abs(d_fitted - self.d_true)
                
                info["angle_error"] = best_angle
                info["offset_error"] = float(offset_error)
                info["inlier_recovery_rate"] = inlier_recovery_rate
                info["false_inlier_rate"] = false_inlier_rate
                info["achieved_inlier_ratio"] = achieved_ratio
                
                # Scoring
                # 1. Normal angle score: exponential decay
                tau = 5.0 # degrees
                normal_angle_score = np.exp(-best_angle / tau)
                
                # 2. Inlier match score (increased penalty for false inliers to overcome recovery drop)
                inlier_ratio_match_score = inlier_recovery_rate * (1.0 - 3.0 * false_inlier_rate)
                
                w1, w2 = 1.0, 1.0
                self.current_score = w1 * normal_angle_score + w2 * inlier_ratio_match_score

        except Exception as e:
            self.inlier_ratio = 0.0
            self.mean_residual = 0.0
            self.current_score = 0.0
            info["error"] = str(e)
            
        # Reward against the running best, not the previous step. The
        # earlier "asymmetric penalty vs. previous step" fix (see git
        # history / SESSION_PROGRESS_LOG.md) made the collapse *worse*,
        # not better: current_score is the LATEST attempt with no running
        # best, so a bad RANSAC draw on a later step didn't just cost a
        # penalty -- it overwrote and banked a worse final result, no
        # matter how good an earlier step had been. Under call-to-call
        # RANSAC noise, that gives refining negative expected value even
        # when the underlying parameter change is genuinely good, purely
        # from the bookkeeping. Comparing against self.best_score instead
        # bounds the downside of a bad step to exactly -step_penalty (you
        # keep your best), which is what makes refining a rational bet
        # again. This also subsumes the earlier step-1-untaxed asymmetry
        # for free -- step 1 now compares against best_score=0 like every
        # other step, so it needs no separate handling.
        step_penalty = 0.01
        if self.reward_mode == "old_flat":
            reward = (self.current_score - self.prev_current_score) - step_penalty
        else:
            score_gain = max(0.0, self.current_score - self.best_score)
            reward = score_gain - step_penalty
        self.prev_current_score = self.current_score

        # best_score/best_info bookkeeping runs in BOTH modes -- it's pure
        # measurement (what's the best result found, and did a later step
        # set it), independent of which reward the policy is actually paid.
        # This is what lets verify_reward_redesign_v2.py measure realized
        # improvement under the OLD reward too, for a real paired baseline.
        if self.current_score > self.best_score:
            self.best_score = self.current_score
            self.best_info = dict(info)
            self.best_info["epsilon"] = eps
            self.best_info["min_support"] = min_supp
            self.best_info["normal_thresh"] = norm_th

        terminated = stop or (self.step_count >= self.max_steps)

        # The episode's reported result must agree with what the reward
        # above is actually paying for: the best step found, not whatever
        # the last step happened to be. Reporting best_info here (rather
        # than this step's raw `info`) is what makes evaluate_synthetic.py
        # and every other consumer of `info["score"]`/`info["angle_error"]`/
        # etc. see a mutually consistent bundle -- fixing only the reward
        # while leaving this line reading the latest attempt would silently
        # reintroduce the same last-vs-best inconsistency this whole change
        # exists to remove. `error` is reported live (this step's own
        # outcome), not best-gated, since "did this attempt crash" is a
        # different question from "what's the best result so far".
        result = dict(self.best_info)
        result["error"] = info["error"]
        result["true_inlier_ratio"] = self.true_inlier_ratio
        result["true_noise_sigma"] = self.true_noise_sigma
        result["score"] = self.best_score

        return self._get_obs(), float(reward), terminated, False, result
