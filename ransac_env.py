import os
import glob
import sys
import time
import csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from plyfile import PlyData
import open3d as o3d

# Add schnabel_cython/ to path so we can import the compiled .pyd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))
import schnabel_ransac

from features.scene_features import compute_scene_features

def load_ply_xyz(filepath, voxel_size=0.05):
    """Load raw XYZ point cloud from a .ply file and apply voxel downsampling."""
    ply = PlyData.read(filepath)
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    
    # Phase 1: fixed voxel_size = 0.05m
    if voxel_size is not None and voxel_size > 0.0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pts = np.asarray(pcd.points, dtype=np.float32)
        
    return pts

def find_ground_plane(shapes, points, z_mode="z_down", horizontal_thresh=0.80):
    if not shapes:
        return None, None, None, None

    candidates = []
    for shape in shapes:
        mask = shape["inlier_mask"]
        plane_pts = points[mask]
        if len(plane_pts) < 10:
            continue
        cov = np.cov(plane_pts.T)
        evals, evecs = np.linalg.eig(cov)
        normal = evecs[:, np.argmin(evals)]
        z_align = abs(float(normal[2]))
        mean_pt = np.mean(plane_pts, axis=0)
        avg_z = float(mean_pt[2])
        
        # Calculate exactly how thick/noisy the plane is (Mean Absolute Error)
        # Distance = | (Point - Mean) dot Normal |
        distances = np.abs(np.dot(plane_pts - mean_pt, normal))
        residual = float(np.mean(distances))
        
        candidates.append({
            "shape": shape,
            "avg_z": avg_z,
            "z_align": z_align,
            "residual": residual,
            "n_points": shape["n_points"],
        })

    horizontal = [c for c in candidates if c["z_align"] >= horizontal_thresh]
    if not horizontal:
        horizontal = sorted(candidates, key=lambda c: c["n_points"], reverse=True)[:1]
        if not horizontal:
            return None, None, None, None

    reverse = (z_mode == "z_down")
    horizontal.sort(key=lambda c: c["avg_z"], reverse=reverse)
    best = horizontal[0]
    return best["shape"], best["avg_z"], best["z_align"], best["residual"]

EPS_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
MIN_SUPPORT_LEVELS = [50, 100, 200, 300, 500, 800]
# Matches ADAPTIVE_RL_PLAN.md's Phase 2 spec exactly. Moved into Phase 1 (this
# is a 4-action MultiDiscrete now, not the original 3) because epsilon and
# normal_thresh are directly coupled -- loosening epsilon to catch more
# distant points usually needs a looser normal_thresh too, since those fringe
# points have noisier normals. With normal_thresh pinned at a fixed 0.90, the
# agent had no way to act on that coupling, which plausibly explains some of
# the rough-terrain / Supermarket-style failures observed under the 3-action
# version (v1, v2_reward, v2_reward_heldout).
NORM_THRESH_LEVELS = [0.80, 0.85, 0.88, 0.90, 0.93, 0.95]


class RansacEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Controls the RANSAC parameters for point cloud ground segmentation.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, data_dir=None, log_name="evaluation_metrics.csv", fixed_normal_thresh=None,
                 include_envs=None, exclude_envs=None, split=None, test_frac=0.15, gap_frac=0.02,
                 z_mode="z_down", uniform_folder_prob=0.2, coverage_log_interval=1000):
        """
        uniform_folder_prob: fraction of resets that pick a folder uniformly at
            random, ignoring the adaptive EMA-reward weighting below. Pure
            adaptive sampling can fully starve a folder that looks "good enough"
            early on -- e.g. a 51,200-step training run was found to visit only
            ~3,600 of 44,103 episodes' worth of *distinct* folder/frame combos in
            practice being far lower than expected, traced to exactly this
            starvation risk (compounded by a since-fixed frame_id collision bug
            in logging). This floor guarantees every folder gets picked with at
            least this frequency regardless of its reward history.
        coverage_log_interval: print a running "X/Y unique frames visited"
            summary every this many episodes (0 disables). Real-time visibility
            into coverage during training, instead of only discoverable by
            analyzing the CSV after the fact.
        z_mode: passed straight through to find_ground_plane() to pick which
            horizontal candidate counts as "ground" -- "z_down" (default)
            matches TartanAir's synthetic data (ground = highest z, a NED-style
            convention) and preserves existing training/eval behavior exactly.
            RELLIS-3D's real Ouster LiDAR data is the opposite: standard sensor
            z-up convention, ground = lowest/most-negative z, so real-world
            eval scripts must pass z_mode="z_up" or find_ground_plane silently
            picks the *topmost* horizontal surface (e.g. tree canopy, a roof)
            as "ground" instead.
        include_envs / exclude_envs: optional lists of environment names (e.g.
            ["Downtown", "Office"]) to restrict which environments' frames this
            instance draws from -- for an *environment-level* train/test split
            (train with exclude_envs=[held-out envs], evaluate on those same
            held-out envs via include_envs=[...] or just point data_dir at
            them directly).
        split / test_frac / gap_frac: optional *frame-level* train/test split
            within each remaining environment. Each environment's frames are
            sorted (chronological order within its trajectory) and laid out
            as [ ...train... | gap (used by neither) | ...test... ], with the
            last test_frac fraction reserved for split="test" and the
            gap_frac fraction immediately before it dropped entirely. None
            (default split) uses every frame -- the original, unsplit
            behavior. A chronological tail-split (not a random per-frame
            split) is used deliberately: consecutive LiDAR frames from a
            moving sensor are sometimes near-duplicates (measured as little
            as 2-7cm of sensor movement between frames on this data), so a
            random split could scatter near-identical frames across
            train/test throughout the whole trajectory. The tail-split
            confines that risk to a single seam -- and gap_frac removes a
            small buffer of frames straddling that seam so even the one
            remaining boundary can't leak a near-duplicate pair.
        """
        super(RansacEnv, self).__init__()

        # None (default) preserves the RL training/eval behavior exactly: normal_thresh
        # fixed at 0.90 for every step. Set this to override it -- used by
        # baseline_evaluator.py so Standard (0.85) and Loose (0.80) match
        # BASELINE_CONFIG.md instead of silently running at 0.90 too.
        self.fixed_normal_thresh = fixed_normal_thresh
        self.z_mode = z_mode
        self.uniform_folder_prob = uniform_folder_prob
        self.coverage_log_interval = coverage_log_interval

        # Action space: epsilon (8 levels), min_support (6 levels), normal_thresh (6 levels), stop/continue (2 levels)
        self.action_space = spaces.MultiDiscrete([8, 6, 6, 2])
        
        # Observation space: 31 dims
        # 21 geometric features + 10 feedback features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)
        
        # Feedback state variables
        self.step_count = 0
        self.inlier_ratio = 0.0
        self.prev_inlier_ratio = 0.0
        self.mean_residual = 0.0
        self.z_align = 0.0
        self.plane_normal = np.zeros(3, dtype=np.float32)
        self.prev_epsilon = 0.0
        self.prev_min_support = 0
        self.prev_normal_thresh = 0.0

        # Setup CSV Logging
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, log_name)
        
        # Create CSV header if the file doesn't exist yet, or exists but is
        # empty (e.g. manually cleared without deleting it) -- os.path.exists
        # alone would skip the header in the latter case, leaving a run that
        # appends headerless rows to an empty file.
        if not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0:
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame_id", "epsilon", "min_support", "normal_threshold", 
                    "runtime", "reward", "steps_used", "inlier_ratio", 
                    "plane_normal", "residual"
                ])
        
        # Parameter mappings
        # epsilon: [0.05, 0.5]
        # min_support: [100, 1000]
        # normal_thresh: [0.7, 0.95]
        
        self.data_dir = data_dir
        if self.data_dir is None:
            # Default to the root data directory to train on all environments
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        # Search recursively for all .ply files in any subfolder
        search_pattern = os.path.join(self.data_dir, "**", "*.ply")
        all_files = glob.glob(search_pattern, recursive=True)
        
        # Filter out bad files: cache folders, ground truth labels (_segmented), and outputs (ground.ply, obstacles.ply)
        valid_files = []
        for f in all_files:
            f_lower = f.lower()
            if "cache" in f_lower:
                continue
            if "segmented" in f_lower:
                continue
            if "ground.ply" in f_lower or "obstacles.ply" in f_lower:
                continue
            valid_files.append(f)
            
        self.files = sorted(valid_files)

        if not self.files:
            raise ValueError(f"No .ply files found recursively in {self.data_dir}")

        # Group files by environment (e.g. Office, Hospital, Sewerage) for Adaptive
        # Sampling. Environment folders can have any number of subfolder levels
        # between the environment and the .ply file (e.g. <Env>/Data_omni/P0000/lidar/),
        # so the grouping key is the top-level folder relative to data_dir -- not
        # os.path.basename(os.path.dirname(f)), which would just be the immediate
        # parent ("lidar" for every environment) and collapse all environments into
        # a single bucket.
        all_folders_dict = {}
        for f in self.files:
            folder_name = self._env_folder(f)
            if folder_name not in all_folders_dict:
                all_folders_dict[folder_name] = []
            all_folders_dict[folder_name].append(f)

        # Environment-level filter: restrict to a specific train or test set
        # of environments (e.g. hold a few environments out entirely, never
        # seen during training, for a genuine cross-environment generalization
        # test -- as opposed to the frame-level split below, which still
        # trains on every environment, just not every frame of it).
        if include_envs is not None:
            all_folders_dict = {k: v for k, v in all_folders_dict.items() if k in set(include_envs)}
        if exclude_envs is not None:
            all_folders_dict = {k: v for k, v in all_folders_dict.items() if k not in set(exclude_envs)}
        if not all_folders_dict:
            raise ValueError(
                f"No environments left after include_envs={include_envs}/exclude_envs={exclude_envs} "
                f"filtering (found: {sorted({self._env_folder(f) for f in self.files})})"
            )

        # Frame-level split: within each remaining environment, lay frames out
        # chronologically as [ ...train... | gap | ...test... ], with the gap
        # (used by neither split) absorbing the one seam where a near-duplicate
        # frame pair could otherwise straddle train/test. None (default) keeps
        # every frame -- the original, unsplit behavior.
        self.folders_dict = {}
        for folder_name, folder_files in all_folders_dict.items():
            if split is None:
                self.folders_dict[folder_name] = folder_files
                continue
            n = len(folder_files)
            n_test = max(1, int(round(n * test_frac)))
            n_gap = max(1, int(round(n * gap_frac))) if gap_frac > 0 else 0
            train_end = max(n - n_test - n_gap, 0)
            train_files = folder_files[:train_end]
            test_files = folder_files[n - n_test:]
            chosen = train_files if split == "train" else test_files
            if chosen:
                self.folders_dict[folder_name] = chosen

        self.files = sorted(f for files in self.folders_dict.values() for f in files)
        if not self.files:
            raise ValueError(
                f"No frames left after split={split!r}, test_frac={test_frac} filtering."
            )

        self.folder_names = list(self.folders_dict.keys())
        self.folder_rewards = {folder: 0.0 for folder in self.folder_names}

        # Folder sizes vary a lot (533 to 3,727 frames in the current training
        # pool) but folder *selection* probability doesn't automatically
        # account for that -- simulating the actual sampler against real
        # folder sizes showed equal-per-folder selection only reaches ~76%
        # true coverage after 50,000 steps (the largest folder only gets to
        # ~47% of its own frames, since it's picked no more often than a
        # folder 7x smaller). Full size-proportional selection fixes coverage
        # (~99.9% at the same budget) but overcorrects: a folder with 7x more
        # frames then also gets ~7x more policy updates, which has no
        # justification -- frame count reflects how much data TartanAir
        # happened to generate for that scene, not how important the terrain
        # is. Weighting by sqrt(size) instead caps that ratio at ~2.6x while
        # still substantially improving coverage over equal-weighting.
        # Precomputed once here since folder sizes are static.
        folder_sizes = np.array([len(self.folders_dict[f]) for f in self.folder_names], dtype=np.float64)
        folder_size_weights = np.sqrt(folder_sizes)
        self.folder_size_probs = folder_size_weights / folder_size_weights.sum()

        # Per-folder shuffled queues for without-replacement frame cycling:
        # every frame in a folder is guaranteed to come up once before any
        # repeat, instead of a fresh random-with-replacement draw each time
        # (which can revisit the same few frames while never touching others,
        # even across many draws -- the actual cause of the ~14% raw coverage
        # measured after a 51,200-step run, since sampling was pure
        # random-with-replacement inside whichever folder got chosen). Seeded
        # with an unseeded RNG here since self.np_random isn't guaranteed set
        # until the first reset(); re-shuffles later use self.np_random so
        # they respect the env's seed from then on.
        init_rng = np.random.default_rng()
        self.folder_queues = {}
        self.folder_queue_ptr = {}
        for folder_name, folder_files in self.folders_dict.items():
            shuffled = list(folder_files)
            init_rng.shuffle(shuffled)
            self.folder_queues[folder_name] = shuffled
            self.folder_queue_ptr[folder_name] = 0

        # Coverage tracking (see coverage_log_interval above).
        self.episode_count = 0
        self.visited_frames = set()

        print(f"Loaded {len(self.files)} point cloud frames across {len(self.folder_names)} folders."
              + (f" [split={split}, test_frac={test_frac}, gap_frac={gap_frac}]" if split is not None else "")
              + (f" [include_envs={include_envs}]" if include_envs is not None else "")
              + (f" [exclude_envs={exclude_envs}]" if exclude_envs is not None else "")
              + f" [uniform_folder_prob={uniform_folder_prob}]")

        self.current_points = None
        self.current_file = None
        self.current_features = None

    def _env_folder(self, path):
        """
        Top-level folder name relative to data_dir (e.g. "Downtown", "Office") --
        the environment a frame belongs to, used as the key for adaptive
        per-environment sampling. Not os.path.basename(os.path.dirname(path)):
        that would return the immediate parent folder ("lidar"), which is the
        same for every environment.
        """
        return os.path.normpath(os.path.relpath(path, self.data_dir)).split(os.sep)[0]

    def _next_frame_in_folder(self, folder_name):
        """
        Pop the next frame from this folder's shuffled queue (without
        replacement); reshuffle and start a new pass once exhausted. Guarantees
        every frame in the folder is visited once per pass, regardless of how
        often the folder itself gets chosen -- unlike a fresh random draw each
        time, which can revisit the same frames while never touching others.
        """
        ptr = self.folder_queue_ptr[folder_name]
        if ptr >= len(self.folder_queues[folder_name]):
            shuffled = list(self.folders_dict[folder_name])
            self.np_random.shuffle(shuffled)
            self.folder_queues[folder_name] = shuffled
            ptr = 0
        frame = self.folder_queues[folder_name][ptr]
        self.folder_queue_ptr[folder_name] = ptr + 1
        return frame

    def _decode_action(self, action):
        eps = EPS_LEVELS[int(action[0])]
        min_supp = MIN_SUPPORT_LEVELS[int(action[1])]
        norm_th = NORM_THRESH_LEVELS[int(action[2])]
        stop = bool(action[3] == 0) # 0 is stop, 1 is continue

        return eps, min_supp, norm_th, stop

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state variables
        self.step_count = 0
        self.inlier_ratio = 0.0
        self.prev_inlier_ratio = 0.0
        self.mean_residual = 0.0
        self.z_align = 0.0
        self.plane_normal = np.zeros(3, dtype=np.float32)
        self.prev_epsilon = 0.0
        self.prev_min_support = 0
        self.prev_normal_thresh = 0.0
        self.iou = 0.0
        self.prev_iou = 0.0
        
        # Folder selection: mostly adaptive (biased toward historically
        # low-reward folders), but a uniform_folder_prob fraction of resets
        # ignore reward entirely and pick uniformly -- without this floor, a
        # folder that looks "good enough" early can be starved of any further
        # visits for the rest of training, regardless of how many of its
        # frames were actually ever seen. Both branches are weighted by
        # folder_size_probs (sqrt-of-size, not raw size -- see folder_size_probs
        # for why) -- selection probability alone doesn't guarantee proportional
        # *coverage* when folder sizes vary 7x, since a folder picked no more
        # often than one 7x smaller only gets through a fraction of its own
        # frames in the same budget (verified by simulation before adding this
        # weighting).
        if self.np_random.random() < self.uniform_folder_prob:
            chosen_folder = self.np_random.choice(self.folder_names, p=self.folder_size_probs)
        else:
            # Adaptive sampling: combine size-proportional baseline coverage
            # with the reward-based curriculum bias (lower reward = higher
            # probability), rather than reward alone -- a large folder that's
            # doing "fine" should still get picked roughly as often as its
            # size warrants, not be suppressed toward zero just because its
            # EMA reward looks good.
            rewards = np.array([self.folder_rewards[f] for f in self.folder_names])
            inverted = -rewards
            # Shift to prevent large negative numbers in exp
            inverted = inverted - np.max(inverted)
            exp_inv = np.exp(inverted)
            reward_probs = exp_inv / np.sum(exp_inv)
            combined = self.folder_size_probs * reward_probs
            probs = combined / combined.sum()
            chosen_folder = self.np_random.choice(self.folder_names, p=probs)

        # Pick the next frame in this folder's without-replacement queue
        # (see _next_frame_in_folder) rather than a fresh random draw.
        self.current_file = self._next_frame_in_folder(chosen_folder)
        self.current_points = load_ply_xyz(self.current_file)
        
        mask_file = self.current_file.replace(".ply", "_gt_mask.npy")
        if os.path.exists(mask_file):
            self.current_gt_mask = np.load(mask_file)
        else:
            self.current_gt_mask = np.zeros(len(self.current_points), dtype=bool)

        # Precompute and cache the 21 geometric features so step() is fast
        self.current_features = compute_scene_features(self.current_points)

        # Coverage tracking (see coverage_log_interval).
        self.episode_count += 1
        self.visited_frames.add(self.current_file)
        if self.coverage_log_interval and self.episode_count % self.coverage_log_interval == 0:
            pct = len(self.visited_frames) / len(self.files) * 100
            print(f"[Coverage] episode {self.episode_count}: "
                  f"{len(self.visited_frames)}/{len(self.files)} unique frames visited ({pct:.1f}%)")

        # Compute observation
        obs = self._get_obs()
        info = {"file": self.current_file}
        return obs, info

    def _get_obs(self):
        if self.current_features is None:
            scene_feat = np.zeros(21, dtype=np.float32)
        else:
            scene_feat = self.current_features
            
        feedback_feat = np.array([
            self.inlier_ratio,
            self.mean_residual,
            self.plane_normal[0],
            self.plane_normal[1],
            self.plane_normal[2],
            float(self.step_count),
            self.prev_inlier_ratio,
            self.prev_epsilon,
            float(self.prev_min_support),
            self.prev_normal_thresh
        ], dtype=np.float32)
        
        return np.concatenate([scene_feat, feedback_feat])

    def _potential(self, inlier_ratio, mean_residual, z_align):
        """
        Reward-shaping potential over the parts of the reward that respond to
        the agent's chosen RANSAC parameters this step. Now strictly based on 
        the intersection-over-union (IoU) with the TartanGround semantic truth.
        """
        return self.iou

    def step(self, action):
        start_time = time.time()
        self.step_count += 1
        eps, min_supp, decoded_norm_th, stop = self._decode_action(action)
        # fixed_normal_thresh overrides the agent's own choice -- used by
        # baseline_evaluator.py to force Strict/Standard/Loose's fixed values
        # regardless of what the (untrained-for-that-purpose) action array
        # happens to contain in that slot.
        norm_th = self.fixed_normal_thresh if self.fixed_normal_thresh is not None else decoded_norm_th

        # Scene-qualified, not just the bare filename: every scene numbers its
        # frames starting from 000000, so "000000_lcam_front_lidar.ply" alone
        # collides across all scenes and silently undercounts true unique
        # coverage in any downstream analysis (found this the hard way).
        frame_id = f"{self._env_folder(self.current_file)}/{os.path.basename(self.current_file)}" \
            if self.current_file else "unknown"
        info = {}

        # Potential of the state carried over from the previous step (or the
        # zeroed reset state, for step 1) -- captured BEFORE this step's
        # Schnabel call overwrites self.inlier_ratio/self.mean_residual/self.z_align.
        prev_potential = self.iou

        # Save previous state before running Schnabel
        self.prev_epsilon = eps
        self.prev_min_support = min_supp
        self.prev_normal_thresh = norm_th
        self.prev_inlier_ratio = self.inlier_ratio
        self.prev_iou = self.iou

        if len(self.current_points) < min_supp:
            self.inlier_ratio = 0.0
            self.mean_residual = 0.0
            self.plane_normal = np.zeros(3, dtype=np.float32)
            self.z_align = 0.0
            self.iou = 1.0 if np.sum(self.current_gt_mask) == 0 else 0.0
            z_align = 0.0
            info = {"error": "too_few_points"}
        else:
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
                    max_shapes=20,
                )
                
                ground_shape, avg_z, z_align, residual = find_ground_plane(shapes, self.current_points, z_mode=self.z_mode)
                
                if ground_shape is None:
                    self.inlier_ratio = 0.0
                    self.mean_residual = 0.0
                    self.plane_normal = np.zeros(3, dtype=np.float32)
                    z_align = 0.0
                    self.z_align = 0.0
                    self.iou = 1.0 if np.sum(self.current_gt_mask) == 0 else 0.0
                    info["error"] = "no_ground_found"
                else:
                    mask = ground_shape["inlier_mask"]
                    intersection = np.sum(mask & self.current_gt_mask)
                    union = np.sum(mask | self.current_gt_mask)
                    self.iou = intersection / union if union > 0 else 1.0
                    
                    ground_pts = ground_shape["n_points"]
                    self.inlier_ratio = ground_pts / len(self.current_points)
                    self.mean_residual = residual
                    self.z_align = z_align

                    # Get the actual normal from the inliers
                    plane_pts = self.current_points[mask]
                    cov = np.cov(plane_pts.T)
                    evals, evecs = np.linalg.eig(cov)
                    self.plane_normal = evecs[:, np.argmin(evals)]

                    info.update({
                        "ground_pct": self.inlier_ratio,
                        "z_align": z_align,
                        "residual": residual,
                        "avg_z": avg_z,
                        "epsilon": eps,
                        "min_support": min_supp,
                        "normal_thresh": norm_th,
                        "inlier_mask": mask.copy(),
                        "iou": self.iou
                    })

            except Exception as e:
                self.inlier_ratio = 0.0
                self.mean_residual = 0.0
                self.plane_normal = np.zeros(3, dtype=np.float32)
                z_align = 0.0
                self.z_align = 0.0
                self.iou = 1.0 if np.sum(self.current_gt_mask) == 0 else 0.0
                info["error"] = str(e)

        runtime = time.time() - start_time
        info["stopped"] = stop

        # Decide if we terminate and what the reward is
        terminated = stop or (self.step_count >= 5)

        new_potential = self.iou
        reward = (new_potential - prev_potential) - 0.01

        if terminated:
            # Update folder moving average reward (EMA)
            folder_name = self._env_folder(self.current_file)
            self.folder_rewards[folder_name] = (0.9 * self.folder_rewards[folder_name]) + (0.1 * reward)
        
        # Log to CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id, 
                round(float(eps), 5), 
                int(min_supp), 
                round(float(norm_th), 5), 
                round(runtime, 5), 
                round(float(reward), 5), 
                self.step_count,
                round(float(self.inlier_ratio), 5), 
                round(float(z_align), 5), 
                round(float(self.mean_residual), 5)
            ])
            
        return self._get_obs(), float(reward), terminated, False, info
