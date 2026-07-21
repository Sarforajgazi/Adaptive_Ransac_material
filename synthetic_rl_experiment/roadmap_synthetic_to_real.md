# Roadmap: Synthetic RL-RANSAC → Real-Data Ground Plane Detection

## Current State (Verified)

- **10k-step pilot** produces a PPO policy where epsilon adapts 0.126→0.234 with noise and normal_threshold adapts 0.819→0.675 — monotonic, physically correct, beating Super Strict by ~3x at high noise.
- Generator supports ground/tilted/vertical planes with orthonormal-basis sampling.
- Observation space (33 dims): 21 geometric scene features + 2 noise-correlated surface-variation features (mean + P25) + 10 feedback features.
- Scoring: one ground-truth plane per scene, metrics = angle_error, inlier_recovery_rate, false_inlier_rate, combined score.
- **No changes needed to scoring or environment step() logic for any phase below** — ground truth remains one plane, everything else is clutter with `gt_mask=False`.

---

## Phase 1: Document Current Result

**Goal:** Lock down the 10k pilot as your "before" baseline.

**Tasks:**
1. Save the current `synthetic_eval_results.csv` as `results_phase0_baseline.csv`.
2. Save the 10k model checkpoint separately (e.g., `models/synthetic_ppo_phase0.zip`).
3. Record the exact hyperparameters: `ent_coef=0.01`, `n_steps=2048`, `batch_size=64`, `lr=3e-4`, `max_steps=5`, observation dim=33.

**Time:** 10 minutes. No code changes.

---

## Phase 2: Surface Deformations (Bumps and Craters)

**Goal:** Ground plane is no longer a perfect plane — it has terrain-like features. RANSAC must still fit the best-approximating plane through irregular terrain.

**Why this matters:** Real ground (dirt roads, gravel, grass) is never a perfect plane. If your agent only trains on perfect planes + Gaussian noise, it may learn to set epsilon too tight for real terrain. Bumps teach it that some geometric deviation is terrain, not noise.

**Files to modify:** `data_generator.py` only.

**Implementation:**

In `generate_scene()`, after creating inlier points on the plane and adding Gaussian noise, add localized bumps/craters:

```python
def _add_surface_deformations(self, inliers, inliers_uv, n_true, num_bumps, box_size):
    """
    Add Gaussian bumps and craters to the plane surface.
    These are terrain features — the points are still ground truth inliers.
    
    Args:
        inliers: (N, 3) inlier points already on the plane
        inliers_uv: (N, 2) the local 2D coordinates used to generate them
        n_true: plane normal
        num_bumps: how many bumps/craters to add
        box_size: scene bounding box size
    
    Returns:
        Modified inliers array (same shape, same gt_mask)
    """
    for _ in range(num_bumps):
        # Random center within the plane
        center_uv = self.rng.uniform(-box_size / 3, box_size / 3, 2)
        
        # Random radius (how wide the bump is) and height (how tall/deep)
        radius = self.rng.uniform(0.5, 2.5)
        height = self.rng.uniform(-0.4, 0.4)  # negative = crater, positive = bump
        
        # Gaussian falloff from center
        dist_sq = ((inliers_uv[:, 0] - center_uv[0])**2 + 
                   (inliers_uv[:, 1] - center_uv[1])**2)
        displacement = height * np.exp(-dist_sq / (2 * radius**2))
        
        # Displace along normal
        inliers = inliers + np.outer(displacement, n_true)
    
    return inliers
```

**In `generate_scene()`**, add a `num_bumps` parameter (default 0):

```python
def generate_scene(self, num_points=10000, inlier_ratio=0.5, noise_sigma=0.01,
                   orientation="ground", box_size=10.0, slope_angle_deg=0.0,
                   num_bumps=0):
    # ... existing code to generate inliers on plane ...
    
    # Add surface deformations BEFORE adding Gaussian noise
    if num_bumps > 0:
        inliers = self._add_surface_deformations(
            inliers, inliers_uv, n_true, num_bumps, box_size
        )
    
    # Then add Gaussian noise (existing code)
    noise_magnitudes = self.rng.normal(0.0, noise_sigma, num_inliers)
    inliers = inliers + np.outer(noise_magnitudes, n_true)
    
    # ... rest of existing code unchanged ...
```

**In `synthetic_env.py` `reset()`**, randomize `num_bumps`:

```python
num_bumps = self.np_random.choice([0, 0, 0, 1, 2, 3])  # 50% chance of no bumps
```

**Key detail:** bump points stay in `gt_mask=True`. They are terrain, not clutter. Your scoring in `step()` needs zero changes — recovery rate and false_inlier_rate calculations work exactly as before.

**Verification:**
- Generate 10 scenes with `num_bumps=3`, visualize in Open3D to confirm bumps look realistic.
- Run `check_eps_signal.py` with bumps enabled to confirm the noise-dependent eps gradient still exists.
- Retrain for 10k steps, compare explained_variance trajectory to Phase 0.

**Time:** ~2 hours (implementation + verification + pilot retrain).

---

## Phase 3: Noise Model Diversity

**Goal:** Replace the single Gaussian noise model with a randomized mix, so the agent encounters the noise profiles it will see in real LiDAR data.

**Files to modify:** `data_generator.py` only.

**Implementation:**

Add a noise type parameter and implement each model:

```python
def _apply_noise(self, inliers, n_true, noise_sigma, noise_type="gaussian"):
    """
    Apply noise along the plane normal. All types displace points
    up/down from the surface — they differ in distribution shape.
    """
    num = len(inliers)
    
    if noise_type == "gaussian":
        # Standard — what you have now
        magnitudes = self.rng.normal(0.0, noise_sigma, num)
    
    elif noise_type == "laplacian":
        # Heavy-tailed: occasional large outlier-like displacements
        # Real LiDAR has this from multi-path reflections
        magnitudes = self.rng.laplace(0.0, noise_sigma / np.sqrt(2), num)
    
    elif noise_type == "uniform":
        # Bounded noise: all displacements within a fixed band
        # Simulates quantization or structured sensor error
        band = noise_sigma * np.sqrt(3)  # match variance to Gaussian
        magnitudes = self.rng.uniform(-band, band, num)
    
    elif noise_type == "spatially_varying":
        # Noise increases with distance from sensor (placed at origin)
        distances = np.linalg.norm(inliers, axis=1)
        max_dist = np.max(distances) + 1e-8
        local_sigma = noise_sigma * (0.3 + 1.4 * (distances / max_dist)**2)
        magnitudes = self.rng.normal(0.0, 1.0, num) * local_sigma
    
    elif noise_type == "mixed":
        # 80% Gaussian + 20% heavy-tailed (simulates mostly clean with some bad returns)
        magnitudes = self.rng.normal(0.0, noise_sigma, num)
        heavy_mask = self.rng.random(num) < 0.2
        magnitudes[heavy_mask] = self.rng.laplace(0.0, noise_sigma * 2, heavy_mask.sum())
    
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    return inliers + np.outer(magnitudes, n_true)
```

**In `synthetic_env.py` `reset()`**, randomize noise type:

```python
noise_type = self.np_random.choice([
    "gaussian", "gaussian", "laplacian", "uniform", 
    "spatially_varying", "mixed"
])  # biased toward gaussian (most common)
```

**Verification:**
- For each noise type at noise_sigma=0.10, compute the P25 surface variation feature and confirm it's still above the noise_sigma=0.01 Gaussian baseline (i.e., the observation feature still responds to non-Gaussian noise, not just Gaussian).
- Visualize one scene per noise type in Open3D.

**Time:** ~2 hours.

---

## Phase 4: Structured Clutter (Cylinders and Boxes)

**Goal:** Add non-planar geometric objects near the ground plane as false-positive traps. These have `gt_mask=False` — they are NOT ground truth, they are obstacles the agent must avoid merging into the ground fit.

**Files to modify:** `data_generator.py` only.

**Implementation:**

Add clutter generation methods to `SyntheticPlaneGenerator`:

```python
def _generate_cylinder(self, base_point, n_ground, num_points):
    """
    A vertical cylinder (tree trunk, pole) rising from the ground.
    Points are on the cylinder surface, NOT ground truth.
    """
    radius = self.rng.uniform(0.1, 0.5)
    height = self.rng.uniform(1.0, 4.0)
    
    # Cylinder axis is along the ground normal (perpendicular to ground)
    up = n_ground / (np.linalg.norm(n_ground) + 1e-8)
    helper = np.array([1, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1, 0])
    side1 = np.cross(up, helper)
    side1 /= np.linalg.norm(side1)
    side2 = np.cross(up, side1)
    
    angles = self.rng.uniform(0, 2 * np.pi, num_points)
    heights = self.rng.uniform(0, height, num_points)
    
    pts = (base_point + 
           heights[:, None] * up +
           radius * np.cos(angles)[:, None] * side1 +
           radius * np.sin(angles)[:, None] * side2)
    
    return pts.astype(np.float32)

def _generate_box(self, base_point, n_ground, num_points):
    """
    A rectangular box (barrier, rock) sitting on the ground.
    Points are on the box surface, NOT ground truth.
    """
    width = self.rng.uniform(0.3, 1.5)
    depth = self.rng.uniform(0.3, 1.5)
    height = self.rng.uniform(0.3, 1.5)
    
    up = n_ground / (np.linalg.norm(n_ground) + 1e-8)
    helper = np.array([1, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1, 0])
    side1 = np.cross(up, helper)
    side1 /= np.linalg.norm(side1)
    side2 = np.cross(up, side1)
    
    pts_per_face = num_points // 6
    faces = []
    
    for sign in [-1, 1]:
        for axis, extent in [(side1, width/2), (side2, depth/2), (up, height)]:
            u_ax = side1 if not np.allclose(axis, side1) else side2
            v_ax = side2 if not np.allclose(axis, side2) else up
            u_ext = width/2 if np.allclose(u_ax, side1) else depth/2
            v_ext = depth/2 if np.allclose(v_ax, side2) else height/2
            
            u_coords = self.rng.uniform(-u_ext, u_ext, pts_per_face)
            v_coords = self.rng.uniform(-v_ext, v_ext, pts_per_face)
            face_pts = (base_point + sign * extent * axis +
                       u_coords[:, None] * u_ax + v_coords[:, None] * v_ax)
            faces.append(face_pts)
    
    return np.vstack(faces).astype(np.float32)[:num_points]
```

**Integrating into `generate_scene()`:**

The key insight: clutter points **replace** some uniform outliers, keeping `num_points` constant.

```python
def generate_scene(self, ..., num_cylinders=0, num_boxes=0):
    # ... existing inlier generation ...
    
    # Generate structured clutter
    clutter_points = []
    for _ in range(num_cylinders):
        # Place cylinder base on or near the ground plane
        base_uv = self.rng.uniform(-box_size/3, box_size/3, 2)
        base_3d = p0 + base_uv[0] * u + base_uv[1] * v
        cyl_pts = self._generate_cylinder(base_3d, n_true, 
                                          num_points=self.rng.integers(50, 200))
        clutter_points.append(cyl_pts)
    
    for _ in range(num_boxes):
        base_uv = self.rng.uniform(-box_size/3, box_size/3, 2)
        base_3d = p0 + base_uv[0] * u + base_uv[1] * v
        box_pts = self._generate_box(base_3d, n_true,
                                     num_points=self.rng.integers(50, 200))
        clutter_points.append(box_pts)
    
    if clutter_points:
        clutter = np.vstack(clutter_points)
        # Replace some uniform outliers with structured clutter
        num_structured = min(len(clutter), num_outliers // 2)
        outliers[:num_structured] = clutter[:num_structured]
    
    # ... rest unchanged (combine, shuffle, return) ...
```

**In `synthetic_env.py` `reset()`:**

```python
num_cylinders = self.np_random.choice([0, 0, 0, 1, 1, 2])
num_boxes = self.np_random.choice([0, 0, 0, 0, 1])
```

**Why this is specifically useful for epsilon adaptation:** A cylinder base sitting on the ground plane has points within ~0.1-0.5m of the plane surface. If epsilon is set to 0.40 (Loose baseline), RANSAC will grab those cylinder-base points as ground inliers — inflating false_inlier_rate. The agent should learn to tighten epsilon slightly when structured clutter is nearby. The observation features (surface variation, KNN statistics) will reflect the presence of nearby non-planar structure, giving the agent the signal it needs.

**Verification:**
- Generate scenes with 2 cylinders + 1 box, visualize in Open3D.
- Confirm that `false_inlier_rate` increases for the Loose baseline compared to clutter-free scenes (proving the clutter is actually trapping loose parameter settings).
- Run a 10k pilot, check that the agent doesn't collapse.

**Time:** ~3 hours.

---

## Phase 5: Near-Tangent Intersecting Plane (Wall/Ramp)

**Goal:** A second plane (wall, ramp, embankment) intersects or nearly touches the ground plane. Near the intersection line, the two surfaces are within epsilon of each other — the agent must choose epsilon carefully to avoid merging them.

**Files to modify:** `data_generator.py` only.

**Implementation:**

```python
def _generate_intersecting_plane(self, n_ground, d_ground, p0, u, v, 
                                  box_size, num_points):
    """
    Generate a second plane that intersects or nearly touches the ground.
    Returns points with gt_mask=False (these are NOT ground).
    
    The angle between planes is randomized:
    - Near-tangent (5-15°): hardest case — large overlap zone within epsilon
    - Moderate (30-60°): typical wall/ramp
    - Perpendicular (80-90°): clean wall, easy to separate
    """
    # Choose intersection angle
    angle_deg = self.rng.choice([
        self.rng.uniform(5, 15),    # near-tangent (hardest)
        self.rng.uniform(30, 60),   # moderate
        self.rng.uniform(80, 90),   # perpendicular (easiest)
    ])
    angle_rad = np.radians(angle_deg)
    
    # Rotate ground normal around a random in-plane axis
    # to get the second plane's normal
    rotate_axis = u * self.rng.normal() + v * self.rng.normal()
    rotate_axis /= np.linalg.norm(rotate_axis) + 1e-8
    
    # Rodrigues rotation
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    n_wall = (n_ground * cos_a + 
              np.cross(rotate_axis, n_ground) * sin_a +
              rotate_axis * np.dot(rotate_axis, n_ground) * (1 - cos_a))
    n_wall = n_wall / (np.linalg.norm(n_wall) + 1e-8)
    
    # Place the second plane so it intersects the ground near the center
    d_wall = d_ground + self.rng.uniform(-0.5, 0.5)
    
    # Generate points on the second plane
    u2, v2 = self._create_plane_frame(n_wall)
    p0_wall = d_wall * n_wall
    
    coeffs = self.rng.uniform(-box_size / 3, box_size / 3, (num_points, 2))
    wall_pts = p0_wall + coeffs[:, 0:1] * u2 + coeffs[:, 1:2] * v2
    
    # Add slight noise to the wall too
    wall_noise = self.rng.normal(0, 0.02, num_points)
    wall_pts = wall_pts + np.outer(wall_noise, n_wall)
    
    return wall_pts.astype(np.float32), angle_deg
```

**Integration** follows the same pattern as Phase 4: wall points replace some uniform outliers, `gt_mask=False`.

```python
def generate_scene(self, ..., add_intersecting_plane=False):
    # ... existing code ...
    
    if add_intersecting_plane:
        wall_pts, intersection_angle = self._generate_intersecting_plane(
            n_true, d_true, p0, u, v, box_size,
            num_points=self.rng.integers(200, 800)
        )
        num_wall = min(len(wall_pts), num_outliers // 3)
        outliers[:num_wall] = wall_pts[:num_wall]
    
    # ... combine, shuffle, return ...
```

**In `synthetic_env.py` `reset()`:**

```python
add_intersecting_plane = self.np_random.random() < 0.25  # 25% of episodes
```

**Why near-tangent is the critical test:** At a 10° intersection angle, the two planes share a wide band where their distance is less than most epsilon values. An agent using eps=0.40 will merge the wall into the ground fit across that entire band, tanking false_inlier_rate. An agent using eps=0.08 will correctly reject the wall but may also reject noisy ground points. The optimal epsilon depends on both the noise level AND the proximity of the second surface — this is the most challenging scenario and the one most relevant to real-world data where walls and ramps meet the ground.

**Verification:**
- Generate near-tangent scenes (5-10° angle), visualize showing both planes.
- Compare false_inlier_rate for Loose vs Strict baselines — Loose should be dramatically worse.
- Confirm the Strict baseline doesn't crash or produce degenerate results.

**Time:** ~3 hours.

---

## Phase 6: Combined Retrain

**Goal:** Train a single PPO model on the full distribution — all plane types, all noise types, bumps/craters, cylinders/boxes, intersecting planes, all randomized per episode.

**Files to modify:** `synthetic_env.py` (reset only), `train_synthetic.py` (timestep budget).

**Implementation in `reset()`:**

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    
    if options:
        # Controlled evaluation mode — pin parameters
        noise_sigma = options.get("noise_sigma", self.np_random.uniform(0.01, 0.25))
        inlier_ratio = options.get("inlier_ratio", self.np_random.uniform(0.05, 0.90))
        orientation = options.get("orientation", self.np_random.choice(["ground", "ground", "vertical"]))
        slope = options.get("slope_angle_deg", self.np_random.uniform(0.0, 45.0))
        generator_seed = options.get("generator_seed", None)
        if generator_seed is not None:
            self.generator.rng = np.random.default_rng(generator_seed)
    else:
        # Full random training distribution
        noise_sigma = self.np_random.uniform(0.01, 0.25)
        inlier_ratio = self.np_random.uniform(0.05, 0.90)
        orientation = self.np_random.choice(["ground", "ground", "vertical"])
        slope = self.np_random.uniform(0.0, 45.0)
    
    noise_type = self.np_random.choice([
        "gaussian", "gaussian", "laplacian", "uniform",
        "spatially_varying", "mixed"
    ])
    num_bumps = self.np_random.choice([0, 0, 0, 1, 2, 3])
    num_cylinders = self.np_random.choice([0, 0, 0, 1, 1, 2])
    num_boxes = self.np_random.choice([0, 0, 0, 0, 1])
    add_intersecting_plane = self.np_random.random() < 0.25
    
    self.true_inlier_ratio = inlier_ratio
    self.true_noise_sigma = noise_sigma
    
    pts, mask, n_true, d_true = self.generator.generate_scene(
        num_points=10000,
        inlier_ratio=inlier_ratio,
        noise_sigma=noise_sigma,
        orientation=orientation,
        slope_angle_deg=slope,
        noise_type=noise_type,
        num_bumps=num_bumps,
        num_cylinders=num_cylinders,
        num_boxes=num_boxes,
        add_intersecting_plane=add_intersecting_plane
    )
    
    # ... rest of reset unchanged ...
```

**Training budget:** 50k-100k steps as first real run. At 0.29 s/step, 100k steps ≈ 8 hours. Start with 50k (~4 hours), evaluate, extend to 100k if the curves are still improving.

**Evaluation:** Run `evaluate_synthetic.py` with controlled sweeps:
- Noise sweep (existing): noise_sigma on x-axis, all clutter types mixed.
- Clutter ablation: same noise_sigma, but separate runs with (a) no clutter, (b) cylinders only, (c) intersecting plane only.
- This shows whether the agent degrades gracefully when clutter is present vs absent.

**Time:** 4-8 hours (training) + 1 hour (evaluation).

---

## Phase 7: Transfer to TartanAir

**Goal:** Apply the trained synthetic model to simulated LiDAR scenes from TartanAir. No retraining — pure zero-shot transfer to test if synthetic training generalizes.

**What needs to change:**

1. **Point cloud loading:** TartanAir provides depth images, not point clouds. You need a depth-to-pointcloud conversion using the camera intrinsics. You already have this infrastructure from your main pipeline.

2. **Ground truth:** TartanAir has semantic labels. The "ground" class label gives you `gt_mask` directly — no synthetic generation needed.

3. **Observation features:** `compute_scene_features()` and the surface-variation features work on any point cloud. No changes needed as long as the point cloud is the same size (subsample or pad to ~10000 points).

4. **Create `evaluate_tartanair.py`:**
   - Load TartanAir scenes from your existing 20-scene dataset.
   - For each scene: compute observation features → model.predict() → get RANSAC parameters → run RANSAC → compute recovery/false_inlier_rate against semantic ground truth.
   - Compare against the same fixed baselines (Super Strict/Strict/Standard/Loose).

**Critical detail: VecNormalize statistics.** Your model was trained with VecNormalize, which rescales observations using running statistics computed during training. At inference on TartanAir data, the observation statistics (means, variances) will be different from synthetic data. You MUST either:
- Load the saved VecNormalize stats from training (the `.pkl` file saved alongside the model), or
- Retrain with a few TartanAir scenes mixed in so the normalization covers real data too.

**Expected outcome:** The model will likely underperform on TartanAir relative to synthetic data, because real LiDAR has complexities (occlusion, scan patterns, non-uniform density) that synthetic data doesn't model. But if it adapts epsilon at all (not flat), that's evidence of transfer, and the gap tells you what's missing for Phase 8.

**Time:** ~1 day (mostly data pipeline work, not model changes).

---

## Phase 8: Fine-Tune on RELLIS-3D

**Goal:** Adapt the synthetic-pretrained model to real off-road LiDAR data.

**What RELLIS-3D provides:**
- Real Ouster OS1-64 LiDAR point clouds.
- Semantic labels including ground/terrain classes.
- Challenging off-road terrain (grass, mud, puddles, bushes).

**Approach:**

1. **Use the synthetic-trained model as initialization** — don't train from scratch on real data, because you have far fewer labeled real scenes than synthetic ones. The synthetic pretraining gives the agent a good prior on how RANSAC parameters relate to noise and geometry.

2. **Wrap RELLIS-3D scenes in the same `SyntheticRansacEnv` interface:**
   - `reset()` loads a random RELLIS-3D frame instead of generating synthetic data.
   - `n_true` and `d_true` come from fitting a plane to the ground-labeled points (using robust PCA or RANSAC itself on the ground-truth subset).
   - `gt_mask` comes from the semantic labels.
   - Everything else (observation features, RANSAC call, scoring) is identical.

3. **Fine-tune for 10k-20k steps** on RELLIS-3D scenes, starting from the synthetic checkpoint.

4. **Evaluate:** Compare fine-tuned model vs synthetic-only model vs fixed baselines on held-out RELLIS-3D frames.

**Time:** ~2 days (data pipeline + fine-tuning + evaluation).

---

## Summary: Implementation Order

| Phase | What | Modifies | Scoring Changes | Time |
|-------|------|----------|-----------------|------|
| 1 | Document current result | Nothing | None | 10 min |
| 2 | Bumps and craters | `data_generator.py` | None | ~2 hrs |
| 3 | Noise type diversity | `data_generator.py` | None | ~2 hrs |
| 4 | Cylinders and boxes | `data_generator.py` | None | ~3 hrs |
| 5 | Intersecting planes | `data_generator.py` | None | ~3 hrs |
| 6 | Combined retrain (50k-100k) | `synthetic_env.py`, `train_synthetic.py` | None | 4-8 hrs |
| 7 | TartanAir zero-shot eval | New `evaluate_tartanair.py` | None | ~1 day |
| 8 | RELLIS-3D fine-tune | New `rellis_env.py` | None | ~2 days |

**Key principle throughout:** `step()` and scoring logic never change. One ground-truth plane, one set of metrics. Everything new is either a generator extension (Phases 2-5) or a data source swap (Phases 7-8).

**After each phase:** run a 10k pilot before committing to a full train, verify no NaN rewards, confirm explained_variance still climbs, check that the eps gradient from `check_eps_signal.py` survives the new complexity.
