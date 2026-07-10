# Adaptive RANSAC — RL Implementation Plan

> Building a Deep RL agent that wraps the Schnabel C++ RANSAC engine and learns to select the right parameters per scan, decide when the result is good enough, and improve using context from previous frames.

> **This is the original planning document, kept as-is below for history.**
> Phase 1 has since been implemented in [`ransac_env.py`](ransac_env.py), and
> in a few places the implementation ended up different from what's planned
> here — each divergence is called out in a `> **Implementation note:**`
> callout right after the relevant section, explaining what changed and why,
> rather than silently editing the plan. For the actual, current behavior,
> [RL_PIPELINE_OVERVIEW.md](RL_PIPELINE_OVERVIEW.md) is the source of truth;
> this file documents intent and reasoning at the time Phase 1 was designed.

---

## The Core Idea

Standard RANSAC uses fixed parameters for every scan. But real LiDAR data is not uniform — scan density changes with distance, terrain roughness varies, and what works on a flat road fails on a slope. The agent's job is to observe each scan, pick parameters that suit it, and decide when to stop refining.

The key design choice: the agent does **sequential decision-making**, not one-shot prediction. It runs Schnabel, looks at the result, then decides whether to stop or try again with adjusted parameters — up to 5 times per scan. This lets it recover from bad initial guesses rather than being locked into one prediction.

---

## The Backend

**Schnabel C++ via the existing Cython bridge.** It is already running on TartanAir/TartanGround (13,049 frames processed). The RL agent sits on top of it in Python and controls only what parameters it receives. The agent never touches the internal algorithm.

`bitmap_epsilon` is always set automatically to `2 × epsilon` — no separate control needed.

---

## Action Space

4 parameters the agent controls per step. All are cheap to change — no re-running normal estimation:

| Action | Type | Range | Why the agent controls it |
|---|---|---|---|
| `epsilon` | 8 discrete levels | 0.05m → 0.5m | Primary distance threshold. Loose on rough terrain, tight on smooth. Note: Schnabel uses 3× this value internally for global scoring |
| `min_support` | 6 discrete levels | 50, 100, 200, 300, 500, 800 | Minimum inlier count to accept a shape. Must scale with scan density — a sparse scan needs a lower bar |
| `normal_thresh` | 6 discrete levels | 0.80 → 0.98 | How strictly point normals must align with the plane. Loosen on rough/uneven ground, tighten on clean flat surfaces |
| `stop / continue` | Binary | {0, 1} | The core RL decision — accept the current result or run Schnabel again with new params |

**Fixed parameters (agent does not control these):**

| Parameter | Fixed value | Reason |
|---|---|---|
| `kNN` | 20 | Changing it requires recomputing all normals — too expensive per step |
| `m_probability` | 0.001 | Controls search exhaustiveness; unlock in Phase 2 |
| Candidates per round | 200 (hardcoded in C++) | Marginal effect — not worth the complexity |
| `bitmap_epsilon` | `2 × epsilon` (auto) | Derived from epsilon; no independent value in exposing it |

> **Implementation note:** the table above (4 actions, including
> `normal_thresh`) is this plan's *overall* target across phases — the
> "Implementation Phases → Phase 1" section further down already scopes
> Phase 1 down to just 3 actions (`epsilon`, `min_support`, `stop/continue`)
> and defers `normal_thresh` to Phase 2. The shipped Phase 1 code
> (`ransac_env.py`, `action_space = MultiDiscrete([8, 6, 2])`) matches that
> Phase 1 scoping exactly: `normal_thresh` is fixed at a constant `0.90`
> for every step rather than agent-controlled. Not a divergence from the
> plan, just easy to misread if you only look at this top-level table. See
> [RL_PIPELINE_OVERVIEW.md's Action Space section](RL_PIPELINE_OVERVIEW.md#the-action-space--every-term-explained)
> for the currently-shipped action space.

---

## State Space (28-dim Phase 1 → 33-dim Phase 2)

What the agent observes at each step.

> **Implementation note:** the shipped Phase 1 state (`ransac_env.py` +
> [`features/scene_features.py`](features/scene_features.py)) ended up
> **21 scene + 10 feedback = 31-dim**, not the 28-dim (18 scene + 10
> feedback) planned here. This wasn't a deliberate redesign — it's what
> the planned feature list turned into once it was actually implemented,
> and it's worth tracking why rather than pretending the plan and the code
> always matched:
> - **Dropped:** `intensity_mean`, `intensity_std` — TartanAir's per-frame
>   LiDAR `.ply` files (the actual data this env trains on) don't carry an
>   intensity channel, so these two planned features were never
>   implementable and were simply left out.
> - **`eigenvalue_ratio_1`/`eigenvalue_ratio_2`** (ratios of PCA
>   eigenvalues) became **`eig_0, eig_1, eig_2`** (the three raw
>   eigenvalues, unratioed) — 1 extra dimension. Ratios are cheap to
>   recover from the raw values downstream (and the network can learn a
>   ratio-like feature on its own from three raw numbers), but keeping the
>   raw eigenvalues avoids deciding the "right" ratio pairing up front.
> - **Added `normal_x_std, normal_y_std, normal_z_std`** (per-axis std of
>   estimated normals) — not in this plan's scene-feature list, but present
>   in the shipped code as an extra proxy for "how noisy/curved is this
>   scene's geometry," alongside the `normal_consistency` feature this plan
>   already specified (which the shipped code keeps, and also reuses
>   directly in the reward function — see below).
> - **`n_neighbours_mean`** (planned) became **`mean_knn_dist`** (shipped)
>   — same intent (a local-density proxy from kNN), renamed for clarity
>   about what's actually being averaged (distance, not neighbour count).
> - **`z_density_ground`**'s threshold changed from the plan's `z < mean_z
>   + 0.5` to the shipped `z > z_max - 0.5` — both aim at "fraction of
>   points near the ground," but the shipped version accounts for
>   TartanAir's NED (Z-down) convention, where the ground is the
>   *highest*-Z region, not `mean_z`-relative. Using `mean_z` as the plan
>   describes would have picked up points from well above the actual floor
>   in any scene where the sensor sees a lot of open space above ground
>   level.
> - **Added `bbox_volume`** (`dx × dy × dz`) alongside the plan's separate
>   `bbox_dx/dy/dz` — not a replacement, an extra derived dimension kept
>   because `point_density = N / bbox_volume` needed it computed anyway.
>
> Net effect: 18 planned scene features → 21 shipped
> (−2 intensity, +1 from the eigenvalue-ratio→3-raw-eigenvalues swap,
> +3 normal_x/y/z_std, +1 bbox_volume = 18 − 2 + 1 + 3 + 1 = 21). See
> [RL_PIPELINE_OVERVIEW.md's Observation Space section](RL_PIPELINE_OVERVIEW.md#the-observation-space--every-term-explained)
> for the exact, currently-shipped 31-dim layout.

### Scene Features — computed from raw point cloud before running Schnabel

| Feature | How | Why |
|---|---|---|
| `height_mean` | mean(z) | Ground level indicator |
| `height_std` | std(z) | Flat vs hilly terrain |
| `height_min`, `height_max` | min/max(z) | Vertical extent of scan |
| `point_density` | N / volume | High density → can afford higher min_support |
| `eigenvalue_ratio_1` | λ1/λ3 of covariance | Planarity of the scene |
| `eigenvalue_ratio_2` | λ2/λ3 | Linearity vs planarity |
| `normal_consistency` | mean(|dot(nᵢ, nⱼ)|) for kNN pairs | Low = noisy normals = loosen normal_thresh |
| `z_density_ground` | fraction of points with z < mean_z + 0.5 | How much of the scan is near-ground level |
| `intensity_mean` | mean(intensity) | Road vs vegetation reflectivity |
| `intensity_std` | std(intensity) | Surface type variation |
| `bbox_dx`, `bbox_dy`, `bbox_dz` | bounding box extents | Scene scale — affects what epsilon means in real units |
| `scan_range_mean` | mean(‖p‖) | Average distance from sensor |
| `scan_range_std` | std of above | Near/far distribution |
| `ground_slope_estimate` | PCA on lowest 10% of points | Tilt of terrain — slope affects normal direction |
| `n_neighbours_mean` | mean kNN distances | Local density proxy |

### Feedback Features — computed from Schnabel's output after each step

| Feature | Source | Why |
|---|---|---|
| `inlier_ratio` | n_inliers / n_total | How much of the scan was accepted as ground |
| `mean_residual` | mean point-to-plane distance of inliers | Low = tight fit, high = noisy plane |
| `plane_normal_x/y/z` | Detected plane's normal | Is the ground roughly horizontal? |
| `step_count` | Current refinement step (0–4) | Agent knows how many attempts remain |
| `prev_inlier_ratio` | Inlier ratio from previous step | Did the last action improve things? |
| `prev_epsilon` | The epsilon chosen in the previous step (0 at step 0) | Agent must know what it just tried to adjust effectively |
| `prev_min_support` | The min_support chosen in the previous step (0 at step 0) | Same — without this, the agent can't learn "I tried X and it failed, try Y" |
| `prev_normal_thresh` | The normal_thresh chosen in the previous step (0 at step 0) | Completes the previous-action context |

### Temporal Features — from previous frame (Phase 2)

| Feature | Why |
|---|---|
| `prev_frame_normal_x/y/z` | Ground plane is stable across frames — a big deviation signals an unusual frame |
| `prev_frame_epsilon` | What worked last frame is a strong prior for this frame |
| `prev_frame_inlier_ratio` | Baseline quality reference |

**Phase 1 uses scene (18) + feedback (10) = 28-dim. Phase 2 adds temporal (5) = 33-dim.**
*(As shipped: scene (21) + feedback (10) = 31-dim — see the implementation note above this section for why.)*

---

## Reward Function (Self-Supervised)

No ground-truth labels required. Reward is given only at the terminal step (stop action or max steps reached). Per-step reward is zero.

```
reward = α × inlier_ratio
       − β × runtime
       − γ × mean_residual
       + δ × normal_consistency
       − ζ × step_penalty
```

| Term | Weight | Purpose |
|---|---|---|
| `inlier_ratio` | α = 1.0 | More ground points detected = better |
| `runtime` | β = 0.1 | Penalise slow solutions |
| `mean_residual` | γ = 0.5 | Penalise loose fits where points barely pass threshold |
| `normal_consistency` | δ = 0.3 | Reward stable plane normals across the scan |
| `step_penalty` | ζ = 0.05 per step | Penalise using extra steps — without this, the agent always uses all 5 |

**When ground-truth labels are available (TartanAir):**
```
reward = IoU(predicted_ground_mask, gt_ground_mask)
```
Train on self-supervised reward; evaluate with IoU to measure real quality.

> **Implementation note (Day 10):** the terminal-only formula above (same
> five terms, same weights) is exactly what Phase 1 shipped with initially,
> and it worked for training a policy that beat the fixed-parameter
> baselines — but it had one specific failure this plan didn't anticipate:
> with `reward = 0.0` on every non-terminal step, the agent had no signal
> about whether an intermediate refinement step helped, so `mean_steps`
> stayed pinned at 1.0 across every dataset — the 5-step refinement loop
> this whole plan is built around was never actually being used. The fix
> (kept the same five terms and weights, but paid `inlier_ratio` and
> `mean_residual` out incrementally every step via potential-based
> shaping, `Φ(new) - Φ(prev)`, instead of once at the end) is **not
> reflected above** — this section is left as originally planned since it
> was a genuinely reasonable starting design, not a mistake caught before
> shipping. See
> [RL_PIPELINE_OVERVIEW.md's Reward Function section](RL_PIPELINE_OVERVIEW.md#the-reward-function--every-term-explained)
> for both the original and current (shaped) formulas side by side, and the
> Day 10 entry in that file's Known Issues for the full diagnosis.
>
> The IoU-based reward above (for when ground-truth labels are available)
> is unaffected by this change and remains Phase 3/SemanticKITTI-only, as
> planned — it hasn't been implemented yet either way.

> **Implementation note (Day 12):** the `δ × normal_consistency` term above
> is also no longer part of the reward — it's been removed entirely, not
> just moved. It turned out to be a static per-frame descriptor (computed
> once in `reset()`, before any RANSAC call), so adding it as a flat bonus
> only at the terminal step gave the agent a reward for terminating that
> barely varied with — and couldn't respond to — its own actions. That
> breaks potential-based shaping's telescoping guarantee and, confirmed
> live in the `v3_normthresh_heldout` run's training log, produced the same
> `mean_steps` collapse toward 1 that Day 10/11 had already fought, this
> time under the expanded 4-action space. It was replaced with `z_align`
> (the actually-detected plane's normal alignment, recomputed every step)
> folded directly into the per-step potential instead of bolted onto the
> terminal transition. See
> [RL_PIPELINE_OVERVIEW.md's Reward Function section](RL_PIPELINE_OVERVIEW.md#the-reward-function--every-term-explained)
> for the current formula and the Day 12 entry in that file's Known Issues
> for the full diagnosis.

---

## Episode Structure

```
1 episode = 1 LiDAR scan frame

reset():  load raw frame
          [Phase 3 only] agent picks voxel_size → downsample once
          [Phase 1/2]    apply fixed/rule-based voxel downsample
          compute scene features from downsampled cloud

step 0:  observe state → choose (epsilon, min_support, normal_thresh, stop/continue)
              ↓ if continue:
         run Schnabel → get result → update feedback features → form new state
step 1:  observe new state → choose actions again
              ↓ if continue:
         run Schnabel again with new params
...
step 4:  observe state → forced stop (max steps reached)
              ↓
         compute terminal reward → end episode
```

Max 5 steps per episode. Each step calls Schnabel C++ once via the Cython bridge.
**Voxel size is fixed for the entire episode** — it is chosen once at `reset()`, not per step.

---

## Architecture

```
LiDAR frame  (raw, ~100k pts)
       │
       ▼
Voxel Downsample
  Phase 1: fixed voxel_size = 0.05m
  Phase 2: rule-based adaptive  (targets ~15k pts, clipped to [0.02, 0.15]m)
  Phase 3: RL-controlled  (5 discrete levels, chosen once per episode)
       │
       ▼
State Builder
  • 18 scene features     (from downsampled points — computed once per frame)
  • 10 feedback features  (from last Schnabel run, includes prev actions)
  • 5 temporal features   (from previous frame) ← Phase 2 only
  ──────────────────────────────────────────────
  = 28-dim state (Phase 1) → 33-dim state (Phase 2)
       │
       ▼
PPO Policy  (MLP: 28 → 64 → 64 → action heads)
  Action heads:
    • epsilon        — 8-way softmax
    • min_support    — 6-way softmax
    • stop/continue  — binary
    • normal_thresh  — 6-way softmax  ← Phase 2
    • voxel_size     — 5-way softmax  ← Phase 3 (one-shot, chosen at reset)
       │
       ▼
Cython Bridge  →  Schnabel C++ Detect()
       │
       ▼
Result: inlier mask, plane normal, inlier count, mean residual
       │
       ▼
Reward (terminal): inlier_ratio − β·runtime − γ·residual + δ·normal_consistency − ζ·steps
```

> **Implementation note:** this diagram's numbers are the original plan and
> are left as-is, but two of them are now stale against the shipped Phase 1
> code (`ransac_env.py`):
> - **"18 scene features" / "28-dim state" / "MLP: 28 → 64 → 64"** → shipped
>   as **21 scene features / 31-dim state / MLP: 31 → 64 → 64**. Same cause
>   as the State Space section above (dropped intensity features, added
>   `bbox_volume` + normal-std features, eigenvalue ratios became 3 raw
>   eigenvalues).
> - **"Reward (terminal): ..."** → as of Day 10, this formula is no longer
>   terminal-only; every step now gets a shaped, non-zero reward via
>   potential-based shaping (`Φ(new) - Φ(prev)`). As of Day 12, the
>   `δ·normal_consistency` term shown here is gone too — it's been replaced
>   by `z_align` folded into `Φ` itself, so it no longer appears as a
>   separate terminal-only addition at all. See the Reward Function section
>   above and
>   [RL_PIPELINE_OVERVIEW.md](RL_PIPELINE_OVERVIEW.md#the-reward-function--every-term-explained)
>   for the full current formula.
>
> Everything else in this diagram (voxel downsample strategy, action heads,
> Cython bridge, result fields) still matches Phase 1 as shipped.

**Why PPO:** Mixed action space (discrete levels + binary) is stable with PPO. SAC requires continuous actions and is harder to tune for this structure.

---


## Parameter Coupling Rules

These parameters interact. The agent will learn the coupling, but initialising them sensibly avoids early training instability:

| Coupling | Rule |
|---|---|
| `epsilon` ↔ `normal_thresh` | Loosening epsilon (catching more distant points) usually requires loosening normal_thresh too — those fringe points tend to have noisier normals |
| `epsilon` ↔ `min_support` | Larger epsilon → more inliers → can afford higher min_support without starving the search |
| `epsilon` ↔ `bitmap_epsilon` | Always `bitmap_epsilon = 2 × epsilon` automatically — never expose separately |
| `min_support` ↔ `point_density` | Normalise min_support by point_density in the state vector so the agent sees a scale-independent signal |
| `voxel_size` ↔ `min_support` | **Critical (Phase 3):** A coarser voxel (fewer points) requires a lower min_support floor — the agent must learn not to pick large min_support after choosing a coarse voxel, or enforce via action masking |

---

## Implementation Phases

### Phase 1 — Core RL Loop

**Goal:** Working agent on TartanAir with 3 actions. Establish the baseline gap between fixed params and adaptive params.

1. Build the Gym environment
   - `reset()` — load a frame, **apply fixed voxel downsample (voxel_size=0.05m)**, compute scene features, return 28-dim state *(shipped: 31-dim, see implementation note under State Space)*
   - `step(action)` — call Schnabel via Cython bridge, compute feedback, return `(next_state, reward, done)`
2. Implement 28-dim state (scene + feedback only) *(shipped: 31-dim)*
3. Implement self-supervised reward *(shipped, then reworked Day 10 — see implementation note under Reward Function)*
4. Action space: `epsilon` (8 levels) + `min_support` (6 levels) + `stop/continue` (binary)
5. Train PPO
6. Compare against fixed-param baseline on 13,049 TartanAir frames

> **Voxel note:** Use a single fixed voxel_size=0.05m for all frames in Phase 1. This keeps ~10k–20k points per frame, which is fast enough for 5 Schnabel calls per episode. Do not vary it — consistent density is what allows the agent to learn stable min_support values.

---

### Phase 2 — Richer Actions + Temporal Memory

**Goal:** Close the remaining gap with the full action space and cross-frame context.

1. Add `normal_thresh` as a 4th action (6 discrete levels: 0.80, 0.85, 0.88, 0.90, 0.93, 0.95)
2. Add temporal features — grow state from 28-dim to 33-dim
3. Add `m_probability` as an action (3 levels: 0.001, 0.005, 0.01) — controls how exhaustively Schnabel searches
4. **Replace fixed voxel with rule-based adaptive downsampling:**
   ```python
   def adaptive_voxel_size(points, target_points=15_000):
       volume = compute_bbox_volume(points)
       voxel = (volume / target_points) ** (1/3)
       return np.clip(voxel, 0.02, 0.15)
   ```
   This stabilises point count across near/far and dense/sparse frames without any training cost.
5. Run ablations: which state features contribute most? Which action matters most? Does adaptive voxel improve mean inlier ratio vs fixed?

---

### Phase 3 — Deep Geometric Understanding (Conditional)

**Gate condition:** Pursue Phase 3 only if Phase 2 bad frames > 100 OR mean inlier ratio plateaus below 0.7. If Phase 2 already meets targets, skip to SemanticKITTI benchmarking only.

**Goal:** Replace handcrafted heuristics with a unified deep learning backbone (PointNet++) to deeply understand the 3D geometry, acting as both a feature extractor and a per-point estimator.

1. **SemanticKITTI Integration** — Integrate KITTI `.bin` + `.label` files. Switch to IoU-based reward for training. Enables comparison against published baselines and provides supervised labels for pre-training the deep backbone.
2. **Unified Deep Backbone Training** — Pre-train a PointNet++ model in a supervised manner on SemanticKITTI. Freeze the backbone weights for RL training to maintain PPO stability. *Note: Aggressively downsample the point cloud (e.g., 2048 points) to maintain real-time performance (< 50ms). Prefer PointNet++ over DGCNN — it is faster and ball query naturally handles varying density.*
3. **Deep RL State Extractor** — Replace the 18 handcrafted scene features with a 128-dim global embedding generated by the frozen PointNet++ backbone. Concatenate with feedback features. The PPO agent uses this richer representation to choose parameters.
4. **Deep Terrain Classifier** — Add a classification head to the PointNet++ backbone to classify the frame into terrains (e.g., Flat, Rough, Slope). Route the frame to a specialized PPO sub-policy based on the classification.
5. **PointImportance (Groundness Prediction)** — Use the segmentation architecture of PointNet++ to predict a per-point ground probability score. Feed these weights into Schnabel's octree to bias the sampling towards likely ground points. Requires modifying the Cython bridge.
6. **kNN adaptation** — update kNN every N frames as a slow outer loop (not per step). Each change requires recomputing `calcNormals()` over all points.

---

## Success Metrics

| Metric | Phase 1 target | Phase 2 target | Phase 3 target |
|---|---|---|---|
| Mean inlier ratio (TartanAir) | Beat fixed-param baseline | Beat Phase 1 | Beat Phase 2 |
| Bad frames (<1% ground coverage) | Reduce from 285 / 678 | < 50 | < 20 |
| Mean steps used per frame | < 3.0 | < 2.5 | < 2.0 |
| Runtime overhead vs fixed params | < 3× | < 2× | < 3× (GPU inference added) |
| IoU vs GT (SemanticKITTI) | — | Beat fixed-param baseline | Beat published baselines |

---

## Evaluation: Normal Architecture vs PointNet++ Backbone

A head-to-head comparison across two axes — **runtime** and **accuracy** — to justify whether the added complexity of Phase 3 is worth it.

### Runtime Comparison

Measure wall-clock time per frame (mean ± std over the full evaluation set). All timings on the same hardware.

| Component | Normal MLP (Phase 1–2) | PointNet++ (Phase 3) | Notes |
|---|---|---|---|
| **Voxel downsample** | ~2–5 ms (fixed/rule-based) | ~2–5 ms (same) | Identical — not architecture-dependent |
| **Feature extraction** | ~1–3 ms (handcrafted 18-dim) | ~15–40 ms (PointNet++ forward pass on 2048 pts, GPU) | Major cost difference — PointNet++ adds a neural forward pass |
| **Policy inference** | <1 ms (MLP 28→64→64→heads) | ~1–2 ms (MLP 128+10→64→64→heads) | Slightly larger input dim but negligible difference |
| **Schnabel C++ call (per step)** | ~5–20 ms | ~5–20 ms (+ groundness-weighted sampling overhead) | PointImportance adds ~2–5 ms if octree weighting is enabled |
| **Total per step** | ~10–30 ms | ~25–70 ms | PointNet++ roughly 2–3× slower per step |
| **Total per episode (avg 2.5 steps)** | ~25–75 ms | ~60–175 ms | Includes one-time feature extraction + N × step cost |
| **Throughput (frames/sec)** | ~15–40 fps | ~6–16 fps | PointNet++ still real-time for LiDAR (10 Hz) if ≤2 steps |
| **GPU memory** | None (CPU only) | ~200–500 MB (PointNet++ model) | MLP is deployable on CPU-only edge devices |

**Key runtime questions to answer:**
- Does PointNet++ stay under the 100 ms/frame budget for real-time LiDAR (10 Hz)?
- Does the improved accuracy reduce mean steps used, partially offsetting the per-step cost?
- Is batch inference (multiple frames queued) viable to amortise GPU overhead?

### Accuracy Comparison

Evaluate on both TartanAir (self-supervised) and SemanticKITTI (supervised ground-truth).

| Metric | Fixed Baseline | Normal MLP (Phase 1) | Normal MLP (Phase 2) | PointNet++ (Phase 3) |
|---|---|---|---|---|
| **Mean inlier ratio (TartanAir)** | Baseline | > Baseline | > Phase 1 | > Phase 2 |
| **Bad frames < 1% coverage (TartanAir, /678)** | 285 | < 285 | < 50 | < 20 |
| **Mean IoU (SemanticKITTI)** | Fixed-param IoU | — | > Fixed-param IoU | > Phase 2 IoU |
| **Precision (ground)** | Baseline | Measure | Measure | Measure |
| **Recall (ground)** | Baseline | Measure | Measure | Measure |
| **F1 (ground)** | Baseline | Measure | Measure | Measure |
| **Mean residual (point-to-plane dist)** | Baseline | < Baseline | < Phase 1 | < Phase 2 |
| **Normal angle error (detected vs GT plane)** | Baseline | Measure | Measure | Measure |
| **Worst-case frame IoU (5th percentile)** | Baseline | Measure | Measure | Measure |
| **Per-terrain accuracy (Flat / Rough / Slope)** | — | — | — | Measure (terrain classifier enables this) |

### Evaluation Protocol

1. **Datasets:**
   - TartanAir: 13,049 frames — self-supervised metrics (inlier ratio, residual, bad frames)
   - SemanticKITTI: sequences 00–10 — supervised metrics (IoU, Precision, Recall, F1)

2. **Runs per configuration:** 3 seeds × full evaluation set → report mean ± std

3. **Statistical tests:** Paired t-test or Wilcoxon signed-rank on per-frame IoU to confirm Phase 3 gains are significant (p < 0.05)

4. **Ablation checklist:**

   | Ablation | What it tests |
   |---|---|
   | MLP + handcrafted features vs PointNet++ embedding | Is learned geometry better than hand-designed features? |
   | PointNet++ global embedding only vs + PointImportance | Does per-point groundness prediction add value beyond the global feature? |
   | PointNet++ + terrain routing vs single policy | Does specialised sub-policies per terrain type help? |
   | Frozen backbone vs fine-tuned backbone | Is fine-tuning stable under PPO, or does it hurt? |
   | 2048 pts vs 4096 pts for PointNet++ | Accuracy vs latency trade-off for the backbone input size |

5. **Visualisation:**
   - Per-frame scatter plot: IoU (MLP) vs IoU (PointNet++) — points above the diagonal = PointNet++ wins
   - Runtime histogram: distribution of per-frame latency for both architectures
   - Failure case gallery: frames where MLP fails but PointNet++ succeeds (and vice versa)
