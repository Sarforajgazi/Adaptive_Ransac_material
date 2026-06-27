# Adaptive RANSAC — RL Implementation Plan

> Building a Deep RL agent that wraps the Schnabel C++ RANSAC engine and learns to select the right parameters per scan, decide when the result is good enough, and improve using context from previous frames.

---

## The Core Idea

Standard RANSAC uses fixed parameters for every scan. But real LiDAR data is not uniform — scan density changes with distance, terrain roughness varies, and what works on a flat road fails on a slope. The agent's job is to observe each scan, pick parameters that suit it, and decide when to stop refining.

The key design choice: the agent does **sequential decision-making**, not one-shot prediction. It runs Schnabel, looks at the result, then decides whether to stop or try again with adjusted parameters — up to 5 times per scan. This lets it recover from bad initial guesses rather than being locked into one prediction.

---

## The Backend

**Schnabel C++ via the existing Cython bridge.** It is already running on TartanAir/TartanGround (678 frames processed). The RL agent sits on top of it in Python and controls only what parameters it receives. The agent never touches the internal algorithm.

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

---

## State Space (28-dim Phase 1 → 33-dim Phase 2)

What the agent observes at each step.

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

---

## Episode Structure

```
1 episode = 1 LiDAR scan frame

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

---

## Architecture

```
LiDAR frame
       │
       ▼
State Builder
  • 18 scene features     (from raw points — computed once per frame)
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

---

## Implementation Phases

### Phase 1 — Core RL Loop

**Goal:** Working agent on TartanAir with 3 actions. Establish the baseline gap between fixed params and adaptive params.

1. Build the Gym environment
   - `reset()` — load a frame, compute scene features, return 24-dim state
   - `step(action)` — call Schnabel via Cython bridge, compute feedback, return `(next_state, reward, done)`
2. Implement 24-dim state (scene + feedback only)
3. Implement self-supervised reward
4. Action space: `epsilon` (8 levels) + `min_support` (6 levels) + `stop/continue` (binary)
5. Train PPO
6. Compare against fixed-param baseline on 678 TartanAir frames

---

### Phase 2 — Richer Actions + Temporal Memory

**Goal:** Close the remaining gap with the full action space and cross-frame context.

1. Add `normal_thresh` as a 4th action (6 discrete levels: 0.80, 0.85, 0.88, 0.90, 0.93, 0.95)
2. Add temporal features — grow state from 24-dim to 32-dim
3. Add `m_probability` as an action (3 levels: 0.001, 0.005, 0.01) — controls how exhaustively Schnabel searches
4. Run ablations: which state features contribute most? Which action matters most?

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
