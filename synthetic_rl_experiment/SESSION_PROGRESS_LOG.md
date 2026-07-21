# Adaptive RANSAC RL — Synthetic Experiment: Session Progress Log

This document logs the full debugging, fixing, and development process carried
out on `synthetic_rl_experiment/` in this session: the problem that triggered
the investigation, every bug found and how it was diagnosed, every fix
applied and how it was verified, the pilot training results, the real-world
generalization investigation, and the Phase 2-6 synthetic-realism work. It is
written for reuse in report/thesis writing — each claim below is backed by a
number or a direct code reference produced during the session, not a
recollection.

---

## 1. Starting Problem

An existing evaluation plot (`adaptivity_results_all.png`, 6 panels: chosen
`epsilon`/`min_support`/`normal_threshold` vs. `noise_sigma`, and
`angle_error`/`inlier_recovery_rate`/`score` vs. `noise_sigma`) showed the
`RL_Agent` line completely **flat** across all three top-row parameter
panels — constant at `eps=0.15`, `min_support=150`, `normal_thresh=0.85`
regardless of noise level — while the fixed baselines were flat by design
(they never adapt). This defeated the entire premise of the project: an RL
agent that doesn't vary its chosen RANSAC parameters with the scene isn't
adaptive.

Two questions were posed: (1) why is the RL parameter selection constant,
and (2) is the implementation plan (`ADAPTIVE_RL_PLAN.md`,
`STRATEGY_AND_IMPROVEMENTS.md`) sound.

---

## 2. Root-Cause Investigation

### 2.1 Confirmed: this matched a previously-documented policy collapse

`synthetic_rl_experiment/syntheticRL.md` §4 already documented an earlier
10k-step run that suffered **policy collapse**: the agent output the exact
same action (`eps=0.15, min_supp=150, norm_th=0.85`) for every state. That
document's own root-cause: `explained_variance` of the PPO value network
stayed at `5.96e-8` (effectively zero) throughout training, because the
Schnabel RANSAC C++ extension's internal seed is not exposed to Python
(`srand(time(NULL))`, 1-second resolution) — its stochasticity (SD ≈ 0.20)
dominated the true geometric reward gradient (≈0.15-0.25), so the value
network could never learn a state→return mapping. The model checkpoint on
disk (`synthetic_ppo.zip`, timestamped before this session's fixes) matched
this exactly.

### 2.2 A second, independently-found bug: `train_synthetic.py` never had the fix already proven to work

`ransac_env.py`/`train_rl.py` (the main, non-synthetic pipeline) had already
hit and fixed this *exact* symptom — "agent outputs one constant action for
100% of frames" — with `ent_coef=0.01` (entropy bonus, discourages
deterministic collapse) plus `VecNormalize(norm_obs=True, norm_reward=True,
clip_obs=10.0)` (the 33-dim synthetic observation mixes wildly different
feature scales, e.g. `bbox_volume` vs. a 0-1 `normal_consistency` ratio).
`train_synthetic.py` had neither — it trained with plain PPO defaults.

### 2.3 `evaluate_synthetic.py`: the eval script itself never actually tested what it claimed to

`run_baseline()`/`run_rl()` opened with their own `env.reset()` call, which
**discarded** the deliberately pinned scene set moments earlier by a
`force_scene()` helper. So the `noise_sigma`/`inlier_ratio` columns written
to `synthetic_eval_results.csv` were the *intended* sweep labels, not a
description of the point cloud RANSAC actually ran on. This alone would
flatten any apparent noise-vs-parameter trend, independent of the collapsed
policy.

### 2.4 `check_eps_signal.py`: noise was never actually controlled

The script set `env.noise_sigma = noise` and `env.inlier_ratio = 0.50` as
plain instance attributes, but `SyntheticRansacEnv` has no such attributes —
only `self.true_noise_sigma`/`self.true_inlier_ratio`, which `reset()`
overwrites unconditionally with fresh `uniform(0.01, 0.20)` /
`uniform(0.03, 0.9)` draws. Both the "noise=0.01" and "noise=0.20" test arms
were silently sampling from the *same* full random range — never two
distinct conditions.

### 2.5 `check_eps_signal.py`: printed eps values were also mislabeled

A hardcoded local `eps_vals = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
0.40, 0.50]` had drifted out of sync with the real `EPS_LEVELS` in
`synthetic_env.py` (`[0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
0.60]`). Every printed "Eps X" value was the environment's actual value
shifted one index over — e.g. index 0 was printed as "0.05" but the
environment actually ran `0.08`. This directly explained a suspicious prior
finding (from a separate validation pass) that "eps=0.10 is unified-optimal
across both noise levels" — an artifact of both bugs (2.4 and 2.5) compounding,
not a real property of the geometry.

---

## 3. Fixes Applied (Round 1 — mechanics)

| File | Fix |
|---|---|
| `synthetic_env.py` | `reset(seed, options)` now reads `noise_sigma`/`inlier_ratio`/`orientation`/`slope_angle_deg`/`generator_seed`/`num_points` from `options` when present, instead of always randomizing. Verified: pinning `noise_sigma=0.01` vs `0.20` produced real generated-scene residual stds of `0.0099` vs `0.1989` (previously indistinguishable). |
| `evaluate_synthetic.py` | Removed the double-reset bug; `run_baseline`/`run_rl` now accept `reset_options` and pass them straight into the fixed `reset()`. |
| `check_eps_signal.py` | Imports `EPS_LEVELS`/`NORM_THRESH_LEVELS` directly from `synthetic_env` (single source of truth, no more hand-copied list) and uses `reset(options=...)` for real noise control. |
| `train_synthetic.py` | Added `ent_coef=0.01` and `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)`, matching the already-proven main-pipeline fix. Also added per-episode `epsilon` logging and NaN-reward counting to the training callback, and wall-clock timing, so a pilot run's mechanics can be checked directly from its log. |

### 3.1 `EPS_LEVELS` range was also too narrow

A corrected (noise-controlled) eps sweep showed score still rising
monotonically all the way to the *lowest* tested eps (`0.08`) at
`noise_sigma=0.01` — the true low-noise optimum was outside what the action
space could express. `EPS_LEVELS` was extended to
`[0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]` (9 → 11
levels; `MultiDiscrete` action space updated to match). A follow-up sweep
confirmed real, interior optima at both noise extremes:

| noise_sigma | best eps (in-range) | shape |
|---|---|---|
| 0.01 | 0.05 | craters at 0.03 (score 0.040, RANSAC fails to find a valid plane), peaks at 0.05, declines smoothly to 0.60 |
| 0.20 | 0.20 | craters at 0.03/0.05 (score 0.000, no valid plane found), rises to a real interior peak at 0.20, declines afterward |

### 3.2 GPU vs CPU

Measured directly rather than assumed: full `env.step()` = 196.3 ms, of
which `compute_scene_features()` (Open3D KD-tree + per-point PCA) = 125.1 ms
(64%) and the C++ RANSAC call = 36.3 ms (18%) — both 100% CPU-bound. GPU
(`GTX 1650 Ti`, confirmed available) offers no benefit — the `MlpPolicy`
network is too small for GPU transfer overhead to pay off. `train_synthetic.py`
now hardcodes `device="cpu"`.

---

## 4. 10k-Step Pilot Training Run (first trustworthy run under the fixes)

Command: `train_synthetic.py --timesteps 10000` (later relabeled
`synthetic_ppo_pilot`).

**Mechanical health checks (all passed):**
- Wall-clock: 2885.1s total (48.1 min), 0.2885 s/step.
- **Zero NaN rewards** across all 10,240 steps / 5,813 episodes.
- Unique eps values chosen across the entire run: all 11 action-space
  levels used — no collapse.
- Last 20 episodes' chosen eps still spanned the full range
  (`0.03` to `0.6`) — the policy had not prematurely collapsed even at the
  end of the pilot.

**The critical result:** `explained_variance` climbed
**-0.453 → 0.659 → 0.719 → 0.76** across the run's 4 PPO updates — the exact
metric that stayed pinned near zero (`5.96e-8`) for the entire duration of
the old collapsed run. This is direct, measured evidence that the fixes in
§3 let the value network learn a real state-dependent mapping this time,
not just a hope that they would.

**Open item flagged (not resolved in this session):** `rollout/ep_len_mean`
ended at **1.6** (of a max of 5 refinement steps per episode) — the agent
mostly stops after 1-2 steps rather than using its multi-step budget. Not
necessarily a bug (RANSAC's own stochasticity means "try again" isn't a
guaranteed win, and 10k steps / 4 gradient updates may simply not have had
time to develop this behavior), but worth watching in the next real
training run.

**A naming bug found and fixed during this run:** `train_synthetic.py`'s
`--out` argument handling used `args.out.startswith("../models")` instead
of checking for the literal default value — so *any* custom `--out` value
starting with that prefix (including the intended `../models/synthetic_ppo_pilot`)
was silently coerced back to the hardcoded `synthetic_ppo` name, **overwriting
the previous (collapsed, pre-fix) model with no warning.** Fixed by resolving
`--out` relative to the script's own directory and refusing to overwrite an
existing model unless `--force` is passed. The pilot's output was renamed
`synthetic_ppo_pilot.zip`/`_vecnormalize.pkl` to correct the record (no data
was actually lost — the overwritten file was the already-known-collapsed
model, valueless as a checkpoint).

---

## 5. Full Evaluation Sweep + Visualization (post-pilot)

`evaluate_synthetic.py` was also fixed to auto-detect the most recently
trained model (glob + file mtime) instead of a hardcoded name, so it would
actually find `synthetic_ppo_pilot.zip` after the rename.

**Result — RL_Agent parameter adaptivity, mean values per noise bin
(20 trials/bin, 5 noise levels × 5 inlier-ratio levels):**

| noise_sigma | eps | min_support | normal_thresh | score |
|---|---|---|---|---|
| 0.0100 | 0.126 | 37.8 | 0.819 | 1.769 |
| 0.0575 | 0.142 | 51.6 | 0.798 | 1.748 |
| 0.1050 | 0.175 | 45.4 | 0.728 | 1.684 |
| 0.1525 | 0.221 | 50.2 | 0.677 | 1.584 |
| 0.2000 | 0.234 | 46.4 | 0.676 | 1.482 |

Compared to the old collapsed run (flat at `eps=0.15`, `min_support=150`,
`norm_th=0.85` for every noise level): **eps now genuinely increases 0.126 →
0.234 (1.9x, monotonic)** and **normal_thresh decreases 0.819 → 0.676
(monotonic)** with noise. `min_support` stayed roughly flat — interpreted as
the agent having learned that eps and normal_thresh are the noise-sensitive
levers in this environment, not min_support (this was not independently
verified in isolation at that point — see §9 for the later, corrected
verification methodology that would be needed to check a specific parameter
like this).

On performance: RL_Agent tracked close to the best fixed baseline
(`Standard`) across noise levels and pulled clearly ahead of `Super Strict`
at high noise, where `Super Strict`'s score collapsed toward 0.5 while
RL_Agent stayed around 1.48 — meaningful given this was only a 10k-step
pilot, not a converged policy.

**New capability added:** `visualize_synthetic_plane.py` extended to load a
trained model, run it through a real episode, and render the true plane
(blue) vs. the RL-fitted plane (purple) in Open3D, printing the chosen
`eps`/`min_support`/`normal_thresh` and step count. Falls back to the old
fixed-baseline (`eps=0.15, min_support=100, normal_thresh=0.85`) rendering
if no model is found.

---

## 6. Real-Point-Cloud Test (does this generalize to real data?)

Tested the pilot model directly against a real 102,249-point TartanGround
scene reconstruction (`schnabel_cython/tartanair_data/Sewerage/Sewerage_rgb_original.ply`,
bounding box 104.7m × 75.9m × 28.9m — an entire multi-room/multi-surface
scene, not a single dominant plane).

**Result: min_support fragmentation.** The RL agent chose
`eps=0.1, min_support=75, normal_thresh=0.85` for this scene (computed via a
manual single-step inference, bypassing `step()`'s ground-truth-dependent
shape selection — see below). Run against the real cloud:

| | RL-chosen params | Hand-tuned heuristic (`eps=0.15, min_support=2000, normal_thresh=0.9`) |
|---|---|---|
| Shapes found | 30 (hit the `max_shapes` cap) | 1 |
| Largest shape | 6,177 pts (6.04% of cloud) | 9,124 pts (8.92% of cloud) |
| Total coverage | 34.6% of cloud, fragmented | one coherent structure, found immediately |

**Root cause:** `min_support` was a raw absolute point count
(`[30, 50, 75, 100, 150, 200]`), calibrated against 10,000-point training
clouds (75 = 0.75% of the cloud). Applied unchanged to a 102,249-point real
scene, 75 points is **0.07%** of the cloud — two orders of magnitude looser
relative to scene scale than during training, so RANSAC accepted dozens of
small coincidental clusters as "planes" instead of the scene's actual large
structural surfaces.

**A second, structural blocker (not a calibration issue — a design gap):**
`SyntheticRansacEnv.step()` selects which of RANSAC's candidate shapes to
score by comparing to a *known* ground-truth normal (`self.n_true`), which
only exists because training data is synthetic and labeled. For a real,
unlabeled cloud there is nothing to compare against — the current code
cannot run its normal multi-step inference loop on real data without this
selection logic being replaced (e.g. largest-support, or most-horizontal-normal
for a ground-plane-specific task).

---

## 7. Relative-Decode Fix — Implemented, Verified, Then Shelved by Design Decision

### 7.1 `min_support` as a fraction of point count (kept — safe, no retrain needed)

`MIN_SUPPORT_LEVELS` changed from `[30, 50, 75, 100, 150, 200]` (absolute
counts) to `[0.003, 0.005, 0.0075, 0.01, 0.015, 0.02]` (fractions).
`_decode_action()` now computes `min_supp = int(fraction * len(current_points))`.
Because every synthetic training/eval scene uses exactly `num_points=10000`
everywhere in the codebase, this reproduces the **exact same absolute
values** in the synthetic domain — verified directly (`0.0075 * 10000 =
75`, matching the old value exactly at every inlier ratio 0.03-0.9 tested).
No retrain was ever needed for this half of the fix.

### 7.2 `eps` as a multiple of a scale reference (implemented, verified, then reverted)

First attempt used the observation's existing `mean_knn_dist` feature as the
scale reference. **This was wrong and caught by the session's own
verification step, not shipped uncaught:** `mean_knn_dist` is a whole-cloud
mean, dominated by sparse outlier spacing. Measured directly: inlier-only
mean NN distance ≈ 0.113m, but the whole-cloud mean ≈ 0.575m — using it
would have made every eps value 5-10x too large.

Fixed with a dedicated, standalone `estimate_eps_scale_reference()` (a
low-percentile, p10, k-NN distance over a random sample — resistant to
outlier domination since dense inlier neighborhoods are still found at low
percentiles even when outliers dominate the mean). Deliberately kept
separate from `compute_scene_features()` so the model's fixed 33-dim
observation space was never touched.

Verified on the real Sewerage scene: `eps` now ranged **0.047m to 0.931m**
(physically sensible for architecture) instead of the fixed absolute
`0.03m-0.6m` applied blind to scene scale, and `min_support` ranged
**306 to 2,044 points** — directly bracketing the 2,000-point heuristic
that had worked earlier.

**Caveat found and documented:** this scale reference is not perfectly
clean — it still correlates with `inlier_ratio` (varied ~5.7x across the
training range, 0.10m at high inlier ratio to 0.58m at low), since there is
no way to unsupervised-perfectly separate "inlier spacing" from "outlier
spacing" without ground truth.

### 7.3 Decision: reverted the eps part, kept the min_support part

Unlike `min_support`, changing `eps`'s decode function **changes
training-time semantics** — the scale reference varies per synthetic
episode (unlike point count, which is always 10000) — so the already-trained
pilot model's eps calibration no longer matched, and a fresh retrain would
be needed to validate it before trusting any result built on top of it.

Given the project's priority (build a strong synthetic-only result for
Phases 2-6 first; defer real-point-cloud work to Phases 7-8, which is the
only thing the relative-eps fix was for), the decision was made to **revert
just the eps decode** back to fixed absolute meters (matching what
`synthetic_ppo_pilot.zip` was actually trained under), while keeping the
safe `min_support` fraction fix. `estimate_eps_scale_reference()` was left
in `synthetic_env.py`, unused, so this work is not lost when Phases 7-8 are
eventually tackled — just deferred.

Verified after reverting: `_decode_action()` once again produces exactly
`eps=0.1` regardless of scene (matching what the pilot was trained under),
`min_support` still correctly gives `75` at index 2, and the pilot model
loads and runs a full episode end-to-end without error.

---

## 8. Infrastructure: Consistent Naming / Versioning

Built a single naming convention, `run_name_for(tag)` in `train_synthetic.py`
(`None` → `"synthetic_ppo"`, `"v2"` → `"synthetic_ppo_v2"`), shared by every
script so nothing is hand-kept-in-sync:

| Artifact | Path |
|---|---|
| Model | `models/<run_name>.zip` |
| VecNormalize stats | `models/<run_name>_vecnormalize.pkl` |
| Training log (auto-written via a `Tee` to both stdout and file, not manual shell redirection) | `synthetic_rl_experiment/logs/<run_name>_train.log` |
| Eval results | `synthetic_rl_experiment/logs/<run_name>_eval.csv` |
| Adaptivity plot | `synthetic_rl_experiment/plots/<run_name>_adaptivity.png` |

`train_synthetic.py --tag <name>` refuses to overwrite an existing model
unless `--force` is passed (directly closing the bug from §4).
`evaluate_synthetic.py`/`plot_adaptivity.py`/`visualize_synthetic_plane.py`
all accept `--tag`, and auto-detect the most recently modified matching
file (by mtime) when no tag is given, rather than a hand-maintained
candidate list.

**Point-cloud persistence added:** `data_generator.py` gained
`save_scene()`/`load_scene()` (writes a colored `.ply` — green=true inlier,
gray=outlier/clutter — plus a `_meta.json` with `n_true`, `d_true`,
`gt_mask`, and every generation parameter, for exact reproducibility).
Wired into `visualize_synthetic_plane.py` via `--save_scene <name>` →
`synthetic_rl_experiment/saved_scenes/<name>.ply`. Verified: round-trip
save→load reproduces the original points/mask/normal/offset exactly.

---

## 9. Roadmap Discovery: `roadmap_synthetic_to_real.md`

An existing 8-phase roadmap was found already covering the "make this work
on real point clouds" question:

| Phase | What | Modifies |
|---|---|---|
| 1 | Document current result (baseline) | — |
| 2 | Surface deformations (bumps/craters) | `data_generator.py` |
| 3 | Noise model diversity | `data_generator.py` |
| 4 | Structured clutter (cylinders/boxes) | `data_generator.py` |
| 5 | Near-tangent intersecting plane | `data_generator.py` |
| 6 | Combined 50k-100k retrain | `synthetic_env.py`, `train_synthetic.py` |
| 7 | Zero-shot transfer to TartanAir | new `evaluate_tartanair.py` |
| 8 | Fine-tune on RELLIS-3D | new `rellis_env.py` |

**A gap flagged in the roadmap's own Phase 7 plan:** it proposes handling
real-point-cloud scale mismatch by "subsample or pad to ~10000 points" —
this only normalizes point *count*, not physical *scale/density*. Sewerage's
bounding box (105m × 76m × 29m) is far larger than the training generator's
fixed 20m × 20m × 20m box; a 10,000-point sample of Sewerage would still
have much sparser local point spacing than a 10,000-point training sample,
so `eps` (an absolute-distance parameter under the reverted, currently-active
decode) would likely still be miscalibrated even after count-subsampling.
Not yet resolved — relevant when Phase 7 is eventually reached.

**Decision:** defer Phases 7-8 (real-point-cloud work) to the end. Proceed
through Phases 2-6 first, to produce a complete, defensible synthetic-only
result. This also motivated reverting the eps relative-decode work (§7.3) —
that work was specifically for Phase 7/8 and not needed for 2-6.

---

## 10. Phases 2-5 Implementation (all `data_generator.py`-only changes)

### 10.1 Phase 3 — Noise Model Diversity (implemented first, out of roadmap order, at explicit request)

`SyntheticPlaneGenerator._apply_noise(inliers, n_true, noise_sigma, noise_type)`
added, supporting:
- `gaussian` — the original model.
- `laplacian` — heavy-tailed, models multi-path-reflection-style outlier displacements.
- `uniform` — bounded noise, models quantization/structured sensor error.
- `spatially_varying` — noise grows with distance from a sensor placed at the origin.
- `mixed` — 80% Gaussian + 20% heavy-tailed (Laplacian at 2x sigma).

Wired into `synthetic_env.py`'s `reset()`, randomized per episode (biased
2:1 toward gaussian). **Verified:** all 5 types generate with no NaN/Inf,
sensible residual scales (std ≈ 0.10 at `noise_sigma=0.10` for
gaussian/laplacian/uniform); the observation's P25 surface-variation feature
stayed clearly above the low-noise baseline (`0.0006`) for every type
(range `0.0204`-`0.0583`), confirming the observation space stays
informative under all 5 noise models. Full env ran 20 clean episodes with
`noise_type` actively randomized.

### 10.2 Phase 2 — Surface Deformations (bumps/craters)

Implemented after being flagged as skipped (Phase 3 was done first per an
explicit "add noise now" request; Phase 2 was then added to catch up to
roadmap order before Phase 4/5).

`_add_surface_deformations(inliers, inliers_uv, n_true, num_bumps, box_size)`
adds Gaussian-falloff bumps/craters (random center, radius `0.5-2.5`,
height originally `±0.4` — **later corrected to `±0.10`, see §11**),
applied to inlier points *before* noise, keeping `gt_mask=True` (terrain,
not clutter). Wired into `reset()`, 50% chance of `num_bumps=0`, otherwise
1-3. **Verified:** `num_bumps=0` reproduces the pre-existing pure-noise
residual std exactly; `num_bumps≥1` shows real terrain deviation; `gt_mask`
integrity held for all bump-affected points.

### 10.3 Phase 4 — Structured Clutter (cylinders/boxes)

`_generate_cylinder()` and `_generate_box()` added — vertical cylinders
(radius 0.1-0.5m, height 1-4m) and boxes (width/depth/height 0.3-1.5m)
based near the ground plane, `gt_mask=False` (false-positive traps for
loose eps values, not ground truth). A `_create_plane_frame(normal)` helper
was extracted (also used to DRY-refactor `generate_scene()`'s own basis
construction) to derive the two in-plane perpendicular vectors for any
normal.

**A real bug found during verification, not just a smoke test:** both
generators initially wrote `up, side1 = self._create_plane_frame(n_ground)`
— but `_create_plane_frame` returns the two vectors *perpendicular* to its
input, not the input itself, so `up` was silently assigned an in-plane
direction instead of the actual ground normal. Cylinders were squashed flat
(max height 0.134m measured, instead of up to ~4m) rather than crashing —
confirming this class of change is lower-risk than the env/scoring bugs in
§2 (it didn't corrupt the training loop) but still required a real
geometric check, not just "did it run without an exception," to catch.
Fixed by explicitly normalizing `n_ground` as `up` and deriving
`side1, side2 = self._create_plane_frame(up)`.

**Post-fix verification:** cylinder points showed a constant radial spread
matching the sampled radius (0.134m, matching its `[0.1, 0.5]` draw) and a
proper vertical extent; box points fell within the expected height range.
Clutter concentration near the ground was confirmed quantitatively:
**23.2%** of `gt_mask=False` points fell within 1.5m of the true plane with
clutter enabled, vs. **15.3%** in a no-clutter baseline scene.

### 10.4 Phase 5 — Near-Tangent Intersecting Plane

`_generate_intersecting_plane(n_ground, d_ground, p0, u, v, box_size, num_points)`
added — a second plane (wall/ramp) at a randomized intersection angle:
5-15° (near-tangent, hardest case), 30-60° (moderate), or 80-90°
(perpendicular, easiest), built via Rodrigues rotation of the ground normal
around a random in-plane axis, `gt_mask=False`. Wired into `reset()` at 25%
probability per episode. **Verified in isolation:** correct angle sampling
(tested at 83°), wall points spread consistently with the chosen angle
(distance-to-ground-plane range 0.017m-3.469m for an 83° near-perpendicular
case, matching `sin(83°) × box_size/3 ≈ 3.3m` expected).

### 10.5 A proactive correctness fix: outlier-index collision

Both Phase 4 (clutter) and Phase 5 (wall) write their points into the
`outliers` array by replacing a slice starting at index 0
(`outliers[:budget] = ...`). If both were active in the same scene without
care, Phase 5 would silently overwrite some of Phase 4's clutter (a
same-class "silent failure," not a crash). Fixed with a running `offset`
counter so both write into disjoint index ranges, capped so neither can
exceed the available `num_outliers` budget.

**Combined verification:** full env ran 15-20 clean episodes with all of
Phases 2-5 (bumps, noise diversity, clutter, intersecting plane)
simultaneously randomized — no NaN, stable 33-dim observation shape.

---

## 11. Pre-Phase-6 Stress Test and the Bump-Amplitude Bug

Before committing to the 4-8 hour Phase 6 combined retrain, `check_eps_signal.py`
was extended with `--full_complexity` (forces bumps + clutter + intersecting
plane ON, rather than relying on `reset()`'s partial random inclusion which
would dilute easy and hard scenes together) and `--seeds` (reduced sample
size for a faster go/no-go check rather than the original 250-seed rigor).

**First run (bump height `±0.4`, 50 seeds/eps):** gradient magnitude was
large and clearly outside SE bars (1.74 at low noise, 1.41 at high noise —
not washed into statistical noise), **but the peak eps converged to the
same value (0.15) at both `noise_sigma=0.01` and `noise_sigma=0.20`**
(0.20 tied with 0.15 within SE at high noise) — a much weaker
noise-dependent signal than the pre-Phase-2-5 result, where the peak had
clearly shifted `0.05 → 0.20` (4x) between the same two noise levels.

**Diagnosis:** bump height (`±0.4m`) was up to **40x** `noise_sigma` at the
low end (`0.01`) — since bumps are `gt_mask=True` (they count toward
`inlier_recovery_rate`, unlike clutter/wall which only affect
`false_inlier_rate`), their amplitude directly set a noise-independent floor
on how tight `eps` could be before RANSAC started rejecting real terrain.
Verified directly: bump+noise residual std was **0.0317** at
`noise_sigma=0.01` — **3x** pure-noise-alone std (0.0100).

**Fix:** `_add_surface_deformations`'s height range reduced from
`uniform(-0.4, 0.4)` to `uniform(-0.10, 0.10)` — comparable to the
`noise_sigma` range (0.01-0.20) and realistic for actual terrain roughness
(centimeter-scale, not boulder-scale). Verified: residual std at
`noise_sigma=0.01` dropped to **0.0092**, matching pure noise (0.0100)
almost exactly.

**Rerun after the fix (50 seeds/eps):** clean separation restored —

| | noise_sigma=0.01 | noise_sigma=0.20 |
|---|---|---|
| Best eps | **0.05** (score 1.8898 ± 0.0057) | **0.20** (score 1.5175 ± 0.0160) |
| eps=0.05 score | 1.8898 ± 0.0057 | 0.1629 ± 0.0571 |
| eps=0.20 score | 1.8314 ± 0.0103 | 1.5175 ± 0.0160 |

Tighter eps clearly wins at low noise, looser eps clearly wins at high
noise, separation far outside SE bars in both directions — matching the
pre-Phase-2-5 clean-environment result (`0.05` / `0.20`) almost exactly,
while keeping the full Phase 2-5 complexity (reduced-amplitude bumps,
clutter, intersecting plane) active.

---

## 12. Evaluation Script Readiness: Stratification Columns + a Second Fairness Bug

Prepared ahead of Phase 6 completing, so evaluation could run immediately
rather than requiring script changes afterward.

`synthetic_env.py`: added `true_orientation`, `true_noise_type`,
`true_num_bumps`, `true_num_cylinders`, `true_num_boxes`,
`true_add_intersecting_plane` readback attributes to `reset()` (mirroring
the existing `true_inlier_ratio`/`true_noise_sigma` pattern). Previously
these scene-composition choices were local variables inside `reset()`,
never exposed — nothing outside the function could find out what a given
episode actually generated.

**A second fairness bug found while wiring this up (same category as §2.3):**
`evaluate_synthetic.py`'s trial loop never included `noise_type`/`num_bumps`/
`num_cylinders`/`num_boxes`/`add_intersecting_plane` in `reset_options`. Since
each of the 5 methods (4 baselines + RL) calls `env.reset()` independently
within a trial, and these fields were absent from `reset_options`, each call
drew its own scene composition from the environment's continuously-advancing
`self.np_random` stream — meaning baselines and RL were silently being
scored on *different* scene compositions even within "the same trial,"
despite `noise_sigma`/`inlier_ratio`/`generator_seed` being correctly pinned
identically.

**Fix:** scene composition (orientation, noise model, bumps, clutter,
intersecting plane) is now drawn once per trial from a dedicated seeded RNG
(`np.random.default_rng(seed)`, matching `reset()`'s own default
probabilities) and pinned identically into `reset_options` for every method
evaluated in that trial. New CSV columns added: `plane_type`, `noise_type`,
`has_bumps`, `has_clutter`, `has_intersecting_plane` — enabling stratified
plots (eps vs. noise with/without clutter, score by plane type) once Phase 6
data exists.

**Verified:** the readback attributes match the requested composition and
are identical across repeated `reset()` calls with the same options
(confirming the fairness fix); a full `run_baseline()` call with
clutter+wall+vertical-orientation simultaneously active runs end-to-end
with no errors.

## 13. `visualize_synthetic_plane.py` Extended for Phase 2-5

Added `--noise_type`, `--num_bumps`, `--num_cylinders`, `--num_boxes`,
`--add_intersecting_plane` CLI flags — each defaults to unset (randomizes
the same way training does, via `env.reset()`'s own defaults) unless
explicitly forced, so specific Phase 2-5 features can be deliberately
inspected rather than waiting for a random scene to include one.

`choose_params_rl()` now also returns the *actually realized* scene
composition (via the new `synthetic_env.py` readback attributes), not just
the CLI-requested one — necessary because some fields may be left unset and
randomized. `--save_scene`'s metadata now records this too, for full
reproducibility of saved scenes.

**Verified:** a forced composition (`num_bumps=2, num_cylinders=1,
num_boxes=1, add_intersecting_plane=True, noise_type=mixed`) propagates
correctly through `env.reset()` and is faithfully reported back via the
readback attributes.

## 14. Qualitative-Demo Replication Check (a near-miss caught before it reached the report)

A four-run qualitative analysis using `visualize_synthetic_plane.py` (one
seed per condition) claimed the RL agent: (a) tightens/loosens
`normal_thresh` based on noise type, and (b) raises `min_support`
specifically to resolve plane ambiguity when bumps and an intersecting
plane are both present — described as "a qualitatively different strategy"
demonstrating capability no fixed baseline has.

**Timeline check, done first:** confirmed only `synthetic_ppo_pilot.zip`
exists on disk, timestamped *before* Phases 2-5 were added to the code
(§10-11 postdate it). All four runs therefore used a policy that was never
trained on bumps, clutter, non-Gaussian noise, or an intersecting plane —
any behavior observed is pilot-model generalization to out-of-distribution
complexity, not learned complexity-handling. This alone means the original
causal framing ("the agent reads conditions and adjusts") could not be
correct as stated, independent of whether the numbers were accurate.

**Replication test:** each of the four described conditions was rerun at
N=15 seeds instead of N=1, using the pilot model, `inlier_ratio=0.5`:

| Condition | `normal_thresh` mean ± std | `min_support` mean ± std | values seen |
|---|---|---|---|
| A: gaussian, bumps+cylinders | 0.827 ± 0.040 | 68.0 ± 58.4 | 30, 50, 150, 200 |
| B: laplacian, cylinders+box+wall | 0.653 ± 0.034 | 38.0 ± 9.8 | 30, 50 |
| C: gaussian, intersecting plane | 0.767 ± 0.072 | 50.7 ± 49.9 | 30, 50, 150, 200 |
| D: mixed, bumps+wall | 0.613 ± 0.022 | 34.0 ± 8.0 | 30, 50 |

**`normal_thresh` finding confirmed:** std (0.02-0.07) is tight relative to
the spread across conditions (0.61-0.83) — a real, consistent, replicable
correlation between noise/complexity type and the pilot's chosen
`normal_thresh`. Kept, but reframed as pilot-model generalization
(correlation with shifted observation features), not deliberate
noise-reading.

**`min_support` "strategic raising" claim refuted:** condition D (the
closest match to the original "raised min_support" example) shows
`min_support` mean = 34 — the *lowest* of the four conditions, not raised.
Conditions A and C, which lack the "plane ambiguity" story, show *larger*
mean `min_support` and far higher variance (std ≈ mean, close to uniform
across the full action range). No condition shows a reliable rise in
`min_support` under plane ambiguity. This specific claim does not survive
replication and was discarded rather than used in reporting.

**`angle_error` reframed:** the original single-draw values (0.07°-0.24°)
were real but not representative of a tight distribution — std was often
comparable to or larger than the mean (e.g. condition D: mean 0.189°, std
0.130°, max 0.430°). Correct framing is "typically small, sometimes noisy,"
not "excellent across the board."

**Methodological takeaway:** a single-seed qualitative demonstration is not
sufficient evidence for a specific causal claim about learned behavior.
Checking via N≥15 replication in this one pass caught one real, keepable
signal and one false claim that would otherwise have gone into the report
unverified — worth citing as a validation-methodology point in its own
right.

---

## 15. Current Status (updated)

- All fixes in §3 are applied and verified.
- Pilot model `synthetic_ppo_pilot.zip` (10k steps) is trained, verified
  mechanically healthy, and fully valid under the current (reverted-to-absolute)
  eps/min_support decode — no retrain needed to use it as-is.
- Phases 2-5 are fully implemented in `data_generator.py`/`synthetic_env.py`,
  individually and jointly verified, with the bump-amplitude bug found and
  fixed, and the noise-driven eps gradient confirmed to survive the full
  combined complexity.
- `evaluate_synthetic.py` and `visualize_synthetic_plane.py` are both fully
  updated for Phase 2-5 (stratification columns, fairness fix, CLI controls)
  and verified — ready to use the moment a Phase 6 model exists.
- **Phase 6 is RUNNING as of this entry (launched after explicit go-ahead;
  see §24) — no longer staged/pending.** `train_synthetic.py --tag v2
  --timesteps 50000`, bundling the reward redesign (§22), the density/
  observation fix (§17), and Phases 2-5+sinusoidal (§10, §23) into one run.
  Not yet complete; §24 has the live status and will get the final result
  once it finishes.
- **Any claim about the agent's response to bumps/clutter/noise-diversity/
  intersecting-planes must be attributed to the Phase-6 (`v2`) model once
  it exists, not the pilot.** Current qualitative results (§14) are
  pilot-model out-of-distribution generalization only, and only the
  `normal_thresh` pattern has been replication-checked; the `min_support`
  claim was refuted.
- Phases 7-8 (real-point-cloud transfer) are deliberately deferred. The
  relative-decode groundwork (`estimate_eps_scale_reference()`,
  fraction-based `min_support`) remains in the code, unused for eps, ready
  to be re-activated (with a required retrain to re-validate eps
  calibration) when that phase is reached.
- Real-point-cloud testing so far (§6) used the *pre-Phase-2-5* pilot model
  against the Sewerage scene; it has not been repeated against any
  post-Phase-6 model, and the ground-truth-dependent shape-selection
  blocker in `step()` (§6) is still unresolved.
- **(§17) `num_points` is randomized per episode (log-uniform, 10k-100k)
  and the `prev_min_support` observation feedback bug (raw count instead
  of fraction — density-OOD) is fixed.** This invalidated the pilot's
  weights against the new 33-dim observation semantics, so it's bundled
  into the Phase 6 retrain (§24) rather than a separate pass — that retrain
  is now running.

## 16. Open Items / Known Limitations (for future work section)

1. `ep_len_mean` staying near 1.6-2.0 of a 5-step budget (§4) — unresolved,
   worth checking again after Phase 6.
2. `min_support`'s lack of a noise-dependent signal was inferred from the
   pilot's flat parameter trend (§5), never independently verified via its
   own controlled sweep (unlike eps, which was explicitly isolated and
   tested in §3.1 and §11). `run_se_sweep.py`/`run_final_sweep_mp.py` were
   identified as having the same reset-pinning bug as the original
   `check_eps_signal.py` (§2.4) and were never fixed in this session.
3. `step()`'s ground-truth-dependent shape selection (§6) blocks real
   multi-step RL inference on unlabeled data — needs a deployment-appropriate
   heuristic (largest-support, or most-horizontal-normal) before Phase 7.
4. The eps scale reference (§7.2) is an approximation — it correlates with
   `inlier_ratio`, not purely with noise/density, since true inlier-vs-outlier
   separation is unrecoverable without ground truth. Worth keeping in mind
   if/when Phase 7/8 reactivates it.
5. The roadmap's own Phase 7 plan (§9) may not fully solve real-scene eps
   calibration via point-count subsampling alone, since it doesn't address
   physical-scale mismatch — flagged but not resolved.
6. §14's replication check adds direct evidence to item 2: on the pilot
   model, `min_support` showed std comparable to or larger than its mean in
   2 of 4 tested conditions (essentially uniform across the action range),
   while `normal_thresh` was consistently tight (std 0.02-0.07). Whether
   `min_support` develops a real, low-variance noise/complexity-dependent
   signal after Phase 6 training is an open, specific, testable question —
   not yet answered for the pilot, and worth checking again once a
   Phase-6 model exists, ideally via `evaluate_synthetic.py`'s new
   stratification columns (§12) rather than another ad hoc replication script.
7. §17's `num_points` randomization has an unmeasured training-cost impact —
   `compute_scene_features()`'s Open3D normal estimation and
   `schnabel_ransac.detect()` both scale with point count, so episodes near
   100k points will take meaningfully longer than the fixed-10k pilot's did.
   No wall-clock estimate exists yet for the retrain this fix is bundled
   into; worth timing a small batch of high-N episodes before committing to
   a full run.

---

## 17. Density-Generalization Fix: `prev_min_support` OOD Bug + `num_points` Randomization

**Trigger:** a direct question — "I trained on 10k-point clouds; what happens
if the same model is run on a 100k-point cloud?" — surfaced a real,
previously-undiagnosed bug rather than just a hypothetical.

**What transfers and what doesn't, mechanically:**
- `eps` (`EPS_LEVELS`) is absolute meters — density-invariant by
  construction, transfers fine as long as noise scale/sensor is unchanged.
- `min_support`'s *action* output is already a fraction of point count
  (`MIN_SUPPORT_LEVELS`, converted via `len(self.current_points)` in
  `_decode_action()`, §7.1) — also density-invariant, transfers fine.

**The actual bug:** `_get_obs()` (`synthetic_env.py`) fed the *decoded
absolute count* (`self.prev_min_support`, e.g. up to ~200 at 10k points)
back into the observation vector, not the fraction the action space itself
uses. At 100k points the same action index decodes to ~10x that count —
a value never seen in training, pushing that one observation dimension
out-of-distribution starting at step 2 of every episode (step 1's value is
always 0, so it's unaffected).

**Why this went unnoticed:** the current pilot model is effectively
one-shot — per-step/runtime penalties collapsed it to a single meaningful
pick, so step-2+ observations barely influence the output today. The bug
was latent, not absent. It becomes load-bearing the moment the
in-progress reward redesign (`check_refinement_benefit.py`) restores
genuine iterative refinement across steps — at which point a density sweep
run *without* this fix would misattribute the resulting degradation to
"the reward change didn't generalize" rather than to this observation
feature. Fixing it now, ahead of that retrain, preempts that
misdiagnosis.

**Second gap found in the same discussion:** `num_points` had been
hardcoded to `10000` in *every* episode this project has ever run, training
or eval (see the §3.1-era comment this superseded). Density had literally
never varied, so density-robustness was assumed, not trained for or tested.

**Fixes applied (`synthetic_env.py`), code-only, no training launched:**
1. `_decode_action()` now also returns `min_supp_frac` (the raw fraction
   from `MIN_SUPPORT_LEVELS`), alongside the existing absolute `min_supp`
   used for the actual RANSAC call.
2. The observation feedback field is renamed `prev_min_support` →
   `prev_min_support_frac` and now stores the fraction, not the absolute
   count — density-invariant like `eps` and the `min_support` action.
3. `reset()` now log-uniform samples `num_points` per episode (over
   `[10000, 100000]`, unless pinned via `options["num_points"]`) instead of
   a fixed `10000`. Log-uniform, not linear-uniform, so both orders of
   magnitude get equal sampling weight. A `self.true_num_points` readback
   attribute was added, matching the existing `true_*` convention (§8), for
   eventual density-stratified eval.
4. Verified with a smoke test: `true_num_points` samples spanned ~11k-97k
   across 8 seeded resets, `prev_min_support_frac` stayed in `[0, 1]` after
   a step, and the observation shape remained the required 33-dim.

**This changes the semantics of an existing observation dimension, so the
pilot's weights are invalid against it — a retrain is required regardless.**
Per the discussion that produced this fix, it's intentionally bundled into
the same retrain as the reward redesign rather than run as a separate pass.
**No training has been run for this yet** — same standing instruction as
Phase 6: hold for explicit go-ahead.

**Two non-retrain mitigations were discussed but NOT implemented** (available
if 100k-point inference is needed before the retrain lands):
- Wrap inference with a canonical-density view: subsample/voxel-downsample
  the incoming cloud to ~10k for parameter *selection* only, then apply the
  chosen `eps`/`min_support` fraction to the full-resolution cloud for the
  actual fit. Valid specifically because both action outputs are already
  density-invariant.
- A zero-training diagnostic to isolate the bug's real-world impact: run the
  current pilot at 100k twice, once as-is and once with the old
  `prev_min_support` observation value manually rescaled by `10000/N` before
  being fed to the policy. A recovery toward 10k-quality behavior in the
  rescaled run (and not the as-is run) would confirm and quantify this
  feature's contribution, isolated from every other variable. Not run.

---

## 18. Density-Throughput Cost Model (`measure_density_throughput.py`)

**Trigger:** before committing to Phase 6 under the new log-uniform
`num_points` distribution (§17), the old wall-clock estimate (0.29s/step,
measured at a fixed 10k) no longer applies once density varies per episode
— needed a real measurement, not a back-calculation from a distribution
that no longer exists.

**Method:** a single real PPO rollout (2048 steps, no checkpoint saved) run
under the actual training path (`DummyVecEnv` → `VecNormalize` → `PPO`,
same hyperparameters as `train_synthetic.py`), logging `(true_num_points,
step_wall_time)` per step instead of only a mean, since a short window's
mean would be biased by whatever mix of densities happened to land in it.

**Result — cost model `time_per_step ≈ a + b·N`:**
- `a = 0.31436`, `b = 0.0000054`, R² = 0.25 on the raw per-step fit (n=2047).
- Low R² is expected and not a problem for the budget figure: the noise is
  scene-complexity variance (bumps/clutter/intersecting-plane also
  randomize per episode) that swamps the density signal *per step*, but the
  budget is a sum over tens of thousands of steps, so its precision is
  governed by the standard error of the mean, not per-step scatter — with
  2047 points spanning the full range, `b` is many standard errors from
  zero. Confirmed with quantile-binned means: mean step time rises
  cleanly and monotonically from 0.376s (lowest N-decile, ~10-12k) to
  0.794s (highest, ~79-100k); Pearson r(N, dt) = 0.50.
- Warmup was checked, not assumed away: refitting after dropping the first
  20/50/100/200 steps moved `a` by <0.5% and left `b` unchanged to 4
  significant figures. Cold-start effects are not biasing this fit.
- N-range actually covered: 10,091-99,940 (full nominal range hit),
  percentiles 10/25/50/75/90 = 12.4k/17.4k/28.6k/54.8k/79.4k across 2047
  step observations and 1089 episodes.

**The a-vs-b·N split, at the correct expected density:** `E[N]` under
log-uniform sampling is `(H-L)/ln(H/L) ≈ 39,087` — **not** the geometric
mean (~31,623), which is the log-uniform's *median*, not its mean; using
the geometric mean would have under-projected the density-dependent
budget term by ~24%. At `E[N] = 39,087`: fixed per-step overhead (`a`) is
60% of cost, density (`b·N`) is 40% — a real, load-bearing contribution,
not negligible and not dominant either.

**Budget impact — old (0.29s/step, fixed 10k) vs new (0.525s/step, `E[N]`):**

| | 50k steps | 100k steps |
|---|---|---|
| Old (roadmap-stated) | 4.03h | 8.06h |
| New (measured) | 7.30h | 14.59h |
| Ratio | 1.81× | 1.81× |

This 1.81× is the honest cost of the density-generalization work in §17 —
worth stating alongside the generalization result itself in any report,
not just the generalization win on its own.

**Episode length reference (measured in the same run, old reward):** mean
1.881 steps/episode across 1089 episodes, distribution `[531, 333, 111,
52, 62]` for lengths 1-5 — 77% of episodes ending by step 2. This is the
"before" number §19's reward redesign is trying to move, and it's a floor:
it reflects the one-shot collapse, not any ceiling on what iterative
refinement would cost.

**Two open items flagged, not yet resolved:**
- A 10× density range may need more training timesteps to converge, not
  just more wall-clock per timestep — the pilot's fixed timestep count
  can't reveal this; watch the learning curve in the real retrain.
- The budget projection uses `E[N]` at the *episode* level, which assumes
  episode length is independent of density. True today (episode length is
  ~constant under the one-shot collapse), but if the reward redesign (§19)
  makes harder/denser scenes need more refinement steps, high-N episodes
  would get longer, shifting the effective per-*step* mean N above the
  per-*episode* `E[N]=39.1k` and causing `b·N` to under-project. Cheap
  fix when it matters: re-weight `E[N]` by per-step occupancy rather than
  reusing the per-episode mean, once the retrained model's timing is
  re-measured.

**Confirmed separately: `step_penalty` (§19) is the only penalty term in
the codebase** (grepped every `.py` file in `synthetic_rl_experiment/` for
`reward|penalty`) — no wall-clock-based or episode-end penalty exists
anywhere, so the density-coupling risk that motivated this check does not
apply to the reward function itself, only to the wall-clock cost of
collecting steps, which is what this section quantifies.

**Budget definition confirmed from code, not assumed:** `train_synthetic.py`
stops purely on `total_timesteps` (`model.learn(total_timesteps=timesteps)`)
— there is no episode-count or coverage-based stopping condition anywhere
in this harness. So the cost-model figures above apply directly to whatever
`--timesteps` value Phase 6 launches with, with no unit-mismatch risk.

---

## 19. Reward Redesign: Asymmetric Step Penalty (fixing the one-shot collapse)

**Motivation:** §18's own measurement (1.881 mean steps/episode, 77% of
episodes ending by step 2) confirmed what had been suspected — the agent
almost never uses the 5-step refinement budget the environment was built
to offer. Root cause: `reward = (current_score - prev_score) -
step_penalty` applied a flat `-0.01` to *every* step unconditionally. Since
`normal_angle_score = exp(-angle/5)` is very flat once angle error is
already small, achievable per-step score gains are often well under 0.01
— making a further refinement step net-negative in expectation even when
it genuinely improved the result. The agent's optimal response was to grab
a decent first guess and stop.

**Fix applied (`synthetic_env.py`, `step()`):** the penalty is now
asymmetric — applied only when a step fails to improve the score, not
unconditionally:

```python
score_delta = self.current_score - prev_score
reward = score_delta if score_delta > 0 else (score_delta - step_penalty)
```

A step that improves the score is never taxed now; a step that doesn't
(including a wasted terminal stop) still is — preserving the original
intent (discourage aimless continuation) without punishing refinement that
works.

**Smoke-tested** (random actions, not a trained policy) to confirm the
mechanism: a step improving score by +0.017 now nets its full +0.017
(previously would have net to +0.007 under the old flat penalty); a
step that dropped score by -0.157 nets -0.167 (penalized, as intended).

**This changes reward semantics, so it's bundled into the same retrain as
§17's observation fix** rather than run separately — consistent with the
existing plan.

**Verification pass launched (`verify_reward_redesign.py`):** before
committing either fix to the full Phase 6 retrain, a real (not
random-action) verification run — 16,384 timesteps, 8 real PPO
gradient updates, density pinned at a **fixed N=10,000** (not the new
log-uniform range) to isolate the reward effect cleanly, per the agreed
methodology of keeping density and reward as separate, additive
comparisons rather than conflating them in one number. No checkpoint
saved. Tracks mean steps/episode in blocks of 50 episodes across training,
to show whether it actually trends up off the 1.881 baseline rather than
producing only a single before/after number.

**Status: completed. Result was a regression — see §21.** The asymmetric
fix in this section was superseded by the best-of-episode reformulation
in §22 as a direct consequence.

---

## 20. Real-World (RELLIS-3D) Evaluation Infrastructure Audit

**Context:** separate from `synthetic_rl_experiment/`, but relevant to
report-writing — a question came up about improving the real-world
evaluation story (currently a raw IoU number from a TartanGround-trained
model run on RELLIS-3D, felt to be "not a proper evaluation") by adding a
Strict/Standard/Loose-vs-RL bar-chart comparison and a visual comparison,
matching the methodology already used in this project's synthetic
evaluation.

**Finding: most of the infrastructure for this already exists** in the
repo root (a separate codebase from `synthetic_rl_experiment/`):
- `eval_rellis3d.py` already runs strict/standard/loose baselines *and*
  the RL policy against real RELLIS-3D ground-truth labels
  (nearest-neighbor-matched to the raw, undownsampled point cloud),
  computing genuine IoU/precision/recall/F1 (not the heuristic
  `inlier_ratio` used elsewhere), and saves both per-mode CSVs and an
  overall `RELLIS3D_summary.csv`.
- `plot_comparison.py` already implements the exact "grouped column bar,
  RL vs Strict/Standard/Loose" chart pattern (built for the TartanGround
  pipeline) — adapting it to read `RELLIS3D_summary.csv` instead of
  `full_comparison_summary.csv` is a light adaptation, not new work.
- `visualize_inference.py` already does Open3D point-cloud rendering of
  predicted inlier masks for this pipeline — extending it to overlay the
  *real* RELLIS-3D ground-truth mask (`<frame>_gt.npy`, already produced
  by `eval_rellis3d.py`) is a natural, low-risk extension.

**The actual gap, and why the current result reads as "not proper":**
`logs/RELLIS3D_summary.csv` currently contains only two rows, both RL,
**no baseline rows at all**:

```
mode,iou,precision,recall,f1,n_frames
rl_v3_model,0.469,...,n_frames=13556
rl_v4_model,0.499,...,n_frames=1224
```

`rl_v3_model` covers the full dataset (all 5 sequences, 13,556 frames);
`rl_v4_model` covers only 1,224 frames (a partial/`--limit` run). So the
existing number isn't just missing a chart — it has no baseline
comparison on disk at all, and the two RL entries aren't even comparable
to each other (different frame counts). This, not a flaw in the IoU
metric itself, is why it read as insufficient.

**Recommended next step (not yet executed, pending direction on which
model to target):** a clean rerun of `eval_rellis3d.py` — full frame set
(13,556 frames × 4 modes: strict/standard/loose/one chosen RL model),
baselines included — to produce a genuinely comparable summary, before
building the bar chart or visual comparison from it. This is flagged as
a real compute cost (13,556 frames × 4 modes of real RANSAC calls) worth
confirming before launching, distinct from the bar-chart/visualization
script-writing itself, which is cheap. **Deferred by explicit user
request** ("I will run this later") in favor of finishing the
`synthetic_rl_experiment` reward-redesign verification first.

---

## 21. First Verification Result: the Asymmetric-Penalty Fix Made Things Worse

`verify_reward_redesign.py` (§19) completed: 16,384 steps, fixed N=10,000,
9,786 episodes. **Result was a regression, not an improvement:**

- Overall mean steps/episode: **1.674**, down from the 1.881 old-reward
  baseline (measured in a different run, §18) — wrong direction.
- The decline is a real, consistent trend across training, not noise:
  first-50-episode block mean = 2.04, last full block ≈ 1.48. Checked
  statistically over all 9,786 episodes: Pearson r(episode index, length)
  = **-0.17**, linear fit goes from ~1.98 at episode 0 to ~1.37 by episode
  9,786. First-500 vs. last-500 episode means: 1.904 → 1.458. The policy
  was actively *learning* to collapse toward fewer steps, more
  aggressively than under the old reward.

**Diagnosis (confirmed correct by the user, who identified the more
fundamental of the two causes):**

1. **Asymmetry / level-shift (minor).** `reward = score_delta if
   score_delta > 0 else score_delta - penalty` compares each step to the
   *previous* step. Step 1 always starts from `prev_score = 0`, and
   `current_score >= 0` always, so step 1's delta is almost always
   positive — meaning step 1 became untaxed (full reward, no penalty)
   where every other step still paid `-0.01` on failure. This makes
   stopping-at-1 more attractive by a roughly constant amount, but doesn't
   change the *sign* of refining's expected value on its own.
2. **Unbounded-downside overwrite (fundamental — this is what set the
   sign of the regression).** `current_score` is the *latest* attempt,
   with no running best. Under RANSAC's own call-to-call noise (confirmed
   separately in §22's calibration), a refinement step is a coin flip on
   `current_score`. Without a running best, a bad draw doesn't just cost
   `-0.01` — it **overwrites and banks a worse final result**, no matter
   how good an earlier step had been. That gives refining **negative
   expected value even when the underlying parameter change is genuinely
   good**, purely from the bookkeeping, not from anything the policy
   actually learned wrong. The agent was correctly learning that
   refinement was a bad bet, because as coded, it was one. This is why
   the decline *sharpened* under the fix: step 1's reward got bigger while
   refinement's expected value stayed negative, steepening the gradient
   toward one-shot behavior rather than flattening it.

**Methodological note:** this is the second time in this session a
plausible-sounding fix was caught empirically rather than by inspection
(the first was the qualitative-demo replication check, §14). Both times
the run that caught it was cheap relative to the retrain it was gating
(this one: ~20 min vs. the 7-15h Phase 6 retrain it would have been
bundled into un-verified).

---

## 22. Reward Redesign v2: Best-of-Episode Reformulation

**Fix, per the diagnosis in §21:** reward against the running best score,
not the previous step, so a failed refinement has *bounded* downside
(lose `step_penalty`, keep the best result already found) instead of
unbounded downside. This also subsumes §21's asymmetry issue for free —
step 1 now compares against `best_score = 0` like every other step, so it
needs no separate patch (deliberately not stacked on top, to keep
attribution to one structural change per verification pass).

**Reward expression (`synthetic_env.py`, `step()`):**
```python
step_penalty = 0.01
score_gain = max(0.0, self.current_score - self.best_score)
reward = score_gain - step_penalty
if self.current_score > self.best_score:
    self.best_score = self.current_score
    self.best_info = dict(info)
    self.best_info["epsilon"] = eps
    self.best_info["min_support"] = min_supp
    self.best_info["normal_thresh"] = norm_th
```

**Episode-result-read line — extended beyond the scalar score.** Not just
`info["score"]`, but the *entire* result bundle (`angle_error`,
`offset_error`, `inlier_recovery_rate`, `false_inlier_rate`,
`achieved_inlier_ratio`, and the `epsilon`/`min_support`/`normal_thresh`
that produced it) is snapshotted at the best step and returned, so a
best-so-far score can never get paired with a different (possibly worse)
step's angle_error in the CSV — the same class of inconsistency the fix
itself was designed to remove, one level down. `error` is reported live
(this step's own crash/no-plane state), not best-gated, since "did this
attempt crash" is a different question from "what's the best result so
far":
```python
result = dict(self.best_info)
result["error"] = info["error"]
result["true_inlier_ratio"] = self.true_inlier_ratio
result["true_noise_sigma"] = self.true_noise_sigma
result["score"] = self.best_score
```

**`reward_mode` toggle added** (`"best_of_episode"` default, `"old_flat"`
for a real paired baseline) so a fair comparison could be run without
reverting code. `best_score`/`best_info` bookkeeping runs identically in
both modes (pure measurement); only the reward paid differs — isolating
the comparison to exactly the reward signal.

**Mechanical verification (before committing to the full run), all clean:**
1. Monotonicity + reward floor (40 episodes × 2 modes): `best_score` never
   decreases within an episode; `best_of_episode` rewards never fall below
   `-0.01`. Zero violations.
2. `best_info` matches the step that actually set the best (300 checks ×
   2 modes, including the no-step-ever-beat-baseline edge case): zero
   mismatches, once the independent check script was corrected to mirror
   the env's own semantics (seed the comparison with the 0.0 baseline,
   require strict `>` not a tie-inclusive match — the first version of
   this check had 24 false-positive "mismatches" that were bugs in the
   *check*, not the env).
3. Full analysis pipeline (both conditions, 300-step/~1000-episode smoke
   scale): ran end-to-end with no crashes, well-formed output.

**Realized-improvement threshold, pre-registered before the full run:**
"did a step after step 1 raise best_score" needs a floor, or RANSAC's own
call-to-call jitter registers as false "refinement wins" in both
conditions. Calibrated via repeated single-step calls at **identical
scene and parameters** (isolates pure RANSAC jitter, no policy/generator
randomness involved):

| Scene difficulty | jitter std | jitter range |
|---|---|---|
| easy (ir=0.5, noise=0.05, no clutter) | 0.0000 | 0.0000 (bit-identical, 80 calls) |
| medium (ir=0.3, noise=0.12, light clutter) | 0.0003 | 0.0011 |
| hard (ir=0.08, noise=0.18, full clutter) | 0.0154 | 0.0557 |
| hard2 (ir=0.05, noise=0.20, full clutter, vertical) | 0.0528 | 0.1898 |

Jitter is real and highly scene-dependent (near-zero on dominant scenes,
substantial on ambiguous ones), and the hard end is well within what
`inlier_ratio`'s domain-randomization range (0.03-0.9) actually samples.
**Threshold set to 0.05** — at the worst-case single-call jitter std
measured, comfortably above jitter everywhere else, and small relative to
the 0-2.0 score range.

**Verification design (`verify_reward_redesign_v2.py`), per explicit
methodological direction:**
- **Primary metric: realized-improvement rate** — fraction of episodes
  where a step after step 1 raised `best_score` by more than the 0.05
  threshold, and by how much — not raw steps/episode. Episode length
  alone conflates "refining productively" with "wasting steps," and
  length-when-unwarranted would be a different memorized policy, not
  evidence of adaptation.
- **Stratified by exogenous scene-difficulty variables**
  (`true_noise_sigma`, `true_inlier_ratio`), not by step-1's own score.
  Step-1 score is endogenous — a bad step-1 draw can mean "hard scene" or
  "easy scene, unlucky RANSAC draw," so conditioning on it partly measures
  regression-to-the-mean rather than learned adaptive behavior.
- **Step-1-score bucket kept as a secondary view only**, with the
  regression-to-the-mean caveat attached, not used as the headline metric.
- Both conditions (`old_flat`, `best_of_episode`) run at fixed N=10,000,
  16,384 steps each, matching §19's run size for comparability, no
  checkpoints saved.

**Status: full run launched, ~2.5h estimated (sequential, both
conditions). Result not yet known — to be logged here once complete,
alongside the Phase 6 go/no-go decision.**

---

## 23. Sinusoidal Surface Deformation (`data_generator.py`)

Added `_add_sinusoidal_deformation()` as a second terrain-deformation type,
stacking with (not replacing) the existing Gaussian bump/crater from §2/
§11 — rolling/undulating terrain and washboard-gravel patterns, distinct
from the existing function's localized perturbations. Per wave: random
in-plane direction, wavelength (1-4m), phase, and amplitude capped at
0.02-0.10m — deliberately mirroring the corrected bump-amplitude ceiling
from §11 so this can't reintroduce the same noise-independent floor that
the bump-amplitude bug caused there.

`generate_scene(..., num_sine_waves=0)` — new parameter, applied after
bumps, before noise. `synthetic_env.py`'s `reset()` domain-randomizes
`num_sine_waves` (50% chance 0, otherwise 1-2) independently of
`num_bumps`, with a matching `true_num_sine_waves` readback attribute
following the established `true_*` convention (§8) for later eval
stratification.

Verified with noise disabled: a 2-wave scene showed inlier residuals along
the normal bounded at ±0.19 (consistent with two 0.10m-amplitude waves
summing in phase at their peak), std≈0.089 — correctly bounded, non-trivial
signal, no crash.

---

## 24. Phase 6 Launch

**Explicit go-ahead received** ("Yes — go. Launch Phase 6.") after all of
§21's regression, §22's fix, and its full paired verification (mechanism
+ outcome both positive, timestep-axis-robust, benign-decay explained —
see §22's final head-to-head numbers: realized-improvement rate 0.133→
0.179 significant at p≈0.008, mean final_best 1.615→1.653 significant at
t=5.76).

**Launched:** `train_synthetic.py --tag v2 --timesteps 50000` — no
existing `synthetic_ppo_v2.zip` to conflict with, no `--force` needed.
This single run bundles everything built this session: the reward
redesign (§22), the density randomization + observation OOD fix (§17),
and all of Phases 2-5 plus the newly-added sinusoidal deformation (§10,
§23) — every episode independently randomizes orientation, noise type,
bumps, sine waves, clutter, intersecting-plane presence, and point count
(log-uniform 10k-100k), since `train_synthetic.py`'s env is constructed
plain with no fixed overrides.

**Status as of this entry: running, not yet complete.**
- Progress: 34,816 / 50,000 timesteps (~70%), ~4h48m elapsed.
- `ep_len_mean` climbing through training: 1.84 (start) → 2.24 → 2.59 →
  2.39 (natural fluctuation, overall trend still well above baseline) —
  consistent with §22's verification prediction, not just an early spike.
- Zero NaN reward steps across the entire run so far.
- Measured throughput (~0.50-0.52s/step) matches the §18 cost model's
  0.525s/step prediction closely — that calibration held up under the real
  run, not just the pilot-scale check.
- ETA at current rate: total ≈ 6.9-7.2h, in line with the 7.3h estimate.

**Final result — training completed cleanly:**
```
Wall-clock: 25,173.8s (6.99h) -- in line with the 6.9-7.2h in-progress estimate
Throughput: 0.5035s/step -- matches the §18 cost model prediction (0.525s/step)
ep_len_mean (final rollout): 2.62 -- sustained the upward trend through the whole run,
  started at 1.84 (see §24's in-progress snapshots: 1.84 -> 2.24 -> 2.59 -> 2.39 -> 2.62)
explained_variance: 0.989 -- vs. the pilot's 0.76, meaningfully better value-function learning
Total episodes: 21,377
Total NaN reward steps: 0 (entire run)
Unique eps values chosen: [0.0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
  -- the full EPS_LEVELS range, genuinely varied in the last 20 episodes too
  (0.08-0.4, not collapsed) -- directly confirms the original flat-parameter bug
  that started this whole session (§1) is not present in this model.
```
Saved: `models/synthetic_ppo_v2.zip`, `models/synthetic_ppo_v2_vecnormalize.pkl`.

**`evaluate_synthetic.py --tag v2` run — full quantitative results:**

| method | mean score | mean angle error | recovery rate | false-inlier rate |
|---|---|---|---|---|
| RL_Agent | **1.6268** | 0.1624 | **0.8097** | 0.0623 |
| Standard | 1.4833 | 0.1937 | 0.7455 | 0.0892 |
| Strict | 1.4088 | 0.1269 | 0.6092 | 0.0291 |
| Loose | 1.3146 | 0.2504 | 0.7183 | 0.0960 |
| Super Strict | 1.3018 | 0.1875 | 0.5357 | 0.0211 |

RL beats every fixed baseline on overall score and has by far the best
recovery rate. Not #1 on every individual axis (`Strict` has slightly
better raw angle error, `Super Strict` the lowest false-inlier rate) but
those come at the cost of much worse recovery (0.61 and 0.54) — the
correct framing is "better balance than any single fixed setting," not
"wins every sub-metric."

**The result this entire session started to fix:** `eps` now correlates
with noise, monotonically, across the full sweep:
```
noise_sigma:  0.01    0.06    0.11    0.15    0.20
mean eps:    0.092   0.111   0.142   0.158   0.167
```
Pearson r(noise_sigma, eps) = 0.51. This directly contradicts the flat/
constant-across-noise-levels plot that triggered the original
investigation in §1 — confirmed resolved on the full post-Phase-6 model,
not just the pilot's narrower earlier signal (§5).

**`min_support`'s noise signal remains the weaker one** (§16 item 2/6,
pre-dating Phase 6) — trends upward with noise (267→330→293→342→373) but
noisily (std ~220-267) and non-monotonic at one point. Not a new problem,
not fully resolved either; worth being honest about this in the report
rather than only citing eps.

**Scene-composition breakdown (RL_Agent only):** ground vs. vertical
orientation score almost identically (1.630 vs 1.621, no orientation
bias); all noise types close (1.57-1.69); bumps cost a little (1.65→1.61,
expected — harder terrain); intersecting-plane scenes are clearly the
hardest composition (1.65→1.56, also expected — genuine plane ambiguity).
No red flags in any stratification.

**Plot generated:** `plot_adaptivity.py --tag v2` →
`plots/synthetic_ppo_v2_adaptivity.png`.

**Report screenshots generated** (`visualize_synthetic_plane.py --tag v2
--save_screenshot`, 6 representative scenes: low/high noise, bumps,
clutter, intersecting plane, vertical orientation) — saved to
`report_screenshots/`. Scene 02 (high noise) vs. scene 01 (low noise)
visually shows eps adapting (0.08 -> 0.20), the same result as the
quantitative table above.

**Fixed the opacity/camera issue flagged in §26 while generating these** —
the plane meshes were rendering fully opaque and covering most of the
point cloud on well-fit scenes. Fix: shrunk the plane markers to ~40% of
the cloud's extent, nudged true/fitted planes a few cm apart along their
own normals (prevents z-fighting when they're nearly coincident, i.e. a
good fit), and switched the default screenshot camera from top-down to an
oblique angle. This is now the default for anyone using
`--save_screenshot`, not just these six images — a real fix to the
tool, not a one-off workaround.

**Still remaining:** the real-data figure (RELLIS-3D or TartanGround,
§25) — not started, your call on which and when.

---

## 25. Real-Point-Cloud Inference — Still Blocked for `v2`, Same as the Pilot

Question came up: can `synthetic_ppo_v2` run on real point clouds (e.g.
TartanGround/TartanAir scans)? Re-confirmed the same structural blocker
from §6 still applies and Phase 6 does not fix it — no code changed here,
this is a documented clarification, not a bug this session touched.

- `SyntheticRansacEnv.step()`'s shape-selection logic compares RANSAC
  candidates against `self.n_true`, the scene's *known* ground-truth
  normal. Real point clouds have no such label, so the normal multi-step
  refinement loop — the entire behavior the reward redesign (§22) was
  built to enable — **cannot run on real, unlabeled data at all**, not a
  calibration issue, structural. Still open (§16 item 3), unaffected by
  anything done this session.
- What *does* work, same as the original Sewerage test (§6): a manual
  single-shot bypass (`model.predict()` on the observation once, decode
  eps/min_support/normal_thresh, call `schnabel_ransac.detect()` directly,
  skipping `step()`'s ground-truth-dependent selection). Gets one
  parameter choice, not iterative refinement.
- What's genuinely improved for real-data transfer since the Sewerage
  test, independent of Phase 6: `min_support` is fraction-based now (§7.1),
  fixing that test's specific fragmentation failure (75 absolute points
  being 0.07% of a 102k-point real cloud). `eps` is still deliberately
  absolute meters (Phases 7-8 remain deferred), so the scale-calibration
  risk noted in §7.2 is unchanged.
- Distinguishing two "real point cloud" cases that need different
  treatment for report figures: **RELLIS-3D** has real per-point ground
  truth (used by `eval_rellis3d.py`, §20) — supports outcome-colored
  (TP/FP/FN/TN) visualization. **TartanGround** (e.g. Sewerage) has no
  per-point labels — only "predicted ground" coloring is possible, and
  only via the single-shot bypass above.

---

## 26. Report Visualization Tooling: Screenshot Capture

For end-of-internship report figures (professor's suggestion: colored
point-cloud screenshots, real + ground-truth shown distinctly). Found
`visualize_synthetic_plane.py` already had the right coloring scheme for
synthetic scenes (yellow=TP, green=FN/missed, red=FP, gray=TN — a full
confusion-matrix coloring, not just binary) — it only lacked a way to
save an image instead of opening an interactive window.

**Added:** `--save_screenshot PATH` argument. Uses Open3D's `Visualizer`
in hidden-window mode (`visible=False`, needs an actual display session,
not true headless — fine for local desktop use, scriptable in a loop for
batch figure generation instead of manual per-scene GUI screenshots).

**Verified working** (`--no_rl` smoke test, avoided adding CPU load on top
of the running Phase 6 training): produced a valid 1280×960 PNG, visually
confirmed correct — yellow TP points visible through the plane overlap,
gray outliers scattered at the margins, consistent with the reported
0.06° angle error for that test scene.

**Known cosmetic limitation, not yet fixed:** the plane meshes render
fully opaque despite code comments calling them "semi-transparent" (Open
3D's basic `Visualizer` doesn't alpha-blend without extra material setup)
— an inlier-heavy/well-fit scene can end up mostly a solid color block. Two
candidate fixes discussed but not implemented pending preference: an
oblique default camera angle instead of top-down, or shrinking the
rendered plane mesh's footprint. Deferred — not blocking report figure
generation, just a polish item.

**Real-data figures (RELLIS-3D outcome-colored, TartanGround
predicted-ground-only) still need their own script** — not built yet,
same scope split as §25. `evaluate_synthetic.py`/`visualize_synthetic_plane.py`
cover the synthetic side; nothing yet automates real-data screenshot
generation.

---

## 27. Real-World Testing: `visualize_real_pointcloud.py` and the Density-Mismatch Finding

**Built the script §25/§26 flagged as missing**: `visualize_real_pointcloud.py`
implements the single-shot bypass described in §25 — builds the observation
a fresh `reset()` would produce (23-dim scene features + 10 zeros, matching
`step_count=0`), gets one action from `synthetic_ppo_v2`, decodes
eps/min_support (fraction materialized against the REAL cloud's point
count)/normal_thresh, and runs `schnabel_ransac.detect()` directly. Since
there's no ground truth on TartanGround data, shape selection uses
**largest support** (the heuristic flagged as still-needed in §16 item 3),
not the angle-to-ground-truth selection `step()` uses.

**Tested against 9 merged/reconstructed TartanGround scenes**
(`schnabel_cython/tartanair_data/*_rgb_original.ply`, 102k-647k points
each, entire multi-room reconstructions): **only 3 of 9 found any plane at
all** — Sewerage, Hospital, Supermarket. The other 6 (Downtown,
OldScandinavia, OldTownFall, and all 3 SeasonalForest scenes) found zero
candidate shapes with the model's chosen parameters.

**Root cause, confirmed by density correlation, not just asserted:**

| Scene | pts/m² | Found a plane? |
|---|---|---|
| Hospital | 18.21 | yes |
| Sewerage | 12.86 | yes |
| OldTownFall | 9.93 | no (exception) |
| Downtown | 2.45 | no |
| Supermarket | 2.04 | yes (exception) |
| OldScandinavia | 1.78 | no |
| SeasonalForestSpring | 0.68 | no |
| SeasonalForestAutumn | 0.66 | no |
| SeasonalForestWinterNight | 0.63 | no |

All 3 sparsest scenes (<0.7 pts/m², the SeasonalForest set) failed
outright; the two clearest successes are the two densest scenes. This is
the already-known, still-unresolved limitation (§7.2/§16 item 4) made
concrete: `eps` is absolute meters (8cm), calibrated against the dense
synthetic training clouds (10k-100k points packed into a ~10m box). Applied
to a real scene where points spread thin over hundreds of m², an 8cm
tolerance often can't accumulate enough coplanar points to clear
`min_support`, regardless of whether a real flat surface exists there.
Two exceptions don't fit density alone: Supermarket succeeds at low
density likely because its smaller total N (157,802) makes the 0.5%
`min_support` fraction an easier absolute bar (789 pts) than Downtown's
(3,237 pts) despite similar density; OldTownFall fails despite fairly high
density (9.93) with no clear explanation found -- possibly a genuine lack
of a large enough flat surface in that scene's geometry.

**Tested against 12 additional scenes as single-frame LiDAR sweeps
instead** (`data/<scene>/Data_omni/P0000/lidar/000000_*.ply`, 43k-58k
points each — one scan, not a merged reconstruction): **all 12 found at
least one plane.** Confirms the density story directly: single-sweep
density is much closer to synthetic training density than a fully merged
multi-room reconstruction is, and the hit rate difference (3/9 merged vs.
12/12 single-frame) tracks that almost exactly.

**A second, distinct limitation found in this data, not previously
documented:** of those 12 successes, only 10 are plausible ground
(fitted normal within ~12° of vertical). **2 of 12 (Restaurant at 54.3°
from vertical, AbandonedFactory at 45.4°) clearly found a wall, not the
floor** — confirmed both by the angle and visually (screenshot shows a
narrow vertical strip along a building edge, not a ground patch). This is
the **largest-support selection heuristic's actual failure mode**: without
any orientation prior, it will happily pick a large flat wall over a
smaller-but-real floor if the wall has more visible coplanar points in
that particular sweep. Directly actionable: §16 item 3's proposed fix
("largest-support, or most-horizontal-normal") should likely be **both
combined** (largest support *among* roughly-horizontal candidates), not
either alone — this data is the first concrete evidence for that specific
combination rather than an untested guess.

**Screenshots saved:** `report_screenshots/real_<scene>_v2.png` (9 merged
scenes) and `report_screenshots/real_<scene>_frame0_v2.png` (12 single-frame
scenes) — visually confirm the Sewerage/WesternDesertTown results (a
correctly-located ground patch, WesternDesertTown's showing the dense
near-sensor LiDAR rings a spinning scanner produces) and the
Restaurant/AbandonedFactory failure mode (a vertical wall strip, not a
floor).

**Fix applied and verified same-session:** `visualize_real_pointcloud.py`'s
shape selection now ranks by support only *among* candidates within
`HORIZONTAL_TOLERANCE_DEG = 30.0` of vertical, falling back to
largest-overall (with an explicit printed warning) only if literally
nothing qualifies. Confirmed on both known failures:
- **Restaurant**: was 54.3° from vertical (wall) -> now selects a genuinely
  horizontal candidate (0.3° from vertical, 9.0% of cloud) -- 4 of the 10
  original candidates were horizontal and simply weren't the biggest one.
- **AbandonedFactory**: was 45.4° from vertical (wall) -> now 8.1° from
  vertical, 16.0% of cloud -- screenshot now shows the same dense
  near-sensor LiDAR-ring pattern as the other correct detections, not a
  distant wall strip.

**No regression on a known-good case**: re-ran WesternDesertTown after the
fix -- unchanged (38.2% of cloud, 0.7° from vertical, same as before), since
the largest-support shape there was already horizontal.

This closes the open question from §16 item 3/§25 with a concrete answer:
largest-support and most-horizontal-normal needed to be combined (support
ranking restricted to horizontal candidates), not used as alternatives —
confirmed on real data, not just reasoned about.

---

## 28. Real-World Urban Test: `new_data/` (Dense Semantic-Labeled City Tiles)

A new, different real-data source was provided mid-session: `new_data/`,
containing large real point-cloud tiles unrelated to TartanGround/TartanAir
(no synthetic origin at all). Built `view_new_data.py` (plain true-color
viewer) and `visualize_new_data_rl.py` (runs `synthetic_ppo_v2` end-to-end
and shows predicted ground next to the true-color original) to test it.

**What the data actually is**, confirmed by direct `plyfile` inspection
before writing any loading code:
- 2.2M-3.8M points per tile, real RGB (`red/green/blue`, non-degenerate —
  confirmed varying 0-255, not a placeholder), plus `semantic` (int class
  ID, 21 distinct values per tile, no legend embedded in the file — only
  `comment author Yiyi`), `instance`, `visible`, `confidence`.
- Absolute survey-scale coordinates, not sensor-centered — e.g. one tile
  spans x=827-1008, y=3670-3831, z=112-129. This matters (next point).
- Organized as `new_data/static/<name>.ply` (background/scene) and
  `new_data/dynamic/<name>.ply` (moving objects — cars, pedestrians —
  captured in that tile's time window; has an extra `timestamp` field
  static lacks). Most tiles have no dynamic counterpart at all; of the two
  that do, one is genuinely empty (0 points, nothing was moving that
  window) and one has 17,863 real points.
- Semantic sanity check (not used for the RL run, just to understand the
  data): class `7` is 27.8% of one tile's points with an unusually tight,
  low z-band (114.20-114.48m) and wide spatial footprint (153m x 144m) —
  a strong ground/road signature, and independently close to what the RL
  model itself picked out (see below) despite the model never seeing this
  semantic field.

**Coordinate-frame bug caught before it produced silently-bad results:**
`compute_scene_features()` computes `scan_range_mean/std` as
`norm(points)` — distance from the **origin**, an assumption that held for
every dataset used all session (synthetic scenes and single-frame LiDAR
sweeps are both naturally sensor-centered near (0,0,0)). These tiles are
absolute survey coordinates, nowhere near the origin — feeding them in
directly would have produced `scan_range` values around 3700 instead of a
few meters, wildly outside the `VecNormalize` running-stats range the
model trained on, almost certainly producing a degenerate action. Fixed by
recentering on the point cloud's own centroid before feature computation
*and* RANSAC (recentering changes no relative geometry, so plane detection
itself is unaffected — this is purely about matching the observation
distribution the model actually learned from). Display still uses true
world coordinates; only the model-facing copy is recentered.

**Scale handling:** voxel-downsampled at 0.15m before RANSAC (tiles are
too dense — millions of points over ~150-200m — for both speed and for
matching training-scale density). When a non-empty dynamic file exists for
a tile, its points are concatenated onto the downsampled static cloud
(kept at full resolution, since dynamic point counts are small — thousands,
not millions) and highlighted magenta in the true-color view.

**Static-only vs. static+dynamic, tested directly (tile `0000002897_0000003118`,
the one tile with a non-empty dynamic file):**

| Run | Ground found | Angle from vertical |
|---|---|---|
| static + dynamic (17,863 pts) | 9.3% | 3.8° |
| static only | 10.5% | 2.6° |

Nearly identical — the 17,863 dynamic points (~2.4% of that tile's total)
are not what's driving the lower coverage on this tile; it genuinely has
less open ground than the other tiles (more built-up/intersection area),
independent of the dynamic-object question.

**Full results, all 17 tiles tested (same single-shot bypass + largest-
support-among-horizontal selection as §27, `HORIZONTAL_TOLERANCE_DEG=30`):**

| idx | file | pts (downsampled) | dynamic pts | eps | found | ground % | angle from vertical |
|---|---|---|---|---|---|---|---|
| 0 | 0000000002_0000000385 | 843,814 | 0 | 0.08 | yes | 28.3% | 0.1° |
| 1 | 0000000372_0000000610 | 848,460 | 0 | 0.08 | yes | 28.9% | 0.1° |
| 2 | 0000000599_0000000846 | 956,874 | 0 | 0.08 | yes | 28.9% | 0.2° |
| 3 | 0000000834_0000001286 | 695,987 | 0 | 0.08 | yes | 21.3% | 0.2° |
| 4 | 0000002897_0000003118 | 735,241 | 17,863 | 0.08 | yes | 9.3% | 3.8° |
| 5 | 0000002913_0000003233 | 808,164 | 0 | 0.08 | yes | 20.6% | 0.6° |
| 6 | 0000005880_0000006165 | 801,864 | 0 | 0.08 | yes | 23.3% | 0.5° |
| 7 | 0000006387_0000006634 | 498,873 | 0 | 0.08 | yes | 25.1% | 0.3° |
| 8 | 0000007438_0000007605 | 631,561 | 0 | 0.08 | yes | 46.3% | 0.1° |
| 9 | 0000007596_0000007791 | 667,551 | 0 | 0.08 | yes | 44.6% | 0.3° |
| 10 | 0000008278_0000008507 | 599,633 | 0 | 0.08 | yes | 18.6% | 0.5° |
| 11 | 0000009666_0000009895 | 702,431 | 0 | 0.08 | yes | 41.4% | 0.1° |
| 12 | 0000009886_0000010098 | 663,668 | 0 | 0.08 | yes | 48.5% | 0.2° |
| 13 | 0000010078_0000010362 | 585,822 | 0 | 0.08 | yes | 27.4% | 0.4° |
| 14 | 0000010577_0000010841 | 598,877 | 0 | 0.08 | yes | 22.3% | 0.2° |
| 15 | 0000011079_0000011287 | 706,340 | 0 | 0.08 | yes | 48.3% | 0.6° |
| 16 | 0000011278_0000011467 | 662,105 | 0 | 0.08 | yes | 46.5% | 0.7° |

**17/17 tiles found a valid ground plane, 0 failures.** Every single
detection landed under 1° from vertical (best case 0.1°, worst case 3.8°
on the one tile with dynamic objects mixed in) — a much cleaner result
than §27's merged-TartanGround test, consistent with these tiles' density
being close to single-frame-sweep density rather than sparse
multi-room-reconstruction density. Ground-percentage variation (9-49%)
tracks scene content (how much open road vs. buildings is in that
particular tile), not detection failures — visually confirmed across all
17 screenshots: the green predicted-ground region consistently traces the
actual road/street shape, including branching intersections and curved
roads, never a wall or an arbitrary blob.

**Screenshots saved:** `report_screenshots/new_data/tile{00-16}_<name>.png`
(headless via `--batch START END`, added alongside the existing
single-tile interactive mode in the same script).

**Second batch: 13 more tiles added to `new_data/static/` mid-session.**
Non-contiguous indices once re-sorted alongside the original 17 (new files
interleaved alphabetically among the old ones), so added `--indices i j
k...` (explicit list) alongside the existing `--batch START END`
(contiguous range). Also added `--interactive`, usable with either batch
mode: saves the screenshot as before *and* opens the interactive window
per tile (blocking until closed) so both the automated figure and manual
inspection happen in one run instead of two.

| idx | file | pts (downsampled) | eps | found | ground % | angle from vertical |
|---|---|---|---|---|---|---|
| 0 | 0000000002_0000000125 | 622,155 | 0.15 | yes | 37.2% | 4.0° |
| 1 | 0000000002_0000000208 | 627,237 | 0.08 | yes | 22.4% | 0.6° |
| 3 | 0000000208_0000000298 | 694,908 | 0.15 | yes | 29.2% | 4.3° |
| 4 | 0000000353_0000000557 | 608,758 | 0.08 | yes | 48.5% | 0.1° |
| 6 | 0000000378_0000000466 | 700,548 | 0.15 | yes | 29.6% | 2.6° |
| 8 | 0000000785_0000000870 | 491,527 | 0.08 | yes | 32.9% | 1.7° |
| 10 | 0000001034_0000001127 | 604,017 | 0.08 | yes | 28.5% | 2.3° |
| 11 | 0000001245_0000001578 | 695,062 | 0.08 | yes | 34.8% | 1.4° |
| 12 | 0000001340_0000001490 | 723,063 | 0.08 | yes | 19.0% | 2.2° |
| 13 | 0000001577_0000001664 | 538,594 | 0.15 | yes | 30.5% | 2.2° |
| 14 | 0000001745_0000001847 | 550,859 | 0.08 | yes | 33.9% | 0.6° |
| 15 | 0000001872_0000002033 | 515,560 | 0.08 | yes | 41.6% | 0.3° |
| 16 | 0000002395_0000002789 | 533,742 | 0.08 | yes | 49.9% | 0.9° |

**13/13 found a valid ground plane again — running total across all of
`new_data/`: 30/30, 0 failures.** One difference from the first batch
worth flagging: angles here run noticeably higher (up to 4.3° vs. the
first batch's max 0.7°), and the model picked the looser `eps=0.15` on 4
of these 13 tiles instead of the uniform `0.08` seen throughout the first
batch — still comfortably inside the 30° horizontal tolerance and still
visually road-shaped in every screenshot, but consistent with these
particular tiles having somewhat less perfectly-flat ground (mild
slope/undulation) than the first set, not a detection problem.

**Screenshots saved:** `report_screenshots/new_data/tile{00,01,03,04,06,
08,10-16}_<name>.png`, same naming scheme, indices match the file's
position in the full sorted `static/` listing (30 files total) rather
than a separate counter.

---

## 29. Off-Road Dataset Research (For Future Downloadable-`.ply` Testing)

User asked what off-road-terrain point-cloud datasets exist that could be
manually downloaded (same workflow as `new_data/`: drop `.ply` files in a
folder, run the model, look at the result) rather than a full scripted
download+convert pipeline.

**Already have one, easy to forget:** `data/RELLIS3D` (Texas A&M Rellis
Campus, off-road robot terrain — grass/mud/rubble/puddle classes) is
already downloaded and fully integrated (`download_rellis3d.py`,
`rellis3d_convert.py`, `eval_rellis3d.py`), with real quantitative results
already in `logs/RELLIS3D_summary.csv`: `rl_v4_model` — IoU 0.499,
precision 0.747, recall 0.667, F1 0.607, over 1224 frames.

**Researched via web search for something additional/different:**
**WildScenes** (CSIRO Robotics, IJRR 2024) — handheld-LiDAR point clouds
from real natural forest environments (Brisbane, Australia; two forests,
Venman and Karawatha; 21km of walking, 12,148 annotated 3D submaps),
labels distinguish terrain types (dirt vs. gravel vs. other). Confirmed
(via the GitHub repo, not assumed) that the full point clouds are
distributed as **`.ply` files** with intensity, matching the format this
whole `new_data` workflow already expects. Distributed via CSIRO's data
portal as an S3 bucket (organized per walking sequence, e.g. `K-01`,
`V-03`) rather than one monolithic archive, so individual sequences/tiles
should be selectable without pulling the whole dataset — could not verify
the exact bucket file layout directly (the CSIRO data portal returned
403 to automated fetches), so this is inference from standard S3-dataset
practice, not a confirmed screenshot of the file listing.

Not yet downloaded or tested — flagged as the next real-data source to
try once the user pulls a few tiles down, expected to behave differently
from `new_data`'s urban roads (patchier ground, possibly more
wrong-plane/wall picks given uneven natural terrain) which would itself
be a useful, distinct data point for the report.

---

## 30. Off-Road Testing: OpenTopography — 7 Real Surveys, 26 Tiles, a
Success/Failure Split That Tracks Terrain Shape, Not Dataset Labels

WildScenes (sec.29) turned out to require personal CSIRO Data Access
Portal credentials (confirmed via their own docs: login, per-collection
request, ~48h-lived S3 keys — not anonymous), which the user didn't want
to set up. Pivoted to **OpenTopography**, confirmed genuinely anonymous
(their own FAQ: "You do not need an OpenTopography account to access and
process lidar datasets") and confirmed browsable without any tool beyond
plain `curl` — their point-cloud tiles sit in a public, unauthenticated S3
bucket (`https://opentopography.s3.sdsc.edu/pc-bulk/<survey>/`) that
supports anonymous `?list-type=2` listing, so exact filenames and sizes
could be listed before downloading anything, matching what the user
explicitly asked for. This became the go-to real-data source for the rest
of the session — every survey below was found the same way: query
`API/otCatalog?productFormat=PointCloud`, list the bucket, pick tiles by
size, download with plain `curl`.

New tooling: `convert_laz_to_ply.py` (LAZ->PLY via `laspy`, newly installed
with the `lazrs` backend since neither `laspy` nor `pdal` were present;
auto-detects per-file whether `classification`/RGB actually exist, not
just whether the LAS point format nominally includes the field — several
surveys have the RGB schema present but every value genuinely 0, confirmed
via `red.min()==red.max()==0`, not a loading bug) and
`visualize_offroad_rl.py` (single-shot-bypass + recentering, same pattern
as `visualize_new_data_rl.py`; outcome-colored TP/FP/FN/TN when real ASPRS
`classification` exists, `--color_mode height` viridis-by-elevation
otherwise/always-available, with normal estimation + `light_on` + a tilted
camera so real relief is visible instead of flattened by the default
top-down view). Later extended with `SURVEY_INFO` (filename-prefix ->
survey folder + description) and CSV logging (`results.csv` per survey
folder + `ALL_SURVEYS_SUMMARY.csv` combined) so results stay organized and
findable, per explicit user request — final layout:

```
report_screenshots/off_road/
├── ALL_SURVEYS_SUMMARY.csv
├── sawpit_wash/               (5 tiles)
├── san_andreas_fault/         (1 tile)
├── rock_glaciers/             (3 tiles)
├── ridgecrest/                (5 tiles)
├── dune_dune_interactions/    (3 tiles)
├── sacramento_river/          (3 tiles)
├── coastal_dune_erosion/      (3 tiles)
└── deltaic_wetlands/          (3 tiles)
```

**Seven real surveys, 26 tiles total, all downloaded via the same
anonymous S3 listing (no login anywhere), sizes verified against the
bucket's own reported `<Size>`:**

| Survey | Alternate name | Terrain | Tiles | Density / notes |
|---|---|---|---|---|
| Post-Bobcat Fire, Sawpit Wash, CA (2020) | `CA20_DiBiase` | Undeveloped canyon/wash, San Gabriel Mountains | 5 | 2.46 pts/m² |
| Lidar Survey over San Andreas Fault, CA (2017) | `CA17_Brooks` | Fault-zone terrain, 516m relief in 1 tile | 1 (19.7M raw points) | ~4.9 pts/m² post-downsample |
| Mapping Rock Glaciers, San Juan Mtns, CO (2025) | `CO25_Ruef` | Rocky alpine terrain, 531-719m relief per tile | 3 | ~1.7-1.9 pts/m² |
| Mobile Laser Scan over Ridgecrest, CA (2019) | `CA19_Brooks` | Desert town streets (mobile/terrestrial scan) | 5 (of 118 available) | 200+ pts/m² |
| Dune-Dune Interactions, CA (2022) | `CA22_Marvin` | Inland sand dunes | 3 | 90-98% real "ground" class, but undulating |
| Topo-Bathymetric Sacramento River, CA (2023) | `CA23_Sacramento` | River/water body, has real RGB | 3 | 52 pts/m², includes a non-standard class `151` (bathymetric-specific) |
| Coastal Dune Erosion, CA (2025) | `CA25_Sondeno` | Beach/dune, water-adjacent | 3 | ~6 pts/m² |
| Deltaic Wetlands, LA (2023) | `LA23_Davidson` | Muddy marsh/wetland | 3 | only 7-14% real "ground" (rest is marsh vegetation) |

**Full per-tile results (single-shot bypass, model's own chosen
parameters, no manual override unless noted):**

| Survey | Tile | Points | Real ground % | Found? | Angle from vertical |
|---|---|---|---|---|---|
| sawpit_wash | 406000_3781000 | 1,750,063 | 28.7% | NOT FOUND | — |
| sawpit_wash | 408000_3780000 | 1,117,348 | 32.2% | NOT FOUND | — |
| sawpit_wash | 409000_3780000 | 1,079,211 | 23.6% | NOT FOUND | — |
| sawpit_wash | 411000_3780000 | 1,490,237 | 25.7% | NOT FOUND | — |
| sawpit_wash | 413000_3781000 | 1,539,978 | 26.0% | NOT FOUND | — |
| san_andreas_fault | 701000_4006000 | 4,755,959 (0.4m voxel) | 53.7% | NOT FOUND | — |
| rock_glaciers | 261000_4207000 | 1,811,793 | 59.9% | NOT FOUND | — |
| rock_glaciers | 262000_4207000 | 1,855,793 | 43.6% | NOT FOUND | — |
| rock_glaciers | 263000_4202000 | 1,690,825 | 74.5% | NOT FOUND | — |
| ridgecrest | pt_000011_1 | 166,207 (0.1m voxel) | n/a (no gt field) | **FOUND**, 47.9% of cloud | 0.3° |
| ridgecrest | pt_000035_1 | 240,474 (0.1m voxel) | n/a (no gt field) | **FOUND**, 85.4% of cloud | 0.1° |
| ridgecrest | pt_000053_1 | 191,908 (0.1m voxel) | n/a (no gt field) | **FOUND**, 69.8% of cloud | 0.4° |
| ridgecrest | pt_000057_1 | 161,012 (0.1m voxel) | n/a (no gt field) | **FOUND**, 95.8% of cloud | 0.3° |
| ridgecrest | pt_000060_1 | 168,493 (0.1m voxel) | n/a (no gt field) | **FOUND**, 40.2% of cloud | 0.6° |
| dune_dune_interactions | 707000_3755000 | 1,474,888 | 90.1% | NOT FOUND | — |
| dune_dune_interactions | 707000_3756000 | 1,806,770 | 93.2% | NOT FOUND | — |
| dune_dune_interactions | 712000_3760000 | 1,927,092 | 98.2% | NOT FOUND | — |
| sacramento_river | 56554510 | 494,027 | 14.2% | **FOUND**, 1.8% of cloud | 1.4° (IoU=0.050) |
| sacramento_river | 56554705 | 533,952 | 11.8% | NOT FOUND | — |
| sacramento_river | 56654625 | 596,448 | 32.8% | **FOUND**, 13.7% of cloud | 1.9° (IoU=0.227) |
| coastal_dune_erosion | 604000_4053000 | 1,961,904 | 26.8% | **FOUND**, 10.2% of cloud | 0.0° (IoU=0.114) |
| coastal_dune_erosion | 605000_4060000 | 2,112,041 | 47.8% | **FOUND**, 5.8% of cloud | 1.4° (IoU=0.039) |
| coastal_dune_erosion | 608000_4068000 | 1,586,507 | 17.8% | NOT FOUND | — |
| deltaic_wetlands | 252000_3253000 | 1,986,471 | 7.5% | NOT FOUND | — |
| deltaic_wetlands | 253000_3257000 | 2,014,342 | 13.8% | NOT FOUND | — |
| deltaic_wetlands | 254000_3258000 | 1,524,568 | 14.3% | NOT FOUND | — |

**26 tiles, three-way split, not a binary pass/fail:**
- **Clean success (5/5 tiles):** Ridgecrest only. All sub-1° from
  vertical, no ASPRS ground truth available there (confirmed genuinely
  unclassified, not omitted) so judged visually — screenshots show a
  broad flat plane covering most of each ~100x100m mobile-scan segment,
  consistent with real paved streets.
- **Weak/partial success (4/6 tiles found, but low quality):**
  Sacramento River and Coastal Dune Erosion. These *do* sometimes find a
  genuinely near-horizontal plane (angles 0.0-1.9°), but coverage is
  small (1.8-13.7% of the cloud) and IoU against real ASPRS ground is
  consistently weak (0.039-0.227) — a middle ground between clean success
  and outright failure.
- **Complete failure (17/17 tiles found nothing, 0.0%):** Sawpit Wash,
  San Andreas Fault, Rock Glaciers, Dune-Dune Interactions, Deltaic
  Wetlands. The Dune-Dune result is the sharpest evidence in the whole
  session that this is about terrain *shape*, not vegetation/clutter
  confusing the classifier: those tiles are 90-98% real ASPRS "ground"
  (almost the entire cloud, since bare sand has little vegetation to
  misclassify) and still failed outright — undulating dune shape alone is
  enough to break single-plane detection, independent of how much of the
  cloud is technically "ground."

**Root-caused with a manual eps-override diagnostic on the two original
failure sites** (bypassing the model's own eps choice, sweeping
0.15/0.3/0.5/0.6 by hand): shapes *do* start appearing, and the largest
candidate is genuinely near-horizontal in both cases (Sawpit Wash: 1.6°
from vertical at eps=0.5; San Andreas: 3.1° from vertical at eps=0.6) — so
it's not that eps=0.08 is "wrong" and a bigger eps reveals nothing. But
**even the best candidate only reaches modest overlap with real ground
truth**: Sawpit Wash tops out at 9.4% IoU (eps=0.5), San Andreas at 18.2%
IoU (eps=0.6, climbing 5.2%->14.4%->17.3%->18.2% across the sweep,
suggesting San Andreas has more contiguous flatter sub-regions than
Sawpit Wash — plausibly a fault valley/bench floor — but still nowhere
near a clean single-plane match). **Conclusion: not purely an eps-tuning
problem.** Natural, undeveloped terrain doesn't have one dominant flat
"ground" plane the way a road does — real ASPRS-classified "ground" here
spans a genuinely undulating surface, so the single-plane RANSAC
assumption itself is a poor fit, independent of parameter choice.

**Ridgecrest remains the clean counter-example and matters for the
overall story:** despite being part of an "off-road"-labeled desert
survey, these specific mobile-scan segments cover actual paved streets,
and the model correctly finds them at default parameters with the same
sub-1° accuracy seen on `new_data`'s urban tiles. Confirms the model
isn't broadly broken on anything labeled "off-road" — it tracks actual
terrain flatness, succeeding on real flat surfaces and correctly failing
to force a fit onto genuinely non-planar natural terrain. A legitimate,
reportable boundary condition (structured/flat ground: works; undulating
natural terrain: breaks down, not fixable by eps alone; water-adjacent
terrain: partial/weak) rather than a bug.

**Screenshots + `results.csv` saved per survey under
`report_screenshots/off_road/<survey>/`, combined log at
`report_screenshots/off_road/ALL_SURVEYS_SUMMARY.csv`.**

---

## 31. Real-Data Testing on Named Terrain-Type Scenes (Beach/Marsh/Harbor/
Island) — Already-Downloaded TartanGround Scenes, No New Download Needed

User asked for real data covering parks/muddy ground/water bodies/beach —
before downloading anything new, checked whether the project's existing
local TartanGround scenes (`data/<scene>/Data_omni/P0000/lidar/`) already
covered this. They did: `SeasideTown` (beach/coastal town), `GreatMarsh`
(marshland — muddy/wetland), `NordicHarbor` (harbor, water-adjacent), and
`GothicIsland` (island, water-adjacent) were all already downloaded but
never tested this session. Built `visualize_terrain_types_rl.py` (same
single-shot bypass + largest-support-among-horizontal selection + height
colormap as `visualize_real_pointcloud.py`/`browse_real_lidar_frames.py`;
organized output the same way as sec.30: per-scene folder + `results.csv`
+ `ALL_SCENES_SUMMARY.csv`). 3 random frames per scene, seed 0.

| Scene | Terrain | Genuinely horizontal finds | Outright NOT FOUND | False "FOUND" (fallback to non-horizontal) |
|---|---|---|---|---|
| SeasideTown | Beach | 0/3 | 2/3 | 1/3 (82.8° from vertical) |
| GreatMarsh | Muddy wetland | 1/3 (8.3°, only 5.2% coverage) | 2/3 | 0/3 |
| NordicHarbor | Harbor/water | 1/3 (1.7°) | 0/3 | 2/3 (89.2°, 89.9° — essentially vertical, likely docks/boat hulls) |
| GothicIsland | Island/water | **3/3** (0.6°, 10.5°, 1.0°) | 0/3 | 0/3 |

GothicIsland is the clear best performer — clean, plausible ground
detection on all 3 frames. The other three repeat a pattern first seen
with DIODE (sec.32): the largest-support-among-horizontal fallback
sometimes has *no* horizontal candidate to fall back to gracefully, so it
reports "FOUND" on a large near-vertical surface instead (a wall, boat
hull, dock structure) — beach/marsh/harbor scenes evidently have more of
these misleading large flat-but-vertical structures nearby than open
island terrain. Screenshots (height-colored, green predicted-ground
overlay) confirm visually: these single-frame near-sensor LiDAR scans are
much sparser/more cluttered (scattered wall/structure fragments) than the
wide-area LiDAR tiles used elsewhere, consistent with earlier per-frame
LiDAR observations this session.

**Results saved under `report_screenshots/terrain_types/<scene>/` +
`ALL_SCENES_SUMMARY.csv`.**

---

## 32. A Third Real Data Modality: RGB-D → Point Cloud (DIODE Dataset)

User wanted to test the model against data that isn't natively a point
cloud at all — an RGB image + depth map, back-projected via camera
intrinsics — as a genuinely different combination from the LiDAR-native
data used everywhere else this session.

**Dataset: DIODE (Dense Indoor/Outdoor DEpth), diode-dataset.org.**
Real RGB + dense depth from a survey-grade laser scanner (FARO Focus
S350), freely downloadable with no login (confirmed via a direct HTTP
request, `200 OK`). Only the whole-partition archive is downloadable, not
individual scenes — pulled `val.tar.gz` (2,774,625,282 bytes, confirmed
exact match to the server's reported `Content-Length`), then extracted
only `val/outdoor/` (3 scenes, 10 scans, 446 RGB-D crops; 325 indoor
samples in the same archive were left compressed/unextracted, not
needed). Camera intrinsics **do** exist (not obvious from the main
project page — found in `diode-devkit`'s `intrinsics.txt`, fetched
directly): `fx=886.81, fy=927.06, cx=512, cy=384`, matching the
1024x768 RGB/depth resolution.

**New tooling: `visualize_diode_rl.py`.** Back-projects one RGB-D crop to
a point cloud via standard pinhole projection (`X=(u-cx)Z/fx,
Y=(v-cy)Z/fy, Z=depth`), keeping only pixels where `depth_mask==1` (DIODE
has real depth gaps — sky, glare, out-of-range — confirmed one sample
crop was 96.2% valid, not 100%). Camera-frame convention (X right, Y
down-image, Z forward) is swapped to this session's z-up convention
(`points = [x, z, -y]`) so the horizontal-plane check (`select_shape`,
axis index 2) means the same thing here as everywhere else. Same
single-shot bypass as the LiDAR scripts; no ground truth exists for
DIODE's outdoor split, so visual-only (height colormap + green predicted-
ground), same evidentiary standard as Ridgecrest/`new_data`.

**Results, 6 random crops (seed 0), each crop ~640-780k valid points out
of 768x1024=786,432 pixels:**

| Crop | Valid pts | Found? | Pred % | Angle from vertical |
|---|---|---|---|---|
| scene_00024/scan_00202/..._190_000 | 763,272 | FOUND | 49.7% | 19.6° |
| scene_00022/scan_00196/..._330_010 | 617,526 | FOUND | 24.6% | **1.2°** (genuine) |
| scene_00024/scan_00201/..._120_020 | 782,077 | FOUND | 52.7% | 82.5° (fallback) |
| scene_00022/scan_00197/..._160_010 | 716,798 | FOUND | 24.1% | **2.7°** (genuine) |
| scene_00022/scan_00193/..._100_050 | 668,330 | FOUND | 21.9% | 60.3° (fallback) |
| scene_00022/scan_00195/..._290_040 | 652,451 | FOUND | 58.8% | 74.2° (fallback) |

**6/6 technically report "FOUND", but only 3/6 are genuinely horizontal**
(1.2°, 2.7°, and the 19.6° case is borderline-plausible). The other 3
(60.3°, 74.2°, 82.5°) are the largest-support-among-horizontal fallback
kicking in because that specific camera crop's field of view simply
didn't contain any horizontal surface at all (confirmed via
`horizontal_candidate=False` for all three) — e.g. a view pointed at
trees or a wall, not ground. **This is a genuine, newly-surfaced
limitation of the fallback heuristic itself**, not specific to DIODE: when
no horizontal candidate exists, "largest support" silently substitutes a
near-vertical surface and the pipeline still reports success. Same
fallback pattern was then independently confirmed on 3 of the 4 real
terrain-type scenes in sec.31 (SeasideTown, NordicHarbor) — not a DIODE-
specific artifact.

**Results saved under `report_screenshots/rgbd/diode/results.csv` +
screenshots.**

**Second batch, 12 more random crops (seed 1), same script/method:**
10/12 genuinely horizontal (0.8°, 1.9°, 3.8°, 10.8°, 12.9°, 13.1°, 14.6°,
17.0°, 18.4°, 18.9°), only 2/12 fallback (68.7°, 72.0°) — a notably better
ratio than the first batch. **Combined DIODE total: 18 crops, 13/18
(72%) genuinely horizontal, 5/18 (28%) fallback-to-non-horizontal.**
Confirms the fallback issue is real but a minority outcome, not the norm,
once sample size is larger than 6.

---

## 33. RGB-D Dataset Research: What Else Was Considered and Why It Wasn't
Pursued

For completeness — datasets investigated but not downloaded, with the
concrete reason:

- **KITTI depth benchmark** (cvlibs.net) — real road-scene depth
  organized as individually downloadable drive sequences (better
  granularity than DIODE's one-big-archive), but **confirmed to require
  account registration** (all download links route through
  `user_login.php`) — not something that can be done on the user's
  behalf. Sizes for reference if registered: 14GB annotated depth, 5GB
  projected LiDAR, 2GB validation/test subset.
- **ORDSLAM** (Outdoor RGB-D SLAM Dataset) — real ZED stereo-camera data,
  no login needed, but a poor fit on inspection: proprietary `.svo`
  format (needs the ZED SDK to even open), and its actual focus is
  vegetation/lighting-condition variation, not road/terrain, per its own
  description. Dropped.
- **DIML/CVL RGB-D Dataset** — outdoor split described as
  office/dormitory/street/road scenes, not confirmed to include any
  water/mud/beach terrain, and depth is stereo-disparity-based (needs a
  calibration/baseline conversion step, unlike DIODE's direct metric
  depth + published intrinsics). Considered, not pursued — no clear
  advantage over what DIODE already provides.
- **Dedicated beach/muddy-ground/water-body RGB-D dataset** — searched
  specifically, found nothing real and downloadable matching that
  description. This is why sec.31 (existing local TartanGround scenes)
  and sec.30's water-adjacent OpenTopography surveys (Sacramento River,
  Coastal Dune Erosion, Deltaic Wetlands) became the actual path to that
  terrain type instead of a new RGB-D source.

---

## 34. WildScenes — a 4th Data Source, and Easily the Best Real
Natural-Terrain Result This Session

Followed up on sec.29/33's finding that WildScenes needs personal CSIRO
Data Access Portal credentials. User went through the portal themselves
(login, requested S3 access for the WildScenes collection) and pasted the
resulting temporary credentials (Access Key, Secret Access Key, ~48h
validity) directly into the conversation. Used them only via environment
variables for `boto3` — never written to any file in the repo, since
they're temporary and would be dead weight (or a real credential leak)
once expired.

**Confirmed bucket structure by listing it directly** (`s3.data.csiro.au`,
bucket `dapprd`, prefix `000061541v003/data/WildScenes/`):
`WildScenes3d/{K-01,K-03,V-01,V-02,V-03}/{Clouds,Hists,Labels}` (K = Karawatha
forest, V = Venman forest — the two forests from sec.29's earlier
research) plus a separate `Fullclouds/` (raw merged per-scan point clouds,
unlabeled).

**Format confirmed from the official devkit source** (fetched directly
from GitHub, not assumed), which matters because it's *not* quite
SemanticKITTI despite superficially similar `.bin`/`.label` naming:
`Clouds/*.bin` is raw float32 **x,y,z only** (confirmed by byte-count:
661,398 floats / 3 = exactly 220,466 points, not divisible by 4 — no
intensity column, unlike SemanticKITTI's raw sweeps or this project's own
RELLIS-3D conversion). `Labels/*.label` is uint32, one value per point,
matching the `.bin` point count exactly. Real per-point terrain classes
(from `wildscenes/tools/utils3d.py`'s `METAINFO`): bush, dirt, fence,
grass, gravel, log, **mud**, other-object, other-terrain, rock, sky,
structure, tree-foliage, tree-trunk, **water** — includes genuine mud and
water ground truth, the actual reason this dataset was pursued.
`Fullclouds/*.ply` confirmed via direct inspection to have properties
`x, y, z, time, intensity, returnNum` — **no RGB either**, same as most
of the OpenTopography surveys. Real color exists only as separate 2D
camera images (`WildScenes2d`), never fused onto the 3D points in the
downloadable files — would require a camera-to-lidar projection step
using their calibration data, not already done for us.

**New tooling:** `download_wildscenes.py` (lists + randomly samples N
frames per sequence via `boto3`, converts `.bin`+`.label` pairs to `.ply`
with `x,y,z,classification`) and `visualize_wildscenes_rl.py` (same
single-shot bypass + recentering + largest-support-among-horizontal
selection as every other real-data script; "ground" for scoring =
{dirt, grass, gravel, mud, other-terrain, rock}, water tracked as its own
separate category rather than folded into "ground" or "not-ground", since
a water surface being found as a flat plane would be geometrically
correct but semantically distinct and worth seeing on its own).

**Results, 20 tiles (4 per sequence × 5 sequences, seed 0):**

| Seq | Frame | Points | Ground gt % | Found? | Angle | IoU |
|---|---|---|---|---|---|---|
| K-01 | 1624326467... | 131,950 | 77.7% | FOUND | 1.8° | 0.917 |
| K-01 | 1624327476... | 167,716 | 42.3% | FOUND | 4.7° | 0.842 |
| K-01 | 1624327916... | 121,564 | 26.9% | FOUND | 6.7° | 0.702 |
| K-01 | 1624328845... | 81,670 | 13.6% | FOUND | 17.8° | 0.522 |
| K-03 | 1639435021... | 101,993 | 46.6% | FOUND | 4.7° | 0.855 |
| K-03 | 1639436402... | 103,054 | 26.7% | FOUND | 7.6° | 0.845 |
| K-03 | 1639439499... | 78,707 | 8.9% | FOUND | 21.3° | 0.000 |
| K-03 | 1639440831... | 73,513 | 26.5% | FOUND | 13.0° | 0.812 |
| V-01 | 1623378861... | 109,979 | 31.0% | FOUND | 8.6° | 0.559 |
| V-01 | 1623379014... | 74,150 | 43.6% | FOUND | 0.5° | 0.728 |
| V-01 | 1623379177... | 32,698 | 39.9% | FOUND | 0.3° | 0.326 |
| V-01 | 1623379783... | 106,811 | 21.9% | FOUND | 12.3° | 0.712 |
| V-02 | 1623370473... | 107,382 | 35.7% | FOUND | 9.5° | 0.809 |
| V-02 | 1623371286... | 98,607 | 36.4% | FOUND | 10.4° | 0.849 |
| V-02 | 1623371836... | 90,040 | 20.5% | FOUND | 4.8° | 0.664 |
| V-02 | 1623372132... | 44,347 | 30.9% | FOUND | 11.8° | 0.590 |
| V-03 | 1639697445... | 83,415 | 24.4% | FOUND | 7.8° | 0.450 |
| V-03 | 1639699912... | 117,784 | 25.3% | FOUND | 14.9° | 0.675 |
| V-03 | 1639700060... | 85,281 | 23.7% | FOUND | 9.2° | 0.692 |
| V-03 | 1639700261... | 71,994 | 43.7% | FOUND | 28.7° | 0.672 |

**20/20 found a plane (100%), mean IoU ≈0.64 against real ground-truth
terrain classes** (best 0.917, worst 0.000 on one K-03 frame that found a
plane with zero overlap with real ground — likely a different flat
surface like a fallen log or structure, not the forest floor). This is
**by far the best real natural-terrain result all session** — every
OpenTopography survey in sec.30 topped out at 0.05-0.23 IoU even with a
manually loosened eps, while WildScenes reaches 0.64 *at the model's own
default parameters*, no override needed. The likely reason: these are
local handheld-SLAM submaps along a walking trail (tens of meters across),
not sprawling multi-km² area surveys — the forest floor immediately
around the path is much more likely to be one continuous, fairly flat
patch than a whole mountainside or dune field is.

**Water caveat:** 0.0% water in all 20 randomly-sampled frames — real
"mud" and "water" classes exist in the dataset and are tracked separately
by the script, but none of this particular random sample happened to
contain any. Not yet resampled specifically to find water-containing
frames.

**Results saved under `report_screenshots/wildscenes/<seq>/` +
`ALL_SEQUENCES_SUMMARY.csv`.**

---

## 35. A 5th Data Source: TartanGround's Native `image`+`depth` Modality
(Not LiDAR) — and Fixing a Real Environment Blocker to Get There

User wanted to compare the LiDAR-sweep results from the water/mud scenes
(sec.31: SeasideTown, GreatMarsh, NordicHarbor, GothicIsland) against
those same scenes' native camera image + depth map modality, back-projected
to a point cloud the same way DIODE was (sec.32) — a genuinely different
sensing modality of the *same* underlying scenes, not a new dataset.

**Blocker found and fixed: the project's own `tartanair` library couldn't
even be imported.** `import tartanair` crashed with `ImportError: DLL
load failed while importing runtime: An Application Control policy has
blocked this file` — a Windows security policy blocking `cupy`'s CUDA
DLL, pulled in transitively by `tartanair.customizer` (used only for
optional camera-model/optical-flow re-rendering features). Confirmed via
`grep` that `download_ground()` (the only function actually needed) never
touches the customizer, and that `init()` *already* wraps customizer
instantiation in try/except expecting exactly this kind of failure — the
only bug was that the import itself sat at module level (line 4) instead
of being deferred into those try blocks, so the whole package failed
before ever reaching the graceful-degradation code that was clearly
already intended. **Fix: patched the installed
`.venv/Lib/site-packages/tartanair/tartanair.py`** to move
`from .customizer import TartanAirCustomizer, TartanAirFlowCustomizer`
out of the top-level imports and into the two `try:` blocks that
instantiate them. Verified: `import tartanair as ta` now succeeds cleanly,
`download_ground` importable and callable. Local, surgical, reversible
(a fresh `pip install tartanair` would undo it) — not a system-level
change.

**Confirmed real camera intrinsics and depth format from tartanair's own
source** (not guessed): `fx=fy=320, cx=cy=319.5` (`customizer.py`,
hardcoded, used internally for their own re-projection code), image/depth
resolution 640x640. Depth is stored as an RGBA PNG, but — confirmed via
`reader.py`'s `depth_rgba_float32()` — the 4 uint8 channels per pixel are
literally the 4 raw bytes of a little-endian float32 depth-in-meters
value (`depth_rgba.view("<f4")`), decoded with
`cv2.imread(path, IMREAD_UNCHANGED)` (not PIL, to avoid any channel
reordering). Verified sane decoded values on a real sample (0.53-20.6m,
mean 2.88m) before building anything on top of it.

**A real, avoidable oversized-download mistake, caught before it went
too far:** the first `download_ground(..., modality=['image','depth'])`
call (no `camera_name` filter) started pulling *every* camera direction's
depth for GothicIsland alone (`lcam_front`, `lcam_back`, `lcam_bottom`,
`rcam_left`, ...) — already 766MB of depth zips alone before being killed,
for a scene that only needed the front camera. Fixed by adding
`camera_name=['lcam_front']`, which brought GothicIsland down to a
reasonable 0.49GB for both modalities combined. Valid ground modality
list confirmed via the library's own `tartanair_module.py`: `['image',
'meta', 'depth', 'seg', 'imu', 'lidar', 'rosbag', 'sem_pcd', 'rgb_pcd',
'seg_labels']` — `rgb_pcd` (a pre-made colored point cloud) exists as a
simpler alternative to manual back-projection, noted but not used since
the user specifically wanted to do the RGB-D→point-cloud conversion.

**A second real bug, also fixed:** the downloader's own auto-unzip step
crashed with `UnicodeEncodeError` printing a colored warning message on
Windows' default `cp1252` console encoding — this is the same category of
issue the project's *other* downloader script (`download_lidar_frames.py`)
already explicitly works around at the top of the file. First occurrence:
manually extracted the already-fully-downloaded zips with Python's
`zipfile` module instead of re-downloading. Subsequent scenes: fixed
properly by setting `PYTHONIOENCODING=utf-8` on the subprocess, which
let auto-unzip complete cleanly (confirmed on SeasideTown).

**New tooling: `visualize_tartanair_rgbd_rl.py`.** Same
back-projection + single-shot-bypass + z-up-axis-swap pattern as
`visualize_diode_rl.py`. No ground truth available for this modality
(visual-only, true RGB + height-colored/predicted-ground, same standard
as Ridgecrest/DIODE).

**Results, 5 random frames per scene (seed 0), all 4 scenes complete:**

| Scene | Found | Angles from vertical | Coverage |
|---|---|---|---|
| GothicIsland | **5/5, all genuine** | 1.2°, 1.8°, 2.4°, 2.9°, 10.0° | 31.8%-47.0% |
| SeasideTown | **5/5, all genuine** | 0.2°, 0.3°, 1.5°, 11.5°, 12.4° | 2.4%-30.8% |
| GreatMarsh | 5/5 found, **4/5 genuine** | 0.1°, 0.3°, 1.1°, 12.7° genuine; 1 fallback at 43.3° | 9.3%-68.0% |
| NordicHarbor | **5/5, all genuine** | 0.2°, 0.4°, 0.8°, 1.0°, 1.8° | 34.5%-46.9% |

**19/20 frames across all 4 scenes (95%) genuinely horizontal**, with zero
fallback cases on 3 of the 4 scenes — dramatically cleaner than DIODE's 72%
genuine rate or the LiDAR sweeps' frequent fallbacks. Directly comparable
to sec.31's LiDAR results on the *same four scenes*: GothicIsland LiDAR was
3/3 valid but only 5-9% coverage; SeasideTown LiDAR was 0/3 valid (2
outright failures, 1 false-found at 82.8°); GreatMarsh LiDAR was 1/3 valid;
NordicHarbor LiDAR was 1/3 valid with 2 false-founds at ~89°. NordicHarbor
is the starkest single comparison in this report: weakest LiDAR result of
the four scenes, cleanest RGB-D result (tightest angle spread of any
source tested this session, 0.2°-1.8°). **RGB-D back-projection is
dramatically cleaner than sparse 360° single-sweep LiDAR for these
scenes** — plausible reason: a dense forward-facing depth image
concentrates hundreds of thousands of points into one coherent field of
view, where a single omnidirectional LiDAR sweep spreads far fewer returns
across the full 360°, most of it far from anything relevant.

**The one fallback case (GreatMarsh, 43.3°) is informative, not just
noise:** confirms the horizontal_candidate=False mechanism documented in
sec.32/33 is a genuine property of specific scans (no horizontal surface
in that particular camera's field of view), not an artifact unique to
DIODE or to LiDAR — it recurs at a low rate even in the best-performing
modality tested.

**Download reliability note:** Hugging Face throttled/rate-limited the
unauthenticated download requests partway through this session (memory
growth per background-task check went from steady to near-flat over
several checks) — tried `data_source='airlab'` (CMU's own server) as an
alternative, which failed outright with `404 Not Found` (this data
genuinely isn't hosted there, confirmed, not a config error). Reverted to
`huggingface` with `num_workers=1`, slower but reliable — completed
successfully for all 4 scenes, including GreatMarsh's unusually large
4.13GB download (3,727 frames, ~7x GothicIsland's) and NordicHarbor's
2.90GB (3,182 frames). User created a free Hugging Face account and
obtained an access token mid-session as a faster alternative, but the
in-progress download finished on its own via the slow path before the
token was needed.

**Results saved under `report_screenshots/rgbd/tartanair/<scene>/` +
`ALL_SCENES_SUMMARY.csv`.**

---

## 36. Documentation Debt Closed: RELLIS-3D Visual Ground-Truth Comparison
(`visualize_real_lidar_gt.py`)

Built earlier in the session, before §27, but never logged at the time —
flagged as documentation debt in this session's own notes and closed now
while assembling the real-world report (sec below on the report itself).
This is distinct from the *numeric* RELLIS-3D evaluation in
`Adaptive_RANSAC_RL_Report.docx` (`eval_rellis3d.py`, IoU/precision/
recall/F1): this script was built specifically in response to the user's
explicit mid-session instruction to stop reporting IoU/numeric scores for
this real-ground-truth work and show results visually instead.

**Method:** single-shot bypass (same pattern as `visualize_real_pointcloud.py`),
scored against real per-point `<frame>_gt_mask.npy` files that exist for 6
scenes only (`AbandonedFactory, CoalMine, Gascola, House, NordicHarbor,
WesternDesertTown`), TP/FP/FN/TN confusion-matrix coloring (yellow/red/
orange/gray). Point/mask length alignment handled defensively
(`load_points_matching_mask`): tries the raw `.ply` point count first,
falls back to `ransac_env.py`'s 0.05m-voxel-downsampled count, skips the
frame with an explicit message if neither matches, since it was confirmed
empirically that different scenes' masks align to different point counts
(CoalMine to raw, Gascola/House to downsampled) — not assumed to be
uniform. 12 screenshots generated (2 random frames per scene, seed 0) to
`report_screenshots/real_gt/`.

**Re-inspected directly (not from memory) while writing this entry**, one
frame per scene:

| Scene | Observed |
|---|---|
| NordicHarbor | Clean success — yellow (TP) densely overlaps the orange (real ground) near-sensor LiDAR rings. |
| House | Red (FP) ring inside an orange rectangular room outline, zero yellow — found a plane, but not the real ground (likely a wall or central object, not the floor). |
| Gascola | Only orange visible, no red or yellow at all — no plane found this frame, real ground entirely missed. |
| AbandonedFactory | Small red blob near the sensor, no orange visible anywhere in frame — a wrong-plane find; the real ground-truth mask for this specific frame has effectively no ground in view. |
| CoalMine | Same pattern as AbandonedFactory — red ring near sensor, no orange visible. |
| WesternDesertTown | Same pattern — small red patch near sensor, no orange visible. |

Only 1 of the 6 scenes checked (NordicHarbor) shows a clean, unambiguous
real-ground match in this single-frame spot-check; the rest show either a
wrong-plane failure or an outright miss. This is a smaller, harder set of
real-ground-truth frames than the other sources tested this session (only
6 scenes have any mask at all, and several of the sampled frames
apparently contain little to no real "ground" class in view at all,
independent of detection quality) — consistent with, not contradicting,
the broader real-world pattern (§30, §34) that flat, easily-detected
ground is the exception rather than the rule in unstructured real scenes.

**Results saved under `report_screenshots/real_gt/` (12 screenshots, no
CSV log — built before the CSV-logging convention was established in
later sections).**
