# Adaptive RANSAC RL — Strategy & Improvements Reference

Organized by theme (not chronology) for report/thesis writing. For a day-by-day
narrative with exact numbers and dates, see `DAILY_LOG.md`. This document
synthesizes that log plus supporting design docs (`BASELINE_CONFIG.md`,
`ADAPTIVE_RL_PLAN.md`) into sections that map onto typical report structure.

---

## 1. Problem Statement & Motivation

Ground-plane detection in LiDAR point clouds via Efficient RANSAC (Schnabel et
al.) requires hand-tuned parameters — `epsilon` (inlier distance threshold),
`min_support` (minimum inlier count), `normal_threshold` (horizontality
tolerance) — that trade off against each other depending on terrain. A single
fixed configuration cannot be simultaneously correct for a flat indoor office
floor and bumpy outdoor forest terrain: tight parameters (`Strict`) miss rough
ground; loose parameters (`Loose`) over-segment indoors, mislabeling walls and
furniture as ground.

This project frames parameter selection as a sequential decision problem: a
reinforcement learning agent (PPO) observes per-scene geometric features and
chooses RANSAC parameters adaptively, with up to 5 refinement steps per scene
to iteratively adjust its choice based on feedback from the previous attempt.

---

## 2. System Architecture

- **RANSAC engine:** Schnabel efficient RANSAC, implemented in C++ and exposed
  to Python via a Cython binding (`schnabel_ransac.detect()`).
- **Environment:** `ransac_env.py`, a custom OpenAI Gym (`gymnasium`)
  environment (`RansacEnv`) wrapping the Cython call, responsible for loading
  point clouds, computing observation features, calling RANSAC with the
  agent's chosen parameters, and computing reward.
- **Agent:** Proximal Policy Optimization (PPO) via Stable-Baselines3,
  `MlpPolicy`, trained with `VecNormalize` (observation + reward
  normalization) over a `MultiDiscrete` action space.
- **Data:** TartanGround (synthetic, LiDAR-simulated point clouds across 15+6
  named scenes) for training and cross-scene generalization testing;
  RELLIS-3D (real off-road Ouster OS1-64 LiDAR with human-annotated
  point-wise semantic labels) for real-world ground-truth validation.

---

## 3. State Space Design

31-dimensional observation, in two parts:

- **21 scene (geometric) features** — computed once per episode in `reset()`
  from the raw point cloud (`features/scene_features.py`): PCA-based
  descriptors (bounding-box volume, normal variance, etc.), computed via
  Open3D. Originally planned around 18 features using an intensity/
  reflectivity channel; the dataset's `.ply` files don't carry that channel,
  so this was replaced with 21 purely geometric features instead.
- **10 feedback features** — updated every step: current and previous
  `inlier_ratio`, `mean_residual`, the detected plane normal's 3 components,
  `step_count`, and the previous step's chosen `epsilon`/`min_support`/
  `normal_threshold`. These give the agent memory of what it already tried
  and how well it worked, which is what makes the 5-step sequential
  refinement meaningful rather than 5 independent guesses.

`VecNormalize(norm_obs=True, ...)` normalizes this state before it reaches
the policy network, because the 21+10 features mix wildly different scales
(e.g. `bbox_volume` vs. a 0–1 `normal_consistency` ratio) — see §8.1.

---

## 4. Action Space Evolution

| Version | Action space | Change |
|---|---|---|
| Initial | `MultiDiscrete([8, 6, 2])` | `epsilon` (8 levels), `min_support` (6 levels), stop/continue |
| Current | `MultiDiscrete([8, 6, 6, 2])` | + `normal_threshold` (6 levels: `[0.80, 0.85, 0.88, 0.90, 0.93, 0.95]`) |

`normal_threshold` was fixed at `0.90` for the first several training
iterations. It was promoted into the action space because `epsilon` and
`normal_threshold` are directly coupled in practice: loosening `epsilon` to
admit more distant points usually needs a correspondingly looser
`normal_threshold`, since fringe points have noisier estimated normals. With
`normal_threshold` pinned, the agent had no way to act on this coupling —
plausibly a real contributor to persistent rough-terrain / `Supermarket`-style
failures observed under every earlier version. A smoke test confirmed the
added dimension isn't a no-op: three different `normal_threshold` choices on
one real frame produced clearly distinct rewards and inlier ratios (`0.80` →
reward 0.403; `0.95` → reward 0.285).

`EPS_LEVELS`, `MIN_SUPPORT_LEVELS`, `NORM_THRESH_LEVELS` are defined once as
module-level constants in `ransac_env.py` and imported everywhere else that
needs them (`visualize_inference.py`, `baseline_evaluator.py`) — consolidated
after finding the levels were previously being duplicated/hardcoded
inconsistently across scripts.

---

## 5. Reward Function: Three Iterations

### 5.1 Iteration 1 — terminal-only reward
```
reward = (1.0 * inlier_ratio) - (0.1 * runtime) - (0.5 * mean_residual)
         + (0.3 * normal_consistency) - (0.05 * step_count)
```
Computed only at episode termination (`stop` action or 5-step cap). **Failure
mode:** every non-terminal step received a reward of exactly `0.0` — the
agent had no signal about whether an intermediate refinement step helped at
all, only a lump-sum payout at the end. Combined with the flat
`-0.05 * step_count` penalty (the only thing actually influencing the
stop/continue decision), the agent learned to always stop after 1 step
(`mean_steps` pinned at 1.0 across every dataset), never exercising the
5-step sequential refinement the whole project is built around.

### 5.2 Iteration 2 — potential-based shaping
Redefined a potential function over the parts of the reward that respond to
the agent's actions each step:
```
Φ(state) = 1.0 * inlier_ratio - 0.5 * mean_residual
reward    = [Φ(new) - Φ(prev)] - 0.1 * runtime - 0.01
```
with `normal_consistency` kept as a separate `+0.3` bonus applied only at the
terminal step. Every step now carries real, dense credit-assignment signal
instead of zero. Verified by hand-tracing the telescoping sum before
training: the total episode return recovers the same terms as Iteration 1,
just paid out incrementally rather than as one lump sum — a reward-shaping
change, not a new objective. Confirmed via smoke test (a step with genuine
improvement got `+0.097`; steps with no further improvement got a small
negative tie-breaker `~-0.025`).

The `-0.1 * runtime` term was later removed entirely (both here and in the
`too_few_points` early-return path): it taxed every step equally rather than
specifically discouraging unhelpful continuation, and `mean_steps` was still
drifting back down toward 1.0 despite the shaping fix.

### 5.3 Iteration 3 — fixing a structurally invalid terminal bonus
A live training run's logs surfaced a case where reward was strongly
*positive* (`+0.197`) despite plane quality clearly getting *worse* that step
(`Φ` dropped from `0.0897` to `0.00975`, a genuine `-0.09` shaping penalty).
Root cause: `normal_consistency` is a **static per-frame scene feature**,
computed once in `reset()` before any RANSAC call — it cannot respond to the
agent's chosen parameters at all, and it sits near `~0.99` for almost every
frame. As a flat bonus applied only at the terminal transition, it broke
potential-based shaping's telescoping-sum property (a term that doesn't
cancel across the episode isn't policy-invariant) and functioned as a
near-constant reward for terminating, regardless of actual detection quality
— directly re-causing the `mean_steps → 1.0` collapse a second time, under
the 4-action-space version.

**Fix:** replaced the static `normal_consistency` bonus with `z_align` — the
*actually-detected* plane's `|normal_z|` from `find_ground_plane()`, computed
fresh every step from the real Schnabel output, and therefore genuinely
action-sensitive. Folded directly into the potential function so it
telescopes like the other two terms:
```
Φ(state) = (1.0 * inlier_ratio) - (0.5 * mean_residual) + (0.15 * z_align)
reward    = [Φ(new) - Φ(prev)] - 0.01
```
The weight was set at `0.15`, not the original bonus's `0.3` — a distribution
analysis across 141,435 logged steps showed `z_align` is close to bimodal
(a direct consequence of `find_ground_plane`'s `horizontal_thresh=0.80`
gate), so a full gate-crossing swing at `0.3` would contribute up to `0.3` to
potential — 2–3x the typical meaningful `inlier_ratio` improvement (p75–p25
≈ 0.15, max ever observed 0.447) — and would dominate the term actually being
optimized.

---

## 6. Training Data Sampling Strategy

The sampling strategy went through four distinct problems and fixes, each
uncovering the next:

### 6.1 Adaptive (hard-negative-mining) folder selection
Introduced early: files are grouped by top-level scene folder, an
exponential moving average (EMA) tracks each folder's recent reward, and
folder selection is biased toward historically low-reward (harder) folders
via softmax over `-reward`. Rationale: force the agent to keep practicing on
scenes it's currently weak at (e.g. `Sewerage`) rather than spending most of
its budget on scenes it has already mastered.

### 6.2 The `frame_id` collision (measurement artifact, not a real bug)
Coverage was logged using the bare filename (e.g. `"000000_....ply"`), but
every scene's frames are numbered starting from `000000` — so frames from all
15 scenes collided onto the same IDs in the log, making a raw coverage
reading (~14%) look far worse than reality. Fixed by scene-qualifying the
logged ID (`"<folder>/<filename>"`); true coverage was then re-measured via
proper episode-boundary reconstruction rather than a naive column-mean.

### 6.3 Within-folder replacement (a real bug)
Once the measurement artifact was removed, a genuine coverage problem
remained: frame selection *within* a chosen folder was a fresh
random-with-replacement draw every time, which can revisit the same handful
of frames repeatedly while other frames in that folder are never drawn at
all, even across thousands of episodes. **Fix:** `_next_frame_in_folder()` —
a per-folder shuffled queue consumed without replacement, reshuffled only
once fully exhausted. Guarantees every frame in a folder is seen exactly once
per pass before any repeat.

### 6.4 Folder-selection frequency ignoring folder size (a second real bug)
Folder sizes in the training pool range 7x (533 to 3,727 frames across 15
folders, 24,528 frames total), but folder-selection probability (both the
`uniform_folder_prob` floor and the adaptive EMA branch) treated all folders
equally regardless of size. Simulating the real sampler against real folder
sizes showed equal-per-folder selection reaches only ~76% true coverage at a
50,000-step budget, with the largest folder (`GreatMarsh`) reaching just
~47% of its own frames — being picked no more often than a folder 7x smaller
means it only gets through a fraction of its own frames in the same budget.

**First fix (rejected on further review):** weight folder selection directly
by folder size (`probability ∝ frame_count`). Verified via simulation
(~99.9% coverage) and directly against the running environment (3,000 real
`reset()` calls matched predicted size proportions within ~1 percentage
point). This *did* solve coverage, but introduced a new problem (§6.5).

### 6.5 The proportional-sampling bias, and the sqrt(size) resolution
Weighting selection by raw folder size means a folder 7x larger also
receives ~7x more policy updates during training — purely because the
dataset happened to contain more captured frames for that scene, not because
that terrain is more important to learn. This is a real bias: it implicitly
tells the optimizer "this terrain matters 7x more," with no principled
justification. (It also isn't a new problem introduced by the fix — the
*original* equal-per-folder scheme had the mirror-image bias: treating a
3,727-frame scene and a 533-frame scene as equally important selection
*targets* systematically under-covers the larger scene's actual frame
diversity. There is no bias-free option here, only a choice of which
imbalance to accept.)

**Resolution:** weight folder selection by `sqrt(folder_size)` rather than
raw size. This caps the largest-vs-smallest folder's selection-probability
ratio at **~2.64x** (`GreatMarsh` 10.6% vs. `GothicIsland` 4.0% of draws),
down from ~7x under full proportional weighting, while still reaching
**~100% simulated single-pass coverage at 50,000 steps** (vs. ~76% under the
original equal-weighting) — verified against the real running environment.
This is the sampling strategy the next full training run (`v4_sampler_heldout`)
uses.

| Scheme | Max/min folder selection ratio | Simulated coverage @ 50k steps |
|---|---|---|
| Equal-per-folder (original) | 1x | ~76% (worst folder ~47%) |
| Full size-proportional | ~7x | ~99.9% |
| **sqrt(size)-weighted (adopted)** | **~2.64x** | **~100%** |

---

## 7. Training Strategy

### 7.1 Algorithm and hyperparameters
PPO (`stable_baselines3.PPO`), `MlpPolicy`, over the `MultiDiscrete([8,6,6,2])`
action space (§4). Key settings in `train_rl.py`:

| Setting | Value | Why |
|---|---|---|
| `learning_rate` | 0.0003 | SB3 default; not swept — out of scope for this project |
| `ent_coef` | 0.01 | Added to fix policy collapse (§8.1 — a `0.0` default let the agent settle on one constant action) |
| `VecNormalize` | `norm_obs=True, norm_reward=True, clip_obs=10.0` | Normalizes the 31-dim state's mismatched feature scales (§3); also fixes policy collapse |
| `device` | `"auto"` (resolves to CUDA) | Left as default after profiling showed the bottleneck is CPU-bound RANSAC, not the policy network (§10) — changing it would not help |
| Environment vectorization | `DummyVecEnv`, single instance | `SubprocVecEnv` multi-core parallelism identified as a real speedup option but deliberately not adopted (§10) |

### 7.2 Train/test split: environment-level only, no frame-level split
Two independent ways to hold data out were available: **environment-level**
(exclude whole scenes from training, e.g. `Gascola`) and **frame-level**
(within an included scene, reserve a chronological tail of frames as a
held-out test set — `RansacEnv` supports this via `split`/`test_frac`/
`gap_frac`, built early in the project). The final methodology uses
**environment-level holdout only**: every frame of every in-sample scene is
used for training, and generalization is tested purely via the 6 fully
held-out scenes (§9.2) plus RELLIS-3D (§9.3). This was an explicit choice
over adding a second, frame-level split dimension on top — kept the
train/test methodology simple and easier to reason about and defend in a
report, at the cost of not also measuring in-scene (same-environment,
unseen-frame) generalization specifically. The frame-level split
infrastructure remains in the code (unused by the current training command)
in case that comparison is wanted later.

### 7.3 Non-destructive versioning via `--tag`
Every training run is tagged (`--tag <name>`), which namespaces its
checkpoints, final model, `VecNormalize` stats, and training CSV log
(`ppo_ransac_<tag>_final.zip`, `evaluation_metrics_<tag>.csv`, etc.) so a new
run can never silently overwrite a previous one's files. This let each
architecture/reward/sampler change (§4, §5, §6) become its own clean,
comparable checkpoint rather than destroying the evidence needed to compare
before/after:

| Tag | What it captured | Action space | Reward | Sampler |
|---|---|---|---|---|
| `ppo_ransac_final` (untagged, v1) | First trustworthy run post-collapse-fix (§8.1) | 3-action | Iteration 1 (terminal-only) | Equal-per-folder, EMA-adaptive |
| `v2_reward` | Potential-shaping pilot | 3-action | Iteration 2 | Equal-per-folder |
| `v2_reward_heldout` | + held-out scenes, runtime term removed | 3-action | Iteration 2 (no runtime term) | Equal-per-folder |
| `v3_normthresh_heldout` | + `normal_threshold` action dimension | 4-action | Iteration 2 (still had the `normal_consistency` bug, §5.3) | Equal-per-folder |
| `sampler_pilot` | Measurement-only pilot for the sampler fixes | 4-action | Iteration 3 (`z_align`) | Mixed, pre-size-weighting |
| `v4_sampler_heldout` (in progress) | Full run under all current fixes | 4-action | Iteration 3 | sqrt(size)-weighted (§6.5) |

### 7.4 Checkpointing and resume strategy
`CheckpointCallback` saves a full checkpoint (model + `VecNormalize` stats)
every 1,000 steps. Combined with the `reset_num_timesteps=(args.load is None)`
fix (§8.5), this made it possible to resume a run from its exact last
checkpoint with no data loss and no discontinuity in the step counter —
which mattered in practice, not just in theory: a training run survived at
least two unexplained mid-run process deaths (no traceback, cause never
identified) by resuming from the latest checkpoint each time.

### 7.5 Step budget
The current run (`v4_sampler_heldout`) uses **100,000 timesteps**, chosen by
weighing measured throughput against training benefit rather than picking a
round number: at the measured ~0.517 sec/step (§10), 50,000 steps already
reaches ~100% single-pass frame coverage under the sqrt-weighted sampler
(§6.5), so 100,000 steps buys roughly two full passes over the 24,528-frame
training pool for further policy convergence, at an estimated ~14.4 hours
wall-clock. The run does not have to be treated as final at 100,000 steps —
since resume works cleanly (§7.4), it can be extended from its final
checkpoint with more steps later if the reward curve is still trending
upward when it completes.

**Status at time of writing:** `v4_sampler_heldout` is in progress
(817 steps logged as of the most recent check).

---

## 8. Infrastructure & Correctness Fixes

These aren't strategy choices, but they materially affect how trustworthy
every result up to the point of each fix is — worth citing explicitly when
discussing which results are reliable.

### 8.1 Policy collapse (entropy + normalization)
The first full-scale evaluation appeared to show RL beating every fixed
baseline. Deeper inspection revealed the agent was outputting the **exact
same action** for 100% of ~15,900 evaluated frames across all 10
environments — a fully collapsed policy that had auto-discovered one
decent static parameter combination, not learned adaptive behavior. Root
cause: `ent_coef` defaulted to `0.0` (no entropy bonus discouraging output
collapse) and no `VecNormalize` was applied (the 31-dim state mixes wildly
different feature scales unnormalized). Fixed with `ent_coef=0.01` and
`VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)`. Confirmed
fixed by checking the agent now genuinely varies `epsilon` per scene under
deterministic inference.

### 8.2 `find_ground_plane()` 3-vs-4-value return mismatch
Two early-return paths returned 3 values where the caller unpacked 4. This
silently crashed and was swallowed by a blanket `except Exception` in
`step()`, meaning "no ground plane found" and "the code actually crashed"
were indistinguishable in every evaluation log prior to the fix.

### 8.3 Evaluation scripts feeding unnormalized observations
Once training used `VecNormalize`, the trained model's weights only make
sense on normalized observations — but `rl_evaluator.py` and
`visualize_inference.py` were feeding raw, unnormalized state. Fixed with
`load_obs_normalizer()`, which replicates SB3's exact
`(obs - mean) / sqrt(var + eps)` formula from the saved `.pkl` stats file.

### 8.4 `baseline_evaluator.py` action-encoding rot
After the `MultiDiscrete` action-space refactor, `baseline_evaluator.py`
still called a removed method (`env._unnormalize_action()`), left over from
an earlier continuous-action design. Rewritten around a `BASELINE_MODES`
dict and `build_action()` matching the current discrete action space exactly.
A second, deeper bug found in the same pass: `step()` hardcoded
`normal_thresh=0.90` regardless of mode, which would have silently run the
`Standard`/`Loose` baselines at the wrong `normal_thresh` even after the
encoding fix. Fixed with a `fixed_normal_thresh` override parameter on
`RansacEnv` (defaults to `None`, zero behavior change for RL training/eval).

### 8.5 Checkpoint-overwrite data loss on resume
Resuming training via `--load` didn't pass `reset_num_timesteps=False`, so
Stable-Baselines3 defaulted to resetting the step counter to 0 — every
resumed run's checkpoints then silently overwrote the original run's
same-numbered files (7 checkpoints lost from one pilot run before this was
caught). Fixed with `reset_num_timesteps=(args.load is None)`. This fix is
also what made it safe to survive several unexplained mid-training process
deaths later in the project (see §7.4) by simply resuming from the latest
checkpoint with no data loss.

### 8.6 RELLIS-3D silently swept into the training pool
A recursive `data_dir=None` scan for training data collected every `.ply`
file under `data/`, which — after RELLIS-3D was downloaded for evaluation —
included 13,556 real-world frames that should never be trained on. Fixed
robustly: `NEVER_TRAIN_ENVIRONMENTS` (held-out scenes + RELLIS-3D) is now
always excluded inside `train_rl.py` regardless of what `--exclude_envs`
is passed, specifically to prevent the "forgot to type it in the exclude
list" class of bug from recurring as more data is added later.

### 8.7 Invalid LiDAR returns corrupting RELLIS-3D ground truth
Ouster's KITTI-bin export format zero-pads no-return beams as `(0,0,0,0)`
rather than dropping them, to preserve a fixed per-scan point count. ~40% of
every raw RELLIS-3D scan was this padding, sitting at the sensor origin with
meaningless fallback semantic labels — traced from an earlier oddity where
label `17` ("person") had implausibly high counts. Fixed by dropping
exact-zero points during conversion (131,072 raw points → 78,658 valid per
frame; ground fraction corrected from a misleading 29% of *all* points to a
real 49% of *valid* points).

### 8.8 NED vs. standard sensor Z-convention (a real, significant bug)
The first RELLIS-3D smoke test came back with `Loose` scoring *worse* than
`Strict` (inverted from the expected relationship), and one sequence scoring
exactly `0.0000` IoU across every mode. Root cause: `find_ground_plane`'s
`z_mode="z_down"` was hardcoded, tuned for TartanAir's apparent NED-style
simulation convention (ground = *highest* z). RELLIS-3D's real Ouster
sensor data uses the opposite, standard sensor z-up convention (ground =
*lowest* z). Confirmed directly on the failing case: RANSAC found the real
ground plane fine (23,761 points, mean z −1.04) but `find_ground_plane`
picked a tree-canopy/overhang plane instead (10,239 points, mean z +1.69)
purely because it was sorting for the wrong sign. Fixed with a `z_mode`
constructor parameter on `RansacEnv` (`"z_down"` default preserves all
existing TartanAir behavior exactly; RELLIS-3D evaluation passes `"z_up"`).

### 8.9 No exposed random seed in the Schnabel C++ call
Confirmed empirically, not assumed: four back-to-back calls on the
*identical* frame and parameters returned wildly different inlier ratios
(1%, 55%, 55%, 55%). Not a bug introduced during this project — an inherent,
previously-undocumented property of the underlying RANSAC implementation, and
it affects every result in the project, not just RELLIS-3D evaluation. Its
practical implication: **any single-frame comparison is unreliable; only
aggregates over many frames should be trusted or cited.**

---

## 9. Evaluation Methodology

Three independent evaluation tracks, deliberately kept separate:

### 9.1 Fixed-parameter baselines (`Strict` / `Standard` / `Loose`)
Defined in `BASELINE_CONFIG.md` — see table below. Run on raw, unfiltered
point clouds (no voxel downsampling) to represent the worst-case
fixed-parameter scenario, across the full training-environment set.

| Parameter | Strict | Standard | Loose |
|---|---|---|---|
| `epsilon` | 0.10 m | 0.15 m | 0.25 m |
| `min_support` | 800 | 500 | 200 |
| `normal_threshold` | 0.90 | 0.85 | 0.80 |

### 9.2 Cross-scene generalization (held-out scenes)
6 TartanGround scenes (`Gascola`, `House`, `WesternDesertTown`, `CoalMine`,
`AbandonedFactory`, `NordicHarbor`) are excluded from training entirely and
reserved purely for evaluation — an environment-level split, not a
frame-level one (the project deliberately trains on every frame of every
in-sample scene rather than holding out a frame-level tail, to keep the
methodology simple and avoid a second, harder-to-reason-about split
dimension). `NordicHarbor` specifically was kept held-out (rather than added
to the 15-scene training pool) even though it's coastal like the in-sample
`SeasideTown`, specifically to test generalization *across* similar coastal
terrain types, not just "has the model ever seen water at all."

### 9.3 Real-world ground truth (RELLIS-3D)
Every evaluation prior to this track scored predictions against a heuristic
(`find_ground_plane`'s own PCA/z-height guess) standing in for ground truth —
never an actual accuracy number. RELLIS-3D (`unmannedlab/RELLIS-3D`) provides
real off-road Ouster OS1-64 LiDAR with genuine point-wise human semantic
labels (20 classes; ground truth defined here as raw label IDs
`1/3/10/23/31/33` = dirt/grass/asphalt/concrete/puddle/mud, verified directly
against the dataset's own `rellis.yaml` rather than assumed). IoU / Precision
/ Recall / F1 are computed by nearest-neighbor-matching each downsampled
predicted point back to the raw, undownsampled ground-truth cloud — avoiding
any dependency on Open3D's voxel downsampling being index-stable.

**100-frame smoke test result** (20 frames × 5 sequences), benchmarking
`v3_normthresh_heldout` — the model trained *before* the §5.3 reward fix —
as a documented "before" data point:

| Mode | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| Loose | 0.631 | 0.878 | 0.712 | 0.719 |
| RL (`v3_normthresh_heldout`) | 0.348 | 0.952 | 0.368 | 0.423 |
| Standard | 0.317 | 0.951 | 0.338 | 0.390 |
| Strict | 0.231 | 0.978 | 0.237 | 0.305 |

The RL model underperformed the simple fixed `Loose` baseline on real
off-road terrain at this stage, but the *shape* of the gap is informative
beyond the headline number: RL's precision (0.952) is on par with `Strict`'s
(0.978), while recall is low (0.368) — the sim-trained policy is being too
conservative on real terrain, confidently claiming only a small slice of the
true ground extent rather than its full area. Consistent with a policy whose
risk tolerance was learned on cleaner synthetic geometry not transferring
directly to real terrain's bumpiness. This result predates both the §5.3
reward fix and the §6 sampler fixes, and is documented as a "before" point,
not a final result — a full-scale re-run against the corrected model is
still pending training completion.

---

## 10. Synthetic-Data Results Summary (pre real-world validation)

Two evaluation passes against TartanAir/TartanGround synthetic data, using
the (uncalibrated, heuristic) `find_ground_plane` scoring:

**Pass 1 (collapsed policy, later found invalid — see §8.1):** appeared to
show RL beating every baseline (mean inlier ratio 0.129 vs. Strict's best
0.079), but the agent was outputting one constant action for every frame —
not evidence of adaptive behavior.

**Pass 2 (post-collapse-fix, trustworthy):** mean inlier ratio 0.135 vs.
Strict's 0.079; 2,098s total runtime vs. Strict's 4,438s. Per-frame,
head-to-head: RL wins the individual frame outright 53.1% of the time
(Strict 16.0%, Standard 19.2%, Loose 11.7%); beats Strict on 65.9% of
frames, Standard on 68.5%, Loose on 80.6%. Per-dataset: wins 8 of 10,
sometimes by a wide margin (`Sewerage` 0.235 vs. Strict's 0.055; `Hospital`
0.213 vs. 0.067), but loses narrowly on `Downtown` and `Supermarket`.
`Supermarket` remained a genuine weak point (66.2% bad frames) — but for an
honest reason this time: the model explores 4 different action combinations
there, and simply keeps reaching for `min_support=800`, which doesn't suit
that scene's point distribution well.

This pass predates the reward-function fixes in §5.2/§5.3 and the sampler
fixes in §6 — cite as an intermediate result, not a final one.

---

## 11. Compute & Performance Considerations

Measured, not assumed, before deciding whether to invest engineering time
in acceleration:

- **Per-step wall-clock throughput:** ~0.517 sec/step, measured directly
  from checkpoint file timestamps across a real training run (5 consecutive
  1,000-step intervals, no drift).
- **Where that time actually goes:** mean logged Schnabel-RANSAC-only
  runtime is 0.401 sec/step (6,342 real steps) — ~78% of total step time is
  the CPU-bound C++ RANSAC call and point-cloud I/O (loading + voxel
  downsampling), not the neural network.
- **GPU:** available (NVIDIA GTX 1650 Ti) and already selected by default
  (`device="auto"` resolves to `cuda`), but concluded to be of negligible
  benefit — the MlpPolicy network is far too small (31-dim input, small
  hidden layers) for GPU compute to matter, and the actual bottleneck is
  entirely outside the neural network's control.
- **The real available speedup:** `SubprocVecEnv`-based multi-core
  parallelism (the environment currently runs as a single `DummyVecEnv`
  despite 8 CPU cores being available; independent Schnabel calls across
  parallel environment copies could run concurrently for a near-linear
  speedup). Identified and scoped, but not implemented — kept out of scope
  by explicit decision, to avoid changing the training setup this close to
  a full run.

---

## 12. Open Questions / Future Work

- Full-scale RELLIS-3D evaluation (all 13,556 frames, not just the 100-frame
  smoke test) against a model trained under the corrected (§5.3) reward
  function and sampler (§6.5) — the central "did the fixes actually help"
  question, currently unanswered because training hasn't completed.
- An isolated A/B of the reward-shaping changes alone (§5.2 vs. §5.3) has
  not been run — every real-world comparison to date also changed the
  sampler and/or action space in the same run, so the reward function's
  individual contribution is not yet cleanly isolated.
- Whether `v3_normthresh_heldout`'s RELLIS-3D result (§9.3) reflects a real
  effect or measurement noise is unresolved — the deterministic baseline
  configs showed comparable-magnitude run-to-run variation in earlier
  spot checks, and Schnabel's unseeeded randomness (§8.9) means single-run
  comparisons at this scale carry real uncertainty.
- `Supermarket`'s persistent weakness (both in early and post-collapse-fix
  evaluations) has an identified proximate cause (over-reliance on
  `min_support=800`) but no resolution yet — worth revisiting once the
  `normal_threshold` action dimension (§4) and corrected reward (§5.3) have
  both had a full training run to influence that behavior.
