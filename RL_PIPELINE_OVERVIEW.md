# Adaptive RANSAC RL Pipeline — Complete Overview

> A from-scratch explanation of this project's reinforcement learning
> system: what problem it solves, how the agent is trained, exactly what
> every term in the state/action/reward means, and what every file in the
> repo does.
> For the RANSAC algorithm internals, see [EFFICIENT_RANSAC_BREAKDOWN.md](EFFICIENT_RANSAC_BREAKDOWN.md).
> For the Cython/C++ wrapper this RL system calls into every step, see
> [SCHNABEL_CYTHON_BREAKDOWN.md](SCHNABEL_CYTHON_BREAKDOWN.md).

---

## Table of Contents

1. [The Problem, In Plain Terms](#the-problem-in-plain-terms)
2. [Why RL Instead of Just Picking Good Defaults](#why-rl-instead-of-just-picking-good-defaults)
3. [The RL Formulation](#the-rl-formulation)
4. [The Observation Space — Every Term Explained](#the-observation-space--every-term-explained)
5. [The Action Space — Every Term Explained](#the-action-space--every-term-explained)
6. [The Reward Function — Every Term Explained](#the-reward-function--every-term-explained)
7. [How the Agent Decides to Stop or Continue](#how-the-agent-decides-to-stop-or-continue)
8. [One Episode, Step by Step](#one-episode-step-by-step)
9. [The Training Loop (PPO)](#the-training-loop-ppo)
10. [Adaptive Sampling Across Environments](#adaptive-sampling-across-environments)
11. [Evaluation and Comparison Against Baselines](#evaluation-and-comparison-against-baselines)
12. [File-by-File Guide](#file-by-file-guide)
13. [Data Layout](#data-layout)
14. [How to Run Things](#how-to-run-things)
15. [Known Issues / Project History Worth Knowing](#known-issues--project-history-worth-knowing)

---

## The Problem, In Plain Terms

A LiDAR sensor on a drone/robot produces a **point cloud** every frame — a
few tens of thousands of `(x, y, z)` points describing everything the
sensor can see: the ground, walls, furniture, trees, etc. To navigate
safely, the robot needs to know **which points are the ground** (safe to
drive/fly over) and which are **obstacles**.

RANSAC (specifically Schnabel's Efficient RANSAC) can find the ground
plane in a point cloud — but it needs several parameters set correctly:
how close a point must be to the plane to count (`epsilon`), how many
points a plane needs to be considered real (`min_support`), and how
aligned a point's surface normal must be (`normal_thresh`). **The right
values for these parameters are different in every scene**: a flat indoor
office floor tolerates a tight `epsilon`; a bumpy forest floor needs a
loose one. Fixed parameters that work well indoors fail outdoors and vice
versa (this is exactly what `BASELINE_CONFIG.md`'s Strict/Standard/Loose
comparison demonstrates).

**This project trains an RL agent to look at a scene and choose good
RANSAC parameters for it, frame by frame, instead of using one fixed
setting for every scene.**

---

## Why RL Instead of Just Picking Good Defaults

You could imagine a simpler rule-based approach ("if point density is low,
use a loose epsilon"). RL is used instead because:

- The mapping from scene statistics → good parameters isn't obvious or
  linear, and hand-tuning a rule for it is itself a search problem.
- The agent gets to **refine its choice over multiple attempts within one
  frame** (up to 5 steps) — first try a parameter set, see the resulting
  inlier ratio and residual, and adjust, rather than committing to one
  guess blind. This is a sequential decision problem, which is what RL is
  built for.
- There's no ground-truth label for "the correct ground plane" in this
  data, so this is trained with a **self-supervised reward** derived from
  properties of the fit itself (inlier ratio, residual, normal alignment)
  rather than supervised learning against labels.

---

## The RL Formulation

This is a standard [Gymnasium](https://gymnasium.farama.org/) environment
(`RansacEnv` in [`ransac_env.py`](ransac_env.py)), trained with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)'s **PPO**
(Proximal Policy Optimization) algorithm.

```
┌─────────────────────────────────────────────────────────────────┐
│                         One Episode                              │
│                                                                    │
│   reset()                                                         │
│     → pick one .ply LiDAR frame (adaptive sampling, see below)    │
│     → load + voxel-downsample it                                  │
│     → precompute 21 scene features (expensive, done once)         │
│     → return initial 31-dim observation                           │
│                                                                    │
│   step(action)  [repeated up to 5 times]                          │
│     → decode action → (epsilon, min_support, stop/continue)       │
│     → run schnabel_ransac.detect() with those parameters           │
│     → find the ground plane among the detected shapes             │
│     → update feedback state (inlier_ratio, residual, normal, ...) │
│     → if stop==True or step 5 reached: compute terminal reward     │
│       else: reward = 0 (no signal until the episode ends)          │
│     → return (new observation, reward, terminated, info)          │
└─────────────────────────────────────────────────────────────────┘
```

One **episode** = one LiDAR frame, attempted for up to 5 **steps**. Each
step is one full call into the Schnabel RANSAC engine with a chosen
`(epsilon, min_support)` pair (plus a fixed `normal_thresh`), plus a binary
decision of whether to accept this result (`stop`) or try again with
different parameters (`continue`).

---

## The Observation Space — Every Term Explained

`observation_space = Box(shape=(31,))` — 31 floating-point numbers, split
into two groups:

### Group 1 — Scene Features (21 dims, from `features/scene_features.py`)

Computed **once per episode** (cached in `self.current_features`), since
recomputing them every step would be wasteful — they describe the raw
point cloud, which doesn't change between the agent's attempts on the
same frame.

| # | Name | What it measures |
|---|---|---|
| 0–2 | `bbox_dx, bbox_dy, bbox_dz` | Size of the point cloud's bounding box along each axis (metres) — is this a small room or a huge outdoor scene? |
| 3 | `bbox_volume` | `dx × dy × dz` — total bounding-box volume. |
| 4 | `point_density` | `N / bbox_volume` — how densely packed the points are. Sparse outdoor scans vs. dense indoor scans need very different `epsilon`/`min_support`. |
| 5 | `z_mean` | Mean height of all points. |
| 6 | `z_std` | Standard deviation of height — high in bumpy/uneven terrain, low on a flat floor. |
| 7 | `z_min` | Minimum Z value. **Note the coordinate convention:** TartanAir uses NED (North-East-Down), where Z increases *downward*. So `z_min` is actually the highest physical point in the scene (a ceiling, treetops), not the lowest. |
| 8 | `z_max` | Maximum Z value — the lowest physical point (the floor), since Z-down means "down" = larger Z. |
| 9 | `scan_range_mean` | Mean Euclidean distance of points from the sensor origin — how far the sensor can see in this scene on average. |
| 10 | `scan_range_std` | Spread of that distance — uniform open space vs. cluttered near/far mix. |
| 11–13 | `eig_0, eig_1, eig_2` | The three eigenvalues of the full point cloud's covariance matrix (PCA), sorted descending. Describes the overall "shape" of the point cloud's spatial distribution — e.g. a long corridor has one dominant eigenvalue; an open room has three similar ones. |
| 14–16 | `normal_x_std, normal_y_std, normal_z_std` | Standard deviation of estimated surface normals along each axis — a rough proxy for how "chaotic" vs. "flat/regular" the scene's surfaces are. |
| 17 | `normal_consistency` | Average `\|dot(normal_i, normal_of_nearest_neighbor_i)\|` over 1000 sampled points. Close to 1.0 means neighbouring points have well-aligned normals (smooth, clean surfaces); lower means noisy or highly-curved geometry. Still part of the observation vector, but **as of Day 12 no longer reused inside the reward function** — it's a static per-episode descriptor that can't respond to the agent's actions, which turned out to make it a poor terminal-bonus term (see [Reward Function](#the-reward-function--every-term-explained) and the Day 12 entry in [Known Issues](#known-issues--project-history-worth-knowing)). |
| 18 | `z_density_ground` | Fraction of all points within the bottom 0.5m of the scene (in Z-down, that means `z > z_max - 0.5`) — roughly, "what fraction of this scan is probably floor." |
| 19 | `ground_slope_estimate` | PCA is run on just the lowest 10% of points (by height) to estimate the *local* ground plane's tilt, reported as an angle in radians from perfectly horizontal. This is a much more targeted "is the floor flat or ramped" signal than the whole-cloud eigenvalues above. |
| 20 | `mean_knn_dist` | Mean distance to the 4 nearest neighbours (excluding self), averaged over the same 1000 sampled points used for `normal_consistency` — another density-like signal, but local rather than global like `point_density`. |

### Group 2 — Feedback Features (10 dims, computed fresh in `ransac_env.py::_get_obs()`)

These change **every step** — they tell the agent what happened on its
*previous* attempt at this same frame, so it can adjust:

| # | Name | What it is |
|---|---|---|
| 21 | `inlier_ratio` | Fraction of the frame's points currently assigned to the detected ground plane (0 if no plane was found yet this episode). |
| 22 | `mean_residual` | Mean absolute distance of the ground plane's inlier points to the fitted plane — how "noisy"/thick the detected surface is. Lower is a tighter, cleaner fit. |
| 23–25 | `plane_normal[0..2]` | The `(x, y, z)` components of the detected ground plane's normal vector (from a fresh PCA over its inliers). |
| 26 | `step_count` | How many steps have been taken so far in this episode (1 to 5) — lets the agent know "I'm running out of attempts." |
| 27 | `prev_inlier_ratio` | The inlier ratio from *before* the current step ran (i.e., the previous step's result) — lets the agent compare "did my last parameter change help or hurt." |
| 28 | `prev_epsilon` | The `epsilon` value used on the previous step. |
| 29 | `prev_min_support` | The `min_support` value used on the previous step. |
| 30 | `prev_normal_thresh` | The `normal_thresh` value used on the previous step (currently always `0.90` in Phase 1 — see [Action Space](#the-action-space--every-term-explained)). |

The full observation is `np.concatenate([scene_features, feedback_features])` — see `_get_obs()` in `ransac_env.py`.

**Why normalize?** `train_rl.py` wraps the environment in
`VecNormalize(norm_obs=True, norm_reward=True)`, which tracks a running
mean/variance for each of the 31 dimensions and rescales them to roughly
unit variance before the policy network sees them. This matters a lot
here specifically because the 31 raw dimensions have wildly different
natural scales — e.g. `bbox_volume` can be in the thousands while
`normal_consistency` is bounded in `[0, 1]` — and neural nets trained on
unnormalized, mismatched-scale inputs tend to effectively ignore the
small-scale features. **Any script that loads a trained model must
re-apply the exact same normalization** the model was trained under (see
`rl_evaluator.py`'s `load_obs_normalizer()`) — feeding a trained policy
raw, unnormalized observations produces meaningless actions, since the
policy has only ever seen normalized inputs.

---

## The Action Space — Every Term Explained

```python
action_space = MultiDiscrete([8, 6, 2])
```

A `MultiDiscrete` action is 3 independent discrete choices per step, each
picked from its own small menu:

| Sub-action | Levels | Meaning |
|---|---|---|
| `action[0]` — epsilon level | 8 choices | Index into `EPS_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]` (metres). This is the RANSAC distance threshold — how far a point can be from the candidate plane and still count as ground. |
| `action[1]` — min_support level | 6 choices | Index into `MIN_SUPPORT_LEVELS = [50, 100, 200, 300, 500, 800]`. Minimum number of inlier points required for a plane to be accepted as real. |
| `action[2]` — stop/continue | 2 choices | `0` = **stop** (accept this attempt's result and end the episode), `1` = **continue** (this result becomes the "previous" state, and the agent gets to try again with a new epsilon/min_support next step, up to 5 total steps). |

**`normal_thresh` is fixed at `0.90` for every step in Phase 1** — it's
*not* an action the agent controls yet. `RansacEnv.__init__` accepts a
`fixed_normal_thresh` override (used only by `baseline_evaluator.py` to
match `BASELINE_CONFIG.md`'s Standard/Loose modes, which specify 0.85/0.80)
but during actual RL training and evaluation it's always exactly 0.90.
This was a deliberate scoping decision — see
[EFFICIENT_RANSAC_BREAKDOWN.md's "Adaptive RANSAC Opportunities" section](EFFICIENT_RANSAC_BREAKDOWN.md#6-calcnormalsradius-knn--upstream-of-everything)
for the reasoning: `normal_thresh` is cheap to vary per-step, but `kNN`
(surface normal estimation neighbourhood size) is expensive to vary
per-step, so both were deferred to keep Phase 1's action space simpler and
faster to train.

Discrete levels (rather than a continuous action space) were chosen
because Schnabel's algorithm has highly non-linear sensitivity to these
parameters near certain thresholds, and a small fixed menu of
"meaningfully different" values is easier for PPO to explore reliably than
a continuous range where most nearby values behave almost identically.

---

## The Reward Function — Every Term Explained

### Original formula (Day 1–9, superseded)

Reward was **only computed when an episode terminated** (either the agent
chose `stop`, or it hit the 5-step cap) — every non-terminal step returned
`reward = 0.0`:

```python
reward = (1.0 * inlier_ratio) \
       - (0.1 * runtime) \
       - (0.5 * mean_residual) \
       + (0.3 * normal_consistency) \
       - (0.05 * step_count)
```

| Term | Sign | What it rewarded / penalized |
|---|---|---|
| `1.0 * inlier_ratio` | **+** | The main objective: the fraction of the frame's points successfully classified as ground. |
| `0.1 * runtime` | **−** | Wall-clock seconds the RANSAC call took. |
| `0.5 * mean_residual` | **−** | Mean absolute distance of inlier points to the fitted plane (metres). |
| `0.3 * normal_consistency` | **+** | Scene feature (index 17) — rewards operating in scenes where surface normals are already well-behaved. |
| `0.05 * step_count` | **−** | Penalized using more of the 5 available refinement steps. |

Failure cases (too few points for `min_support`) got a flat `reward = -1.0`
sentinel instead of the formula above.

**Why this was replaced (Day 10):** with `reward = 0.0` on every
non-terminal step, the agent had zero signal about whether an intermediate
refinement step actually helped — the only thing steering "continue or
stop" was the flat `-0.05 * step_count` penalty, which dominated given how
small this task's inlier ratios typically are (0.03–0.3). The measurable
symptom: `mean_steps` stayed pinned at 1.0 across every dataset — the
agent never used the multi-step refinement the environment was built
around (see the Day 10 entry in
[Known Issues](#known-issues--project-history-worth-knowing) for the full
diagnosis). The fix kept the exact same terms (nothing above was judged
"wrong" — inlier ratio, residual, runtime, and normal consistency are all
still in the formula) but changed *when* they're paid out, per below.

### Day 10–11 formula (potential-based shaping, superseded Day 12)

Every step returned a **non-zero reward**, from `ransac_env.py`'s
`_potential()` and `step()`, as of Day 10:

```python
def _potential(inlier_ratio, mean_residual):
    return (1.0 * inlier_ratio) - (0.5 * mean_residual)

# per step:
reward = (_potential(new_state) - _potential(prev_state)) \
       - (0.1 * runtime) \
       - 0.01

# only on the terminal step (stop==True or step_count reaches 5), add:
reward += 0.3 * normal_consistency
```

| Term | Sign | What it rewarded / penalized |
|---|---|---|
| `Φ(new) - Φ(prev)` where `Φ = 1.0*inlier_ratio - 0.5*mean_residual` | **±** | The core signal: how much this step's attempt actually improved (or worsened) the fit vs. the previous attempt (or vs. the zeroed reset state, on step 1). Rewards genuine refinement instead of paying out only at the end. |
| `0.1 * runtime` | **−** | Wall-clock seconds the RANSAC call took, charged every step. **Removed Day 11** — evidence from the `v2_reward` pilot showed `mean_steps` still drifting down toward 1.0 despite the Day 10 shaping; this term was taxing every step equally rather than specifically discouraging unhelpful continuation, competing with rather than reinforcing the potential-based signal. `runtime` is still computed and logged to CSV, just no longer part of the reward. |
| `0.01` | **−** | A flat per-step tie-breaker (replaces the old `0.05 * step_count` penalty). Small because real signal (the potential delta) now does most of the work of discouraging pointless extra steps. |
| `0.3 * normal_consistency` | **+** | The scene feature described above (index 17), added **only once, on the terminal step**. **Removed Day 12** — see [Current formula (Day 12+)](#current-formula-day-12-z_align-replaces-normal_consistency) below. |

Because `Φ(new) - Φ(prev)` telescopes across an episode's steps, the total
return for an episode still sums to (within the runtime/step-count
constant swap) the same quantity the old terminal-only formula computed —
this was a reward-shaping change that gave credit incrementally, not a
different objective. That telescoping property is exactly what the Day 12
fix below turned out to hinge on.

### Current formula (Day 12+, z_align replaces normal_consistency)

```python
def _potential(inlier_ratio, mean_residual, z_align):
    return (1.0 * inlier_ratio) - (0.5 * mean_residual) + (0.15 * z_align)

# per step, including the terminal step -- no separate terminal-only term:
reward = (_potential(new_state) - _potential(prev_state)) - 0.01
```

| Term | Sign | What it rewards / penalizes |
|---|---|---|
| `Φ(new) - Φ(prev)` where `Φ = inlier_ratio - 0.5*mean_residual + 0.15*z_align` | **±** | Same as Day 10/11, plus `z_align` — `find_ground_plane()`'s `\|normal_z\|` for the plane actually detected *this step*, recomputed fresh every step (previously computed but discarded). Telescopes across the episode exactly like the other two terms. |
| `0.01` | **−** | Flat per-step tie-breaker, unchanged from Day 10/11. |

**Why `normal_consistency` was replaced rather than reweighted or
threshold-gated (Day 12):** it's computed once per episode in `reset()`,
before any RANSAC call runs — it can't respond to `eps`/`min_support`/
`norm_th`, or to which plane actually got detected. A flat bonus added
only at termination gave the agent a near-constant reward for stopping,
regardless of the quality of what it found — not just "usually close to
1," but *identical* for every step of a given frame. Confirmed live: the
`v3_normthresh_heldout` run's training log (`logs/evaluation_metrics_v3_normthresh_heldout.csv`)
showed `steps_used=1` for the overwhelming majority of recent episodes —
the same collapse Day 10/11 had already fought once, resurfacing under the
Day 11 4-action expansion. Structurally, an untelescoped flat term added
only at one transition isn't valid potential-based shaping (it can't
cancel out of the sum the way `Φ(new) - Φ(prev)` does across every other
step), so it directly biased the "stop vs. continue" decision rather than
just adding noise.

`z_align` was chosen as the replacement because it's the one already-computed,
per-step, action-sensitive quantity that measures the same underlying
thing (plane-normal alignment) `normal_consistency` was meant to proxy for
— it's just measured on the *actual detection result*, not the raw scene.
Weight set to `0.15`, not the original `0.3`: across all 141,435 logged
steps in `logs/evaluation_metrics*.csv`, `z_align` is close to bimodal
(p25 = 0.03, median = 0.95 — a direct consequence of `find_ground_plane`'s
`horizontal_thresh = 0.80` gate) while `inlier_ratio` — the primary
objective — rarely exceeds ~0.35 (p90 = 0.234, max = 0.447 across the
entire dataset). At `0.3`, a single `horizontal_thresh` gate-crossing could
contribute more to `Φ` than the *entire observed range* of `inlier_ratio`,
letting plane-alignment swings dominate over actual coverage/fit quality.

**Failure cases still short-circuit, but go through the shaped formula
rather than a fixed sentinel, same as Day 10/11:**
- If the current frame has fewer points than the chosen `min_support`,
  `inlier_ratio`/`mean_residual`/`plane_normal`/`z_align` are reset to 0
  and the episode terminates immediately with
  `reward = (Φ(zeroed) - Φ(prev)) - 0.01` (`info["error"] = "too_few_points"`).
- If RANSAC runs but no horizontal plane passes `find_ground_plane()`'s
  filter, or an exception occurs, `inlier_ratio`/`mean_residual`/`plane_normal`/`z_align`
  are all reset to 0 for that step and the same shaped formula runs on
  whatever state resulted.

**Note for anyone comparing evaluation runs across this change:** the
`reward` column `RansacEnv.step()` logs to CSV is computed by whichever
formula is live in `ransac_env.py` at the time the environment runs — it
is *not* frozen to whatever formula a given model was actually trained
under. Evaluating an old (Day 10/11-trained) model after this fix lands
will log `reward` values computed by the Day 12 formula, which is not the
objective that model was optimized for. `inlier_ratio`, `mean_residual`,
and `z_align` are computed directly from the RANSAC result regardless of
reward formula, so those remain valid to compare across model versions —
`reward` does not.

**`find_ground_plane()`** (in `ransac_env.py`) is the logic that picks
*which* detected plane is "the ground," since Schnabel's `detect()` can
return several planes per frame (walls, tabletops, the actual floor):
1. Estimate each candidate plane's normal via PCA over its inlier points.
2. Keep only planes whose normal is "horizontal enough"
   (`|normal_z| >= horizontal_thresh`, default `0.80`) — filters out
   vertical walls.
3. Among the horizontal candidates, in `z_down` mode (TartanAir's NED
   convention), pick the one with the **highest** average Z — since Z
   increases downward, the highest-Z horizontal plane is physically the
   lowest one, i.e. the floor.
4. If nothing passes the horizontality filter, fall back to the single
   largest detected plane by point count.

---

## How the Agent Decides to Stop or Continue

It's worth separating three different questions that are easy to conflate:
what information the agent has available, what actually computes the
stop/continue decision, and how it learns to use that information well.

### What the agent can actually see

At each step, the agent's only window into "how did my last attempt go" is
the 10 feedback features described above:

- `inlier_ratio`, `mean_residual`, `plane_normal` — the result of the
  attempt it *just* ran.
- `prev_inlier_ratio` — the result from the attempt *before that*.
- `prev_epsilon`, `prev_min_support`, `prev_normal_thresh` — what
  parameters produced that previous result.
- `step_count` — how many of its 5 attempts are already used up.

That's the entire basis for the decision. There is no ground-truth
"correct" ground-plane label to check against — only these numbers.

### What actually computes stop vs. continue

**There is no formula or threshold anywhere in the code for this.**
`action[2]` (stop/continue) comes entirely out of the trained neural
network's output given the observation above — `terminated = stop or (step_count >= 5)`
in `ransac_env.py::step()` just executes whatever the network decided.
There's no line that says "if `inlier_ratio > 0.8`, stop." So "how does it
know when to stop" really means "what did the network learn to do with
those numbers," which is a fundamentally different thing than a rule you
could point to in the source.

### The subtlety that makes this a real bet, not a lookup

**The environment does not remember the best attempt across steps.**
`self.inlier_ratio` and `self.mean_residual` are overwritten on every
`step()` call with whatever the *most recent* attempt produced — there is
no "keep the best result seen so far" logic anywhere in `RansacEnv`. Since
Day 10's reward-shaping change, every step (not just the terminal one) is
scored against the *previous* step's state via `Φ(new) - Φ(prev)` (see
[Reward Function](#the-reward-function--every-term-explained)), so a step
that makes things worse is penalized immediately rather than only being
reflected in a single end-of-episode number — but there is still no
mechanism that lets the agent fall back to an earlier, better attempt if a
later one regresses. If the agent continues and its 3rd attempt is worse
than its 1st, and it then stops, the episode's cumulative return reflects
that regression; nothing rescues it. This means "should I continue" is
still a genuine bet: continuing only pays off if the agent believes
changing `epsilon`/`min_support` will push `inlier_ratio` up / `mean_residual`
down enough to beat what it already has, after accounting for:

- the flat `-0.01` per-step tie-breaker,
- the extra `-0.1 * runtime` cost of another RANSAC call,
- and the risk of landing on a *worse* combination than its current one.

### How training shapes the decision

During PPO training, the agent runs thousands of these 5-step episodes,
observes the resulting terminal rewards, and gradient-updates its policy
so that action sequences that led to higher reward become more likely.
Concretely, patterns like *"inlier_ratio is already high and residual is
low → the expected extra reward from continuing doesn't outweigh the
step/runtime cost → output stop"* get reinforced, while *"inlier_ratio is
still low → there's more to gain than to lose → output continue"* gets
reinforced too. This association is stored purely as learned weights — the
same actor-critic mechanism PPO uses for every other part of the action:
the critic estimates "how much value is left to gain from this state," and
the actor is nudged toward whichever action historically led to higher
return from similar states.

**In short:** the *inputs* to the stop/continue decision are the feedback
features (current/previous fit quality + steps used); the *decision
itself* is a learned function of those inputs with no hand-coded rule; and
it's shaped by the fact that there's no safety net for a bad late attempt —
which is exactly why the flat `-0.01` per-step tie-breaker matters: it's
the thing pushing the network away from "always burn all 5 steps just in
case" whenever a step's genuine potential-delta signal is near zero.

This is also precisely why a poorly-trained policy can degenerate into
*never* using the refinement loop at all (see the Day 8 collapse in
[Known Issues](#known-issues--project-history-worth-knowing)) — if the
network hasn't learned a useful association between the feedback features
and future reward, "always stop after step 1" is a locally safe fallback
that avoids the risk of making things worse, even though it forfeits the
whole point of the multi-step design.

---

## One Episode, Step by Step

Concretely, for one call to `env.reset()` then repeated `env.step(action)`:

1. **`reset()`**: Adaptive sampling (see below) picks a folder, then a
   random `.ply` file within it. `load_ply_xyz()` reads raw XYZ and
   voxel-downsamples at `0.05m`. `compute_scene_features()` runs once and
   is cached. All feedback state (`inlier_ratio`, `step_count`, etc.)
   resets to zero. Returns the initial 31-dim observation.

2. **`step(action)`** (repeat up to 5×):
   - Decode `action` → `(epsilon, min_support, stop)`, with `normal_thresh`
     fixed at 0.90.
   - Compute `Φ(prev)` from the feedback state *before* this step runs
     (see [Reward Function](#the-reward-function--every-term-explained)),
     then save that state as "previous" (for the next observation's
     feedback features) before overwriting it.
   - If the frame has fewer points than `min_support`: terminate
     immediately, `inlier_ratio`/`mean_residual`/`plane_normal` reset to 0,
     reward computed from the shaped formula (effectively a large negative
     `Φ(new) - Φ(prev)` if `Φ(prev)` was already positive).
   - Otherwise call `schnabel_ransac.detect(shapes=["plane"], relative_epsilon=False, epsilon=eps, normal_thresh=0.90, min_support=min_supp, probability=0.001, normal_knn=20, max_shapes=20)`.
   - Run `find_ground_plane()` on the results to pick the actual floor
     plane (or `None`).
   - Update `inlier_ratio`, `mean_residual`, `plane_normal` from the
     chosen plane (or reset to 0 if none found).
   - `terminated = stop or (step_count >= 5)`.
   - Compute `reward = (Φ(new) - Φ(prev)) - 0.1*runtime - 0.01` for
     **every** step (terminal or not); if terminated, add
     `0.3 * normal_consistency` and update this frame's **folder's**
     exponential moving average reward (used by adaptive sampling, see
     next section).
   - Log a row to the CSV log file (`logs/<log_name>.csv`) regardless.
   - Return `(new_observation, reward, terminated, truncated=False, info)`.

---

## The Training Loop (PPO)

`train_rl.py` is the entry point:

1. Build one `RansacEnv` instance, wrap it in SB3's `Monitor` (tracks
   episode-level stats like total reward and length), then in
   `DummyVecEnv` (SB3's vectorized-env interface, even for a single env),
   then in `VecNormalize` (running observation/reward normalization,
   `clip_obs=10.0`).
2. Construct a `PPO("MlpPolicy", vec_env, learning_rate=3e-4, ent_coef=0.01, ...)`.
   - **`MlpPolicy`**: a small feedforward neural network (multi-layer
     perceptron) maps the 31-dim observation to action probabilities —
     appropriate here since the observation is a flat feature vector, not
     an image or point cloud directly (the point cloud itself is
     collapsed into the 21 scene features long before the policy ever
     sees it).
   - **`ent_coef=0.01`**: an entropy bonus added to PPO's loss that
     rewards the policy for keeping its action distribution less certain
     (higher entropy = more exploration). **This specific value was added
     as a bug fix, not a default choice** — see [Known Issues](#known-issues--project-history-worth-knowing)
     below; without it, the policy collapsed to outputting the exact same
     action for every input.
3. `CheckpointCallback` saves a model snapshot (plus matching
   `VecNormalize` stats) every 1000 timesteps to `models/`.
4. `model.learn(total_timesteps=...)` runs the actual PPO training loop:
   collect a batch of episodes under the current policy, compute
   advantages, update the policy network by gradient ascent on the PPO
   clipped objective, repeat.
5. On completion (or on `KeyboardInterrupt`), the final model
   (`ppo_ransac_final.zip`) and its matching `VecNormalize` stats
   (`ppo_ransac_final_vecnormalize.pkl`) are saved together — **these two
   files must always be loaded as a pair**; the model's learned behaviour
   only makes sense relative to the observation scaling it was trained
   under.

Resuming training: `--load <model.zip> --vecnormalize <stats.pkl>` loads
both together rather than reinitializing `VecNormalize`'s running
statistics from scratch, which would otherwise silently invalidate the
loaded policy's calibration.

**Versioning runs with `--tag` (added Day 10).** `train_rl.py --tag v2_reward`
bakes the tag into every output filename (`ppo_ransac_v2_reward_final.zip`,
`ppo_ransac_v2_reward_model_<N>_steps.zip`, `evaluation_metrics_v2_reward.csv`,
...) instead of the untagged defaults (`ppo_ransac_final.zip`,
`evaluation_metrics.csv`), so a new run — e.g. after changing the reward
function, as happened on Day 10 — never overwrites a previous run's model
or logs. `rl_evaluator.py --tag v2_reward` matches on the evaluation side,
writing `<env>_rl_v2_reward.csv` instead of `<env>_rl.csv`.
`compare_results.py` and `per_frame_comparison.py` auto-discover every
`<env>_rl*.csv` present in `logs/` (see `discover_rl_modes()` /
`discover_modes()`), so any tagged evaluation run shows up in comparisons
automatically without editing those scripts. Omit `--tag` on either script
to keep using the original untagged filenames.

### How the Policy Network Actually Produces an Action

It's easy to say "the policy decides the action" without explaining what
that means mechanically. Here's the actual forward pass, end to end, for
`MlpPolicy` on this project's `MultiDiscrete([8, 6, 2])` action space:

1. **The network itself.** SB3's default `MlpPolicy` architecture (not
   overridden anywhere in `train_rl.py`) is two hidden layers of 64
   neurons each with `tanh` activations. There are actually two such
   networks trained together:
   - **The actor (policy net)** — produces the action.
   - **The critic (value net)** — estimates "how much total reward can I
     expect from this state onward." It only matters during training (see
     below); it has no effect on what action gets chosen at inference
     time.
2. **The forward pass.** The 31-dim observation (already rescaled by
   `VecNormalize`) flows through: `input(31) → Linear → tanh → Linear → tanh → output layer`.
3. **Splitting into three action heads.** The output layer doesn't
   produce one number — for `MultiDiscrete([8, 6, 2])`, SB3 splits it into
   three independent groups of raw scores ("logits"): 8 for the epsilon
   choice, 6 for min_support, 2 for stop/continue.
4. **Softmax → probabilities.** Each group's logits are passed through a
   softmax, converting raw scores into a probability distribution that
   sums to 1. The stop/continue head becomes something like
   `P(stop)=0.23, P(continue)=0.77` — two competing probabilities, not a
   yes/no rule.
5. **Turning probabilities into one final action.**
   - **During training**, the action is *sampled* from this distribution
     — this randomness is what gives PPO its exploration; even a
     77%-continue state occasionally outputs stop on purpose, so the
     algorithm can learn from both outcomes.
   - **During evaluation** (`model.predict(obs, deterministic=True)`, used
     by `rl_evaluator.py`/`evaluate.py`), it instead takes the **argmax** —
     whichever of stop/continue has the higher probability wins, with no
     randomness.

So "the policy network says continue" concretely means: a sequence of
matrix multiplications and a softmax turned 31 input numbers into a
probability like `{stop: 0.23, continue: 0.77}`, and then either a random
draw from that distribution or the larger of the two was picked.

**How the weights behind that computation get set** — this is the PPO
training loop that step 4 above glossed over:

1. **Rollout.** Run many episodes under the current network, recording
   every `(observation, action, reward)`.
2. **Advantage estimation.** For each action taken, the critic estimates
   whether the outcome was better or worse than it expected for that
   state (the "advantage"). E.g. if the network output `continue` from a
   given observation and the episode's eventual reward was much higher
   than the critic predicted, that's a strongly positive advantage for
   `continue` in states that look like that one.
3. **Policy update.** PPO nudges the actor's weights so actions with
   positive advantage become more probable (their logit rises) and
   actions with negative advantage become less probable — but *clipped*,
   so no single batch of updates swings the policy too aggressively
   (that clip is the "P" in PPO).
4. **Repeat**, thousands of times. Gradually the weights come to encode
   "in observations that look like X (e.g. low `inlier_ratio`, few steps
   used), `continue` has historically had higher advantage → raise
   `continue`'s logit for inputs like X."

Nothing in this process is a symbolic rule like `if inlier_ratio < 0.5`.
It's a smooth function approximator whose weights were nudged, one
gradient step at a time, toward whatever observation→action mapping
empirically correlated with higher reward during training. This is also
exactly why the Day 8 collapse happened (see
[Known Issues](#known-issues--project-history-worth-knowing)): with
`ent_coef=0` and unnormalized observations, this same gradient process
converged to a degenerate mapping — the same output regardless of input —
instead of a genuinely input-sensitive one.

---

## Adaptive Sampling Across Environments

`RansacEnv` doesn't train on one dataset — it recursively discovers every
`.ply` file under `data_dir` (default: the whole `data/` folder, i.e. every
downloaded TartanAir environment at once), grouped by parent folder name
(`Office`, `Hospital`, `Sewerage`, etc.).

On every `reset()`, instead of picking a file uniformly at random, it uses
a **softmax over inverted per-folder reward** to bias sampling toward
folders the agent is currently doing *worst* on:

```python
inverted = -folder_rewards            # worse reward → larger inverted value
probs = softmax(inverted)             # convert to a probability distribution
chosen_folder = np.random.choice(folder_names, p=probs)
```

`folder_rewards[folder]` is an **exponential moving average (EMA)**
updated every terminal step: `folder_rewards[f] = 0.9 * folder_rewards[f] + 0.1 * reward`.
This means the agent effectively practices more on whichever terrain type
it currently struggles with most (e.g. if it's bad at `Sewerage`, it gets
shown `Sewerage` frames more often until its average reward there catches
up) — a form of automatic curriculum / hard-negative mining, rather than
uniform random sampling across all frames from all datasets.

---

## Evaluation and Comparison Against Baselines

Three **fixed-parameter baselines** (no learning, no per-step
refinement — a single attempt per frame with `stop=0` always) establish a
performance floor, defined in `BASELINE_CONFIG.md` and implemented in
`baseline_evaluator.py`:

| Mode | epsilon | min_support | normal_thresh | Intent |
|---|---|---|---|---|
| Strict | 0.10m | 800 | 0.90 | Conservative — good on smooth indoor floors, fails on rough outdoor terrain. |
| Standard | 0.15m | 500 | 0.85 | The typical "textbook" fixed setting — main comparison point. |
| Loose | 0.25m | 200 | 0.80 | Aggressive — better outdoors, more false positives indoors. |

`rl_evaluator.py` runs the trained agent deterministically (no
exploration noise) over every frame of every downloaded dataset and logs
identical CSV columns, so results are directly comparable.

`compare_results.py` and `per_frame_comparison.py` then join these CSVs:
- `compare_results.py` — aggregate stats per dataset/mode (mean inlier
  ratio, mean residual, bad-frame count, total runtime).
- `per_frame_comparison.py` — a stricter, frame-by-frame join across all
  four modes on shared `frame_id`, reporting genuine per-frame win rates
  rather than only comparing averages (which can hide a method that wins
  narrowly most of the time but loses big occasionally, or vice versa).

Two more scripts compare against a **completely independent, non-RANSAC**
ground/obstacle classifier (`traversability.py`, a standalone grid-based
method) as a sanity check that Schnabel-based ground detection agrees with
a differently-built approach:
- `compare_on_complete_cloud.py` — on pre-merged whole-scene reconstructions.
- `compare_on_trajectory_map.py` — on trajectory maps built by
  `accumulate_trajectory.py` from the same per-frame data the RL agent
  trains on, stitched together with pose transforms + ICP refinement.

---

## File-by-File Guide

### Core RL pipeline

| File | Purpose |
|---|---|
| [`ransac_env.py`](ransac_env.py) | The Gymnasium environment. Defines observation/action spaces, `reset()`/`step()`, the reward formula, `find_ground_plane()`, adaptive folder sampling, and CSV logging. This is the heart of the whole project. |
| [`features/scene_features.py`](features/scene_features.py) | `compute_scene_features(points)` — computes the 21 scene-description numbers documented above, from bounding box/density/height/PCA/normal statistics via numpy and Open3D. |
| [`train_rl.py`](train_rl.py) | Trains a PPO agent on `RansacEnv`. Handles `VecNormalize` wrapping, checkpointing, and resuming from a saved model. |
| [`rl_evaluator.py`](rl_evaluator.py) | Runs a trained model deterministically across every frame of every dataset, logging results to per-dataset CSVs (`<env>_rl.csv`). Includes `load_obs_normalizer()`, which replicates SB3's own observation normalization formula from a saved `VecNormalize` `.pkl` so the model sees correctly-scaled inputs outside of a live `VecNormalize`-wrapped env. |
| [`baseline_evaluator.py`](baseline_evaluator.py) | Runs the three fixed-parameter baselines (Strict/Standard/Loose) across every dataset, one attempt per frame, using the same `RansacEnv.step()` code path as RL evaluation for a fair comparison. |
| [`evaluate.py`](evaluate.py) | A lighter-weight, single-dataset "smoke test" evaluator — loads a model, runs it on a handful of random frames from one environment, and prints per-frame results to the console (no CSV logging). Useful for a quick sanity check without running the full multi-dataset sweep. |

### Data acquisition

| File | Purpose |
|---|---|
| [`download_lidar_frames.py`](download_lidar_frames.py) | Downloads the 9 main TartanAir environments' per-frame LiDAR data (via the `tartanair` package) into `data/<Env>/Data_omni/P0000/lidar/`. Defines `ENVIRONMENTS` (the canonical dataset list used by most evaluation scripts) and `count_frames()`. |
| [`download_tartan_ground.py`](download_tartan_ground.py) | An earlier/alternate downloader for TartanGround data (predates the main TartanAir per-frame pipeline). |
| [`load_tartan_ground.py`](load_tartan_ground.py) | Inspects downloaded TartanGround point clouds — an early exploratory script from before `ransac_env.py` existed. |

### Comparison / analysis / visualization

| File | Purpose |
|---|---|
| [`compare_results.py`](compare_results.py) | Aggregates Strict/Standard/Loose/RL CSVs per dataset into one summary table (mean inlier ratio, bad-frame count, mean residual, mean steps, total runtime). |
| [`per_frame_comparison.py`](per_frame_comparison.py) | Frame-by-frame join of all four modes; reports per-frame win rates, not just averages. Auto-discovers any tagged RL evaluation runs (`<env>_rl_<tag>.csv`) alongside the untagged `<env>_rl.csv`, so multiple RL versions can be compared side by side. |
| [`plot_comparison.py`](plot_comparison.py) | Reads `compare_results.py`'s/`per_frame_comparison.py`'s output CSVs (`logs/full_comparison_summary.csv`, `logs/ALL_per_frame_comparison.csv`) and renders static PNG charts (mean inlier ratio, bad-frame rate, per-frame win rate, overall summary) to `plots/`, for use in reports/slides. |
| [`traversability.py`](traversability.py) | A standalone (RL-independent) grid-based ground/obstacle classifier: splits points into horizontal-ish vs. vertical-ish by normal direction, grids them into cells, fits a local plane per cell (either a lightweight custom RANSAC or, optionally, the real Schnabel engine at cell scale), and flags cells as traversable/obstacle/unknown based on slope, roughness, and step height vs. neighboring cells. Used purely as an independent point of comparison against the Schnabel-based approach — never touches the RL side. |
| [`accumulate_trajectory.py`](accumulate_trajectory.py) | Merges a trajectory's per-frame LiDAR scans into one world-frame point cloud, using each frame's pose file to transform points into a shared frame, then ICP (point-to-plane, seeded by the pose transform) to correct residual misalignment before merging. |
| [`compare_on_complete_cloud.py`](compare_on_complete_cloud.py) | Runs Schnabel and `traversability.py` on the pre-built "complete" multi-view reconstructions in `schnabel_cython/tartanair_data/` and compares them directly. |
| [`compare_on_trajectory_map.py`](compare_on_trajectory_map.py) | Same comparison, but on a trajectory map built by `accumulate_trajectory.py` from per-frame data instead of the pre-made reconstructions. |
| [`visualize_inference.py`](visualize_inference.py) | Loads a trained model + its `VecNormalize` stats, runs it on one specific frame, and opens an Open3D window showing the resulting ground/obstacle segmentation. |
| [`check_frame.py`](check_frame.py) | Opens an Open3D window comparing a raw frame side-by-side with its Schnabel-segmented output (from `batch_segment_lidar.py`'s pre-computed results) — green=ground, grey=obstacles. |
| [`visualize_segmentation.py`](visualize_segmentation.py) | Visualizes the separate `segment_ground.py`/pyRANSAC-3D output (`ground.ply`/`obstacles.ply`) — green=ground, red=obstacles. Independent of the Schnabel/RL path. |
| [`plot_logs.py`](plot_logs.py) | Reads TensorBoard event files from a training run and plots mean episode reward over timesteps as a PNG. |
| [`batch_segment_lidar.py`](batch_segment_lidar.py) | Bulk-runs Schnabel RANSAC (fixed parameters, not RL) over an entire trajectory's frames, saving colour-coded segmented PLYs and a summary CSV — used to generate the data `check_frame.py` visualizes. |
| [`segment_ground.py`](segment_ground.py) | An early, simpler ground segmentation script using `pyransac3d` (pure Python/NumPy RANSAC) instead of the Schnabel C++ engine — predates `schnabel_cython`'s integration into this pipeline. |

### Tests

| File | Purpose |
|---|---|
| [`test_env.py`](test_env.py) | Smoke test for `RansacEnv`: initializes it, takes a few random actions, prints observations/rewards/info to confirm nothing crashes. |
| [`test_features.py`](test_features.py) | Smoke test for `compute_scene_features()`: loads one real frame, computes its 21 features, prints each by name, checks for NaN/Inf. |

---

## Data Layout

```
data/
└── <Environment>/                       # e.g. Office, Hospital, Sewerage, ...
    └── Data_omni/P0000/
        ├── lidar/                       # per-frame .ply point clouds (RL training/eval input)
        │   ├── 000000_lcam_front_lidar.ply
        │   └── ...
        ├── pose_lcam_front.txt          # per-frame pose (x,y,z,qx,qy,qz,qw), NED — used by accumulate_trajectory.py
        └── schnabel_segmented/          # output of batch_segment_lidar.py (colour-coded PLYs + summary.csv)

schnabel_cython/tartanair_data/
└── <Environment>/
    ├── <Env>_rgb_original.ply           # pre-built, pre-merged "complete" scene reconstruction
    ├── <Env>_rgb_segmented.ply          # an earlier Schnabel run's output on that reconstruction (NOT ground truth)
    └── labels/                          # semantic label files bundled with the dataset

logs/
├── evaluation_metrics.csv               # default RansacEnv log during ad-hoc/training runs
├── evaluation_metrics_<tag>.csv         # train_rl.py --tag <tag> training log (e.g. _v2_reward)
├── <Env>_strict.csv / _standard.csv / _loose.csv   # baseline_evaluator.py output
├── <Env>_rl.csv                         # rl_evaluator.py output (untagged)
├── <Env>_rl_<tag>.csv                   # rl_evaluator.py --tag <tag> output, compared side by side
│                                         # with untagged/other-tagged runs by compare_results.py /
│                                         # per_frame_comparison.py
└── eval_run_logs/                       # console output captured from long evaluation runs

models/
├── ppo_ransac_final.zip                 # current trained policy
├── ppo_ransac_final_vecnormalize.pkl    # matching observation/reward normalization stats
├── ppo_ransac_model_<N>_steps.zip       # periodic training checkpoints
├── ppo_ransac_<tag>_final.zip           # train_rl.py --tag <tag> run, kept separate from the above
│                                         # (e.g. ppo_ransac_v2_reward_final.zip, Day 10's reward-shaping retrain)
└── pre_entropy_fix/                     # archived model from before the entropy/VecNormalize fix (Day 8)
```

**Coordinate convention gotcha, worth internalizing early:** the per-frame
sensor data (`data/<Env>/.../lidar/*.ply`) uses TartanAir's **NED**
convention — Z increases *downward*, so the ground is the plane with the
**highest** Z among horizontal candidates. But other data sources in this
repo are **not** in that convention — the pre-built "complete"
reconstructions in `schnabel_cython/tartanair_data/` are Z-up (ground =
lowest Z), and trajectory maps built by `accumulate_trajectory.py` from
pose-transformed per-frame data also empirically come out Z-up. Each
comparison script hardcodes the correct `z_mode` for its specific data
source — this was verified empirically per source (see comments at the
top of `compare_on_trajectory_map.py`), not assumed, and is worth
re-verifying if you introduce a new data source rather than assuming NED
everywhere.

---

## How to Run Things

```bash
# 1. One-time: download the datasets
python download_lidar_frames.py

# 2. Sanity-check the environment and features work
python test_env.py
python test_features.py

# 3. Train (from scratch)
python train_rl.py --timesteps 50000

# 3b. Resume training from a checkpoint
python train_rl.py --timesteps 50000 \
    --load models/ppo_ransac_model_20000_steps.zip \
    --vecnormalize models/ppo_ransac_model_20000_steps_vecnormalize.pkl

# 3c. Train a separate tagged version (e.g. after a reward-function change)
# without touching the untagged run's files
python train_rl.py --timesteps 50000 --tag v2_reward

# 4. Quick single-dataset smoke test of a trained model
python evaluate.py --env_path data/Office/Data_omni/P0000/lidar --num_tests 5

# 5. Full evaluation across all datasets (RL + all 3 baselines)
python rl_evaluator.py --env all
# or, for a tagged model: python rl_evaluator.py --env all --model models/ppo_ransac_v2_reward_final.zip \
#     --vecnormalize models/ppo_ransac_v2_reward_final_vecnormalize.pkl --tag v2_reward
python baseline_evaluator.py strict
python baseline_evaluator.py standard
python baseline_evaluator.py loose

# 6. Compare results (auto-discovers any tagged RL runs alongside the untagged one)
python compare_results.py
python per_frame_comparison.py --env all

# 7. Visualize one frame's inference
python visualize_inference.py --dataset Office --frame 000032

# 8. Watch training progress
python plot_logs.py
# or: tensorboard --logdir logs/

# 9. Generate static comparison charts (PNG) from the results in step 6
python plot_comparison.py
```

---

## Known Issues / Project History Worth Knowing

These aren't abstract caveats — they're bugs that actually happened during
this project's development (see `DAILY_LOG.md` for the full day-by-day
account) and materially changed how results should be interpreted:

**Policy collapse (fixed, Day 8–9).** An earlier trained model
(`models/pre_entropy_fix/`) picked the *exact same action*
(`epsilon=0.2, min_support=800`) for 100% of ~16,000 evaluated frames
across all 10 environments — it looked like it was "beating baselines" in
aggregate purely because that one static combo happened to be decent, not
because it was actually reading the scene and adapting. Root cause: PPO's
`ent_coef` defaulted to `0.0` (no incentive to keep exploring diverse
actions) and the environment's raw 31-dim observation had no normalization
(so wildly different feature scales like `bbox_volume` vs.
`normal_consistency` likely drowned out the smaller-scale features). Fixed
by adding `ent_coef=0.01` and wrapping training in `VecNormalize`. **If you
ever see a trained policy outputting the same action for every frame, this
is the first thing to check** — it's a real failure mode this exact
architecture has hit before, not a hypothetical.

**Silent exception swallowing (fixed, Day 8).** `find_ground_plane()` had
two early-return paths that returned 3 values instead of the 4 the caller
unpacked — a `ValueError` on every call to those paths, caught by a
blanket `except Exception` in `RansacEnv.step()`. This meant "no ground
plane found" and "the code actually crashed" were indistinguishable in
every log produced before the fix. Worth remembering if old CSVs ever look
suspicious: check whether they predate this fix.

**`normal_thresh` silently ignored per-mode (fixed, Day 9).**
`baseline_evaluator.py`'s Standard/Loose modes specify `normal_thresh` of
0.85/0.80 respectively in `BASELINE_CONFIG.md`, but `RansacEnv.step()`
originally hardcoded `0.90` regardless of what the caller intended. Fixed
by adding the `fixed_normal_thresh` constructor override (defaults to
`None`, which preserves exact RL training/eval behaviour — only
`baseline_evaluator.py` passes a non-`None` value).

**Observation/action space is not yet fully exploited.** As of the last
recorded evaluation (Day 9), the trained agent still leans heavily on
`min_support=800` in weaker environments (e.g. 81% of frames in
Supermarket) rather than exploring the full range adaptively everywhere,
and does not make heavy use of the multi-step refinement budget. The
per-frame win-rate numbers in `DAILY_LOG.md` Day 9 are the most trustworthy
account of current agent quality — treat any earlier (pre-Day-9) reported
numbers as reflecting the collapsed policy, not genuine adaptive
behaviour.

**Reward redesigned to fix `mean_steps` pinned at 1.0 (Day 10, in progress).**
The Day 9 agent never used its 5-step refinement budget — every episode
terminated after exactly one step. Root cause: the old reward gave `0.0`
on every non-terminal step, so the only thing steering "continue or stop"
was a flat `-0.05 * step_count` penalty with no compensating signal for
genuine improvement, which dominated given how small this task's inlier
ratios typically are (0.03–0.3). Fixed by switching to **potential-based
reward shaping** (see [Reward Function](#the-reward-function--every-term-explained)) —
every step now gets `Φ(new) - Φ(prev)` credit for whether it actually
helped. This is a **new, separately-tagged training run** (`--tag
v2_reward`), trained from scratch rather than resumed from the Day 9
model, since continuing from a policy trained under the old incentive
structure risks carrying over its "always stop at step 1" bias. A
`v2_reward`-tagged model has completed an initial run (see
`models/ppo_ransac_v2_reward_final.zip`,
`logs/evaluation_metrics_v2_reward.csv`), but per-frame comparison against
the Day 9 baseline (via `rl_evaluator.py --tag v2_reward` +
`per_frame_comparison.py`) has not yet been written up here — treat
`mean_steps`/win-rate claims elsewhere in this doc as still describing the
Day 9 (untagged) model until that comparison lands. The Day 9 model and
its data (`ppo_ransac_final.zip`, `<env>_rl.csv`) remain untouched
throughout.

**Terminal-only `normal_consistency` bonus biased termination regardless
of plane quality (fixed, Day 12).** While reviewing the reward formula
against the live `v3_normthresh_heldout` run, found a concrete case where a
step's plane quality clearly got *worse* (inlier ratio 0.1416→0.0139,
residual 0.1038→0.0083 — i.e. `Φ` dropped from 0.0897 to 0.00975, a genuine
~-0.09 shaping penalty) yet the logged reward was a large **positive**
+0.197, because the terminal-only `+0.3 * normal_consistency` bonus
(`normal_consistency` sits at ~0.99 for nearly every frame) overwhelmed the
shaping penalty. Root cause: `normal_consistency` (scene feature index 17)
is computed once per episode in `reset()`, before any RANSAC call runs —
it's a static per-frame descriptor, completely independent of
`eps`/`min_support`/`norm_th` or which plane actually got detected. Adding
it as a flat bonus only at the terminal transition isn't valid
potential-based shaping (it doesn't telescope/cancel across the episode
the way `Φ(new) - Φ(prev)` does), so it gave the agent a near-constant
reward for terminating, irrespective of final detection quality. Confirmed
this wasn't theoretical: the tail of
`logs/evaluation_metrics_v3_normthresh_heldout.csv` from the live,
still-running training process (PID at the time, ~17,240/40,000 steps into
a resumed run) showed `steps_used=1` for the overwhelming majority of
recent episodes — the same `mean_steps`-collapse symptom Day 10/11 had
already fought once, resurfacing under the Day 11 4-action expansion.
Fixed by removing the standalone bonus and folding `z_align` (the
actually-detected plane's per-step normal alignment, already computed in
`find_ground_plane()` but previously discarded) directly into `_potential()`
instead, weighted at `0.15` rather than `0.3` after distribution analysis
across 141,435 logged steps showed `z_align` is close to bimodal (median
0.95, p25 0.03, from `find_ground_plane`'s `horizontal_thresh=0.80` gate)
while `inlier_ratio` rarely exceeds ~0.35 — at `0.3` a single gate-crossing
could outweigh the entire observed range of the primary objective. See
[Reward Function](#the-reward-function--every-term-explained) for the full
current formula. **Not retroactively applied** to the `v3_normthresh_heldout`
run already in progress at the time of the fix — that run was left to
finish under the old (biased) reward rather than interrupted, so its final
model should be treated as reflecting the pre-Day-12 bias, and any
follow-up training run intended to benefit from this fix should start from
a fresh/earlier checkpoint rather than resume from
`ppo_ransac_v3_normthresh_heldout_final.zip`.
