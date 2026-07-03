# Adaptive RANSAC RL: Daily Tracking Log

This file tracks our day-to-day progress, detailing what was planned, what was actually done, the challenges we faced, how we fixed them, and the final results. We will update this log continuously as we progress through the 3-week timeline.

---

## Week 1

### Day 1
* **Planned:** Project kickoff, review existing C++ Schnabel Cython wrapper, and ensure TartanAir/TartanGround data is correctly downloaded and loading.
* **Done:** Verified the data pipeline. Confirmed the Python wrapper correctly loads and processes `.ply` XYZ point clouds without issues.
* **What Went Wrong:** No major issues.
* **Fixed:** N/A.
* **Result:** Data ingestion is stable, serving as a solid foundation for the Gym environment.

### Day 2
* **Planned:** Build the custom Gym Environment (`ransac_env.py`). Implement `reset()` and basic `step()` function bridging to Cython.
* **Done:** Created `ransac_env.py` to wrap the Schnabel Cython module for RL.
* **What Went Wrong:** The initial implementation was built as a "1-shot" continuous environment (ending after 1 step), which failed to support the Phase 1 requirement of a 5-step sequential decision process.
* **Fixed:** Refactored the environment heavily. Converted the action space to `MultiDiscrete([8, 6, 2])` (epsilon, min_support, stop/continue) and implemented logic to allow up to 5 steps per episode.
* **Result:** The environment correctly holds state across steps and properly delays episode termination.

### Day 3
* **Planned:** Implement the 28-dim State Builder (Scene & Feedback features).
* **Done:** Implemented `scene_features.py` which calculates geometric properties of the point cloud.
* **What Went Wrong:** 
  1. The dataset `.ply` files lacked an intensity/reflectivity channel, meaning the planned 18 scene features could not be used verbatim. 
  2. The initial prototype completely missed the critical 10 "feedback features" (e.g., previous actions, inlier ratio history).
* **Fixed:** 
  1. Replaced the missing intensity features with 21 robust geometric features (using Open3D for normal variance and PCA) to compensate for the lost data. 
  2. Injected the 10 feedback features directly into `ransac_env.py`'s observation array.
* **Result:** We now have an even richer **31-dimensional state space** that correctly tracks the agent's historical context.

### Day 4
* **Planned:** Implement the Self-Supervised Reward Function.
* **Done:** Added reward calculation logic to `step()` in the Gym environment.
* **What Went Wrong:** The original prototype used a very basic heuristic `(ground_pct * 10.0) + (z_align * 2.0)` applied at every single step, which would have broken sequential learning.
* **Fixed:** Overwrote the logic to strictly follow the Phase 1 formula: `(1.0 * inlier_ratio) - (0.1 * runtime) - (0.5 * mean_residual) + (0.3 * normal_consistency) - (0.05 * step_count)`. Forced the reward to only be calculated on terminal states (`stop` action or max steps reached).
* **Result:** The environment properly penalizes the agent for running too slowly or using too many steps, and rewards tight, accurate ground fits.

### Day 5
* **Planned:** Set up the PPO Agent architecture (`train_rl.py`) for the 3-action space.
* **Done:** Developed `train_rl.py` using Stable Baselines3's `PPO` and `MlpPolicy`.
* **What Went Wrong:** There was a concern that SB3 wouldn't handle the new `MultiDiscrete` action space we introduced on Day 2.
* **Fixed:** Verified that Stable Baselines3 inherently supports `MultiDiscrete` spaces natively without any custom architecture changes.
* **Result:** Wrote a test script (`test_step.py`) verifying that the environment processes actions flawlessly. `train_rl.py` is verified and ready for the first major training session.

---
*End of Week 1 logs. Ready to begin Day 6 (PPO Training).*

## Week 2

### Day 6
* **Planned:** Enhance the environment sampling logic to support multi-folder training and adaptive difficulty (Hard-Negative Mining).
* **Done:** Updated `ransac_env.py` to recursively discover `.ply` files across all dataset folders and implemented an Exponential Moving Average (EMA) to track the agent's performance on each terrain.
* **What Went Wrong:** A uniform random sampling strategy across all files would cause massive datasets to overshadow small ones, preventing the agent from learning diverse terrains. Additionally, temporary `cache` folders from dataset downloads risked breaking the script.
* **Fixed:** 
  1. Filtered out any paths containing `"cache"`.
  2. Implemented an Adaptive Softmax Sampling mechanism in `reset()`. The environment now explicitly groups files by their parent folder, checks their average reward, and heavily biases the next random selection toward the folders with the worst scores.
  3. Added the fixed `0.05m` Voxel Downsampling step using `Open3D` to the `.ply` loader as dictated by the Phase 1 architectural diagram.
* **Result:** The agent now actively hunts down its weaknesses. If it struggles on "Sewerage", it will force itself to practice on "Sewerage" until it masters it. Downsampling has also drastically sped up RANSAC execution time.

### Day 7
* **Planned:** Run the trained PPO agent (50,000-step checkpoint from `PPO_3`) against the fixed-parameter baseline across all TartanAir datasets and produce a head-to-head comparison table.
* **Done:** Fixed `rl_evaluator.py`, which was still pointing at `ppo_ransac_final.zip` — a stale 5,000-step smoke-test model saved before the observation-space refactor, incompatible with the current 31-dim state. Repointed it at `ppo_ransac_model_50000_steps.zip`, ran the full evaluation (`--env all` + `--env Office`, ~15,900 frames across 10 datasets), and patched around a gap in `compare_results.py` to fold Office into the aggregate table.
* **What Went Wrong:** `rl_evaluator.py` and `compare_results.py` both loop over `download_lidar_frames.ENVIRONMENTS`, which excludes `Office` even though the baseline benchmark covers it. The first-ever evaluation attempt (before this session) silently produced an empty `Office_rl.csv` (header only, 0 rows) with no visible error — every frame was throwing an exception under the hood that got swallowed (see Day 8).
* **Fixed:** Ran the evaluator as two passes to cover all 10 datasets; manually merged Office's baseline + RL CSVs into the comparison output since the script itself still has the `ENVIRONMENTS` gap (not yet fixed in code).
* **Result:** RL beat every fixed baseline on aggregate: mean inlier ratio 0.129 vs Strict's 0.079 (best baseline), bad frames down to 2,090 vs 2,994, runtime 2.5x faster than Strict. Looked like a clean Phase 1 win — but see Day 8, this turned out not to mean what it looked like it meant.

### Day 8
* **Planned:** Investigate why `Supermarket` had a much higher bad-frame rate (67%, vs 15% for the best baseline there) despite the RL agent's strong Day 7 aggregate numbers.
* **Done:** Discovered the RL agent picks the *exact same action* (`epsilon=0.2, min_support=800`) for **100% of all 15,915 evaluated frames**, across all 10 environments — Hospital, forests, Sewerage, Downtown, everything. It isn't reading the scene state at all; it's a fully collapsed, constant-output policy. It only "beat" baseline in Day 7 by having auto-discovered one decent static combo, not through adaptive behaviour. Root-caused to two gaps in `train_rl.py`'s PPO setup: `ent_coef` defaulting to `0.0` (no entropy bonus discouraging output collapse) and no `VecNormalize` (the raw 31-dim state mixes wildly different feature scales, e.g. `bbox_volume` vs `normal_consistency`, unnormalized). Also confirmed the agent never uses more than 1 of its available 5 refinement steps — the sequential-refinement behaviour central to the whole project pitch has never actually been exercised.
* **What Went Wrong:**
  1. While building `visualize_inference.py` to inspect individual frames, found a real bug in `ransac_env.py`'s `find_ground_plane()`: two early-return paths returned 3 values instead of the 4 the caller unpacks. This has been silently crashing and getting swallowed by a blanket `except Exception` in `RansacEnv.step()` this whole time — meaning "no ground found" and "code actually crashed" were indistinguishable in every evaluation log to date.
  2. Separately found `baseline_evaluator.py` is currently broken — it calls `env._unnormalize_action()`, a method that no longer exists after the Day 2 `MultiDiscrete` action-space refactor. Baselines cannot currently be regenerated or extended to new datasets without fixing this script first.
* **Fixed:**
  1. Patched `find_ground_plane()`'s two early returns to consistently return 4 values (`None, None, None, None`).
  2. Fixed `visualize_inference.py` to load the correct trained model and accept `--dataset`/`--frame` args for on-demand inspection.
  3. Added `ent_coef=0.01` and wrapped the training env in `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)` in `train_rl.py`, plus a `--vecnormalize` resume flag and matching stat-saving on both the normal-completion and `KeyboardInterrupt` exit paths (needed so eval scripts can later load the same normalization stats the model was trained with).
  4. `baseline_evaluator.py`'s `_unnormalize_action` bug is identified but **not yet fixed**.
* **Result:** `train_rl.py` is ready for a retrain with the entropy/normalization fix, but retraining has **not** been run yet. The Day 7 "RL beats baseline" result is not trustworthy as evidence of adaptive behaviour — it needs to be re-measured after retraining. `baseline_evaluator.py` still needs a fix before baselines can be regenerated or extended.

### Day 9
* **Planned:** Retrain with the `ent_coef`/`VecNormalize` fix, confirm the action collapse is actually gone, fix `baseline_evaluator.py`, and produce a trustworthy RL-vs-baseline comparison.
* **Done:**
  1. Retrained (50,000 steps, completed cleanly, no `KeyboardInterrupt` this time) -> `ppo_ransac_final.zip` + `ppo_ransac_final_vecnormalize.pkl`. Confirmed under deterministic inference that the agent genuinely varies `epsilon` per scene now (Office consistently picks level 6; Hospital and Supermarket pick different, lower levels, even frame-to-frame) — a real fix, not just more training-log noise.
  2. Fixed `baseline_evaluator.py`: replaced the broken `_unnormalize_action()`/continuous-action call with a `BASELINE_MODES` dict + `build_action()` matching the current `MultiDiscrete` action space exactly. Also fixed a second, deeper bug found along the way — `step()` hardcoded `normal_thresh=0.90` for every call regardless of mode, which would have silently run Standard/Loose at the wrong `normal_thresh` (should be 0.85/0.80) even after the action-encoding fix. Added a `fixed_normal_thresh` override to `RansacEnv` (defaults to `None`, so RL training/eval behavior is unchanged) so baseline evaluation can now faithfully reproduce `BASELINE_CONFIG.md`. Confirmed via smoke test the existing historical baseline CSVs are correct — they were generated by an earlier, then-compatible version of the environment before the Day 2 refactor; only the script had gone stale, not the data.
  3. Fixed `rl_evaluator.py` and `visualize_inference.py`: both were feeding the model raw, unnormalized observations — the retrained model only makes sense on observations scaled by the `VecNormalize` running stats it trained with. Added `load_obs_normalizer()` (replicates SB3's exact `(obs - mean) / sqrt(var + eps)` formula from the saved `.pkl`) and applied it before every `model.predict()` call. Also repointed both scripts at the new model.
  4. Ran the full RL evaluation (`rl_evaluator.py --env all` + `--env Office`) against the corrected model. Hit a snag: an earlier attempt died silently mid-run (no traceback — likely something external killed the process), and restarting without clearing the old partial CSVs caused every dataset's `_rl.csv` to end up with each frame duplicated exactly twice. Deduplicated (`drop_duplicates(subset="frame_id", keep="last")`) rather than re-running the full ~85-minute evaluation a third time — verified the duplicate rows were consistent (same action, same `inlier_ratio`/`residual`/`plane_normal`, only trivial runtime-measurement noise), so safe to dedupe rather than rerun.
  5. Built `per_frame_comparison.py` — joins Strict/Standard/Loose/RL on `frame_id` (all four share the same schema) for true frame-by-frame win rates, not just aggregate averages.
* **Result:** A trustworthy comparison, finally. Overall: RL beats every baseline on mean inlier ratio (0.135 vs Strict's 0.079) and is far faster (2,098s total vs Strict's 4,438s). Per-frame: **RL wins the individual frame outright 53.1% of the time** (vs Strict 16.0%, Standard 19.2%, Loose 11.7%), and beats Strict head-to-head on 65.9% of frames, Standard on 68.5%, Loose on 80.6%. Per-dataset it's a genuine mixed picture now (not the old collapsed model's "wins everywhere by luck"): **wins on 8 of 10 datasets**, sometimes hugely (Sewerage 0.235 vs Strict's 0.055, Hospital 0.213 vs 0.067), but **loses narrowly on Downtown and Supermarket**. Supermarket in particular is still a genuine weak point (66.2% bad frames, almost unchanged from the collapsed-model run) — but now for an honest reason: the model actually explores 4 different action combos there, yet keeps reaching for `min_support=800` (81% of the time), which just doesn't suit Supermarket's point distribution well. `baseline_evaluator.py` is fixed but not re-run (existing historical baseline data confirmed still valid, no need to regenerate).
