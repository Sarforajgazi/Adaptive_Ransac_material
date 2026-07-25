# Synthetic RL Experiment — Adaptive RANSAC on Synthetic Data

> **START HERE.** This file is the entry point for this folder. Read it
> before the other docs so you know what's current and what's historical.

This is a **self-contained reinforcement-learning experiment**, separate
from the repo's main pipeline (see the root [`README.md`](../README.md) →
"Two RL Tracks" for how they relate). It trains a PPO agent to choose
Schnabel RANSAC parameters (`epsilon`, `min_support`, `normal_thresh`)
adaptively per scene — but on **procedurally-generated synthetic point
clouds** where the ground-truth plane is known exactly. That known ground
truth is the whole point: it lets us measure *quantitatively* whether the
agent adapts to noise/clutter, which unlabeled real LiDAR can't.

---

## Current status (what's true right now)

- **The current best model is [`../models/synthetic_ppo_v2.zip`](../models/)**
  (with its paired `synthetic_ppo_v2_vecnormalize.pkl`), from the Phase-6
  training run.
- **It is genuinely adaptive.** On the synthetic evaluation it varies its
  parameters with scene noise (`eps` rises ~0.13→0.23, `normal_thresh`
  falls ~0.82→0.68 as noise increases) and **beats every fixed baseline**
  on overall score (RL 1.63 vs. best baseline 1.48), with the best
  inlier-recovery rate of any method. Full numbers: `SESSION_PROGRESS_LOG.md`
  §24.
- **Real-world generalization is a separate, still-mixed-results thread.**
  The synthetic model has been spot-checked against several real datasets
  (RELLIS-3D, WildScenes, DIODE, off-road surveys, TartanGround) — results
  vary and are documented in `SESSION_PROGRESS_LOG.md` §25–36. This is
  ongoing, not a finished claim.

---

## Which doc to read, in what order

| Order | Doc | What it is | Trust level |
|---|---|---|---|
| 1 | **this README** | Orientation + current status | Current |
| 2 | [`SESSION_PROGRESS_LOG.md`](SESSION_PROGRESS_LOG.md) | The authoritative, detailed log — every bug, fix, verification, and result, each backed by a number or code reference | **Authoritative** |
| 3 | [`roadmap_synthetic_to_real.md`](roadmap_synthetic_to_real.md) | Forward-looking 8-phase plan (synthetic → real transfer) | Plan (not all done) |
| — | [`syntheticRL.md`](syntheticRL.md) | **Corrected historical log.** Its original conclusion — that the agent *couldn't* learn adaptivity and "degrades into grid search" — was later **disproven**. It carries a correction banner explaining why. | **Historical — superseded** |

**Key point about `syntheticRL.md`:** it blamed an early training collapse on
RANSAC's C++ PRNG (`srand(time(NULL))`). The real cause was a missing
`ent_coef=0.01` + `VecNormalize` in `train_synthetic.py` — a standard RL bug,
fixed in §1–2 of `SESSION_PROGRESS_LOG.md`. The PRNG quirk is real (and still
unpatched in the C++ backend) but was only ever evaluation *jitter*, not a
learning blocker. Don't cite `syntheticRL.md`'s conclusions as current.

---

## Key files in this folder

| File | Purpose |
|---|---|
| [`synthetic_env.py`](synthetic_env.py) | The Gymnasium environment (33-dim obs, `MultiDiscrete` action, reward, scoring against known ground truth) |
| [`data_generator.py`](data_generator.py) | Generates synthetic scenes: planes + noise models + bumps/craters + cylinders/boxes + intersecting walls, all with a `gt_mask` |
| [`train_synthetic.py`](train_synthetic.py) | Trains PPO (`--tag <name>` versions output files; refuses to overwrite without `--force`) |
| [`evaluate_synthetic.py`](evaluate_synthetic.py) | Runs the trained model vs. fixed baselines across a noise/inlier sweep, writes `logs/<run>_eval.csv` |
| [`plot_adaptivity.py`](plot_adaptivity.py) | Renders the adaptivity plots (`plots/<run>_adaptivity.png`) |
| [`visualize_synthetic_plane.py`](visualize_synthetic_plane.py) | Open3D view of true plane vs. RL-fitted plane for one episode |
| `visualize_*_rl.py`, `eval_real_lidar_frames.py` | Real-dataset spot-checks (the §25–36 generalization thread) |

---

## Quick start

```bash
# from repo root, with .venv activated

# Evaluate the current best model against baselines (writes logs/synthetic_ppo_v2_eval.csv)
python synthetic_rl_experiment/evaluate_synthetic.py --tag v2

# Plot its adaptivity curves
python synthetic_rl_experiment/plot_adaptivity.py --tag v2

# Visualize one episode (true plane vs. RL-fitted plane)
python synthetic_rl_experiment/visualize_synthetic_plane.py --tag v2

# Retrain from scratch (long; ~7h for 50k steps — see SESSION_PROGRESS_LOG.md §18/§24 for the cost model)
python synthetic_rl_experiment/train_synthetic.py --tag v3 --timesteps 50000
```
