# Learning to RANSAC: Deep RL for Adaptive Ground Segmentation in LiDAR Point Clouds

> This document captures the architecture and research plan from a **separate codebase** for a Deep RL-based adaptive RANSAC system. Kept here for comparison with the current Schnabel C++ RANSAC pipeline so the best ideas from both can be combined.

---

## The Core Idea

Standard RANSAC for LiDAR ground segmentation uses **fixed parameters** (distance threshold, iteration count, etc.). This project builds a **Deep Reinforcement Learning agent** that learns to adaptively control those parameters — and more — on a per-scan basis, in real time.

The key difference from prior work: instead of observing a scan and outputting one set of parameters (single-shot prediction), the agent does **sequential decision-making** — it runs RANSAC, looks at the result, then decides whether to stop or refine further (up to 5 iterations per scan).

---

## The Problem With Existing Approaches

Prior work treats RL as a simple parameter predictor:
```
observe scan → output parameters → run RANSAC once → done
```

This project argues that is insufficient. The agent instead runs a loop:
```
observe scan → run RANSAC → evaluate result → refine or stop → repeat (up to 5x)
```

This is the **key novelty claim**: genuine multi-step RL outperforms single-shot parameter prediction.

---

## The 5 Novelty Pillars

| # | What | Why it's novel |
|---|---|---|
| 1 | **Multi-step episodes** | RL genuinely outperforms single-shot parameter prediction |
| 2 | **Adaptive stopping** | Agent decides *when* RANSAC is good enough, not just *what* parameters |
| 3 | **Learned sampling weights** | PointImportance model biases RANSAC toward likely ground points |
| 4 | **Self-supervised rewards** | No ground-truth labels needed — deploys on any LiDAR dataset |
| 5 | **Hierarchical policy** | Terrain classifier (flat/slope/rough) routes to specialized sub-policy |

---

## The Architecture Stack

```
LiDAR scan (.bin)
       │
       ▼
TemporalExtractor  →  24-dim state vector
  • 16 handcrafted features: height stats, point density, eigenvalues,
                              normal consistency, etc.
  • 8 context features:      prev plane normal, prev threshold,
                              current step count, prev inlier ratio, etc.
       │
       ├──► PointNet-lite / DGCNN  →  32-dim learned features
       │         (jointly trained or pre-trained)
       │
       └──► Combined: 48-dim state  (24 handcrafted + 24 from PointNet)
       │
       ▼
TerrainClassifier
  → flat / slope / rough
  (routes to a specialized sub-policy for each terrain type)
       │
       ▼
RL Policy  (PPO or SAC)
  Action space:
    • threshold    (8 discrete levels, e.g. 0.1m → 0.5m)
    • iterations   (7 discrete levels, e.g. 100 → 2000)
    • stop/continue (2 actions — the adaptive stopping decision)
    • ransac_n     (3 levels — minimum points to define a plane)
       │
       ▼
WeightedRANSAC  (custom, replaces Open3D uniform sampling)
  ↑ uses PointImportance model for per-point sampling probability weights
  (ground-likely points are sampled more often)
       │
       ▼
HypothesisScorer  (learned accept/reject for plane candidates)
  (replaces the fixed algebraic inlier-count criterion)
       │
       ▼
Inlier mask  →  Reward signal
  • Supervised:       IoU-based (requires GT labels like SemanticKITTI)
  • Self-supervised:  inlier_ratio − β·runtime − γ·residual + normal_consistency
                      (no labels needed — works on any LiDAR dataset)
```

**Episode structure:** 1 episode = 1 LiDAR scan, up to **5 RANSAC refinement steps**.

---

## The Reward Function (Self-Supervised)

```
reward = inlier_ratio
       − β × runtime
       − γ × mean_residual
       + δ × normal_consistency
```

Where:
- `inlier_ratio`        — fraction of scan points accepted as ground inliers
- `runtime`             — penalises slow solutions
- `mean_residual`       — penalises points that barely pass the threshold (noisy fits)
- `normal_consistency`  — rewards planes whose normals are stable across frames

10 reward variants were implemented for ablation study.

---

## Action Space Detail

| Action dimension | Options | What it controls |
|---|---|---|
| `threshold` | 8 discrete levels | Distance threshold ε (metres) |
| `iterations` | 7 discrete levels | Max RANSAC iterations |
| `stop/continue` | 2 binary | Agent decides to accept current result or keep refining |
| `ransac_n` | 3 levels | Minimum sample size to define a candidate plane |

---

## The 3-Week Sprint Plan

### Week 1 — Environment + Scaffolding (Mostly Done)
- [x] Multi-step RL environment built
- [x] 10 reward variants implemented
- [x] TemporalExtractor with 24-dim state
- [x] HypothesisScorer scaffolded
- [x] WeightedRANSAC with PointImportance scaffolded
- [x] 42 unit tests passing
- [ ] **BLOCKER:** No real data yet — need SemanticKITTI `.bin` + `.label` files

### Week 2 — Training
- [ ] Train PPO agent
- [ ] Train SAC agent
- [ ] Integrate PointNet-lite / DGCNN learned features
- [ ] Train hierarchical terrain classifier + sub-policies

### Week 3 — Evaluation + Paper
- [ ] Full ablations: reward variants, feature types, action spaces
- [ ] Benchmarks vs baselines:
  - Fixed RANSAC (Open3D default)
  - Grid search over parameters
  - Supervised MLP predictor
  - Supervised Random Forest predictor
- [ ] Publication figures
- [ ] Paper draft

---

## Current Blocker

All code is scaffolded and tested but **no real data yet**.

**What is needed:**
1. Download SemanticKITTI `.bin` + `.label` files
2. Place in `data/processed/`
3. Run `src/utils/io.make_splits()` to generate train/val/test split

That is the gate before any RL training can begin.

**SemanticKITTI** provides:
- `.bin` files — raw Velodyne LiDAR point clouds (x, y, z, intensity)
- `.label` files — per-point semantic labels including ground class (label 40)

---

## Comparison With Current Schnabel C++ Pipeline

| Dimension | Current Project (Schnabel) | RL Adaptive RANSAC |
|---|---|---|
| **Algorithm** | Efficient RANSAC (Schnabel 2007 C++) | Standard RANSAC + RL controller |
| **Parameters** | Fixed per run | Learned per scan, per step |
| **Iterations** | Probabilistic stopping (internal) | Agent decides stop/continue (up to 5 steps) |
| **Sampling** | Uniform random | Weighted by PointImportance model |
| **Speed** | Very fast (native C++) | Slower (Python RL + RANSAC loop) |
| **Shapes** | Plane, sphere, cylinder, cone, torus | Plane only (ground) |
| **Labels needed** | None | Optional (self-supervised reward available) |
| **Dataset used** | TartanAir / TartanGround | SemanticKITTI (planned) |
| **Maturity** | Running — 678 frames segmented | Scaffolded — no training yet |

### Ideas Worth Combining

- **Take from Schnabel:** The C++ speed. Use Schnabel's C++ as the underlying RANSAC engine inside the RL loop instead of Open3D's Python RANSAC — much faster per iteration.
- **Take from RL:** The adaptive stopping and per-scan parameter selection. Even a simple version (predict ε and min_support from scan features) would improve Schnabel results on the bad frames (~285 frames with <1% ground coverage).
- **Take from RL:** The self-supervised reward as an automatic quality metric — exactly what is needed to judge whether the 285 "bad" frames are actually bad or just viewing angles with little floor visible.
- **Take from RL:** The PointImportance sampling bias — could be applied on top of Schnabel's octree sampling to focus on likely-ground regions first.
