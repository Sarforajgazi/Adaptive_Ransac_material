# RANSAC Material — Project Overview

A 3D point cloud **ground segmentation pipeline** for robotics and autonomous navigation. The project wraps Ruwen Schnabel's 2007 Efficient RANSAC C++ library in Cython so Python can call it at native C++ speed, applies it to LiDAR data from the TartanAir and TartanGround simulation datasets, and — on top of that — trains a **reinforcement learning agent** to pick the RANSAC parameters adaptively per scene instead of using one fixed setting everywhere.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Run Ground + Wall Detection on Your Own Point Cloud](#run-ground--wall-detection-on-your-own-point-cloud)
- [Documentation Map](#documentation-map)
- [Folder Structure](#folder-structure)
- [Dataset Locations](#dataset-locations)
- [How the C++ → Cython → Python Chain Works](#how-the-c--cython--python-chain-works)
- [The Full Pipeline](#the-full-pipeline)
- [The Adaptive RL Pipeline](#the-adaptive-rl-pipeline)
- [Two RL Tracks (important — don't confuse them)](#two-rl-tracks-important--dont-confuse-them)
- [The Synthetic RL Experiment](#the-synthetic-rl-experiment-synthetic_rl_experiment)
- [Virtual Environment](#virtual-environment)
- [Quick Start](#quick-start)

---

## What This Project Does

Given a 3D LiDAR point cloud (a list of `[X, Y, Z]` coordinates from a sensor), this project:

1. Detects the **ground plane** using RANSAC (Random Sample Consensus)
2. Separates **ground points** from **obstacle points** (walls, objects, vehicles, etc.)
3. Saves and **visualizes** the result in 3D with ground painted green and obstacles in red

Two RANSAC implementations are used and compared:
- **pyransac3d** — pure Python/NumPy (simple, slow)
- **schnabel_ransac** — Schnabel's 2007 C++ algorithm wrapped in Cython (complex, fast)

On top of that base pipeline, a **PPO reinforcement learning agent** (root-level `ransac_env.py`, `train_rl.py`, etc.) learns to choose Schnabel's `epsilon`/`min_support` parameters per-frame — see [The Adaptive RL Pipeline](#the-adaptive-rl-pipeline) below and [RL_PIPELINE_OVERVIEW.md](RL_PIPELINE_OVERVIEW.md) for the full breakdown.

---

## Run Ground + Wall Detection on Your Own Point Cloud

This is the fastest path to trying the project on your own data. `detect_ground_and_walls.py` is the RL-tuned, wall-capable entry point — it picks Schnabel RANSAC parameters per-scene (instead of one fixed setting), then detects the ground plane, any wall (vertical-plane) structures, and — depending on model variant — cylinders, all in one pass.

```bash
python detect_ground_and_walls.py --ply your_cloud.ply --model_variant synthetic
```

Don't have a point cloud handy? Fetch a small sample first (one-time, ~8MB, via Open3D's own asset downloader — not committed to the repo directly):

```bash
python schnabel_cython/download_data.py
python detect_ground_and_walls.py --ply schnabel_cython/office_with_ground.ply --model_variant synthetic
```

Or drop your own file in [`my_point_clouds/`](my_point_clouds/) and point `--ply` at that instead.

which prints a per-shape summary (e.g. `ground=2971 pts (47.6%), 3 wall(s) (1916 pts), 0 cylinder(s)`) and opens an Open3D viewer window with ground in green, each wall in a distinct color, and cylinders (if any) in gold — rotate with the left mouse button, press **P** to save a screenshot, close the window to exit.

**Flags that matter:**

| Flag | What it does |
|---|---|
| `--model_variant synthetic` | Recommended default — trained on procedurally-generated synthetic data, never overfit to one real dataset, and the variant this project's own real-world testing standardized on. (`main`, the script's technical default, was trained on real TartanAir LiDAR specifically.) |
| `--z_mode z_up` or `z_down` | `z_down` (default) is correct for LiDAR. Most other real point clouds — RGB-D, terrain surveys, indoor scans — are `z_up` and need this set explicitly, or ground/height-dependent features get computed backwards. |
| `--interactive` | For the batch input modes below (`--env`, `--s3dis_room`, `--ply_list`), also opens a viewer per frame instead of just logging results. (`--ply` single-file mode always opens a viewer, with or without this flag.) |
| `--csv_log path.csv` | Append a results row (point/ground/wall counts, chosen parameters, timing) instead of or alongside viewing. |
| `--no_cylinders` | Cylinder detection is on by default under `--model_variant main` only — `synthetic` never requests cylinders (a degenerate cylinder hypothesis was found to sometimes win a region before a real plane hypothesis was ever tried there, see the comment at `detect_ground_and_walls.py:127`). |

This is inference only — you get a detection, not an accuracy score, unless you separately compare against your own ground-truth labels. For batches of files instead of one, see `--ply_list path1.ply path2.ply ...` (generic files), `--s3dis_room <dir>` (S3DIS rooms), or `--env <name> --indices <n...>` (indexed TartanAir/TartanGround frames) in `python detect_ground_and_walls.py --help`.

**New here and just want to get something running?** See [`GETTING_STARTED.md`](GETTING_STARTED.md) for a step-by-step setup guide (install, run, troubleshoot, choosing a `--voxel` size) — this section above is the short version.

---

## Documentation Map

Every doc in this project, in one place:

| Doc | What's in it |
|---|---|
| [`README.md`](README.md) (this file) | Project overview, architecture, folder structure, both RL tracks |
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | Practical setup/run guide — install, run detection with `synthetic_ppo_v2.zip`, the interactive viewer, voxel downsampling guidance |
| [`EFFICIENT_RANSAC_BREAKDOWN.md`](EFFICIENT_RANSAC_BREAKDOWN.md) | Deep-dive on the vendored Schnabel C++ algorithm itself — every parameter, every hardcoded constant, the octree/candidate/stopping-criterion internals |
| [`SCHNABEL_CYTHON_BREAKDOWN.md`](SCHNABEL_CYTHON_BREAKDOWN.md) | Deep-dive on the Cython wrapper — the full Python→Cython→C++ call chain, build system, gotchas |
| [`RL_PIPELINE_OVERVIEW.md`](RL_PIPELINE_OVERVIEW.md) | Deep-dive on the main RL track — observation/action/reward spaces, training loop, bugs found and fixed |
| [`DAILY_LOG.md`](DAILY_LOG.md), [`STRATEGY_AND_IMPROVEMENTS.md`](STRATEGY_AND_IMPROVEMENTS.md), [`ADAPTIVE_RL_PLAN.md`](ADAPTIVE_RL_PLAN.md), [`ADAPTIVE_RANSAC_RL_IDEA.md`](ADAPTIVE_RANSAC_RL_IDEA.md) | Process journals from the main track's development — historical, not current-state reference |
| [`BASELINE_CONFIG.md`](BASELINE_CONFIG.md) | The fixed Strict/Standard/Loose parameter baselines used throughout evaluation |
| [`TRAVERSABILITY_COMPARISON.md`](TRAVERSABILITY_COMPARISON.md) | `traversability.py`'s grid-based classifier vs. the Schnabel-based pipeline |
| [`synthetic_rl_experiment/README.md`](synthetic_rl_experiment/README.md) | **Start here** for the synthetic RL track — current status, doc reading order |
| [`synthetic_rl_experiment/SESSION_PROGRESS_LOG.md`](synthetic_rl_experiment/SESSION_PROGRESS_LOG.md) | Authoritative, detailed log for the synthetic track — every bug, fix, and result backed by a number |
| [`schnabel_cython/README.md`](schnabel_cython/README.md) | The Cython wrapper folder's own structure + demo scripts |
| [`features/README.md`](features/README.md) | What `compute_scene_features()` computes and the z-up/z-down gotcha |
| [`models/README.md`](models/README.md) | Both tracks' model version histories, which checkpoint to actually use |
| [`data/README.md`](data/README.md) | Pointer to the Dataset Locations table below |
| [`logs/README.md`](logs/README.md), [`plots/README.md`](plots/README.md), [`screenshots/README.md`](screenshots/README.md), [`scratchpad/README.md`](scratchpad/README.md) | Short notes on each output/working folder |
| [`paper_efficient_RANSAC_Schnabel/README.md`](paper_efficient_RANSAC_Schnabel/README.md) | The two reference PDFs |

---

## Folder Structure

```
Ransac_material/
│
├── Efficient-RANSAC-for-Point-Cloud-Shape-Detection/   # Original C++ library (Schnabel 2007)
├── schnabel_cython/                                    # Cython wrapper (the core engineering)
├── synthetic_rl_experiment/                            # Second RL track: adaptive RANSAC on synthetic data (self-contained)
├── pyRANSAC-3D/                                        # Pure Python RANSAC (alternative)
├── paper_efficient_RANSAC_Schnabel/                    # Reference papers (PDF)
├── data/                                               # Dataset storage (TartanAir LiDAR)
├── .venv/                                              # Single Python virtual environment
│
├── download_tartan_ground.py   # Step 1: Download TartanAir LiDAR data
├── load_tartan_ground.py       # Step 2: Inspect downloaded point clouds
├── segment_ground.py           # Step 3: Run RANSAC ground segmentation
├── visualize_segmentation.py   # Step 4: Open 3D visualization window
│
├── features/                   # Scene-feature extraction used by the RL agent
├── models/                     # Trained PPO models + VecNormalize stats
├── logs/                       # Training/evaluation CSV logs and TensorBoard runs
├── plots/                      # Static PNG comparison charts (plot_comparison.py output)
├── screenshots/                # Open3D screenshots, redirected here by screenshot_utils.py
├── scratchpad/                 # Ad-hoc one-off scripts from past work sessions
│
├── detect_ground_and_walls.py   # Ground + wall + cylinder detection on any point cloud (see Quick Start above)
├── screenshot_utils.py           # Wraps Open3D's viewer so its screenshot key lands in screenshots/, not scattered around
├── ransac_env.py                # Gymnasium environment wrapping schnabel_ransac for RL
├── train_rl.py                  # Trains the PPO agent
├── rl_evaluator.py               # Evaluates a trained agent across all datasets
├── baseline_evaluator.py         # Evaluates fixed-parameter (Strict/Standard/Loose) baselines
├── compare_results.py            # Aggregates RL vs. baseline results
├── per_frame_comparison.py       # Frame-by-frame RL vs. baseline win rates
└── plot_comparison.py            # Renders the above comparisons as PNG charts
```

> The RL pipeline (everything above the `features/`/`models/`/`logs/`/`plots/` line) is documented in full in [RL_PIPELINE_OVERVIEW.md](RL_PIPELINE_OVERVIEW.md) — that file is the primary reference for how the agent's observation/action/reward spaces work and how to train or evaluate it.

> Root also holds a number of one-off scripts from the project's development history — debugging aids (`debug_alignment.py`), report-generation helpers (`add_images_to_docx.py`), superseded download/eval scripts, and similar. They're not needed to use the pipeline; the scripts named above and in the Quick Start section are the ones that matter for actually running something.

---

### `Efficient-RANSAC-for-Point-Cloud-Shape-Detection/`

The **original C++ source code** by Ruwen Schnabel and Roland Wahl (University of Bonn, 2007/2009). This is not a Python package — it is raw C++ that gets compiled into the Cython extension.

**This code is never run directly.** It is compiled into `schnabel_ransac.pyd` and called from Python transparently.

Key components:

| File / Folder | Purpose |
|---|---|
| `RansacShapeDetector.h/.cpp` | Main detector: probabilistic stopping + octree sampling loop |
| `PointCloud.h/.cpp` | Holds 3D points and normals; runs PCA-based normal estimation |
| `Plane/Sphere/Cylinder/Cone/Torus .cpp` | Geometric math for each shape type |
| `*PrimitiveShapeConstructor.cpp` | Factory classes that fit each shape from minimal point samples |
| `GfxTL/` | Template graphics math library: KD-Trees, Octrees, matrices, spatial indexing |
| `MiscLib/` | Utility library: reference-counted pointers, custom vectors, RNG |
| `main.cpp` | Standalone C++ demo — excluded from the Cython build |

**Supported shape types:** Plane (0), Sphere (1), Cylinder (2), Cone (3), Torus (4)

**Reference paper:**
> Ruwen Schnabel, Roland Wahl, Reinhard Klein.
> "Efficient RANSAC for Point-Cloud Shape Detection."
> *Computer Graphics Forum, 26:2 (214–226), June 2007.*

---

### `schnabel_cython/`

The **hand-written Cython wrapper** — the main engineering contribution of this project. It compiles the Schnabel C++ library into a Python-importable binary module.

#### Bridge Layer (C++)

| File | Purpose |
|---|---|
| `bridge.h` | Declares the flat C ABI: `DetectedShape` struct and `detect_shapes()` function |
| `bridge.cpp` | Implements `detect_shapes()`: copies float array → PointCloud, computes normals, runs RANSAC, maps results back to original point indices |

#### Cython Layer

| File | Purpose |
|---|---|
| `schnabel_ransac.pxd` | Cython declaration file — tells Cython about `bridge.h` types |
| `schnabel_ransac.pyx` | The Python-facing wrapper. Exposes `schnabel_ransac.detect(points, ...)`. Converts NumPy arrays to C pointers, releases the GIL, returns list of Python dicts |
| `setup.py` | Build script: compiles `.pyx` + `bridge.cpp` + all Schnabel `.cpp` files into one `.pyd`/`.so` |

#### Compiled Output

| File | Platform |
|---|---|
| `schnabel_ransac.cp311-win_amd64.pyd` | Windows, Python 3.11 (x64) — ready to use |
| `build/lib.win-amd64-cpython-311/schnabel_ransac.cp311-win_amd64.pyd` | Same, in build folder |
| `build/lib.macosx-12.1-arm64-cpython-312/schnabel_ransac.cpython-312-darwin.so` | macOS Apple Silicon, Python 3.12 |

The `.pyd` is a **compiled Windows DLL**. Compilation already happened — `import schnabel_ransac` loads the binary directly with no runtime compilation.

#### Demo and Utility Scripts

| Script | What it does |
|---|---|
| `ground_segmentation.py` | Loads a point cloud via Open3D, calls `schnabel_ransac.detect()`, picks the lowest-elevation plane as ground, visualizes in 3D |
| `tartan_ground_segmentation.py` | Same but specifically for TartanGround `.pcd` files |
| `real_data_demo.py` | Demo on a real Open3D indoor scan |
| `visual_demo.py` | Demo on synthetic data (plane + cylinder + noise) to prove the algorithm works |
| `download_tartan_pcd.py` | Downloads TartanGround environments from Hugging Face |
| `fix_templates.py` | One-time utility that patched C++ template compatibility issues in Schnabel's code |

#### `tartanair_data/`

Pre-downloaded TartanGround point cloud maps (from `theairlabcmu/TartanGround` on Hugging Face). Each environment has:
- `Env_rgb.pcd` — full-scene colored point cloud
- `Env_rgb_original.ply` — raw version
- `Env_rgb_segmented.ply` — after ground segmentation

Environments available: Downtown, Hospital, OldScandinavia, OldTownFall, SeasonalForestAutumn, SeasonalForestSpring, SeasonalForestWinterNight, Sewerage, Supermarket.

---

### `pyRANSAC-3D/`

A **third-party pure Python RANSAC library** used as a simpler alternative to the Schnabel C++ wrapper. No compilation required — runs entirely on NumPy.

```
pyransac3d/
  plane.py      ← used by root-level segment_ground.py
  sphere.py
  cylinder.py
  cone.py
  line.py
  circle.py
  cuboid.py
  point.py
  aux_functions.py
tests/          ← unit tests with sample .ply datasets
```

Already installed in `.venv` as `pyransac3d-0.6.0`. Import with `import pyransac3d as pyrsc`.

**Trade-off vs Schnabel:** Much simpler to use, but significantly slower and only finds one plane at a time. Schnabel C++ finds multiple shapes simultaneously and is orders of magnitude faster on large point clouds.

---

### `paper_efficient_RANSAC_Schnabel/`

Two reference PDFs:
- `schnabel_2007_efficient_310a84c162.pdf` — The original Schnabel 2007 paper that the C++ library implements
- `cstamas_thesis.pdf` — A thesis on RANSAC for point clouds

---

### `data/`

Storage directory populated by `download_tartan_ground.py` (TartanAir LiDAR) and, over the course of this project's real-world testing work, several unrelated real datasets downloaded alongside it. See [Dataset Locations](#dataset-locations) below for the full breakdown of what's actually in here and how big it is.

```
data/Office/Data_omni/P0000/lidar/*.ply
```

---

## Dataset Locations

Real-world data for this project is spread across **6 separate locations**, totaling roughly 139GB — all of it gitignored, so none of it is part of what you clone from GitHub. Sizes below are approximate, measured locally:

| Location | Contents | Size |
|---|---|---|
| `data/<TartanAir env>/` (26 environments) | TartanAir per-frame LiDAR sequences | ~56G |
| `data/RELLIS3D/`, `data/RELLIS3D_raw/` | Real off-road LiDAR — **a separate dataset, unrelated to TartanAir**, that happens to live inside the `data/` folder | 13G + 48G |
| `data/s3dis/`, `data/s3dis_sample/` | Real indoor scans (Stanford S3DIS) — **also unrelated to TartanAir** | 8G + 11M |
| `schnabel_cython/tartanair_data/` | TartanGround full-scene merged maps (9 environments) | 5.0G |
| `synthetic_rl_experiment/new_data/` | Urban semantic-labeled LiDAR tiles (real-world test source) | 1.9G |
| `synthetic_rl_experiment/off_road_data/` | OpenTopography surveys (real-world test source) | 1.0G |
| `synthetic_rl_experiment/rgbd_data/` | DIODE + TartanGround RGB-D (real-world test source) | 6.2G |
| `synthetic_rl_experiment/wildscenes_data/` | WildScenes (real-world test source) | 67M |

**The one thing worth remembering:** `data/RELLIS3D*` and `data/s3dis*` are not TartanAir or TartanGround data — they're independent real datasets that were downloaded into the `data/` folder alongside TartanAir's environment folders, purely as a matter of convention, not because they're related. If you're looking for "the TartanAir data" specifically, it's the environment-named subfolders (`data/Office/`, `data/House/`, etc.), not everything under `data/`.

The `synthetic_rl_experiment/*_data/` folders are separate real-world sources used specifically to test the synthetic-trained RL model's generalization (see [Two RL Tracks](#two-rl-tracks-important--dont-confuse-them) below) — self-contained to that subproject.

### TartanAir vs. TartanGround

Within `data/`'s TartanAir environments and `schnabel_cython/tartanair_data/`'s TartanGround maps specifically, this project uses **two different datasets** from the same CMU AirLab research group, sharing environment names but not the same data:

### TartanAir (`theairlabcmu/tartanair`)

Used by the **root-level scripts** via the `tartanair` Python package.

- A large robotics simulation dataset with full multi-modal sensor streams
- Contains: LiDAR scans, RGB cameras, depth maps, IMU, segmentation masks
- Data is organized as **sequences of per-frame scans** from a moving robot
- Download granularity: environment + version (`omni/diff/anymal`) + trajectory + modality

```python
import tartanair as ta
ta.init('./data')
ta.download_ground(env='Office', version='omni', traj='P0000', modality='lidar', unzip=True)
# → data/Office/Data_omni/P0000/lidar/*.ply  (one .ply per timestep)
```

### TartanGround (`theairlabcmu/TartanGround`)

Used by **`schnabel_cython/download_tartan_pcd.py`** directly via `huggingface_hub`.

- A separate, derived dataset specifically for ground vehicle navigation
- Contains **pre-built, full-scene merged point cloud maps** (not per-frame sequences)
- Files are single large `.pcd` files covering an entire environment

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="theairlabcmu/TartanGround",
                filename="Supermarket/Supermarket_rgb_pcd.zip", ...)
# → schnabel_cython/tartanair_data/Supermarket/Supermarket_rgb.pcd
```

| | TartanAir | TartanGround |
|---|---|---|
| Accessed via | `tartanair` pip package | `huggingface_hub` directly |
| Data type | Per-frame LiDAR sequences | Full-scene merged maps |
| Format | Multiple `.ply` files | Single `.pcd` per environment |
| Already downloaded | No (`data/` is empty) | Yes (`schnabel_cython/tartanair_data/`) |

---

## How the C++ → Cython → Python Chain Works

When you call `schnabel_ransac.detect(points, ...)` from Python, here is what actually executes:

```
Python script
    │
    │  import schnabel_ransac          ← loads .pyd binary (no compilation at runtime)
    │  shapes = schnabel_ransac.detect(points, epsilon=0.02, ...)
    ▼
schnabel_ransac.pyx  (Cython layer)
    │  - converts NumPy float32 array → raw C float* pointer
    │  - validates input shape is (N, 3)
    │  - resolves relative epsilon to absolute distance
    │  - releases Python's GIL  (with nogil:)
    │  - calls C function: detect_shapes(pts_ptr, n_points, epsilon, ...)
    ▼
bridge.cpp  (C++ bridge)
    │  - copies float array → Schnabel PointCloud object
    │  - computes bounding box + padding
    │  - calls pc.calcNormals(radius, knn)   ← KD-Tree PCA normal estimation
    │  - configures RansacShapeDetector with options
    │  - registers requested shape constructors (plane, sphere, etc.)
    │  - calls detector.Detect(pc, 0, pc.size(), &shapes)
    ▼
RansacShapeDetector.cpp  (Schnabel 2007 C++)
    │  - builds Octree over point cloud
    │  - randomly samples minimal point sets
    │  - fits shape candidates via least-squares
    │  - scores candidates against all remaining points
    │  - applies probabilistic adaptive stopping criterion
    │  - returns accepted shapes sorted by inlier count
    ▼
Results flow back:
bridge.cpp → packs into DetectedShape[] structs
Cython     → converts to Python list of dicts:
             [{"type": "plane", "n_points": 4821,
               "inlier_mask": np.array([True, False, ...]),
               "params": np.array([...]), ...}, ...]
```

The `.pyd` file is a pre-compiled Windows DLL. Python loads it once at import time. All RANSAC math — octree construction, normal estimation, candidate fitting, scoring — runs as native compiled C++.

---

## The Full Pipeline

```
                    ┌─────────────────────────────────┐
                    │  Option A: TartanAir LiDAR       │
                    │  (per-frame .ply sequences)       │
                    └──────────────┬──────────────────┘
                                   │ download_tartan_ground.py
                                   ▼
                    data/Office/Data_omni/P0000/lidar/*.ply
                                   │
                    load_tartan_ground.py  (inspect)
                                   │
                    segment_ground.py      (pyransac3d RANSAC)
                                   │
                    visualize_segmentation.py  (Open3D viewer)

                    ┌─────────────────────────────────┐
                    │  Option B: TartanGround maps     │
                    │  (full-scene .pcd files)         │
                    └──────────────┬──────────────────┘
                                   │ schnabel_cython/download_tartan_pcd.py
                                   ▼
                    schnabel_cython/tartanair_data/*/Env_rgb.pcd
                                   │
                    schnabel_cython/tartan_ground_segmentation.py
                    (uses schnabel_ransac C++ wrapper)
                                   │
                                   ▼
                    Env_rgb_segmented.ply  ←  ground separated from obstacles
```

---

## The Adaptive RL Pipeline

Schnabel's RANSAC needs several parameters tuned per scene (`epsilon`,
`min_support`, `normal_thresh`) — good values differ between a flat indoor
office floor and a bumpy outdoor forest floor. Instead of picking one fixed
setting for every scene, this project trains a **PPO reinforcement learning
agent** (via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/))
to look at a frame's scene features and choose good RANSAC parameters for
it, refining its choice over up to 5 attempts per frame.

- **Environment:** [`ransac_env.py`](ransac_env.py) — a Gymnasium env whose
  observation is 31 numbers (21 static scene features from
  [`features/scene_features.py`](features/scene_features.py) + 10 dynamic
  feedback features), whose action picks an `(epsilon, min_support,
  stop/continue)` triple, and whose reward is shaped from inlier ratio,
  fit residual, runtime, and surface-normal consistency.
- **Training:** [`train_rl.py`](train_rl.py) trains PPO on top of that
  environment; [`rl_evaluator.py`](rl_evaluator.py) and
  [`baseline_evaluator.py`](baseline_evaluator.py) evaluate the trained
  agent against three fixed-parameter baselines (Strict/Standard/Loose,
  see `BASELINE_CONFIG.md`).
- **Comparison:** [`compare_results.py`](compare_results.py),
  [`per_frame_comparison.py`](per_frame_comparison.py), and
  [`plot_comparison.py`](plot_comparison.py) aggregate and chart RL vs.
  baseline performance across every downloaded environment.

**For the full walkthrough** — every observation/action/reward term
explained, the training loop, adaptive per-environment sampling, and a
day-by-day account of bugs found and fixed — see
[RL_PIPELINE_OVERVIEW.md](RL_PIPELINE_OVERVIEW.md) and
[DAILY_LOG.md](DAILY_LOG.md).

---

## Two RL Tracks (important — don't confuse them)

This repo contains **two separate reinforcement-learning efforts** that
share the same underlying Schnabel RANSAC engine but are otherwise
independent. New readers often trip over this, so to be explicit:

| | **Main pipeline** (repo root) | **Synthetic experiment** ([`synthetic_rl_experiment/`](synthetic_rl_experiment/)) |
|---|---|---|
| Environment file | [`ransac_env.py`](ransac_env.py) | [`synthetic_rl_experiment/synthetic_env.py`](synthetic_rl_experiment/synthetic_env.py) |
| Data | Real simulated LiDAR (TartanAir/TartanGround `.ply`) | Procedurally-generated synthetic planes (known ground truth) |
| Observation | 31-dim | 33-dim |
| Models (both in `models/`) | `ppo_ransac_*.zip` | `synthetic_ppo_*.zip` |
| Primary doc | [`RL_PIPELINE_OVERVIEW.md`](RL_PIPELINE_OVERVIEW.md) | [`synthetic_rl_experiment/README.md`](synthetic_rl_experiment/README.md) |

They each independently hit — and separately fixed — the *same class* of
early "policy collapse" bug (missing `ent_coef`/`VecNormalize`); those are
**two different events in two different codebases**, not one. Each track's
docs describe only its own collapse. The current best synthetic model
(`synthetic_ppo_v2.zip`) is genuinely adaptive; see the synthetic folder's
own README for its status.

---

## The Synthetic RL Experiment (`synthetic_rl_experiment/`)

A **self-contained** second RL track that trains the same kind of adaptive
RANSAC agent on **procedurally-generated synthetic point clouds** (planes
with controlled noise, bumps, clutter, and intersecting walls) where the
ground-truth plane is known exactly — enabling clean, quantitative
adaptivity measurement that real, unlabeled LiDAR can't provide. It has its
own environment, generator, training/eval scripts, models
(`synthetic_ppo_*`), and logs.

**Start with [`synthetic_rl_experiment/README.md`](synthetic_rl_experiment/README.md)** — it
orders the folder's three internal docs and states the current result. Note
that one of those docs (`syntheticRL.md`) is a **corrected historical log**:
its original conclusion that the agent "couldn't learn adaptivity" was later
disproven; it carries a correction banner and the authoritative account is
in `SESSION_PROGRESS_LOG.md`.

---

## Virtual Environment

There is **exactly one virtual environment** in this project, located at `.venv/` in the project root.

- **Python version:** 3.11 (Windows x64)
- **Activate:** `.venv\Scripts\activate` (PowerShell) or `.venv\Scripts\activate.bat` (CMD)
- **Install dependencies:** `pip install -r requirements.txt`

Key packages installed:

| Category | Packages |
|---|---|
| 3D / Point Cloud | `open3d-0.19.0`, `pyransac3d-0.6.0`, `plyfile-1.1.4` |
| Deep Learning | `torch-2.12.1`, `torchvision-0.27.1`, `cupy-cuda12x-14.1.1` |
| Computer Vision | `opencv-contrib-python-4.13.0`, `kornia-0.8.3` |
| Scientific | `numpy-2.4.6`, `scipy-1.17.1`, `numba-0.65.1` |
| Data / Datasets | `pandas-3.0.3`, `huggingface_hub-1.20.1`, `tartanair-1.4.0` |
| Visualization | `matplotlib-3.11.0`, `plotly-6.8.0`, `dash-4.3.0`, `Pillow-12.2.0` |
| Build (Cython) | `Cython` (for recompiling `schnabel_ransac` if needed) |

> The compiled Cython extension (`schnabel_ransac.cp311-win_amd64.pyd`) is already built and present in `schnabel_cython/`. You do not need to recompile unless you modify `bridge.cpp`, `schnabel_ransac.pyx`, or the Schnabel C++ source.

---

## Quick Start

### Run ground segmentation on TartanGround data (already downloaded)

```bash
# Activate the virtual environment
.venv\Scripts\activate

# Run segmentation on a pre-downloaded environment
cd schnabel_cython
python tartan_ground_segmentation.py tartanair_data/Supermarket/Supermarket_rgb.pcd z_down 0.3
```

### Run segmentation on a generic point cloud

```bash
cd schnabel_cython
python ground_segmentation.py                    # uses Open3D default dataset
python ground_segmentation.py my_scan.ply        # use your own file
```

### Download TartanAir LiDAR data and run the root pipeline

```bash
# Step 1: Download (~404 MB from Hugging Face)
python download_tartan_ground.py

# Step 2: Inspect the downloaded files
python load_tartan_ground.py

# Step 3: Segment ground from obstacles
python segment_ground.py

# Step 4: Visualize result (green = ground, red = obstacles)
python visualize_segmentation.py
```

### Recompile the Cython extension (only if you change C++ or .pyx files)

```bash
cd schnabel_cython
python setup.py build_ext --inplace
```

### Train and evaluate the RL agent

See [RL_PIPELINE_OVERVIEW.md's "How to Run Things"](RL_PIPELINE_OVERVIEW.md#how-to-run-things) for the full sequence (download LiDAR frames, train, evaluate against baselines, compare, visualize). Quick version:

```bash
python download_lidar_frames.py           # one-time dataset download
python train_rl.py --timesteps 50000      # train
python rl_evaluator.py --env all          # evaluate the trained agent
python baseline_evaluator.py standard     # evaluate a fixed-parameter baseline
python compare_results.py                 # compare RL vs. baselines
```
