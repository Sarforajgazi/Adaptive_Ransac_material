# RANSAC Material — Project Overview

A 3D point cloud **ground segmentation pipeline** for robotics and autonomous navigation. The project wraps Ruwen Schnabel's 2007 Efficient RANSAC C++ library in Cython so Python can call it at native C++ speed, and applies it to LiDAR data from the TartanAir and TartanGround simulation datasets.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Folder Structure](#folder-structure)
- [The Two Datasets](#the-two-datasets)
- [How the C++ → Cython → Python Chain Works](#how-the-c--cython--python-chain-works)
- [The Full Pipeline](#the-full-pipeline)
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

---

## Folder Structure

```
Ransac_material/
│
├── Efficient-RANSAC-for-Point-Cloud-Shape-Detection/   # Original C++ library (Schnabel 2007)
├── schnabel_cython/                                    # Cython wrapper (the core engineering)
├── pyRANSAC-3D/                                        # Pure Python RANSAC (alternative)
├── paper_efficient_RANSAC_Schnabel/                    # Reference papers (PDF)
├── data/                                               # Dataset storage (TartanAir LiDAR)
├── .venv/                                              # Single Python virtual environment
│
├── download_tartan_ground.py   # Step 1: Download TartanAir LiDAR data
├── load_tartan_ground.py       # Step 2: Inspect downloaded point clouds
├── segment_ground.py           # Step 3: Run RANSAC ground segmentation
└── visualize_segmentation.py   # Step 4: Open 3D visualization window
```

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

Storage directory for TartanAir LiDAR data downloaded by `download_tartan_ground.py`. Currently contains only a Hugging Face cache marker — the actual LiDAR scans (~404 MB) have not been downloaded yet.

After running `download_tartan_ground.py`, data will appear at:
```
data/Office/Data_omni/P0000/lidar/*.ply
```

---

## The Two Datasets

This project uses **two different datasets** from the same CMU AirLab research group. They have similar names but are not the same thing.

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

## Virtual Environment

There is **exactly one virtual environment** in this project, located at `.venv/` in the project root.

- **Python version:** 3.11 (Windows x64)
- **Activate:** `.venv\Scripts\activate` (PowerShell) or `.venv\Scripts\activate.bat` (CMD)

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
