# Getting Started

A practical setup-and-run guide: clone the repo, install dependencies, and
detect ground + walls on your own point cloud using `synthetic_ppo_v2.zip`
— the current, recommended trained model. For the project's architecture
and how everything fits together, see [`README.md`](README.md) instead;
this doc is purely "how do I run it."

## 1. Prerequisites

- Python 3.11 (Windows x64 is what this project was built/tested on; see
  the platform notes in `requirements.txt` if you're on Linux/macOS)
- Git

## 2. Install

```bash
git clone <this repo>
cd Ransac_material
python -m venv .venv
.venv\Scripts\activate          # PowerShell/CMD; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

`requirements.txt` has a header comment on the handful of packages that
need platform-specific attention (`cupy-cuda12x`/`pywin32` are Windows+CUDA
only; `torch`/`torchvision`/`torchaudio` are pinned to CUDA 12.1 builds from
PyTorch's own index, not plain PyPI; `pyransac3d` is installed editable
from this repo's own `pyRANSAC-3D/` folder). None of these block the
ground+wall detection workflow below if you'd rather skip/substitute them.

**The compiled Schnabel RANSAC extension is already built and committed**
(`schnabel_cython/schnabel_ransac.cp311-win_amd64.pyd` for Windows,
`schnabel_ransac.cpython-312-darwin.so` for macOS) — you do not need to
compile anything to get started. If you're on a platform/Python version
without a matching prebuilt binary, see `schnabel_cython/README.md`'s "How
to Compile" section.

## 3. The model — nothing to download

`models/synthetic_ppo_v2.zip` and `models/synthetic_ppo_v2_vecnormalize.pkl`
are committed to the repo — confirmed git-tracked, not gitignored. A fresh
clone already has everything needed to run detection; there's no separate
model download step.

## 4. Run detection on your own point cloud

```bash
python detect_ground_and_walls.py --ply your_cloud.ply --model_variant synthetic
```

Don't have a `.ply` handy? A small sample (a real office room scan, ~8MB)
is one command away — it's fetched via Open3D's own asset downloader
rather than committed to the repo, so it needs this one-time step first:

```bash
python schnabel_cython/download_data.py
```

Then the exact command below works — verified end-to-end while writing
this guide:

```bash
python detect_ground_and_walls.py --ply schnabel_cython/office_with_ground.ply --model_variant synthetic
```

Or drop your own file in [`my_point_clouds/`](my_point_clouds/) instead
and point `--ply` at that — see that folder's README.

Expected output (a small office scan, 6240 points after the default voxel
downsample):

```
Model variant: synthetic
Loading model: .../models/synthetic_ppo_v2.zip
Loading point cloud: schnabel_cython/office_with_ground.ply
  office_with_ground.ply: 6240 pts, params=(eps=0.08, min_supp=94, norm_th=0.6), N shape(s) -> ground=2971 pts (47.6%), M wall(s) (... pts), 0 cylinder(s) (0 pts)
Opening visualizer window (ground=GREEN, walls=distinct colors, cylinders=gold shades, rest=GRAY).
```

The exact shape/wall count (`N`/`M` above) can vary run to run — the
underlying Schnabel C++ engine reseeds its own random generator every call
and isn't seedable from Python (a known, documented, harmless quirk — see
`EFFICIENT_RANSAC_BREAKDOWN.md`'s `Random.cpp` note). Ground detection
itself is stable; it's specifically the count of smaller secondary wall
candidates that can shift between runs.

### Troubleshooting

| Symptom | Likely cause |
|---|---|
| Ground percentage near 0%, or detection looks inverted | Wrong `--z_mode`. Default is `z_down` (correct for LiDAR); most non-LiDAR real point clouds (RGB-D, terrain scans, indoor scans) are `z_up` — pass `--z_mode z_up`. |
| "Nothing to visualize" / no shapes found at all | Likely a scale mismatch — see the voxel section below. This project's own diagnostic work found detection fails specifically when the chosen distance tolerance ends up smaller than your point cloud's natural point-to-point spacing; a ~20x gap in detection-success rate was measured between "tolerance below spacing" and "tolerance above spacing" cases. |
| No cylinders detected even though you expect some | Cylinder detection is only requested under `--model_variant main`, not `synthetic` — see the comment at `detect_ground_and_walls.py:127` for why (a degenerate cylinder hypothesis was found to sometimes out-compete a real plane hypothesis when both are requested together). |
| Command runs but no window ever appears | You're on a machine with no active display session — the viewer needs one (not true headless). Use `--csv_log path.csv` or the batch modes' screenshot output instead of `--ply` on a headless box. |

## 5. The interactive viewer — same one used throughout this project

Every screenshot and figure produced by this project's own real-world
testing and report generation came from this exact viewer
(`screenshot_utils.draw_geometries()`, a thin wrapper around Open3D's
standard interactive window) — running the command above gets you the
identical experience, not a simplified or different one:

- **Ground** — green
- **Each wall** — its own distinct color
- **Cylinders** (`--model_variant main` only) — gold shades
- **Everything else** — gray
- **Controls**: rotate with left mouse drag, pan with Shift+left mouse
  drag, zoom with scroll, press **P** to save a screenshot (lands in
  `screenshots/` — see that folder's README for why), close the window to
  exit the script.

## 6. Voxel downsampling — how and whether to use it

`--voxel <meters>` controls how aggressively the point cloud is
downsampled before detection (LiDAR-sourced `.ply` files only — the
`--source rgbd`/`--source s3dis` paths are never downsampled, they're
already sparse/point-bounded by construction). Default is `0.05` — this
project's existing baseline for a single sensor-centric LiDAR sweep,
unchanged from before this flag existed unless you pass a different value.

There's no universal right answer — it depends on your point cloud's
density and physical extent. This project already sets it differently
across its own real-world test scripts; use that precedent rather than
guessing:

| Point cloud type | Typical size | Voxel size used in this project |
|---|---|---|
| Single LiDAR sweep (sensor-centric `.ply`) | thousands of points | **0.05m** (the default) |
| RGB-D back-projected (image → point cloud) | bounded by image resolution | **0** (none — already sparse enough, raw per-pixel) |
| Indoor room scan (S3DIS-like) | pre-compressed | **0** (none — already reasonably sized) |
| Large dense tile/map (millions of points, 150-200m span) | millions of points | **0.15–0.3m** — full resolution is far denser than anything a trained model saw during training, and far too slow to even rotate smoothly |
| Already-sparse survey tiles (~1-2 pts/m²) | sparse | **0** (none — downsampling something already sparse doesn't help) |

**Rule of thumb**: if your cloud is a small, already-bounded point set
(hundreds of thousands of points or fewer, e.g. RGB-D or a single indoor
scan), try `--voxel 0` first. If it's a large, dense map or merged
multi-scan reconstruction (millions of points spanning 100+ meters), start
around `--voxel 0.15` and increase toward `0.3` if it's still slow or the
result looks overly fragmented. For anything roughly LiDAR-sweep-like, the
`0.05` default is a reasonable starting point.

**Why this matters beyond speed**: this project's own diagnostic work
found that whether the model's chosen distance tolerance (`eps`) ends up
above or below your point cloud's *actual* point spacing predicts
detection success far better than raw point count alone — voxel size
directly changes that spacing. If detection is failing on a cloud you
believe has real flat surfaces, this is the first thing worth adjusting.
