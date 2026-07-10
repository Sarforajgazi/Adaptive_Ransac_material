# `schnabel_cython/` — Complete Breakdown

> Everything in the `schnabel_cython/` folder: what each file is, how the
> Python → Cython → C++ call chain works, what every parameter means, how
> the module is built, and how it's actually used elsewhere in this repo.
> For the internals of the RANSAC algorithm itself (octrees, candidate
> scoring, stopping criterion), see [EFFICIENT_RANSAC_BREAKDOWN.md](EFFICIENT_RANSAC_BREAKDOWN.md) —
> this document is about the **wrapper**, not the algorithm it wraps.

---

## Table of Contents

1. [What This Folder Is](#what-this-folder-is)
2. [The Three-Layer Architecture](#the-three-layer-architecture)
3. [File-by-File Reference](#file-by-file-reference)
4. [The Call Chain, End to End](#the-call-chain-end-to-end)
5. [`detect()` — Full Parameter Reference](#detect--full-parameter-reference)
6. [The Return Value](#the-return-value)
7. [How the Build Works (`setup.py`)](#how-the-build-works-setuppy)
8. [Data Structures Crossing the Boundary](#data-structures-crossing-the-boundary)
9. [Design Decisions and Gotchas](#design-decisions-and-gotchas)
10. [How the Rest of the Repo Uses This Module](#how-the-rest-of-the-repo-uses-this-module)

---

## What This Folder Is

`schnabel_cython/` turns Ruwen Schnabel's 2007 C++ "Efficient RANSAC for
Point-Cloud Shape Detection" library — template-heavy, header-only in
places, built for a standalone C++ demo — into a normal importable Python
module:

```python
import schnabel_ransac
shapes, n_remaining = schnabel_ransac.detect(points, epsilon=0.15, min_support=500)
```

No subprocess calls, no file I/O round-trip, no reflection/ctypes hacking.
It's a real compiled Python extension module (`.pyd` on Windows, `.so` on
macOS/Linux) — calling it costs a Python→C function call, not a process
launch.

This is the single most performance-critical piece of the whole project:
every RL training step and every baseline evaluation frame calls
`schnabel_ransac.detect()` at least once. The RANSAC C++ algorithm itself
does the heavy lifting (octree sampling, candidate scoring); this folder's
job is just to get data in and out of it as cheaply as possible.

---

## The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Python caller (ransac_env.py, batch_segment_lidar.py, ...)  │
│      import schnabel_ransac                                  │
│      shapes, n_remaining = schnabel_ransac.detect(points, …)  │
└───────────────────────────┬───────────────────────────────────┘
                            │  numpy array (N,3) float32
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — Cython wrapper   schnabel_ransac.pyx               │
│  • validates + converts the numpy array to a flat C pointer   │
│  • resolves "relative epsilon" (% of bbox) → absolute metres  │
│  • builds the shape bitmask from a list of strings            │
│  • allocates output buffers                                   │
│  • releases the GIL, calls into C++                           │
│  • converts the C results back into Python dicts / numpy       │
└───────────────────────────┬───────────────────────────────────┘
                            │  float*, size_t, DetectedShape*
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — C++ bridge   bridge.h / bridge.cpp                 │
│  • builds a Schnabel PointCloud from the flat float array      │
│  • computes the bounding box automatically                    │
│  • calls pc.calcNormals() (PCA-based normal estimation)        │
│  • configures RansacShapeDetector::Options                     │
│  • registers the requested shape constructors                 │
│  • calls detector.Detect() — the actual algorithm              │
│  • serializes each detected shape into a flat DetectedShape    │
│  • maps rearranged point order back to original indices        │
└───────────────────────────┬───────────────────────────────────┘
                            │  PointCloud, RansacShapeDetector API
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — Schnabel's original C++ library                    │
│  ../Efficient-RANSAC-for-Point-Cloud-Shape-Detection/          │
│  RansacShapeDetector, PointCloud, Plane/Sphere/Cylinder/…      │
│  (never modified — treated as a vendored third-party library)  │
└─────────────────────────────────────────────────────────────┘
```

Layer 3 is **not part of this folder** — it lives one directory up in
`Efficient-RANSAC-for-Point-Cloud-Shape-Detection/` and is compiled *into*
`schnabel_cython`'s extension module by `setup.py`, but its source is never
touched. `schnabel_cython/` exists specifically to avoid modifying that
vendored code.

---

## File-by-File Reference

### Core wrapper (the only files that matter for using the module)

| File | Role |
|---|---|
| [`bridge.h`](schnabel_cython/bridge.h) | C header. Declares `DetectedShape` (the flat result struct) and the `detect_shapes()` function signature — the entire API surface between Cython and C++. |
| [`bridge.cpp`](schnabel_cython/bridge.cpp) | Implements `detect_shapes()`. This is where the actual Schnabel API (`PointCloud`, `RansacShapeDetector`, shape constructors) gets driven. ~140 lines; see [walkthrough below](#the-call-chain-end-to-end). |
| [`schnabel_ransac.pxd`](schnabel_cython/schnabel_ransac.pxd) | Cython declaration file. Tells the Cython compiler what `bridge.h`'s C types/functions look like, so `.pyx` code can call them directly with no overhead. |
| [`schnabel_ransac.pyx`](schnabel_cython/schnabel_ransac.pyx) | The actual Python-facing API. Defines `detect()` — argument validation, numpy↔C conversion, GIL release, result marshalling. This is the file that becomes `schnabel_ransac.<function>` in Python. |
| [`setup.py`](schnabel_cython/setup.py) | Build script. Compiles `schnabel_ransac.pyx` + `bridge.cpp` + every `.cpp` file in the vendored Schnabel library into one shared library. |
| [`schnabel_ransac.cpp`](schnabel_cython/schnabel_ransac.cpp) | **Generated file** — Cython transpiles `schnabel_ransac.pyx` into this plain C++ file, which is then compiled by a normal C++ compiler. You never edit this directly; it's regenerated from the `.pyx` every build. |
| `schnabel_ransac.cp311-win_amd64.pyd` | **Compiled binary** (Windows, Python 3.11, x64). This is the actual thing `import schnabel_ransac` loads. Already built — no compilation happens at import time. |
| `schnabel_ransac.cpython-312-darwin.so` | Same, for macOS (Apple Silicon, Python 3.12). |
| `build/` | Build artifacts (`.o`/`.obj` object files, intermediate `.pyd`/`.so` copies) left behind by `setup.py build_ext`. Safe to delete and rebuild. |

### Demo / utility scripts (not imported by the RL pipeline)

| File | Role |
|---|---|
| `ground_segmentation.py` | Standalone demo: loads a point cloud via Open3D, calls `schnabel_ransac.detect()`, picks the lowest plane as "ground," visualizes it. Predates `ransac_env.py`. |
| `real_data_demo.py` | Demo against a real Open3D sample indoor scan, to sanity-check the module against non-synthetic data. |
| `visual_demo.py` | Demo against synthetic data (a plane + a cylinder + Gaussian noise) — used to confirm the algorithm actually recovers known ground-truth shapes before trusting it on real data. |
| `download_data.py` | Downloads TartanGround `.pcd` environments (used by these standalone demos, separate from the main `download_lidar_frames.py` pipeline at the repo root). |
| `fix_templates.py` | One-time source patch applied to the vendored Schnabel `.cpp`/`.h` files to fix C++ template-instantiation issues that blocked compilation on modern compilers. Not run as part of normal builds — a historical record of what had to change to make 2009-era code compile today. |
| `test_wrapper.py`, `test_o3d.py` | Ad-hoc smoke tests for the wrapper and for Open3D integration. |
| `null.o`, `test_out*.log`, `build_errors.log`, `lldb_bt.log` | Build/debug artifacts from diagnosing the macOS Apple Silicon segfault (see [Gotchas](#design-decisions-and-gotchas)). Not part of the module itself. |
| `office_with_ground.ply`, `temp_test.ply` | Sample point clouds used by the demo scripts above. |
| `tartanair_data/` | Pre-built "complete" (multi-frame-merged) reconstructions per TartanAir environment, plus label files. Used by `compare_on_complete_cloud.py` at the repo root — a different data source than the per-frame `data/` directory the RL pipeline trains on. |

---

## The Call Chain, End to End

Walking through what actually happens for one call, e.g.:

```python
shapes, n_remaining = schnabel_ransac.detect(
    points, shapes=["plane"], relative_epsilon=False,
    epsilon=0.15, normal_thresh=0.90, min_support=500,
)
```

**1. Input validation & conversion** (`schnabel_ransac.pyx:117-128`)
The input array is coerced to `float32`, C-contiguous, shape `(N, 3)`. If
it isn't already in that layout, numpy makes a copy here — the one
unavoidable copy in the whole pipeline.

**2. Shape bitmask construction** (`:130-142`)
`["plane"]` → looked up in `_SHAPE_BITS = {"plane": 1, "sphere": 2, "cylinder": 4, "cone": 8, "torus": 16}` → OR'd together into a single `int` bitmask. This is what lets `detect_shapes()` on the C++ side take one `int` parameter instead of five booleans.

**3. Relative-epsilon resolution** (`:144-161`)
If `relative_epsilon=True` (the default for the `detect()` API, though the
RL environment always passes `False`), `epsilon` is interpreted as a
fraction of the point cloud's bounding-box diagonal rather than an
absolute distance. The wrapper computes the bbox here in Python/numpy and
multiplies before ever reaching C++ — the C++ side always receives an
absolute float, it never knows about the "relative" concept.

**4. Buffer allocation** (`:163-178`)
Raw `malloc()` for `max_shapes * sizeof(DetectedShape)` (the output shape
array) and `n_points * sizeof(int)` (the per-point assignment array).
Wrapped in `try/finally` so these are always `free()`'d even if C++ raises
or an exception occurs during result marshalling.

**5. The actual call, with the GIL released** (`:180-190`)
```cython
with nogil:
    n_detected = detect_shapes(pts_ptr, n_points, abs_epsilon, ...)
```
This is the one line that matters most for performance: releasing Python's
Global Interpreter Lock means the C++ code could run multi-threaded
(OpenMP) without fighting Python for the CPU — though see the
[OpenMP gotcha](#design-decisions-and-gotchas) below, since it's currently
disabled by default in this build.

**6. Inside `detect_shapes()`** (`bridge.cpp`)
- Copies the flat float array into a Schnabel `PointCloud`, tracking each
  point's original index (`POINTSWITHINDEX` macro) so results can be
  mapped back later.
- Computes the bounding box from the data directly (with 1% padding) —
  Schnabel's original API requires the caller to supply this manually;
  the bridge automates it so Python callers never have to think about it.
- Calls `pc.calcNormals(normal_radius, normal_knn)` — PCA-based surface
  normal estimation over each point's k-nearest neighbours. **This runs on
  every single call.** There's no caching between calls even if you call
  `detect()` twice on the same cloud with different `epsilon` — this is
  one of the costlier fixed overheads per call.
- Builds a `RansacShapeDetector::Options` struct from the parameters and
  registers only the requested shape constructors (`detector.Add(new PlanePrimitiveShapeConstructor())` etc.).
- Calls `detector.Detect(pc, 0, pc.size(), &shapes)` — this is the entire
  Schnabel algorithm (octree building, candidate generation, statistical
  stopping criterion — all covered in [EFFICIENT_RANSAC_BREAKDOWN.md](EFFICIENT_RANSAC_BREAKDOWN.md)).
- After `Detect()` returns, the point cloud has been **physically
  rearranged in memory** — all of shape 0's inliers are one contiguous
  block, shape 1's another, etc., with unassigned points at the front.
  `bridge.cpp` walks this rearranged layout and, using each point's stored
  original index, fills `out_point_shape_index[original_i] = shape_id`
  (or `-1` if never assigned to any shape).
- Each shape's parameters are serialized into a fixed `float[16]` array via
  `shape->Serialize(params)` — a generic mechanism that works whether the
  shape is a 4-float plane or an 8-float torus (see the
  [parameter counts table](#the-return-value)).

**7. Back in Cython — result marshalling** (`:192-224`)
For each detected shape: build a boolean numpy `inlier_mask` by comparing
the per-point assignment array against the shape's index, decode the
shape's type name from `SHAPE_NAMES`, decode the description string, copy
the serialized float parameters into a numpy array, and package it all
into a plain Python `dict`. This is the only layer where Python objects
(dicts, numpy arrays) get created — everything before this point was raw
C memory.

**8. Cleanup** (`:226-231`)
The `finally` block frees the two `malloc()`'d buffers unconditionally.

---

## `detect()` — Full Parameter Reference

```python
schnabel_ransac.detect(
    points_input,
    epsilon=0.01,
    normal_thresh=0.9,
    min_support=500,
    bitmap_epsilon=-1.0,
    probability=0.001,
    shapes=None,
    relative_epsilon=True,
    normal_radius=3.0,
    normal_knn=20,
    max_shapes=100,
)
```

| Parameter | Type / default | Meaning |
|---|---|---|
| `points_input` | array-like `(N, 3)` | The point cloud. Converted to contiguous `float32` internally — pass numpy arrays directly to avoid an extra copy. |
| `epsilon` | `float`, `0.01` | Distance threshold. A point counts as an inlier if it's within roughly this distance of the candidate shape's surface. **The underlying C++ algorithm multiplies this by 3× internally during final ("global") scoring** — see [EFFICIENT_RANSAC_BREAKDOWN.md § Scoring System](EFFICIENT_RANSAC_BREAKDOWN.md#the-scoring-system) for why. If `relative_epsilon=True`, this is a *fraction of the bounding-box diagonal* (e.g. `0.01` = 1% of the cloud's largest extent); if `False`, it's an absolute distance in the same units as the point cloud (metres, for LiDAR data). |
| `normal_thresh` | `float`, `0.9` | Cosine of the maximum allowed angle between a point's estimated surface normal and the candidate shape's normal. `0.9 ≈ 26°` of tolerance, `0.95 ≈ 18°` (stricter). A point failing this check is rejected as an inlier even if it's within `epsilon` distance — this is what prevents, e.g., a vertical wall from being accepted as part of a horizontal floor plane just because they're geometrically coplanar-ish. |
| `min_support` | `int`, `500` | Minimum inlier count for a shape to be accepted at all. Below this, even a geometrically perfect plane is discarded. This is what filters out tiny spurious flat patches (a book cover, a shelf edge) from being reported as "shapes." |
| `bitmap_epsilon` | `float`, `-1` (auto) | Cell size for the connected-component "is this actually one contiguous surface, not two coplanar surfaces that happen to line up" filter. `-1` means "auto": resolved to `2 × epsilon`. |
| `probability` | `float`, `0.001` | Miss probability for the adaptive stopping criterion — the algorithm keeps generating candidate shapes until it's statistically confident (to within this probability) that no undetected shape with ≥ `min_support` inliers remains. Lower = more exhaustive search, slower. `0.01` is noticeably faster than `0.001` for a small quality cost. |
| `shapes` | `list[str]`, `["plane"]` | Which primitive types to search for. Options: `"plane"`, `"sphere"`, `"cylinder"`, `"cone"`, `"torus"`. The RL pipeline only ever requests `["plane"]` — ground segmentation doesn't need the others, and each additional shape type registered adds real search cost since every sample point gets tested against every registered shape constructor. |
| `relative_epsilon` | `bool`, `True` | Interprets `epsilon`/`bitmap_epsilon` as fractions of the bbox diagonal (`True`) or absolute distances (`False`). **The RL environment always passes `False`** — it works in absolute metres because LiDAR frame scale is roughly consistent, and it makes the parameter directly comparable across frames of different physical size. |
| `normal_radius` | `float`, `3.0` | Neighbourhood radius for PCA-based normal estimation. In the current implementation this mostly affects Gaussian weighting in the normal fit; the hard neighbour cutoff is actually driven by `normal_knn`. |
| `normal_knn` | `int`, `20` | Number of nearest neighbours used to estimate each point's surface normal via local PCA. Larger = smoother/more stable normals but blurs sharp edges and costs more compute; this runs once per `detect()` call over every point, so it's one of the more expensive fixed costs per call. |
| `max_shapes` | `int`, `100` | Hard cap on how many shapes can be returned — just an output buffer size, not an algorithm behaviour. The RL environment uses `max_shapes=20` since it only ever wants the single best ground plane and doesn't need more than a handful of candidates to choose from. |

**The one non-obvious interaction to remember:** if you set `epsilon=0.15`
(absolute), the actual distance a point can be from the plane and still
count as a final inlier is up to `0.45` (`3 × epsilon`) — see
[EFFICIENT_RANSAC_BREAKDOWN.md's scoring section](EFFICIENT_RANSAC_BREAKDOWN.md#the-scoring-system)
for exactly why. This is baked into the vendored C++ and the bridge does
not compensate for it.

---

## The Return Value

`detect()` returns `(detected: list[dict], n_remaining: int)`.

`n_remaining` is how many input points were never claimed by any accepted
shape.

Each entry in `detected` is:

```python
{
    "type": "plane",              # str — shape type name
    "type_id": 0,                 # int — 0=plane,1=sphere,2=cylinder,3=cone,4=torus
    "n_points": 41823,            # int — inlier count
    "description": "...",         # str — human-readable description from the C++ side
    "params": np.ndarray(...),    # float32 — serialized shape parameters (see below)
    "inlier_mask": np.ndarray(...) # bool, shape (N,) — True for this shape's inliers
}
```

Parameter counts per shape type (`params` array length), from the
vendored library's `Serialize()`/`SerializedFloatSize()` implementations:

| Shape | `num_params` | Layout |
|---|---|---|
| Plane | 4 | `[normal_x, normal_y, normal_z, dist_to_origin]` |
| Sphere | 4 | `[center_x, center_y, center_z, radius]` |
| Cylinder | 7 | `[axis_point_xyz, axis_dir_xyz, radius]` |
| Cone | 7 | `[apex_xyz, axis_xyz, half_angle]` |
| Torus | 8 | `[center_xyz, axis_xyz, major_radius, minor_radius]` |

In practice, this codebase's ground-segmentation code (`find_ground_plane()`
in `ransac_env.py`) doesn't decode `params` at all — it re-derives the
plane's normal itself via a fresh PCA over the inlier points selected by
`inlier_mask`, rather than trusting the serialized plane parameters
directly. This is a minor redundancy, not a correctness issue: it produces
the same normal from the same points; the plane parameters returned by
Schnabel's own fit could be used instead if you wanted to skip the extra
PCA call.

---

## How the Build Works (`setup.py`)

```
python setup.py build_ext --inplace
```

This is the **only** build step — there's no CMake, no Makefile. `setup.py`:

1. Globs every `.cpp` file directly in
   `../Efficient-RANSAC-for-Point-Cloud-Shape-Detection/` (excluding
   `main.cpp`, which defines its own `main()` and would conflict), plus
   everything under its `MiscLib/` subfolder.
2. Compiles all of those together with `schnabel_ransac.pyx` (via Cython)
   and `bridge.cpp` into **one single extension module**, `schnabel_ransac`.
3. Passes `-std=c++14` (needed for the vendored code) and `-w` (suppress
   all warnings — the vendored code is from 2009 and not warning-clean on
   modern compilers).
4. Defines the macro `POINTSWITHINDEX=1`, which switches on the
   `size_t index` field in Schnabel's `Point` struct — **without this
   macro, `bridge.cpp` cannot map rearranged points back to their original
   order**, since `Detect()` physically reorders the point array in place.
5. Platform-specific compile flags:
   - **Windows (MSVC):** `/O2` (full optimization) — safe here.
   - **macOS/Linux (Clang/GCC):** `-O0` (**no optimization**) — deliberately,
     because `-O1`/`-O2` trigger a strict-aliasing/alignment segfault
     inside the vendored `KdTree` code on Apple Silicon (arm64). This is a
     real, previously-debugged bug in the 2009 template code interacting
     with modern compiler optimizations, not a style choice — see
     `build_errors.log` / `lldb_bt.log` in this folder for the original
     crash investigation.
6. OpenMP (`-fopenmp`) is **commented out** in both branches — parallel
   candidate generation inside the C++ algorithm is currently disabled.
   The GIL is still released around the call (so it wouldn't block other
   Python threads if you had any), but the RANSAC search itself runs
   single-threaded. Re-enabling it on Linux is a one-line uncomment; on
   macOS it additionally requires `brew install libomp` since Apple Clang
   has no built-in OpenMP support.

Compiled artifacts already exist in the repo for both Windows
(`.pyd`, Python 3.11) and macOS (`.so`, Python 3.12) — a fresh checkout
does **not** need to rebuild before use, only if you change `bridge.cpp`,
`schnabel_ransac.pyx`, or switch Python/platform versions.

---

## Data Structures Crossing the Boundary

```
Python (numpy)              Cython (schnabel_ransac.pyx)         C++ (bridge.cpp / Schnabel lib)
─────────────────────────────────────────────────────────────────────────────────────────────────
ndarray(N,3) float32   →    float* pts_ptr                  →    PointCloud (vector<Point>)
list[str] shapes        →    int shape_mask                  →    detector.Add(...) per bit
scalars (eps, etc.)      →    (converted, e.g. rel→abs)       →    RansacShapeDetector::Options
                                                                    ↓ detector.Detect(...)
                                                                    MiscLib::Vector<pair<Shape,size_t>>
int* point_idx (malloc)  ←    memcpy → ndarray(N,) int32      ←    out_point_shape_index[N]
DetectedShape[max_shapes]←    per-shape dict assembly         ←    out_shapes[i] (type,n_pts,params,desc)
     (malloc)
```

The only heap allocations on the C++ side are the two `malloc()` calls in
`schnabel_ransac.pyx` (freed in the `finally` block) — everything inside
`bridge.cpp`/the Schnabel library uses its own internal containers
(`PointCloud`, `MiscLib::Vector`) that are cleaned up automatically when
`detect_shapes()` returns.

---

## Design Decisions and Gotchas

**Why a flat C struct instead of exposing the C++ classes directly to Cython?**
Schnabel's `PrimitiveShape` hierarchy is polymorphic, template-heavy, and
uses custom reference-counted smart pointers (`MiscLib::RefCountPtr`).
Wrapping that directly in Cython would mean either duplicating that
template machinery in `.pxd` declarations or writing extension-type
wrappers for every shape subclass. `bridge.h`'s `DetectedShape` sidesteps
all of that: it's a plain-old-data struct with a fixed-size float array,
representable identically in C, Cython, and eventually numpy — at the
cost of a hard `params[16]` cap (never actually hit — max real usage is
torus's 8) and a `description[256]` char buffer (also generous headroom).

**Why does `bridge.cpp` compute the bounding box itself instead of requiring the caller to pass it?**
Schnabel's original `PointCloud::setBBox()` requires the caller to supply
min/max corners — there's no auto-detection in the base library. Doing it
in `bridge.cpp` instead of in Python means one fewer round-trip and one
fewer thing calling code has to remember to get right (a bbox that's too
tight relative to the data would break the octree construction silently).

**Normals are recomputed on every single `detect()` call — no caching.**
If you call `detect()` twice on the same point cloud with two different
`epsilon` values (as `traversability.py`'s two-strategy fallback or any
future "retry with looser thresholds" logic might), you pay for
`calcNormals()` twice. This is a real, current inefficiency — not a bug,
but worth knowing if profiling shows normal estimation as a hotspot.

**Points come back reordered internally, but the wrapper hides this from you.**
`Detect()` physically moves points around inside the C++ `PointCloud` to
group each shape's inliers contiguously (this is fundamental to how the
scoring algorithm partitions the cloud, not an implementation accident).
`bridge.cpp` uses each point's tracked original index (`POINTSWITHINDEX`)
to undo this before handing results back, so from Python's perspective
`inlier_mask` always lines up with your original input array order —
`points[shape["inlier_mask"]]` is always correct against the array you
passed in. This "undo" logic is the main reason `bridge.cpp` needs the
`index` field at all; the algorithm itself never uses it.

**Relative vs. absolute epsilon is a Cython-layer concept, not a C++ one.**
The bridge and the underlying algorithm only ever see an absolute float.
`relative_epsilon=True` is pure convenience on the Python side (useful
when you don't know a scan's physical scale in advance and want "1% of
the bounding box" instead of guessing metres) — it has zero representation
on the C++ side of the boundary.

---

## How the Rest of the Repo Uses This Module

Every consumer imports it the same way — by inserting `schnabel_cython/`
onto `sys.path` at runtime (there's no `pip install -e` / package
registration; it's used as a local compiled module):

```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))
import schnabel_ransac
```

| Caller | How it's used |
|---|---|
| `ransac_env.py` (`RansacEnv.step()`) | The core RL environment. Calls `schnabel_ransac.detect(shapes=["plane"], relative_epsilon=False, ...)` once per step, with `epsilon`/`min_support` chosen by the agent's action and `normal_thresh` fixed at 0.90. This is the single most-called call site in the whole project — every training timestep and every evaluation frame goes through here. |
| `baseline_evaluator.py` | Same call path via `RansacEnv.step()`, but with fixed parameters (Strict/Standard/Loose from `BASELINE_CONFIG.md`) instead of an RL action, to establish a non-learned performance floor. |
| `batch_segment_lidar.py` | Calls `detect()` directly (not through `RansacEnv`) with hardcoded `epsilon=0.3, min_support=300, normal_thresh=0.9` to bulk-segment an entire trajectory and save colour-coded PLY files for visual inspection. |
| `compare_on_complete_cloud.py` / `compare_on_trajectory_map.py` | Calls `detect()` on much larger, merged multi-frame reconstructions (not the per-frame data the RL agent trains on) to compare Schnabel's plane-fit quality against the independent grid-based `traversability.py` classifier. |
| `traversability.py` (optional `plane_fit_method="schnabel"` mode) | Calls `detect()` per grid cell (a few dozen points at a time) with parameters scaled way down (`min_support=10`, tiny `epsilon`) — an example of the same module being reused at a completely different scale than the whole-frame RL use case. |
| `schnabel_cython/ground_segmentation.py`, `real_data_demo.py`, `visual_demo.py` | Standalone demos/sanity-checks, independent of the RL pipeline. |

For the full picture of how `ransac_env.py` fits into agent training —
the state space, the reward function, and what each RL-side file does —
see [RL_PIPELINE_OVERVIEW.md](RL_PIPELINE_OVERVIEW.md).
