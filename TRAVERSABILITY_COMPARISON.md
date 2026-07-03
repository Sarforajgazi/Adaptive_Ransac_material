# Schnabel RANSAC vs. Grid-Based Traversability — Comparison Log

Two independent ground-detection methods, compared directly against each other
(not against ground truth — see caveat below).

## Methods

| # | Method | Script | Approach |
|---|---|---|---|
| 1 | **Schnabel RANSAC** | `schnabel_cython` (same engine as `ransac_env.py`) | Single best global plane, picked the same way `find_ground_plane()` does |
| 2 | **Grid-based traversability** | `traversability.py` (standalone, no RANSAC/RL dependency) | Per-cell local plane fit + slope/roughness scoring |

## Caveats to keep in mind

| Caveat | Detail |
|---|---|
| No ground truth | `<Env>_rgb_segmented.ply` (shipped with the complete clouds) was itself produced by an earlier Schnabel run, not an independent source. All numbers below are **direct agreement between two live methods**, not accuracy. |
| Coordinate convention varies by data source | Confirm Z-up vs. Z-down empirically per data source — don't assume it carries over (see Part 1 and Part 2 notes below, they differ). |

---

## At a glance

| Data source | Environment | Schnabel ground % | Grid ground % | Agreement IoU | Grid unknown % |
|---|---|---|---|---|---|
| Complete cloud (Part 1) | Downtown | 11.6% | 6.8% | **0.373** | 56.2% |
| Complete cloud (Part 1) | OldScandinavia | 0.5% | 0.4% | **0.010** | 97.2% |
| Accumulated map (Part 2) | Downtown | 10.0% | 0.0% | **0.000** | 76.0% |
| Accumulated map (Part 2) | OldScandinavia | 0.5% | 0.0% | **0.000** | 75.3% |

**Cross-check:** Schnabel's numbers agree closely between the two independent data
sources (Downtown: 11.6% vs 10.0%; OldScandinavia: 0.5% vs 0.5%) — good evidence
the accumulation pipeline in Part 2 is sound. The grid classifier's collapse in
Part 2 is a data-quality artifact (see below), not a second confirmation of
Part 1's terrain-difficulty finding.

---

## Part 1 — Complete/reconstructed point clouds

Data: `schnabel_cython/tartanair_data/<Env>/<Env>_rgb_original.ply` — pre-made
multi-view reconstructions. **Z-up** convention (rooftops sit higher than street
level) — confirmed by geometric reasoning, independent of the non-authoritative
segmented.ply.

Script: `compare_on_complete_cloud.py`

### Config

| Setting | Downtown | OldScandinavia |
|---|---|---|
| Command | `python compare_on_complete_cloud.py --env Downtown --epsilon 0.5 --min_support 200 --cell_size 2.5` | same, `--env OldScandinavia` |
| Points after 0.05m downsample | 646,690 | 645,946 |
| Schnabel epsilon used | 0.5m (0.15m found **zero** shapes at all — see note) | 0.5m |
| Grid cell size | 2.5m (density-driven: ~2.45 pts/m² average, so the project's usual 18cm cells would be almost entirely empty) | 2.5m |

**Note on epsilon:** `epsilon=0.15` (the project's "standard" baseline tolerance)
found zero shapes across the whole Downtown scene — confirmed via direct
`schnabel_ransac.detect()` call, not a script bug. For OldScandinavia, epsilon
sensitivity was checked explicitly (0.5m → 0.80% ground, 1.0m → 0.35%, 2.0m →
0.43%) — loosening epsilon does *not* unlock a large plane there, confirming the
low result is a real terrain property, not a parameter artifact.

### Results

| Metric | Downtown | OldScandinavia |
|---|---|---|
| Schnabel ground points | 74,774 (11.6%) | 3,034 (0.5%) |
| Grid-based ground points | 44,273 (6.8%) | 2,587 (0.4%) |
| Both agree | 32,340 | 57 |
| Only Schnabel | 42,434 | 2,977 |
| Only Grid-based | 11,933 | 2,530 |
| Agree on not-ground | 559,983 | 640,382 |
| **Agreement IoU** | **0.373** | **0.010** |
| Grid: traversable / obstacle / unknown | 27.1% / 16.7% / 56.2% | 1.1% / 1.7% / 97.2% |
| Total grid cells | 16,700 | 32,795 |

### Read

- **Downtown (moderate agreement, 37%):** "Schnabel-only" points are likely areas
  within the one big global plane's loose tolerance band that the grid method's
  local slope/roughness thresholds rejected as not flat/smooth enough locally.
  "Grid-only" points are likely small flat regions (a plaza, a side street at a
  different elevation) sitting just outside Schnabel's single global plane's
  height band, which the grid method still recognizes as flat on its own terms.
  This is exactly the kind of disagreement that motivated the cell-based approach.
- **OldScandinavia (near-zero agreement, 0.01):** both methods essentially fail.
  This terrain doesn't have a large flat area at any reasonable single-plane
  tolerance, and the grid classifier is also starved for data (97.2% of cells
  have too few points to classify at 2.5m resolution). Rough/sloped terrain is a
  genuine hard case for both approaches, not just RANSAC's.

---

## Part 2 — Accumulated per-frame trajectory maps

A different data source: instead of the pre-made complete clouds, this builds a
merged point cloud directly from the **per-frame LiDAR scans** used for RL
training/eval (`data/<Env>/Data_omni/P0000/lidar/*.ply`), transformed into one
shared world frame using each frame's pose.

Scripts: `accumulate_trajectory.py` (builds the map) + `compare_on_trajectory_map.py`
(runs the same comparison + 3-panel viewer, reusing `compare_on_complete_cloud.py`).

| Question | Answer |
|---|---|
| Why does this need pose data at all? | Per-frame point clouds are sensor-local (ego-centric) — confirmed empirically, centroids stay pinned near the origin across every frame regardless of how far the drone moved. Raw concatenation would be meaningless. |
| Where did pose come from? | `pose_lcam_front.txt` (x,y,z,qx,qy,qz,qw, NED) — downloaded via the `meta` modality. `pose` itself is not a valid modality for this Ground dataset variant. |
| Coordinate convention? | Also turned out to be **Z-up** after the pose transform — NOT Z-down/NED like the untransformed per-frame sensor data. Verified directly: the single largest, cleanest horizontal plane (matching the street) sits at the *lowest* Z among horizontal candidates, not the highest. Don't assume conventions carry over between data sources — check empirically each time. |

### Config

| Setting | Downtown | OldScandinavia |
|---|---|---|
| Command | `python compare_on_trajectory_map.py --env Downtown --stride 10 --final_voxel 0.4 --epsilon 0.5 --min_support 200 --cell_size 2.5 --lowest_cluster_band 0.5` | same, `--env OldScandinavia` |
| Frames merged | 113 of 1,125 (every 10th) | 294 of 2,931 (every 10th) |
| Points after merge + 0.4m downsample | 919,875 | 3,611,265 |
| XYZ extent | 351m × 325m × 31m | 442m × 466m × 69m |
| Grid `lowest_cluster_band` | 0.5m (loosened from the 0.15m project default — see read below) | 0.5m |

### Results

| Metric | Downtown | OldScandinavia |
|---|---|---|
| Schnabel ground points | 91,713 (10.0%) | 18,387 (0.5%) |
| Grid-based ground points | 451 (0.0%) | 1,332 (0.0%) |
| **Agreement IoU** | **0.000** | **0.000** |
| Grid: traversable / obstacle / unknown | 0.3% / 23.8% / 76.0% | 0.5% / 24.3% / 75.3% |
| Total grid cells | 11,113 | 18,792 |

### Read

- **Schnabel checks out:** its result on this accumulated map (10.0% / 0.5%)
  closely matches its result on the independently-sourced complete cloud
  (11.6% / 0.5%) for both environments — good evidence the accumulation
  pipeline itself is sound.
- **Grid classifier fails on both — a data-quality issue, not a terrain finding.**
  Diagnosed the root cause on Downtown:
  - With the project's default 0.15m height-band filter, **64.6%** of cells had
    enough horizontal points but lost most of them to the band filter — the same
    physical floor surface lands at slightly different heights across different
    frames' contributions, because this merge has no ICP/registration
    refinement, just a raw pose transform.
  - Loosening the band to 0.5m let more points through, but those points carry
    the same registration noise, which then reads as high local *roughness* —
    pushing most of those cells into "obstacle" instead of "unknown."
  - **Conclusion:** this naive multi-frame merge isn't a clean enough substrate
    for the grid classifier's 5cm-scale roughness threshold. Schnabel tolerates
    it fine (RANSAC's inlier search is inherently noise-robust); the grid
    method's per-cell roughness check is much more sensitive to exactly this
    kind of noise. The real fix is ICP (or similar registration refinement)
    between frames before merging — not further threshold tuning.
