# Baseline Benchmark Configuration Reference

## Overview
Three fixed-parameter baselines were evaluated across all 9 TartanAir datasets
to establish a performance floor before RL training. All baselines run on
**raw, unfiltered point clouds** (no voxel downsampling) to represent the
worst-case fixed-parameter scenario.

---

## Baseline Configurations

| Parameter | Strict | Standard | Loose |
|---|---|---|---|
| `epsilon` | **0.10 m** | **0.15 m** | **0.25 m** |
| `min_support` | **800** | **500** | **200** |
| `normal_threshold` | **0.90** | **0.85** | **0.80** |
| `kNN` | 20 | 20 | 20 |
| `bitmap_epsilon` | Auto (2 × ε) | Auto (2 × ε) | Auto (2 × ε) |
| `probability` | 0.001 | 0.001 | 0.001 |
| `relative_epsilon` | False (absolute) | False (absolute) | False (absolute) |
| Point cloud input | Raw (no downsampling) | Raw (no downsampling) | Raw (no downsampling) |

---

## Parameter Justification

### Strict (epsilon=0.10, min_support=800, normal_threshold=0.90)
- **Purpose:** Conservative detection. Enforces near-perfect flatness and
  requires a large number of inlier points before accepting a plane.
- **Expected behaviour:** Works well on smooth indoor floors (Office,
  Hospital, Supermarket). Will systematically fail on outdoor terrain
  (OldScandinavia, SeasonalForest) where the ground is naturally bumpy.

### Standard (epsilon=0.15, min_support=500, normal_threshold=0.85)
- **Purpose:** The main benchmark. Represents the industry-standard
  parameters widely used in robotics and 3D reconstruction literature.
- **Expected behaviour:** Moderate performance across all environments.
  Acts as the primary comparison point for the RL agent.

### Loose (epsilon=0.25, min_support=200, normal_threshold=0.80)
- **Purpose:** Aggressive detection. Tolerates more roughness and accepts
  smaller planes.
- **Expected behaviour:** Works better on outdoor/rough terrain but will
  produce false positives indoors (walls and furniture incorrectly labelled
  as ground).

---

## Datasets Evaluated

| # | Dataset | Frames | Environment Type |
|---|---|---|---|
| 1 | Downtown | 1125 | Urban outdoor |
| 2 | Hospital | 981 | Indoor |
| 3 | OldScandinavia | 2482 | Outdoor, rough terrain |
| 4 | OldTownFall | 1125 | Urban outdoor, autumn |
| 5 | Office | 678 | Indoor, flat floor |
| 6 | SeasonalForestAutumn | 1125 | Outdoor, forest |
| 7 | SeasonalForestSpring | 1125 | Outdoor, forest |
| 8 | Sewerage | 1045 | Underground, narrow |
| 9 | Supermarket | 761 | Indoor, large open space |
| **Total** | | **10,447** | |

---

## Output CSV Naming Convention

Each run produces one CSV per dataset, tagged with the mode name:

```
logs/
├── Downtown_standard.csv
├── Downtown_strict.csv
├── Downtown_loose.csv
├── Hospital_standard.csv
├── Hospital_strict.csv
...
```

## CSV Column Reference

| Column | Description |
|---|---|
| `frame_id` | Filename of the LiDAR frame |
| `epsilon` | Distance threshold used (metres) |
| `min_support` | Minimum inlier count required |
| `normal_threshold` | Cosine similarity threshold for horizontality |
| `runtime` | Wall-clock time for RANSAC to complete (seconds) |
| `reward` | Computed RL reward signal |
| `steps_used` | Number of RANSAC iterations used |
| `inlier_ratio` | Fraction of total points classified as ground (0–1) |
| `plane_normal` | Z-component of detected plane normal (1.0 = perfectly flat) |
| `residual` | Mean Absolute Error of inlier points to fitted plane (metres) |

---

## Run Status

| Baseline | Status | Duration |
|---|---|---|
| Standard | ✅ Complete | 128.82 minutes |
| Strict | 🔄 Running | ~2 hours |
| Loose | ⏳ Pending | ~2 hours |
