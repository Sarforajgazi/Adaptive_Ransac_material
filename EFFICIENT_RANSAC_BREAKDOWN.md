# Efficient RANSAC — Complete Codebase Breakdown

> Deep technical breakdown of every file, every parameter, every hardcoded constant, and every decision point in `Efficient-RANSAC-for-Point-Cloud-Shape-Detection/`. Written to inform design of an Adaptive RANSAC system.

---

## Table of Contents

1. [File Map](#file-map)
2. [Data Structures](#data-structures)
3. [The Algorithm — Step by Step](#the-algorithm--step-by-step)
4. [All Parameters](#all-parameters)
5. [All Hardcoded Constants](#all-hardcoded-constants)
6. [What the User Must Provide](#what-the-user-must-provide)
7. [What the Algorithm Decides Internally](#what-the-algorithm-decides-internally)
8. [The Scoring System](#the-scoring-system)
9. [The Stopping Criterion](#the-stopping-criterion)
10. [Shape-Specific Details](#shape-specific-details)
11. [Support Libraries](#support-libraries)
12. [Adaptive RANSAC Opportunities](#adaptive-ransac-opportunities)

---

## File Map

```
Efficient-RANSAC-for-Point-Cloud-Shape-Detection/
│
│── CORE ALGORITHM
│   ├── RansacShapeDetector.h/.cpp   ← The main detector class (the brain)
│   ├── Candidate.h                  ← A candidate shape with statistical bounds
│   ├── ScoreComputer.h              ← Gaussian-weighted point-to-shape scoring
│   ├── ScorePrimitiveShapeVisitor.h ← Visitor that walks the octree and scores a shape
│   ├── ScoreAACubeTreeStrategy.h    ← Octree traversal strategy for scoring
│   ├── FlatNormalThreshPointCompatibilityFunc.h ← Point vs. shape compatibility check
│
│── DATA STRUCTURES
│   ├── PointCloud.h/.cpp            ← Container for 3D points + normals
│   ├── Octree.h                     ← Type aliases for the two octree types used
│   ├── IndexIterator.h              ← Indexed iteration utility
│
│── PRIMITIVE SHAPES (one group per shape type)
│   ├── BasePrimitiveShape.h         ← Common interface shared by all shapes
│   ├── PrimitiveShape.h             ← Abstract base class
│   ├── PrimitiveShapeConstructor.h  ← Abstract factory (builds shape from samples)
│   ├── BitmapPrimitiveShape.h/.cpp  ← Base for shapes that use bitmap connectivity
│   │
│   ├── Plane.h/.cpp                         ← Geometric math for a plane
│   ├── PlanePrimitiveShape.h/.cpp           ← Plane as a RANSAC primitive
│   ├── PlanePrimitiveShapeConstructor.h/.cpp ← Builds plane from 1 sample point
│   │
│   ├── Sphere.h/.cpp
│   ├── SpherePrimitiveShape.h/.cpp
│   ├── SpherePrimitiveShapeConstructor.h/.cpp
│   │
│   ├── Cylinder.h/.cpp
│   ├── CylinderPrimitiveShape.h/.cpp
│   ├── CylinderPrimitiveShapeConstructor.h/.cpp
│   │
│   ├── Cone.h/.cpp
│   ├── ConePrimitiveShape.h/.cpp
│   ├── ConePrimitiveShapeConstructor.h/.cpp
│   │
│   ├── Torus.h/.cpp
│   ├── TorusPrimitiveShape.h/.cpp
│   ├── TorusPrimitiveShapeConstructor.h/.cpp
│   │
│   ├── LowStretchSphereParametrization.h/.cpp  ← UV param for sphere bitmap
│   ├── LowStretchTorusParametrization.h/.cpp   ← UV param for torus bitmap
│   ├── SimpleTorusParametrization.h/.cpp
│   ├── Bitmap.h/.cpp                ← 2D bitmap for connected component analysis
│
│── FITTING
│   ├── LevMarFitting.h              ← Levenberg-Marquardt iterative fitting
│   ├── LevMarFunc.h                 ← Function interface for LevMar
│   ├── LevMarLSWeight.h             ← Weighted least-squares for LevMar
│
│── MATH / SPATIAL LIBRARIES
│   ├── GfxTL/                       ← Template-heavy graphics math library
│   │   ├── KdTree.h/.hpp            ← KD-Tree (used for normal estimation)
│   │   ├── AAKdTree.h/.hpp          ← Axis-aligned KD-Tree
│   │   ├── AACubeTree.h/.hpp        ← Axis-aligned cube tree (the octree base)
│   │   ├── MatrixXX.h               ← Matrix operations
│   │   ├── VectorXD.h               ← N-dimensional vector
│   │   ├── Plane.h/.hpp             ← Geometric plane (used in PCA)
│   │   ├── NearestNeighbors.h       ← kNN search
│   │   ├── Covariance.h             ← Covariance matrix computation
│   │   ├── Jacobi.h                 ← Jacobi eigenvalue decomposition
│   │   └── ... (30+ strategy/kernel files)
│   │
│   └── MiscLib/                     ← Small utility library
│       ├── Random.h/.cpp            ← Custom LCG random number generator
│       ├── Vector.h                 ← std::vector wrapper
│       ├── NoShrinkVector.h         ← Vector that never shrinks capacity
│       ├── RefCount.h/.cpp          ← Reference counting base
│       ├── RefCountPtr.h            ← Smart pointer (like shared_ptr)
│       ├── RefCounted.h             ← Wraps any object with ref counting
│       └── Performance.h            ← Timing utilities
│
│── ENTRY POINT
│   ├── main.cpp                     ← Standalone C++ demo (excluded from Cython build)
│   └── basic.h                      ← Vec3f type definition
```

---

## Data Structures

### `Point` (PointCloud.h)

The fundamental unit. Every point in the cloud is one of these:

```cpp
struct Point {
    Vec3f pos;      // 3D position (x, y, z) — float32
    Vec3f normal;   // Surface normal vector — computed by calcNormals()
    size_t index;   // Original input index (only if POINTSWITHINDEX is defined)
};
```

**Critical:** `normal` starts as `(0,0,0)` — it is NOT read from the input file. The algorithm computes it internally via `calcNormals()`. If normals are wrong, everything downstream is wrong.

---

### `PointCloud` (PointCloud.h/.cpp)

A `std::vector<Point>` plus a bounding box:

```cpp
class PointCloud : public MiscLib::Vector<Point> {
    Vec3f m_min, m_max;   // Axis-aligned bounding box
};
```

Key methods:
- `setBBox(min, max)` — **must be called manually** before `Detect()`
- `calcNormals(radius, kNN, maxTries)` — runs PCA over kNN neighbours for each point
- `getScale()` — returns `max(dx, dy, dz)` — used to set relative thresholds

---

### `Candidate` (Candidate.h)

A hypothesis: a shape fit to a sample, with statistical upper/lower bounds on how many inliers it might have across the full point cloud.

```
Candidate {
    PrimitiveShape* m_shape      ← the fitted shape object
    float m_lowerBound           ← statistical lower bound on inlier count
    float m_upperBound           ← statistical upper bound on inlier count
    size_t m_subset              ← how many octree subsets have been scored
    size_t m_level               ← octree level at which it was sampled
    bool m_hasConnectedComponent ← whether connectivity filtering was applied
    size_t m_score               ← actual inlier count seen so far
}

ExpectedValue() = (lowerBound + upperBound) / 2
```

The bounds use **hypergeometric distribution** statistics — not a confidence interval in the frequentist sense, but a mathematically derived range on how many inliers the shape would have if scored against the full cloud, given what was observed in a subset.

---

## The Algorithm — Step by Step

### Phase 0: Setup (called once by user)

```cpp
// 1. Fill PointCloud
pc.push_back(Point(Vec3f(x, y, z)));

// 2. Set bounding box (REQUIRED — no auto-detection)
pc.setBBox(Vec3f(min_x, min_y, min_z), Vec3f(max_x, max_y, max_z));

// 3. Compute surface normals
pc.calcNormals(radius=3.0, kNN=20);

// 4. Configure options
RansacShapeDetector::Options opts;
opts.m_epsilon       = 0.01f;
opts.m_normalThresh  = 0.9f;
opts.m_minSupport    = 500;
opts.m_bitmapEpsilon = 0.02f;
opts.m_probability   = 0.001f;

// 5. Add shape constructors
detector.Add(new PlanePrimitiveShapeConstructor());

// 6. Run
detector.Detect(pc, 0, pc.size(), &shapes);
```

---

### Phase 1: Build Octree Hierarchy

Inside `Detect()`, the algorithm builds a **stratified multi-level octree**:

```
Number of subsets = max(floor(log2(N)) - 9, 2)

For N=48,000:  log2(48000) ≈ 15.5  →  subsets = max(6, 2) = 6
For N=10,000:  log2(10000) ≈ 13.3  →  subsets = max(4, 2) = 4
```

Each subset is a **random stratified sample** of the point cloud, half the size of the next:
- Subset 5: full cloud (48,000 pts)
- Subset 4: 24,000 pts (random half)
- Subset 3: 12,000 pts
- Subset 2: 6,000 pts
- Subset 1: 3,000 pts
- Subset 0: 1,500 pts

Each subset has its own octree. Additionally one **global indexed octree** covers the full cloud.

**Octree parameters (hardcoded):**
```cpp
octree.MaxBucketSize() = 20;        // max points per leaf
octree.MaxSubdivisionLevel() = 10;  // max depth
```

---

### Phase 2: Initialize Sampling Level Weights

Each octree level gets equal initial sampling probability:
```
sampleLevelProbability[i] = 1.0 / num_levels   (uniform initially)
```

These weights are **updated adaptively** during the run: levels that produce good candidates get higher probability. This is the algorithm's internal adaptive mechanism.

---

### Phase 3: Main Loop (runs until stopping criterion met)

```
REPEAT:
    1. GenerateCandidates() — draw 200 sample sets, fit shapes, quick-score on smallest subset
    2. FindBestCandidate()  — progressively score best candidates on larger subsets
    3. If a good candidate found:
         - Run least-squares fitting (up to 3 iterations)
         - Run connected-component filtering
         - Accept shape: remove its points from the cloud
         - Update sampling weights
         - Housekeeping: rebuild octrees if >25% of points assigned
    4. Else: increment try counter
UNTIL: stopping criterion satisfied
```

---

### Phase 4: Generate 200 Candidates per Round

In `GenerateCandidates()`, per round (parallelised with OpenMP):

```
FOR 200 iterations:
    1. Pick an octree level (based on current sampling weights)
    2. DrawSamplesStratified(): pick m_reqSamples points from that level's cell
    3. For each registered shape constructor:
         a. Call constructor.Construct(samplePoints) → try to fit shape
         b. Verify: all sample points must pass ε and normal_thresh tests
         c. Quick-score on smallest octree subset
         d. If UpperBound >= m_minSupport: add to candidate list
```

**Hardcoded:** `200` candidates generated per round.

---

### Phase 5: FindBestCandidate

Progressively scores top candidates on increasingly large subsets:

```
1. Sort candidates by ExpectedValue (descending)
2. For each promising candidate:
     ImproveBounds(): score on next larger subset
     Update lowerBound and upperBound
3. Stop when: FailureProbability(best) <= m_probability
4. Final: ConnectedComponent() filtering on the winner
```

---

### Phase 6: Acceptance and Point Removal

When a shape is accepted:
- Its inlier points are **moved to the end** of the PointCloud array
- The `shapeIndex[]` array marks those points as assigned
- `drawnCandidates` is scaled down: `drawnCandidates *= (1 - removed/total)^3`
- All remaining candidates have their bounds **recomputed** against the now-smaller cloud

---

### Phase 7: Housekeeping (triggered when >25% of points are assigned)

```cpp
if(numInvalid > currentSize / 4) {
    // physically compact the array (remove assigned points)
    // rebuild all octrees
    // reindex all remaining candidates
}
```

**Hardcoded threshold:** `currentSize / 4` (25%)

---

### Phase 8: Least-Squares Fitting

After a candidate is accepted, it runs **up to 3 LS refinement iterations**:

```cpp
do {
    shape = Fit(shape, pc, inlier_indices);   // LeastSquaresFit
    newScore = GlobalWeightedScore(shape);
    if(newScore > oldScore) accept_fit;
    ++fittingIter;
} while(newScore > oldScore && fittingIter < 3);
```

**Hardcoded:** `fittingIter < 3` — maximum 3 least-squares refinement iterations.

---

## All Parameters

### User-Configurable (via `RansacShapeDetector::Options`)

| Parameter | Type | Default | What it controls |
|---|---|---|---|
| `m_epsilon` | `float` | `0.01f` | **Distance threshold.** A point is an inlier if its distance to the shape is ≤ `3 * m_epsilon` (global scoring) or ≤ `m_epsilon` (subset scoring). **Critical: the global scorer uses 3× this value internally.** |
| `m_normalThresh` | `float` | `0.95f` | **Normal angle threshold.** cos(max_angle). A point is an inlier only if `|point_normal · shape_normal| > m_normalThresh`. `0.9 ≈ 26°`, `0.95 ≈ 18°`. |
| `m_minSupport` | `unsigned int` | `100` | **Minimum inlier count.** A shape is only accepted if it has ≥ `m_minSupport` inlier points. Also controls: stopping criterion, candidate pruning, stat bucket size. |
| `m_bitmapEpsilon` | `float` | `0.01f` | **Bitmap cell resolution** for connected-component analysis. Typically `2 × m_epsilon`. Controls how large a "cell" is when checking if inliers form a spatially connected region. |
| `m_probability` | `float` | `0.001f` | **Miss probability.** The algorithm stops when the probability of missing a shape with ≥ `m_minSupport` inliers drops below this value. Lower = more thorough but slower. |
| `m_fitting` | `enum` | `LS_FITTING` | Whether to run least-squares refinement after acceptance. Options: `NO_FITTING`, `LS_FITTING`. |

---

### User-Configurable (via `calcNormals`)

| Parameter | Type | Default | What it controls |
|---|---|---|---|
| `radius` | `float` | — | Neighbourhood radius for normal estimation (not a hard cutoff — kNN is used instead, but radius affects Gaussian weighting) |
| `kNN` | `unsigned int` | `20` | K nearest neighbours used for PCA normal estimation per point |
| `maxTries` | `unsigned int` | `100` | Only relevant for the unused `LMS_NORMALS` mode. For `PCA_NORMALS` (the active mode), this parameter has no effect. |

---

### User-Provided (via `detector.Add()`)

Which shape types to detect:

| Constructor | Min samples | Shape |
|---|---|---|
| `PlanePrimitiveShapeConstructor` | 1 | Infinite plane |
| `SpherePrimitiveShapeConstructor` | 2 | Sphere |
| `CylinderPrimitiveShapeConstructor` | 2 | Cylinder |
| `ConePrimitiveShapeConstructor` | 2 | Cone |
| `TorusPrimitiveShapeConstructor` | 3 | Torus |

`m_reqSamples` is set to the maximum `RequiredSamples` across all added constructors.

---

### User-Provided (via `setBBox`)

```cpp
pc.setBBox(Vec3f(min_x, min_y, min_z), Vec3f(max_x, max_y, max_z));
```

**This is mandatory.** The octree cannot be built without it. The algorithm does not auto-compute the bounding box. In our bridge.cpp, we compute it from the data and add a 1% padding.

---

## All Hardcoded Constants

These values are **baked into the C++ code** — not configurable from outside:

| Constant | Value | Location | What it does |
|---|---|---|---|
| `200` | 200 | `RansacShapeDetector.cpp:107` | Candidates generated per main loop iteration |
| `3×` | 3.0 | `RansacShapeDetector.cpp:464` | Global scorer uses `3 * m_epsilon` as threshold |
| `20` | 20 | `RansacShapeDetector.cpp:501,516` | Octree max bucket size (leaf node capacity) |
| `10` | 10 | `RansacShapeDetector.cpp:502,517` | Octree max subdivision level (tree depth) |
| `1/4` | 0.25 | `RansacShapeDetector.cpp:672` | Housekeeping trigger: rebuild if >25% points removed |
| `3` | 3 | `RansacShapeDetector.cpp:650` | Max LS fitting iterations per accepted shape |
| `0.8×` | 0.8 | `RansacShapeDetector.cpp:584` | Second-pass min size = 80% of first candidate size |
| `500` | 500 | `RansacShapeDetector.cpp:702` | Min subset size before merging small subsets |
| `1.4×` | 1.4 | `RansacShapeDetector.cpp:597` | Level weight update: include candidates within 1.4× of best |
| `0.9 / 0.1` | 0.9, 0.1 | `RansacShapeDetector.cpp:79` | Level weight mixing: 90% new + 10% uniform baseline |
| `0.5` | 0.5 | `RansacShapeDetector.cpp:605` | Weight update factor when first candidate found |
| `1.21` | 1.21 | `RansacShapeDetector.cpp:54` | Stat bucket size (logarithmic spacing base) |
| `20` | 20 | `m_maxCandTries=20` | Max retries when drawing stratified samples |
| `40` | 40 | `RansacShapeDetector.cpp:929` | Max retries when picking second/third sample point |
| `9` | 9 | `RansacShapeDetector.cpp:468` | `subsets = floor(log2(N)) - 9` |
| `2` | 2 | `RansacShapeDetector.cpp:468` | Minimum subsets regardless of N |
| `8` | 8 | `PointCloud.cpp:207` | Min neighbours for a valid PCA normal estimate |
| `0.3` | 0.3 | `Candidate.h:191` | Connected component decision threshold (30%) |
| `95%` | 0.95 | `Candidate.h:241` | Recompute threshold: if <95% of inliers still valid, full rebuild |
| `1/9 × eps²` | σ = ε/3 | `ScoreComputer.h:12` | Gaussian weighting sigma = ε/3 |

---

## What the User Must Provide

1. **The point cloud** — raw (x, y, z) floats. No normals needed from user.
2. **The bounding box** — `pc.setBBox(min, max)`. Without this, the octree cannot be built.
3. **`m_epsilon`** — the single most important parameter. In absolute units (metres). **Note: internally, the global score uses 3×ε.**
4. **`m_normalThresh`** — typically 0.85–0.95 depending on how noisy the normals are.
5. **`m_minSupport`** — minimum number of points to consider a shape real.
6. **`m_bitmapEpsilon`** — typically set to `2 × m_epsilon`.
7. **`m_probability`** — typically 0.001 for thorough search, 0.01 for faster search.
8. **Which shapes** — which constructors to add via `detector.Add(...)`.
9. **Normal estimation radius and kNN** — for `pc.calcNormals(radius, kNN)`.

---

## What the Algorithm Decides Internally

Everything below is computed automatically — **not exposed to the user:**

| Decision | How it's made |
|---|---|
| Number of octree subsets | `max(floor(log2(N)) - 9, 2)` |
| Which octree level to sample from | Adaptive: updated based on which levels yield the best candidates |
| How many candidates to generate per round | Fixed: 200 |
| When to stop generating candidates | When `FailureProbability(minSupport, n, drawnCandidates, levels) ≤ m_probability` |
| Whether a candidate is good enough | Statistical bounds test + connectivity filter |
| When to run housekeeping | When >25% of remaining points have been assigned |
| Number of LS fitting iterations | Up to 3, stops when score stops improving |
| Subset merging | When a subset shrinks below 500 points |
| Level weight updates | After each accepted shape (0.5 mixing factor) + after each round (0.9/0.1 smoothing) |

---

## The Scoring System

### Point Compatibility Check (`FlatNormalThreshPointCompatibilityFunc`)

A point is compatible with a shape if **both** conditions hold:
```
distance(point, shape) < epsilon        ← position test
|dot(point_normal, shape_normal)| > normalThresh  ← orientation test
```

### Two Scoring Contexts

| Context | Epsilon used | Where |
|---|---|---|
| Subset scoring (fast, approximate) | `1 × m_epsilon` | During candidate generation |
| Global scoring (exact, final) | `3 × m_epsilon` | After a winner is found |

These are two distinct **phases** of the RANSAC loop, each with a different tolerance:

#### Subset Scoring (candidate evaluation phase)

During the main loop the algorithm does **not** check every point in the cloud on every iteration — that would be too slow. Instead it builds **stratified random subsets** (octree sub-samples at decreasing densities, roughly halving each level). A candidate shape is scored by counting how many points in the current subset fall within `1 × m_epsilon` of the shape.

This is the **tight, conservative** check used repeatedly while candidates compete against each other. The threshold is strict so weak candidates are eliminated quickly.

```cpp
// RansacShapeDetector.cpp:461
subsetScoreVisitor(m_options.m_epsilon, ...)   // 1× epsilon
```

#### Global Scoring (winner confirmation phase)

Once a candidate "wins" — its expected inlier count survives all competing subsets — the algorithm does **one final sweep over the entire point cloud** to confirm it and collect all inliers. This sweep uses `3 × m_epsilon` as the threshold.

```cpp
// RansacShapeDetector.cpp:464
globalScoreVisitor(3 * m_options.m_epsilon, ...)   // 3× epsilon
```

#### Why the 3× difference?

1. **Recall** — after subset-based filtering has already identified the correct shape, you want to capture all real inliers including slightly noisier points at the fringes of the surface.
2. **Robustness** — subsets are sub-sampled, so some near-boundary points may have been missed during the tight competitive phase. The looser final sweep picks them up.

#### Practical implication

If you set `m_epsilon = 0.01`, points up to **0.03** distance from the shape are counted as inliers in the final output. The inlier set you get back reflects the `3 × m_epsilon` threshold, not the `m_epsilon` you set. This matters if you use those inliers for a subsequent least-squares fit or downstream processing.

**The 3× multiplier is hardcoded and non-obvious.** It means a point can be 3× further than your ε and still be counted as an inlier in the final global score.

### Gaussian Weighting (`ScoreComputer.h`)

Each inlier point contributes a **weighted score**, not just +1:

```
weight(d, ε) = exp(-d² / (2/9 · ε²))
             = exp(-d² / (2 · (ε/3)²))
```

This is a Gaussian with `σ = ε/3`. Points exactly on the surface (d=0) contribute 1.0. Points at distance ε contribute `exp(-4.5) ≈ 0.011`.

### Connected-Component Filtering (`m_bitmapEpsilon`)

After global scoring collects all inliers, the algorithm does **one more check**: it verifies that those inliers actually form a single spatially contiguous patch, not a scattered cloud of random points that all happened to be near the same plane/cylinder.

It does this using a 2D bitmap:

#### Step 1 — Project inliers into 2D parameter space

Each shape type has a natural 2D parameterisation:
- Plane → UV coordinates on the plane surface
- Cylinder → (arc angle, height along axis)
- Sphere → (longitude, latitude)

Every inlier point is projected into this 2D space.

#### Step 2 — Build a grid (the bitmap)

The 2D parameter space is divided into a regular grid where each cell is **`m_bitmapEpsilon × m_bitmapEpsilon`** in size. A cell is marked **occupied (1)** if at least one inlier point falls into it, otherwise **empty (0)**.

```cpp
// PlanePrimitiveShape.cpp:197–206
*uextent = ceil((bbox.Max[0] - bbox.Min[0]) / epsilon) + 1;
*vextent = ceil((bbox.Max[1] - bbox.Min[1]) / epsilon) + 1;
// each point maps to cell:
cell.u = floor((param.u - bbox.Min[0]) / epsilon);
cell.v = floor((param.v - bbox.Min[1]) / epsilon);
```

#### Step 3 — Run connected-component labeling

A standard 2D connected-component algorithm runs on the bitmap — adjacent occupied cells (4-connected or 8-connected) form one component. A **morphological closing** (dilate then erode) is applied first to fill tiny gaps.

#### Step 4 — Keep only the largest component

If the bitmap has multiple disconnected occupied regions, only the **largest connected component** is kept. All inlier points that mapped into smaller / isolated regions are discarded.

```cpp
// Candidate.cpp:83–86
void Candidate::ConnectedComponent(const PointCloud &pc, float bitmapEpsilon, ...) {
    size_t connectedSize = m_shape->ConnectedComponent(pc, bitmapEpsilon, m_indices, ...);
}
```

#### What `m_bitmapEpsilon` controls

It is the **cell size of the grid**, which is the **gap tolerance** for connectivity:

| `m_bitmapEpsilon` (cell size) | Effect |
|---|---|
| Small (fine grid) | A small physical gap between two nearby inlier clusters → they land in separate cells → treated as disconnected |
| Large (coarse grid) | Two clusters with a gap → both land in the same cell or adjacent cells → treated as connected |

**Typical rule:** set `m_bitmapEpsilon = 2 × m_epsilon`. This means a gap of up to ~2× the distance tolerance is tolerated before splitting a patch into two components.

#### Why this matters

Without this step, a shape like a plane could get a high inlier count by collecting points from two separate parallel walls that happen to be coplanar. The connected-component filter forces the algorithm to pick only one contiguous surface patch and discard the rest.

### Statistical Bounds (Candidate.h)

The algorithm never scores all N points immediately. It scores on subsets and uses **hypergeometric distribution bounds** to estimate the range `[lowerBound, upperBound]` of inliers in the full cloud:

```
Induct formula (from the code):
  nI = sampleSize × totalCorrectSize
  dev = sqrt(nI × (totalSize - sampleSize) × (totalSize - totalCorrectSize) / (totalSize - 1))
  lower = (nI - dev) / totalSize
  upper = (nI + dev) / totalSize
```

`ExpectedValue = (lower + upper) / 2` — used for all candidate comparisons.

---

## The Stopping Criterion

The algorithm stops when it is statistically confident that no undetected shape with ≥ `m_minSupport` inliers exists:

```
CandidateFailureProbability(s, n, t, L) =
    min(1, (1 - s/(n·L·2^(reqSamples-1)))^t)

where:
  s = minSupport (or candidate size)
  n = current unassigned points
  t = drawnCandidates (total candidates generated so far)
  L = number of octree levels
  reqSamples = max samples needed by any registered shape constructor
```

Stop when `CandidateFailureProbability(m_minSupport, n, t, L) ≤ m_probability`.

**Translation:** "We've drawn enough candidates that if a shape with minSupport inliers existed, we would have found it with probability ≥ (1 - m_probability)."

**Key insight for adaptive RANSAC:** This formula is what you would replace with a learned stopping policy. Currently it's a closed-form statistical expression with no learning.

---

## Shape-Specific Details

### Plane
- **Min samples:** 1 point + its normal (degenerate — can fit from a single point because normal defines orientation)
- **Parameters stored:** `[normal_x, normal_y, normal_z, dist_to_origin]` — 4 floats
- **Fit:** LeastSquaresFit using PCA on inlier point set
- **Bitmap:** 2D rectangular grid in the plane's local coordinate system

### Sphere
- **Min samples:** 2 points + normals
- **Parameters stored:** `[center_x, center_y, center_z, radius]` — 4 floats
- **Fit:** Levenberg-Marquardt or least squares
- **Bitmap:** Spherical UV parameterization (`LowStretchSphereParametrization`)

### Cylinder
- **Min samples:** 2 points + normals
- **Parameters stored:** `[axis_point_x, y, z, axis_dir_x, y, z, radius]` — 7 floats
- **Bitmap:** Unrolled cylinder surface

### Cone
- **Min samples:** 2 points + normals
- **Parameters stored:** `[apex_x, y, z, axis_x, y, z, half_angle]` — 7 floats

### Torus
- **Min samples:** 3 points + normals
- **Parameters stored:** `[center_x, y, z, axis_x, y, z, major_radius, minor_radius]` — 8 floats
- **Bitmap:** `LowStretchTorusParametrization` or `SimpleTorusParametrization`

---

## Support Libraries

### `GfxTL/` — The Spatial Data Structure Engine

Everything is built from composable C++ template strategies (policy-based design):

| File | Role |
|---|---|
| `KdTree.h/.hpp` | KD-Tree used for `calcNormals()` (kNN queries) |
| `AACubeTree.h/.hpp` | The octree base (axis-aligned cube tree) |
| `AAKdTree.h` | Axis-aligned KD-Tree variant |
| `MatrixXX.h` | Matrix ops (used in PCA, Jacobi) |
| `Jacobi.h` | Eigenvalue decomposition (used in normal PCA) |
| `Covariance.h` | Covariance matrix from point set (used in PCA normals) |
| `Plane.h` | Geometric plane fit (used in `calcNormals`) |
| `NearestNeighbors.h` | kNN result container |
| `L2Norm.h` | Euclidean distance metric |
| `VectorXD.h` | N-dimensional float vector |
| Strategy files | `BucketSizeMaxLevel...`, `CellRange...`, `MaxIntervalSplitting...` etc. — composable tree strategies |

### `MiscLib/` — Utilities

| File | Role |
|---|---|
| `Random.h/.cpp` | LCG random number generator (`rn_rand()`, `rn_setseed()`) — seeded with `time(NULL)` |
| `Vector.h` | Wrapper around `std::vector` |
| `NoShrinkVector.h` | Vector that never deallocates on `resize()` — used for hot paths |
| `RefCount.h/.cpp` | Intrusive reference counting base class |
| `RefCountPtr.h` | Smart pointer equivalent (like `shared_ptr`) |
| `RefCounted.h` | Wraps any type `T` to add reference counting |
| `Performance.h` | Timing utilities |

---

## Adaptive RANSAC Opportunities

Based on this complete analysis, here are exactly the places where adaptive control can be inserted:

### 1. `m_epsilon` — The Biggest Lever

**Current:** Fixed float, same for every point in every frame.

**Adaptive opportunity:** Predict ε per scan from scan statistics (height variance, point density, sensor noise estimate). This is the single parameter with the most impact on segmentation quality.

**Note the 3× internal multiplier:** Your adaptive agent must account for this. If you set `ε = 0.1m`, actual global scoring uses `0.3m`. Consider exposing the effective threshold directly.

---

### 2. `m_minSupport` — Controls What Counts as "Real"

**Current:** Fixed integer, same for every scan.

**Adaptive opportunity:** Scale with scan density. Sparse scans (600 pts/frame) need `min_support=50`. Dense scans (50k pts/frame) can use `min_support=500`.

---

### 3. `200` Candidates Per Round — Fixed Throughput

**Current:** Always generates exactly 200 candidates per round.

**Adaptive opportunity:** In easy scans (clear flat ground), 50 candidates might be enough. In noisy scans (cluttered indoor scene), 500 might be needed. An agent could learn to adjust this.

---

### 4. Sampling Level Weights — Already Partially Adaptive

**Current:** Uniform initially, updated with a **fixed** 0.5 mixing factor and 0.9/0.1 smoothing.

**Adaptive opportunity:** The mixing factors (`0.5`, `0.9`, `0.1`) and the "include within 1.4×" threshold are hardcoded. A learned update rule here could significantly speed up convergence.

---

### 5. The Stopping Criterion — The Key RL Target

**Current:** Closed-form statistical formula. Stops when the math says so.

**Adaptive opportunity (the core RL novelty):** Replace the stopping criterion with a learned policy. The agent observes: current best candidate quality, inlier ratio so far, number of candidates drawn, time elapsed — and decides stop/continue. This is exactly the sequential decision-making described in the RL adaptive RANSAC idea.

---

### 6. `calcNormals(radius, kNN)` — Upstream of Everything

**Current:** Same radius and kNN for every scan.

**Adaptive opportunity:** In dense scans, small kNN (8–12) gives sharp normals. In sparse scans, larger kNN (20–40) is needed to get stable PCA. An agent could predict optimal kNN from local density estimates.

#### Why kNN and `m_normalThresh` should be adapted separately (and kNN last)

These two parameters are coupled — `kNN` sets the quality ceiling of the estimated normals, and `m_normalThresh` decides how strictly those normals are judged during scoring. Changing one changes what the other effectively does.

However, the more important reason to keep them separate in an initial RL system is **computational cost**:

| Parameter | When it runs | Cost to change |
|---|---|---|
| `kNN` | `calcNormals()` — runs **once, before** `Detect()`, over every point | Re-runs full normal estimation on the entire cloud — expensive |
| `m_normalThresh` | Evaluated per-point **inside** `Detect()` | Changing it is just changing a float — free |

If your RL agent adapts `kNN`, it must re-run `calcNormals()` every time it changes — rebuilding the KD-tree and recomputing PCA for every point on every episode. This breaks the fast adapt-and-detect loop you need for RL to be practical.

`m_normalThresh` has no such cost. It can be changed between runs, or even between RANSAC iterations if needed, with zero overhead.

**Recommended approach:**
1. **Phase 1:** Keep `kNN` fixed (e.g. 20). Let the RL agent adapt `m_epsilon`, `m_normalThresh`, `m_minSupport`, and the stop/continue decision. This is already a rich action space.
2. **Phase 2:** Once Phase 1 works, explore making `kNN` adaptive — but treat it as a slow, expensive outer loop (e.g. updated between scans, not between iterations), not a per-step action.

The coupling argument (adapting both simultaneously confuses the agent) is real but secondary. The dominant reason is runtime: normal re-estimation is not cheap enough to be an RL action at episode frequency.

---

### 7. Connected Component Bitmap Epsilon

**Current:** `bitmapEpsilon = 2 × ε` (typically). Hardcoded relationship.

**Adaptive opportunity:** In cluttered scenes with many small planar patches, a smaller bitmap epsilon gives tighter connectivity. In open outdoor scenes, a larger value is better. This could be an agent action.

---

### 8. Max LS Fitting Iterations (`3`)

**Current:** Always runs up to 3 iterations.

**Adaptive opportunity:** For a simple flat ground plane in an easy frame, 1 iteration is sufficient and faster. For a noisy scan with a tilted ground, 3+ iterations might be needed.

---

### Summary Table for Adaptive Agent Design

| Parameter | Current value | RL action? | Why |
|---|---|---|---|
| `m_epsilon` | Fixed (e.g. 0.3m) | Yes — continuous | Most impactful parameter |
| `m_minSupport` | Fixed (e.g. 300) | Yes — discrete levels | Depends on scan density |
| `kNN` (normals) | Fixed (20) | Yes — discrete | Depends on local density |
| Candidates per round (200) | Fixed | Yes — discrete | Throughput vs. quality trade-off |
| Stop/continue decision | Statistical formula | Yes — binary | The core RL novelty |
| Bitmap epsilon | `2×ε` | Maybe | Secondary effect |
| Max LS iterations (3) | Fixed | Maybe | Small effect |
| Sampling level weights | Adaptive (fixed rule) | Maybe — replace rule | Already partially adaptive |
| Normal radius | Follows kNN | Maybe — jointly | Coupled with kNN |
| m_normalThresh | Fixed (e.g. 0.95) | Yes — continuous | Coupled with m_epsilon; scan-density and surface-roughness dependent |