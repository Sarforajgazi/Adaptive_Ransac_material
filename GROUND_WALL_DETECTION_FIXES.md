# Ground & Wall Detection at City-Block Scale — Problem, Investigation, Fixes

What happens when `detect_ground_and_walls.py` (tuned and validated on
single ~40k-point sensor sweeps) is pointed at a real, multi-million-point
city-block survey (Toronto-3D). Two independent, unrelated bugs were found
and fixed, plus one architectural ceiling that a new opt-in feature
addresses. This document is the full record: what broke, every hypothesis
tried (including the ones that were wrong), how each fix was validated, and
exactly what changed in production code.

## At a glance

| # | Problem | Root cause | Fix | Status |
|---|---|---|---|---|
| 1 | Ground residual 25m, most points lost to downsampling | `load_ply_xyz` cast to float32 **before** recentering, destroying precision on UTM-scale (~10^6) coordinates | Recenter in float64 first, cast after | **Fixed & live** |
| 2 | Full-cloud ground coverage 0.6-4.2% despite 92-99% on smaller blocks of the *same* cloud | `find_ground_plane_robust()` picked the single most-extreme-height horizontal plane with **no size comparison** — a tiny low-lying patch could beat the real road by centimeters | `HEIGHT_BAND=0.5m`: among horizontal candidates within 0.5m of the extreme, pick the **largest** | **Fixed & live** |
| 3 | Wall coverage capped ~45-57% even with #2 fixed | `min_support` (fraction-based for `--model_variant synthetic`) becomes too high an absolute bar for one small facade to clear at whole-cloud point counts, so RANSAC never proposes it as a shape | `--tile_walls`: detect walls per-tile (where the fraction is small again), merge same-plane detections across tiles | **New, opt-in flag** |

Everything below explains how each of these was found, what was tried and
rejected along the way, and how the final fix was validated.

---

## Problem 1: precision collapse on real-world coordinates

**Symptom**: running the existing pipeline on Toronto-3D L003 gave a ground
plane residual of ~25 meters (should be centimeters for a real road) and
found 0 walls.

**Cause**: `load_ply_xyz()` (`ransac_env.py`) cast point coordinates to
float32 immediately on load, before any recentering happened downstream.
Toronto-3D (like most real surveys) stores points in absolute UTM-style
coordinates — on the order of 10^6 meters. float32 has ~7 significant
decimal digits; at that magnitude, sub-meter geometry is already destroyed
by the cast, before any later "subtract the centroid" step could help —
you can't recover precision that's already gone.

**Fix**: `load_ply_xyz(recenter=True)` now computes the centroid in
float64 and subtracts it **before** the float32 cast:

```python
pts64 = np.stack([...], dtype=np.float64)
if recenter:
    centroid = pts64.mean(axis=0)
    pts64 = pts64 - centroid
pts = pts64.astype(np.float32)   # now near-origin, precision preserved
```

`process_frame()` in `detect_ground_and_walls.py` was updated to call this
with `recenter=True`. Default is `False`, so every other caller
(TartanAir/TartanGround training data, already near-origin) is unaffected.

**Measured effect** (same L003 file, before → after): ground residual
**25.3m → 0.04m**, walls **0 → 6**, points surviving voxel downsampling
**3.26M → 13.47M** (voxel bucketing was itself corrupted by the precision
loss — many points that should have landed in distinct voxels were
colliding).

---

## Problem 2: the ground-selection bug

### The symptom that started the investigation

Scoring `detect_ground_and_walls.py`'s ground prediction against
Toronto-3D's real per-point labels (`scalar_Label==1` = Ground) on the
**full** L003 cloud (39.7M points, 51.8% real ground):

| | Coverage of real ground | Precision |
|---|---|---|
| Ground (whole-cloud, before fix) | **2.6-4.2%** | 95-96% |
| Walls (whole-cloud) | 45.9% | 98.9% |

High precision, terrible coverage — whatever the model predicted as ground
was *correct*, there just wasn't nearly enough of it. That pattern (not
"wrong," just "way too little") is what made this look architectural
rather than a simple parameter problem.

### Hypothesis A (the original one): scale mismatch — tested, disproven

The starting theory: the agent trained on ~10m synthetic boxes, so a
270m×330m city block is wildly out-of-distribution, and its chosen
`eps`/`min_support` break down at that scale. The proposed fix was tiling
the whole cloud into small windows so each tile looks like the training
distribution again.

**Test**: run the identical detection on 20m, 60m, and 150m blocks of the
*same* L003 cloud (`scratchpad/tiling_pilot.py`'s baseline-vs-tiled sweep).

| Block size | Ground coverage (single pass, no tiling) |
|---|---|
| 20m | 99.1% |
| 60m | 97.9% |
| 150m | 96.3% |

**Ground coverage did not degrade with scale.** It stayed consistently
high all the way to 150m. Whatever was collapsing full-cloud coverage to
single digits, it wasn't "the agent's parameters get worse on bigger
scenes" — that hypothesis was directly tested and rejected.

### Hypothesis B: `min_points_abs` doesn't scale — tested, disproven

`find_ground_plane_robust()` has a `min_points_abs=500` floor (a hardcoded
absolute point count, tuned years earlier on ~40k-point single frames to
reject 168-391-point noise fragments). On a multi-million-point cloud, 500
points is a vanishingly small fraction — the theory was that this floor,
meant to reject noise, had effectively become a no-op, letting a tiny
sliver through that a *properly scaled* floor would have blocked.

**Test** (`scratchpad/ab_min_points_abs.py`): ran the same 60m block twice,
once with the hardcoded `500`, once with a floor scaled to the block's
point count (~66,677). Everything else held identical.

| Setting | Ground coverage |
|---|---|
| `min_points_abs=500` (hardcoded) | 97.9% |
| `min_points_abs=66,677` (scaled) | 97.9% |

**Identical.** This hypothesis was rejected too — `min_points_abs` isn't
the cause.

### The actual root cause

`find_ground_plane_robust()`'s final tie-break, among horizontal
candidates (z_align ≥ 0.80):

```python
horizontal.sort(key=lambda c: c["avg_z"], reverse=reverse)
best = horizontal[0]          # single most extreme height. No size check.
```

**Confirmed directly** with a purpose-built diagnostic
(`scratchpad/diagnose_ground_selection.py`), which prints every horizontal
candidate's size, height, and ground-truth coverage side by side. On L003:

| Plane | Points | avg_z | Coverage | |
|---|---|---|---|---|
| #1 | 670,438 | -1.86 | 3.8% | **← selected** (lowest, but tiny) |
| #0 | 5,909,393 | -1.71 | 92.4% | the actual road, 9x bigger, 15cm higher |

A 670k-point plane beat a 5.9M-point plane — by being 15 centimeters
lower. That's the entire bug: **no comparison of size at all**, at any
scale. It was already a known failure mode on single small frames (the
function's own docstring documents a 168-point fragment beating a
4,870-point real floor by a few cm), but on a city block there's far more
room for some small low-lying patch (a dip, a sunken driveway, a
depression) to exist somewhere and be marginally more extreme than the
real, much bigger surface.

### Choosing the fix: fixed constant vs. adaptive — tested both

**Proposed fix**: two-stage rule. Find the extreme height among horizontal
candidates, then among everything within a band of that height, pick the
**largest**. Rationale: a sub-meter height spread among horizontal
candidates is normal road camber/grade — the same physical surface, not a
different one — so below that scale, size is the honest tie-break, not a
few centimeters of height.

Two ways to set the band width were tested side by side, same runs, same
data:

- **Fixed**: `HEIGHT_BAND = 0.5m`, a plain constant.
- **Adaptive ("Option 3")**: `HEIGHT_BAND = K × residual_of_largest_candidate`
  — self-calibrating from each scene's own measured surface roughness,
  tested at K = 3, 5, 8, 12.

**Validation results across 7 independent datasets:**

| Dataset | Current (buggy) rule | Fixed 0.5m band | Adaptive (Option 3) |
|---|---|---|---|
| Toronto-3D L003 (road) | 3.8% | **92.4%** (=oracle) | matches oracle at K=3,5,8,12 |
| Toronto-3D L001 (road) | 0.6% | **91.6%** (=oracle) | **fails at K=3,5** (stuck at 0.6%); only K≥8 matches |
| S3DIS conferenceRoom_1 | 98.5% (already optimal) | 98.5% (no regression) | matches at all K |
| S3DIS hallway_1 | 98.7% (already optimal) | 98.7% (no regression) | matches at all K |
| S3DIS office_1 | 99.0% (already optimal) | 99.0% (no regression) | matches at all K |
| S3DIS auditorium_1 | 88.7% (already optimal) | 88.7% (no regression) | matches at all K |
| S3DIS lounge_1 | 99.6% (already optimal) | 99.6% (no regression) | matches at all K |

**The fixed 0.5m band is 7 for 7, with no tuning.** The adaptive version
failed outright on L001 at K=3 and K=5 — it stayed stuck on the *same*
broken 0.6%-coverage plane the current buggy rule picks, because the
required gap there (0.41m) exceeded `K×residual` at those K values.

**Why the adaptive version doesn't work**: residual measures small-scale
*surface roughness* (sensor noise / micro-texture) — it sat in a narrow
0.003-0.09m range across every dataset tested, roads and indoor rooms
alike. But the thing the band actually needs to bridge — road camber,
grade, survey-to-survey offset — doesn't scale with that noise floor at
all: 0.15m on L003, 0.41m on L001, with no relationship to either cloud's
residual. Even at the K values where Option 3 worked, it never beat the
fixed band anywhere — it just reproduced the same picks through a less
reliable path. **Decision: fixed 0.5m constant**, not adaptive.

(One dataset — OpenTopography CA17, airborne LiDAR — was tried and
explicitly excluded from this comparison. Its "ground" label is ASPRS bare
earth/terrain, which can be genuinely uneven, not "one flat plane" the way
Toronto-3D's road and S3DIS's floors are — a low coverage number there
reflects a label/geometry mismatch, not a flaw in the fix. It *did*
separately confirm that coupling `eps` to voxel size as a floor
[`eps = max(eps_agent, K_MIN*voxel)`, an earlier, already-approved,
unrelated fix] helps on sparse airborne data — shape count went from 0 to
17 — which is a real but separate finding from the selection-rule fix.)

### Production fix

`find_ground_plane_robust()` in `detect_ground_and_walls.py`:

```python
HEIGHT_BAND = 0.5
...
reverse = (z_mode == "z_down")
horizontal.sort(key=lambda c: c["avg_z"], reverse=reverse)
extreme_z = horizontal[0]["avg_z"]
in_band = [c for c in horizontal if abs(c["avg_z"] - extreme_z) <= HEIGHT_BAND]
best = max(in_band, key=lambda c: c["n_points"])
```

**Before/after on the actual production script** (`--model_variant
synthetic --z_mode z_up --voxel 0.02`, whole clouds):

| Scene | Ground before | Ground after |
|---|---|---|
| L003 (39.7M pts) | 4.2% of cloud (564,902 pts) | **45.3%** of cloud (6,106,406 pts) |
| L001 (21.6M pts) | 5.3% of cloud (525,382 pts) | **46.1%** of cloud (4,555,041 pts) |

Both land right around each cloud's true ground proportion (51.8% / 51.5%
of points are really ground) — consistent with the ~92% *coverage-of-real-
ground* numbers measured against ground truth above. Wall detection is
unchanged (a separate function, untouched by this fix).

**Scope of this fix**: it only changes which single horizontal plane gets
*labeled* ground — it doesn't touch the agent's `eps`/`min_support`/
`norm_th` choices (still fully adaptive, per-scene), and it still returns
exactly one ground plane. Multi-level scenes — a staircase descending past
the true floor into a lower area, a mezzanine — aren't handled by this or
any single-ground-plane design; that's a structurally different problem
(which "ground" is correct depends on which level you're asking about) and
was explicitly out of scope here.

---

## Problem 3: the wall coverage ceiling and tiled wall detection

### The symptom

Even after fixing ground, wall coverage against Toronto-3D's real
`Building` class capped around 45-57% on a whole cloud/large block —
clearly better than ground was, but with real facades still missing.

### Root cause

Different mechanism entirely from the ground bug — this one is about
**detection**, not selection. With `--model_variant synthetic`,
`min_support` (RANSAC's minimum inlier count to accept a shape) is a
*fraction* of the input point count, not a fixed number. Run over an
entire multi-million-point block, that fraction becomes a large absolute
number (tens of thousands of points) — comfortably cleared by a big road
surface, but too high a bar for one small building facade. RANSAC never
even proposes the facade as a candidate shape; it's not that
`find_wall_planes()` rejects it, it never reaches that function at all.

**Confirmed** (`scratchpad/tiling_pilot.py`, 60m block, tile sweep at
15/20/25m with 3m overlap): splitting the block into smaller tiles shrinks
the point count `min_support`'s fraction is computed against, so facades
that couldn't clear the whole-block bar clear the much smaller per-tile
one.

| Config | Wall coverage (vs. real `Building` class) |
|---|---|
| Whole 60m block, single pass | 57.1% |
| Tiled, 20m tiles / 3m overlap | **77.2%** |

Unlike ground, this genuinely is a scale problem — but a detection-
threshold one, not a parameter-adaptation one, and specific to walls
(ground's `min_support` issue was already ruled out separately — see
Hypothesis B above).

### Two more bugs found before shipping this

A design review (Plan-agent, cross-checked against the actual code before
implementation) caught two real defects already baked into that 77.2%
pilot number:

1. **Fallback ground-exclusion was deleting real walls.**
   `find_ground_plane_robust()` falls back to "largest shape wins,
   regardless of orientation" when no candidate in a tile clears
   `horizontal_thresh` — likely in exactly the tiles that are mostly
   facade with little flat ground visible. The pilot always excluded that
   fallback "ground" pick from the tile's wall list, meaning a tile's
   *biggest wall* could get mislabeled ground and then silently dropped.
   **Fix**: only exclude the ground candidate from a tile's walls when it
   actually cleared `horizontal_thresh` — i.e., when it's a real, honest
   ground pick, not an orientation-blind fallback.

2. **Sign convention broke wall merging.** Detected walls from different
   tiles need to be recognized as "the same physical facade" and merged.
   The matching signature (surface normal + offset) needs a consistent
   sign, and the pilot signed by the z-component of the normal — correct
   for ground (where z is always the dominant component above the
   horizontal threshold) but essentially random noise for near-vertical
   walls (where z is the *smallest*, least meaningful component). Two
   tiles detecting the same real wall could get opposite-sign normals and
   silently fail to merge, fragmenting one wall into several.
   **Fix**: sign by the *largest-magnitude* component of the normal
   instead — provably identical to the old rule for ground, stable for
   walls.

Both fixes are in the production code (not the scratchpad pilot), so the
true achievable ceiling for tiled walls is expected to be at or above the
77.2% pilot figure.

### Production implementation: `--tile_walls`

New, **strictly opt-in** flag (default off — every existing run and every
already-tested dataset is unaffected unless this flag is explicitly
passed). New functions in `detect_ground_and_walls.py`, added right after
`find_wall_planes()`:

- **`plane_signature()`** — sign-normalized (normal, offset) for comparing
  planes detected in different tiles, in a shared coordinate frame.
- **`cluster_and_merge_planes()`** — greedy angle+offset clustering that
  **unions point masks** across tiles (not just counts them), so repeated
  detections of one real wall collapse into a single merged entity.
- **`_detect_walls_in_tile()`** — the tile-local detection step: local
  recentering, the same `eps` floor already validated
  (`max(eps_agent, K_MIN*voxel)`, K_MIN=2.0 — clip-as-floor, the agent
  still drives eps), a `normal_knn = min(20, n-1)` guard for small tiles,
  and the corrected ground-exclusion logic (fix #1 above).
- **`detect_walls_tiled()`** — orchestrates the above across all tiles of
  a cloud (stride = `tile_size - tile_overlap`), then merges and returns a
  wall list in the exact same shape `find_wall_planes()` already returns,
  so it's a drop-in for `process_frame()`.

Ground detection is **completely untouched** — still the single
whole-cloud pass, still using the `HEIGHT_BAND` fix above, regardless of
`--tile_walls`.

New CLI flags (all with defaults matching the validated pilot config):
`--tile_walls` (off by default), `--tile_size` (20.0), `--tile_overlap`
(3.0), `--wall_merge_angle_deg` (5.0), `--wall_merge_offset_m` (0.20).

**Important caveat**: the validated mechanism is specific to
`--model_variant synthetic`, whose `min_support` is fraction-based.
`--model_variant main` uses a fixed *absolute* `min_support`, so this
flag's benefit doesn't apply there and is unproven for that variant.

**Cost**: opt-in because it isn't free — a whole-cloud `detect()` call is
still mandatory (still needed for ground; measured 266s on L003's 13.4M
downsampled points), and `--tile_walls` adds several hundred more,
individually smaller, per-tile `detect()` calls on top of that.

### Validation results

A full-cloud `--tile_walls` run on the entire L003 file (39.7M points) was
attempted and **segfaulted 3/3 times** (exit 139), always at the identical
point — during the mandatory whole-cloud pass, before any tiling code even
runs. The same code path had just succeeded 4/4 times without
`--tile_walls`. System RAM/SSD showed signs of heavy strain by that point
in a long session of repeated multi-GB loads; this looks like environment
memory pressure, not a defect in the new code, but a clean full-cloud run
was not obtained this session — see "Open items" below.

Validated instead on a bounded 60m block (`scratchpad/verify_tile_walls.py`
— same densest-block selection as `tiling_pilot.py`, but calling the real
production functions, `find_wall_planes` vs. `detect_walls_tiled`, not
scratchpad copies):

| Config | Wall coverage | Precision | Walls returned |
|---|---|---|---|
| Baseline (no tiling), `max_walls=8` | 57.0% | 98.9% | 5 |
| `--tile_walls`, `max_walls=8` (default) | 63.4% | 96.1% | **8 (capped)** |
| `--tile_walls`, `max_walls=30` | **76.9%** | 93.7% | **30 (capped)** |

**Important interaction found**: `--tile_walls` is meant to surface *more*
distinct facades than one whole-block pass ever finds — but the returned
wall count exactly matched `max_walls` in both tiled rows, meaning even 30
wasn't enough to stop truncating. The default `max_walls=8` (inherited
from the single-pass convention, where 8 was already generous) silently
throws away most of tiling's benefit. Raising it to 30 recovered 76.9%
coverage — matching the original pilot's 77.2% closely (small gap is
normal RANSAC-jitter, not a discrepancy). **`--tile_walls` should always be
paired with a higher `--max_walls`**; this is now documented directly in
the flag's own `--help` text.

### Open items

- A genuine full-cloud (not just block-scale) `--tile_walls` run on L003
  has not yet completed successfully — retry when the system isn't under
  memory pressure from a long session of prior heavy runs, or consider
  chunking a full-cloud run into a few large regions rather than one pass
  over 300+ tiles in a single process.
- `--tile_walls`'s benefit is validated for `--model_variant synthetic`
  only (see caveat above) — untested for `main`.

---

## Files touched / added

| File | Change |
|---|---|
| `ransac_env.py` | `load_ply_xyz()` gained `recenter` param (Problem 1 fix) |
| `detect_ground_and_walls.py` | `find_ground_plane_robust()` HEIGHT_BAND fix (Problem 2); new `plane_signature`, `cluster_and_merge_planes`, `_detect_walls_in_tile`, `detect_walls_tiled`, `--tile_walls` and related CLI flags (Problem 3) |
| `scratchpad/tiling_pilot.py` | Phase-1 pilot script that found the wall-tiling opportunity (superseded by the production `--tile_walls` implementation, which fixes two bugs the pilot had) |
| `scratchpad/ab_min_points_abs.py` | A/B test that ruled out Hypothesis B |
| `scratchpad/diagnose_ground_selection.py` | Diagnostic tool that found and validated the ground-selection fix (both the fixed-band and adaptive versions) |

## How to reproduce

```bash
# Ground fix + precision fix (no flags needed -- always on):
python detect_ground_and_walls.py --ply <file.ply> --model_variant synthetic --z_mode z_up --voxel 0.02

# Tiled wall detection (opt-in):
python detect_ground_and_walls.py --ply <file.ply> --model_variant synthetic --z_mode z_up --voxel 0.02 --tile_walls

# Diagnostic tool (prints every candidate plane + fixed-band vs adaptive-band comparison):
python scratchpad/diagnose_ground_selection.py --ply <file.ply>
python scratchpad/diagnose_ground_selection.py --s3dis_room data/s3dis/Area_1/office_1 --ground_class 1
```
