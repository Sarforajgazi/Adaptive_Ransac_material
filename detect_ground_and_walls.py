"""
detect_ground_and_walls.py

ransac_env.py's step() calls schnabel_ransac.detect() with max_shapes=20 --
it actually finds up to 20 planar shapes per frame, at any orientation --
but find_ground_plane() (ransac_env.py:33) then keeps only the single best
horizontal one. Every other detected shape, including walls, is discarded
and never reaches the reward, observation, or any output.

This script reuses the exact same detect() call and the exact same trained
model, but classifies EVERY returned shape by orientation instead of
throwing the non-ground ones away: find_ground_plane_robust() below is a
LOCAL fix of ransac_env.find_ground_plane()'s tie-break rule (that
function is NOT modified -- ransac_env.py is used for real training/eval,
so this fix is scoped to this script only). Confirmed on Supermarket frame
563: ransac_env.find_ground_plane() picks the horizontal candidate with
the lowest average height with no size floor on that comparison, so a
168-point noise fragment beat a genuine 4,870-point floor patch sitting
just 4cm higher. find_ground_plane_robust() only lets a horizontal
candidate compete in that height tie-break if its support is at least
some fraction of the largest horizontal candidate's support -- filters out
noise fragments while still correctly avoiding a large-but-wrong candidate
(e.g. a big ceiling/shelf-top plane) that plain "largest support wins"
would pick instead. find_wall_planes() below is find_ground_plane's
mirror image for the rest -- keeps every candidate whose normal is
near-horizontal (surface near-vertical, i.e. a wall), since a scene can
have several walls at once, unlike a single ground.

schnabel_ransac's C++ backend also genuinely supports curved primitives
(sphere/cylinder/cone/torus -- bridge.cpp:20-23,110-113), but every call
site in this project (until now) requested shapes=["plane"] only -- see
build_realworld_report.py:225's explicit note that this project "restricts
detection to planes only". This script now also requests "cylinder" (the
most relevant curved primitive for real structures: poles, pillars, pipes,
tree trunks) and shows it as a third category, separate from the
plane-only ground/wall classification (a cylinder's inlier points don't
have a meaningful single "normal" the way a plane's do, so they're kept
out of find_ground_plane()/find_wall_planes() entirely and just reported
by point count).

Single-shot bypass, same convention as visualize_real_pointcloud.py: builds
the observation the model would see on a fresh reset() (scene features +
all-zero feedback) and asks for one action, rather than running the full
multi-step env.step() loop -- that loop needs a ground-truth mask to score
its stop/continue refinement against, which most real-world frames don't
have.

Usage:
    python detect_ground_and_walls.py --ply data/Office/Data_omni/P0000/lidar/000000_lcam_front_lidar.ply
    python detect_ground_and_walls.py --ply <file> --wall_thresh 0.25 --max_walls 5
    python detect_ground_and_walls.py --ply <real_sensor_file> --z_mode z_up

    # Headless batch mode -- process many frame indices from one environment's
    # lidar/ folder at once, saving a PNG per frame instead of opening a
    # blocking interactive window each time:
    python detect_ground_and_walls.py --env Office --indices 481 485 491 495 531 617

    # RGB-D source instead of the sparse LiDAR sweep -- back-projects
    # image_lcam_front/ + depth_lcam_front/ into a dense colored point
    # cloud (same method as visualize_tartanair_rgbd_rl.py). Add
    # --interactive to also pop an on-screen window per frame (blocks
    # until closed) in addition to saving a screenshot:
    python detect_ground_and_walls.py --env Office --indices 481 485 --source rgbd --interactive

    # Use the synthetic-plane-trained model instead of the default
    # (main) TartanAir-trained one -- see MODEL_VARIANTS below for exactly
    # what differs. Cylinder detection is auto-disabled for this variant
    # (confirmed it silently steals valid plane detections otherwise).
    python detect_ground_and_walls.py --env Office --indices 481 --source rgbd --model_variant synthetic
"""
import os
import sys
import csv
import time
import pickle
import argparse
import numpy as np
import cv2
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_rl_experiment"))
import schnabel_ransac
import screenshot_utils
from features.scene_features import compute_scene_features
from ransac_env import load_ply_xyz, EPS_LEVELS as MAIN_EPS_LEVELS, \
    MIN_SUPPORT_LEVELS as MAIN_MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS as MAIN_NORM_THRESH_LEVELS
from synthetic_env import EPS_LEVELS as SYN_EPS_LEVELS, \
    MIN_SUPPORT_LEVELS as SYN_MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS as SYN_NORM_THRESH_LEVELS

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

# Two genuinely different trained models exist in this project, with
# DIFFERENT action spaces and DIFFERENT observation spaces -- not just a
# different --model path. Picking the wrong table for a given model's
# actions/features silently produces garbage (wrong eps/min_support
# scale, wrong feature count fed to the network).
#   "main": ppo_ransac_final.zip -- trained on real TartanAir/TartanGround
#     LiDAR data via ransac_env.py. 8 eps levels, min_support is an
#     ABSOLUTE point count (6 levels), scene features truncated to the
#     first 21 (this model predates compute_scene_features()'s later
#     2-feature addition -- see choose_params_single_shot's docstring).
#   "synthetic": synthetic_ppo_v2.zip -- trained on procedurally-generated
#     synthetic planes via synthetic_rl_experiment/synthetic_env.py. 11 eps
#     levels, min_support is a FRACTION of the current cloud's point count
#     (6 levels), and it expects the FULL current 23-dim scene features
#     (trained after that addition, so no truncation).
MODEL_VARIANTS = {
    "main": dict(
        model_path=os.path.join(WORKSPACE, "models", "ppo_ransac_final.zip"),
        vecnormalize_path=os.path.join(WORKSPACE, "models", "ppo_ransac_final_vecnormalize.pkl"),
        eps_levels=MAIN_EPS_LEVELS,
        min_support_levels=MAIN_MIN_SUPPORT_LEVELS,
        min_support_is_fraction=False,
        norm_thresh_levels=MAIN_NORM_THRESH_LEVELS,
        n_scene_features=21,
        detect_cylinders=True,
    ),
    "synthetic": dict(
        model_path=os.path.join(WORKSPACE, "models", "synthetic_ppo_v2.zip"),
        vecnormalize_path=os.path.join(WORKSPACE, "models", "synthetic_ppo_v2_vecnormalize.pkl"),
        eps_levels=SYN_EPS_LEVELS,
        min_support_levels=SYN_MIN_SUPPORT_LEVELS,
        min_support_is_fraction=True,
        norm_thresh_levels=SYN_NORM_THRESH_LEVELS,
        n_scene_features=23,
        # False: confirmed on Office frame 481 that requesting plane+cylinder
        # together in one detect() call lets a degenerate cylinder hypothesis
        # (radius 253m/531m) win a region BEFORE a plane hypothesis is ever
        # tried there, so 2 of 3 genuine large flat surfaces (30.9%, 26.4%
        # of the cloud) vanish entirely -- radius/frac caps can only reject
        # the bad cylinder afterward, they can't recover the plane that was
        # never proposed. Plane-only also matches how this model was
        # actually validated elsewhere in the project (visualize_real_
        # pointcloud.py, visualize_tartanair_rgbd_rl.py never request
        # cylinders either).
        detect_cylinders=False,
    ),
}

# Kept for backwards compatibility with the rest of this file / anything
# importing these names directly; "main" is still the default variant.
DEFAULT_MODEL_PATH = MODEL_VARIANTS["main"]["model_path"]
DEFAULT_VECNORMALIZE_PATH = MODEL_VARIANTS["main"]["vecnormalize_path"]

# TartanAir/TartanGround's native camera intrinsics for the 640x640
# image_lcam_front / depth_lcam_front pair -- confirmed (not guessed) via
# tartanair's own customizer.py, same constants visualize_tartanair_rgbd_rl.py
# uses for its RGB-D back-projection.
FX, FY, CX, CY = 320.0, 320.0, 319.5, 319.5

# Distinct colors per wall so overlapping/adjacent walls stay visually separable.
WALL_COLORS = [
    [1.0, 0.0, 0.0], [0.0, 0.4, 1.0], [1.0, 0.6, 0.0],
    [0.7, 0.0, 0.9], [0.0, 0.8, 0.8], [1.0, 0.0, 0.6],
    [0.5, 0.5, 0.0], [0.3, 0.3, 0.3],
]

# Separate color family for cylinders (curved surfaces) so they read as a
# distinct category from walls at a glance -- yellow/gold shades vs. the
# wall list's reds/blues/purples.
CYLINDER_COLORS = [
    [1.0, 0.85, 0.0], [0.85, 0.65, 0.13], [1.0, 1.0, 0.4],
    [0.72, 0.53, 0.04], [1.0, 0.75, 0.5], [0.6, 0.4, 0.0],
    [0.93, 0.86, 0.51], [0.8, 0.52, 0.25],
]


def load_obs_normalizer(vecnormalize_path):
    """This model was trained inside VecNormalize(norm_obs=True) -- replicates
    SB3's own normalization formula from the saved running stats."""
    if vecnormalize_path is None or not os.path.exists(vecnormalize_path):
        print("No VecNormalize stats found -- using raw observations.")
        return lambda obs: obs
    with open(vecnormalize_path, "rb") as f:
        vec_normalize = pickle.load(f)
    obs_rms = vec_normalize.obs_rms
    clip_obs = vec_normalize.clip_obs
    epsilon = vec_normalize.epsilon
    return lambda obs: np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                                -clip_obs, clip_obs).astype(np.float32)


def choose_params_single_shot(model, normalize_obs, points, variant, z_mode="z_down"):
    # compute_scene_features() currently returns 23 features. The "main"
    # model (ppo_ransac_final) predates the 2 newest features
    # (mean_surface_var, p25_surface_var) and expects them sliced off
    # (n_scene_features=21); the "synthetic" model (synthetic_ppo_v2) was
    # trained after that addition and expects the full 23. Getting this
    # wrong doesn't crash -- VecNormalize's mean/var arrays would just be
    # the wrong length and broadcast-error, or silently misalign if the
    # lengths happen to match by coincidence -- so this MUST track
    # variant["n_scene_features"], never a hardcoded slice.
    #
    # compute_scene_features() hardcodes a z-down (NED) assumption for its
    # two "ground" features (z_density_ground, ground_slope_estimate: both
    # defined as "the region near z_max"). For z_up sources (S3DIS, RGB-D,
    # any --z_mode z_up LiDAR) that region is the CEILING, not the floor --
    # confirmed by reading the feature code, not a guess. Negating z before
    # feature extraction fixes this cleanly: it's an orthogonal reflection,
    # so every other feature (eigenvalues, normal stats, distances, density)
    # is provably unchanged, and only the two z-max-relative "ground"
    # features end up looking at the actual floor. Only the copy fed into
    # feature extraction is flipped -- RANSAC/ground-selection still get the
    # real, unflipped points.
    feat_points = points.copy()
    if z_mode == "z_up":
        feat_points[:, 2] *= -1
    scene_feat = compute_scene_features(feat_points)[:variant["n_scene_features"]]
    feedback_feat = np.zeros(10, dtype=np.float32)  # matches reset()'s defaults exactly, both envs
    obs = np.concatenate([scene_feat, feedback_feat]).astype(np.float32)
    action, _ = model.predict(normalize_obs(obs), deterministic=True)
    eps = variant["eps_levels"][int(action[0])]
    min_supp_raw = variant["min_support_levels"][int(action[1])]
    if variant["min_support_is_fraction"]:
        min_supp = max(1, int(round(min_supp_raw * len(points))))
    else:
        min_supp = min_supp_raw
    norm_th = variant["norm_thresh_levels"][int(action[2])]
    return eps, min_supp, norm_th


def plane_normal_and_residual(points, mask):
    """Callers already skip masks with < 10 points, but a >=10-point mask
    can still be degenerate (near-collinear / near-duplicate points ->
    singular or non-finite covariance), which crashes np.linalg.eig. Treat
    that case as "not horizontal, not a usable plane" (z_align=0.0,
    residual=inf) rather than raising, so one degenerate shape can't take
    down a whole batch run."""
    plane_pts = points[mask]
    mean_pt = plane_pts.mean(axis=0)
    cov = np.cov(plane_pts.T)
    if not np.all(np.isfinite(cov)):
        return np.zeros(3, dtype=np.float64), 0.0, float("inf")
    evals, evecs = np.linalg.eig(cov)
    normal = evecs[:, np.argmin(evals)]
    z_align = abs(float(normal[2]))
    residual = float(np.mean(np.abs(np.dot(plane_pts - mean_pt, normal))))
    return normal, z_align, residual


def find_ground_plane_robust(shapes, points, z_mode="z_down", horizontal_thresh=0.80, min_points_abs=500):
    """Local fix for ransac_env.find_ground_plane()'s tie-break rule.
    ransac_env.py is NOT modified -- it's used for real training/eval, so
    this fix is scoped to this script only.

    The original picks the horizontal candidate (z_align >= horizontal_thresh)
    with the most extreme average height (lowest for z_up, highest for
    z_down) with NO size floor on that comparison. Confirmed empirically on
    Supermarket frame 563: a 168-point noise fragment beat a genuine
    4,870-point floor patch sitting just 4cm higher, simply because it was
    marginally more extreme in height.

    First attempt at a fix used a RELATIVE floor (fraction of the largest
    horizontal candidate's support) -- verified WRONG by a synthetic test
    reproducing frame 563's exact shape sizes: when one candidate is a huge
    but wrong surface (a 151,192-point ceiling/shelf-top), nothing else
    clears any reasonable fraction of IT, so the filter degenerately
    collapsed back to "largest support wins" and picked the ceiling.

    This version uses an ABSOLUTE point-count floor instead (default 500,
    comfortably above the ~168-391 point noise fragments seen and below
    the ~1,000-5,000 point real (if sometimes small/occluded) floors seen
    across Office/Supermarket frames this session) -- verified correct on
    both the frame-563-shaped synthetic case (correctly keeps the
    4,870-point floor, correctly excludes both the 168-point fragment AND
    the 151,192-point ceiling) and a frame-513-shaped case where the real
    floor is legitimately small (2,931 points, no noise fragment involved
    -- confirms this floor doesn't wrongly reject genuine small floors).

    Second fix, same root cause at a bigger scale: on multi-million-point
    full clouds (Toronto-3D L003/L001), "most extreme height wins" -- with
    NO size comparison at all among horizontal candidates -- let a tiny
    low-lying patch (a dip, a sunken area) beat the real, much larger road
    surface by a few centimeters of height, tanking ground coverage to
    0.6-3.8% instead of ~92%. A sub-metre height spread among horizontal
    candidates is normal road camber/grade (the same physical surface, not
    a different one), so below that scale, size -- not a few cm of height
    -- is the honest tie-break. HEIGHT_BAND=0.5m: among horizontal
    candidates within this many meters of the most extreme height, pick
    the LARGEST. Validated via scratchpad/diagnose_ground_selection.py:
    recovers the coverage-optimal plane on both Toronto-3D L003 (3.8% ->
    92.4%) and L001 (0.6% -> 91.6%), with zero regression across 5 S3DIS
    rooms (floor vs ceiling correctly separated in all 5, since a ceiling
    sits >2.5m from the floor -- well outside the band). An adaptive,
    residual-scaled band was also tried and rejected: it failed on L001 at
    low K (stuck on the same wrong plane) and never beat the fixed band
    anywhere it succeeded -- residual measures small-scale surface
    roughness, not how far a real surface's height can legitimately drift
    across a large scene, so it doesn't track the thing that actually
    varies."""
    HEIGHT_BAND = 0.5
    if not shapes:
        return None, None, None, None

    candidates = []
    for shape in shapes:
        mask = shape["inlier_mask"]
        if np.sum(mask) < 10:
            continue
        normal, z_align, residual = plane_normal_and_residual(points, mask)
        avg_z = float(points[mask][:, 2].mean())
        candidates.append({
            "shape": shape, "avg_z": avg_z, "z_align": z_align,
            "residual": residual, "n_points": shape["n_points"],
        })
    if not candidates:
        return None, None, None, None

    horizontal = [c for c in candidates if c["z_align"] >= horizontal_thresh]
    if not horizontal:
        horizontal = sorted(candidates, key=lambda c: c["n_points"], reverse=True)[:1]
    else:
        eligible = [c for c in horizontal if c["n_points"] >= min_points_abs]
        horizontal = eligible if eligible else horizontal

    reverse = (z_mode == "z_down")
    horizontal.sort(key=lambda c: c["avg_z"], reverse=reverse)
    extreme_z = horizontal[0]["avg_z"]
    in_band = [c for c in horizontal if abs(c["avg_z"] - extreme_z) <= HEIGHT_BAND]
    best = max(in_band, key=lambda c: c["n_points"])
    return best["shape"], best["avg_z"], best["z_align"], best["residual"]


def find_wall_planes(shapes, points, wall_thresh=0.3, exclude_mask=None):
    """Mirror image of ransac_env.find_ground_plane(): keeps every candidate
    whose normal is near-horizontal (z_align <= wall_thresh -> the surface
    itself is near-vertical, i.e. a wall), instead of collapsing to one
    winner -- a scene can have several walls, unlike a single ground."""
    walls = []
    for shape in shapes:
        mask = shape["inlier_mask"]
        if exclude_mask is not None and np.array_equal(mask, exclude_mask):
            continue
        if np.sum(mask) < 10:
            continue
        normal, z_align, residual = plane_normal_and_residual(points, mask)
        if z_align <= wall_thresh:
            walls.append({
                "shape": shape,
                "normal": normal,
                "z_align": z_align,
                "residual": residual,
                "n_points": shape["n_points"],
            })
    walls.sort(key=lambda w: w["n_points"], reverse=True)
    return walls


K_MIN = 2.0  # eps floor multiplier for tiled wall detection: eps_used =
# max(eps_agent, K_MIN * voxel). Clip-as-floor, not a replacement -- the
# agent still drives eps. Anchor from scratchpad/tiling_pilot.py: voxel
# 0.02 + eps 0.08 (k=4) worked on L003; voxel 0.3 + eps 0.08 (k=0.27) found
# nothing at all. The floor only fires in that already-proven-dead regime.


def plane_signature(plane_points_block):
    """Sign-normalized (normal, offset) for cross-tile plane comparison, in
    whatever frame plane_points_block's coordinates are already in -- call
    it on points already lifted to the shared GLOBAL frame, not the
    tile-local recentered ones each tile actually detects in, so two tiles
    (each recentered on its own local mean) can be compared directly.

    Signs by the LARGEST-magnitude component of the normal, not a
    hardcoded axis. For ground candidates (z_align >= horizontal_thresh)
    this is provably identical to signing by normal[2]: z_align=|normal[2]|
    >= 0.80 forces normal[2] to already be the dominant component (since
    nx^2+ny^2 <= 0.36 caps |nx|,|ny| at <= 0.6 < 0.8). For near-vertical
    wall candidates, normal[2] is instead the near-zero, noise-dominated
    component -- signing by it would flip unpredictably between tiles and
    silently fragment one real facade into multiple non-merging groups
    (the offset comparison needs d and -d to agree, which they won't).
    Signing by the dominant component fixes this for walls while leaving
    ground-signature behavior unchanged.

    Returns (None, None) on a degenerate/zero normal."""
    all_mask = np.ones(len(plane_points_block), dtype=bool)
    normal, _z, _r = plane_normal_and_residual(plane_points_block, all_mask)
    if not np.all(np.isfinite(normal)) or np.allclose(normal, 0):
        return None, None
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        normal = -normal
    d = float(np.dot(normal, plane_points_block.mean(axis=0)))
    return normal, d


def cluster_and_merge_planes(entries, angle_deg=5.0, offset_m=0.20):
    """Greedy clustering of (normal, offset, global_mask) entries by angle
    and offset agreement -- treats repeated cross-tile detections of the
    same physical plane as one. Unlike a plain fragmentation count, unions
    each entry's mask into its group, so every merged group carries every
    point that belongs to it across every tile it was seen in. A group's
    reference (normal, offset) is fixed at whichever entry started it --
    not recomputed as more entries join -- matching the greedy clustering
    scratchpad/tiling_pilot.py's count_coplanar_groups() already used (there
    only to count fragmentation; here it also drives the actual merge).
    Returns a list of {"normal", "offset", "mask": unioned bool array}."""
    cos_thresh = np.cos(np.deg2rad(angle_deg))
    groups = []
    for normal, d, mask in entries:
        placed = False
        for g in groups:
            if (abs(float(np.dot(normal, g["normal"]))) >= cos_thresh and
                    abs(d - g["offset"]) <= offset_m):
                g["mask"] |= mask
                placed = True
                break
        if not placed:
            groups.append({"normal": normal, "offset": d, "mask": mask.copy()})
    return groups


def _detect_walls_in_tile(model, normalize_obs, local_points, variant, voxel, z_mode,
                           horizontal_thresh, wall_thresh, min_points_abs=None):
    """Tile-local counterpart of find_wall_planes(), adapted by hand from
    scratchpad/tiling_pilot.py's detect_on_points() -- NOT imported: that
    script's own docstring promises it never modifies or depends on this
    file, and scratchpad/README.md documents scratchpad code as not
    guaranteed to stay runnable against the current codebase, so a
    production->scratchpad import would invert both of those.

    Returns find_wall_planes()'s own per-shape list (TILE-LOCAL masks, one
    entry per distinct wall) rather than collapsing them into one OR-mask,
    so each wall can be lifted to global indices and signature-tagged
    individually for cross-tile merging by the caller.

    Unlike the pilot, only passes exclude_mask to find_wall_planes() when
    the tile's ground candidate actually cleared horizontal_thresh.
    find_ground_plane_robust() falls back to "largest shape wins regardless
    of orientation" when no candidate clears horizontal_thresh -- in a
    facade-dominated tile (little/no flat ground visible, exactly the tile
    profile tiling exists to rescue) that fallback is very likely to grab
    the tile's biggest wall and mislabel it "ground", which would then be
    silently excluded from this tile's own wall list by exact-mask-equality
    in find_wall_planes(). Skipping exclude_mask in that case keeps the
    facade in the wall list where it belongs."""
    n = len(local_points)
    if n < 10:
        return []

    eps_agent, min_supp, norm_th = choose_params_single_shot(
        model, normalize_obs, local_points, variant, z_mode=z_mode)
    eps_used = max(eps_agent, K_MIN * voxel) if voxel and voxel > 0 else eps_agent

    try:
        shapes, _ = schnabel_ransac.detect(
            local_points, shapes=["plane"], relative_epsilon=False,
            epsilon=eps_used, normal_thresh=norm_th, min_support=min_supp,
            probability=0.001, normal_knn=min(20, n - 1), max_shapes=30,
        )
    except Exception:
        return []  # one bad tile must not kill the whole tiled run
    if not shapes:
        return []

    min_pts_abs = min_points_abs if min_points_abs is not None else max(25, int(0.01 * n))
    ground_shape, _avg_z, ground_z_align, _res = find_ground_plane_robust(
        shapes, local_points, z_mode=z_mode, horizontal_thresh=horizontal_thresh,
        min_points_abs=min_pts_abs)
    fallback_fired = ground_shape is not None and ground_z_align < horizontal_thresh
    exclude_mask = (ground_shape["inlier_mask"]
                     if (ground_shape is not None and not fallback_fired) else None)

    return find_wall_planes(shapes, local_points, wall_thresh=wall_thresh, exclude_mask=exclude_mask)


def detect_walls_tiled(points, model, normalize_obs, variant, voxel, z_mode,
                        tile_size=20.0, tile_overlap=3.0,
                        wall_thresh=0.3, horizontal_thresh=0.80, max_walls=8,
                        merge_angle_deg=5.0, merge_offset_m=0.20, verbose=True):
    """Drop-in replacement for find_wall_planes(plane_shapes, points,
    wall_thresh, exclude_mask) when tile_walls=True. Ground detection is
    completely untouched either way -- see process_frame().

    Why this exists: with --model_variant synthetic, min_support is a
    FRACTION of point count. Computed over a whole multi-million-point
    cloud, that fraction becomes too high an absolute bar for one small
    building facade to individually clear, so RANSAC never proposes it as
    a shape at all. Confirmed on Toronto-3D L003: whole-block wall coverage
    vs. the real Building class was 57.1%. Tiling into tile_size windows
    shrinks the fraction's base so facades clear the bar -- measured 77.2%
    at 20m tiles / 3m overlap (scratchpad/tiling_pilot.py). Validated with
    --model_variant synthetic ONLY: --model_variant main's min_support is
    already a fixed absolute count, so this mechanism does not apply to it
    and the benefit here is unproven for that variant.

    Tiles points' XY extent into overlapping windows (stride = tile_size -
    tile_overlap), runs _detect_walls_in_tile() per non-empty tile, lifts
    each individual wall's mask to global indices, tags it with
    plane_signature() computed in the shared GLOBAL frame (sidestepping the
    coordinate mismatch between tiles each recentered on their own local
    mean), and merges same-plane detections across tiles via
    cluster_and_merge_planes() before re-deriving z_align/residual per
    merged mask via the existing plane_normal_and_residual() (safe on a
    merged/non-contiguous mask -- pure PCA/covariance, no contiguity
    assumption, already guards non-finite covariance).

    Returns the same shape find_wall_planes() returns -- confirmed nothing
    downstream reads any wall field beyond shape/n_points/z_align/residual
    -- so this is a legitimate drop-in."""
    n = len(points)
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    x1, y1 = points[:, 0].max(), points[:, 1].max()
    stride = tile_size - tile_overlap
    xs = np.arange(x0, x1, stride)
    ys = np.arange(y0, y1, stride)
    n_tiles = len(xs) * len(ys)

    entries = []
    t_start = time.perf_counter()
    tile_i = 0
    for tx in xs:
        for ty in ys:
            tile_i += 1
            m = ((points[:, 0] >= tx) & (points[:, 0] < tx + tile_size) &
                 (points[:, 1] >= ty) & (points[:, 1] < ty + tile_size))
            gidx = np.flatnonzero(m)
            if len(gidx) < 10:
                continue
            local = points[gidx] - points[gidx].mean(axis=0)
            walls = _detect_walls_in_tile(model, normalize_obs, local, variant, voxel, z_mode,
                                           horizontal_thresh, wall_thresh)
            for w in walls:
                global_mask = np.zeros(n, dtype=bool)
                global_mask[gidx[w["shape"]["inlier_mask"]]] = True
                normal, d = plane_signature(points[global_mask])
                if normal is not None:
                    entries.append((normal, d, global_mask))
            if verbose and tile_i % 20 == 0:
                print(f"      [tile_walls] {tile_i}/{n_tiles} tiles, "
                      f"{len(entries)} wall candidate(s) so far, "
                      f"{time.perf_counter()-t_start:.1f}s elapsed")

    if verbose:
        print(f"      [tile_walls] {n_tiles} tiles scanned in "
              f"{time.perf_counter()-t_start:.1f}s, {len(entries)} wall "
              f"candidate(s) found, merging ...")

    groups = cluster_and_merge_planes(entries, angle_deg=merge_angle_deg, offset_m=merge_offset_m)

    walls = []
    for g in groups:
        mask = g["mask"]
        if mask.sum() < 10:
            continue
        normal, z_align, residual = plane_normal_and_residual(points, mask)
        walls.append({
            "shape": {"inlier_mask": mask, "n_points": int(mask.sum())},
            "normal": normal, "z_align": z_align, "residual": residual,
            "n_points": int(mask.sum()),
        })
    walls.sort(key=lambda w: w["n_points"], reverse=True)
    return walls[:max_walls]


def find_cylinders(shapes, n_total_points, min_points=10, max_frac=None, max_radius=None):
    """Pulls out every detected cylinder shape (curved surfaces: poles,
    pillars, pipes, tree trunks, ...). Unlike find_ground_plane()/
    find_wall_planes(), this doesn't fit a normal -- a cylinder's inlier
    points don't have one meaningful surface normal the way a plane's do,
    so classification here is purely by shape["type"], not by any computed
    orientation.

    radius is read from params[6] -- confirmed against the actual vendored
    source (Efficient-RANSAC-for-Point-Cloud-Shape-Detection/Cylinder.cpp,
    Cylinder::Serialize(float*)): array[0:3]=axisDir (unit vector),
    array[3:6]=axisPos, array[6]=radius, array[7]=angularRotatedRadians.
    This is verified from source, not inferred/guessed.

    max_frac: if set, drops any cylinder whose inlier count exceeds this
    fraction of the total cloud -- a real single curved object rarely fills
    more than a small fraction of a whole scene, so a shape claiming e.g.
    >20% is far more likely a loose-epsilon degenerate fit across many
    unrelated surfaces than a genuine large cylinder.

    max_radius: if set (meters), drops any cylinder whose fitted radius
    exceeds this -- empirically the stronger signal of the two (verified on
    Office frame 491: radii up to 120m and 37m in a scene spanning a few
    meters, i.e. locally indistinguishable from a plane, while point-frac
    alone missed some of these since a few were small-support/high-radius).

    Both None (default): no cap, raw/unfiltered output."""
    cylinders = []
    for s in shapes:
        if s["type"] != "cylinder" or s["n_points"] < min_points:
            continue
        frac = s["n_points"] / n_total_points
        params = s["params"]
        radius = float(params[6]) if len(params) > 6 else None
        if max_frac is not None and frac > max_frac:
            continue
        if max_radius is not None and radius is not None and radius > max_radius:
            continue
        cylinders.append({**s, "radius": radius, "frac": frac})
    cylinders.sort(key=lambda s: s["n_points"], reverse=True)
    return cylinders


def backproject_rgbd(img_path, depth_path):
    """Back-projects a TartanGround image+depth pair into a colored point
    cloud. Depth format and intrinsics confirmed via tartanair's own
    reader.py/customizer.py (see visualize_tartanair_rgbd_rl.py's docstring):
    the "depth" PNG is RGBA-uint8 but the 4 bytes/pixel are a little-endian
    float32 meters value. Camera-frame (X right, Y down-image, Z forward) is
    swapped to this project's z-up convention -- NOT the z-down convention
    the raw LiDAR .ply sweeps use, so callers must pass z_mode="z_up" for
    find_ground_plane() on this source."""
    rgb = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.0
    depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth_rgba.view("<f4").squeeze(-1)

    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    z = depth
    x = (uu - CX) * z / FX
    y = (vv - CY) * z / FY
    points = np.stack([x, z, -y], axis=-1).reshape(-1, 3).astype(np.float32)
    colors = rgb.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (z.reshape(-1) > 0.05) & (z.reshape(-1) < 200)
    return points[valid], colors[valid].astype(np.float64)


def load_s3dis_room(room_dir):
    """Loads a preprocessed S3DIS room (Pointcept/s3dis-compressed layout:
    coord.npy/color.npy/segment.npy per room, real z-up indoor scan -- same
    convention as RELLIS-3D, NOT TartanAir's z-down NED convention). Returns
    (points float32 Nx3, colors float64 Nx3 in [0,1])."""
    coord = np.load(os.path.join(room_dir, "coord.npy")).astype(np.float32)
    color_path = os.path.join(room_dir, "color.npy")
    if os.path.exists(color_path):
        colors = np.load(color_path).astype(np.float64) / 255.0
    else:
        colors = np.full_like(coord, 0.5, dtype=np.float64)
    return coord, colors


def process_frame_s3dis(model, normalize_obs, room_dir, horizontal_thresh, wall_thresh, max_walls,
                         variant, max_cylinders=8, max_cylinder_frac=None, max_cylinder_radius=None, verbose=True):
    """S3DIS counterpart of process_frame_rgbd(): loads a preprocessed room
    directly (real XYZ + RGB, no back-projection needed) instead of a LiDAR
    .ply or an RGB-D pair. Always z_mode="z_up" -- see load_s3dis_room's
    docstring. Same two-panel (true-RGB + classified) output as RGB-D."""
    frame_name = os.path.basename(room_dir.rstrip("/\\"))
    points_world, colors_rgb = load_s3dis_room(room_dir)
    centroid = points_world.mean(axis=0)
    points = points_world - centroid

    eps, min_supp, norm_th = choose_params_single_shot(model, normalize_obs, points, variant, z_mode="z_up")

    shapes, _ = schnabel_ransac.detect(
        points,
        shapes=["plane", "cylinder"] if variant["detect_cylinders"] else ["plane"],
        relative_epsilon=False,
        epsilon=eps,
        normal_thresh=norm_th,
        min_support=min_supp,
        probability=0.001,
        normal_knn=20,
        max_shapes=30,
    )
    plane_shapes = [s for s in shapes if s["type"] == "plane"]

    summary = {
        "file": frame_name,
        "n_points": len(points),
        "eps": eps, "min_supp": min_supp, "norm_th": norm_th,
        "n_shapes": len(shapes),
        "ground_pts": 0, "ground_pct": 0.0, "ground_z_align": None,
        "n_walls": 0, "wall_pts": 0,
        "n_cylinders": 0, "cylinder_pts": 0,
    }

    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(points_world)
    pcd_rgb.colors = o3d.utility.Vector3dVector(colors_rgb)

    if not shapes:
        if verbose:
            print(f"  {frame_name}: {len(points)} pts, RANSAC found nothing to classify.")
        return pcd_rgb, [], summary

    ground_shape, avg_z, ground_z_align, ground_residual = find_ground_plane_robust(
        plane_shapes, points, z_mode="z_up", horizontal_thresh=horizontal_thresh
    )
    ground_mask = ground_shape["inlier_mask"] if ground_shape is not None else None

    walls = find_wall_planes(plane_shapes, points, wall_thresh=wall_thresh, exclude_mask=ground_mask)
    walls = walls[:max_walls]

    cylinders = find_cylinders(shapes, len(points), max_frac=max_cylinder_frac,
                                max_radius=max_cylinder_radius)[:max_cylinders]

    if ground_shape is not None:
        summary["ground_pts"] = ground_shape["n_points"]
        summary["ground_pct"] = ground_shape["n_points"] / len(points) * 100
        summary["ground_z_align"] = ground_z_align
    summary["n_walls"] = len(walls)
    summary["wall_pts"] = sum(w["n_points"] for w in walls)
    summary["n_cylinders"] = len(cylinders)
    summary["cylinder_pts"] = sum(c["n_points"] for c in cylinders)

    if verbose:
        print(f"  {frame_name}: {len(points)} pts, params=(eps={eps}, min_supp={min_supp}, norm_th={norm_th}), "
              f"{len(shapes)} shape(s) -> ground={summary['ground_pts']} pts ({summary['ground_pct']:.1f}%), "
              f"{len(walls)} wall(s) ({summary['wall_pts']} pts), {len(cylinders)} cylinder(s) ({summary['cylinder_pts']} pts)")
        for i, w in enumerate(walls):
            pct = w["n_points"] / len(points) * 100
            print(f"      wall[{i}]: {w['n_points']} pts ({pct:.1f}%), z_align={w['z_align']:.3f}, "
                  f"residual={w['residual']:.4f}m")
        for i, c in enumerate(cylinders):
            radius_str = f"{c['radius']:.3f}m" if c["radius"] is not None else "?"
            print(f"      cylinder[{i}]: {c['n_points']} pts ({c['frac']*100:.1f}%), radius={radius_str}")

    geometries = []
    classified_mask = np.zeros(len(points), dtype=bool)
    if ground_mask is not None:
        pcd_ground = o3d.geometry.PointCloud()
        pcd_ground.points = o3d.utility.Vector3dVector(points_world[ground_mask])
        pcd_ground.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(pcd_ground)
        classified_mask |= ground_mask

    for i, w in enumerate(walls):
        mask = w["shape"]["inlier_mask"]
        pcd_wall = o3d.geometry.PointCloud()
        pcd_wall.points = o3d.utility.Vector3dVector(points_world[mask])
        pcd_wall.paint_uniform_color(WALL_COLORS[i % len(WALL_COLORS)])
        geometries.append(pcd_wall)
        classified_mask |= mask

    for i, c in enumerate(cylinders):
        mask = c["inlier_mask"]
        pcd_cyl = o3d.geometry.PointCloud()
        pcd_cyl.points = o3d.utility.Vector3dVector(points_world[mask])
        pcd_cyl.paint_uniform_color(CYLINDER_COLORS[i % len(CYLINDER_COLORS)])
        geometries.append(pcd_cyl)
        classified_mask |= mask

    remaining = ~classified_mask
    if np.any(remaining):
        pcd_rest = o3d.geometry.PointCloud()
        pcd_rest.points = o3d.utility.Vector3dVector(points_world[remaining])
        pcd_rest.paint_uniform_color([0.6, 0.6, 0.6])
        geometries.append(pcd_rest)

    extent = points_world.max(axis=0) - points_world.min(axis=0)
    offset = np.array([extent[0] * 1.15, 0.0, 0.0], dtype=np.float64)
    for g in geometries:
        g.points = o3d.utility.Vector3dVector(np.asarray(g.points) + offset)

    return pcd_rgb, geometries, summary


def process_frame_rgbd(model, normalize_obs, img_path, depth_path, horizontal_thresh, wall_thresh, max_walls,
                        variant, max_cylinders=8, max_cylinder_frac=None, max_cylinder_radius=None, verbose=True):
    """RGB-D counterpart of process_frame(): back-projects image+depth
    instead of loading a LiDAR .ply, always uses z_mode="z_up" (see
    backproject_rgbd's docstring), and returns an extra true-RGB point
    cloud alongside the classified geometries for a two-panel view."""
    frame_name = os.path.basename(img_path)
    points_world, colors_rgb = backproject_rgbd(img_path, depth_path)
    centroid = points_world.mean(axis=0)
    points = points_world - centroid  # detection runs recentered, same as visualize_tartanair_rgbd_rl.py

    eps, min_supp, norm_th = choose_params_single_shot(model, normalize_obs, points, variant, z_mode="z_up")

    shapes, _ = schnabel_ransac.detect(
        points,
        shapes=["plane", "cylinder"] if variant["detect_cylinders"] else ["plane"],
        relative_epsilon=False,
        epsilon=eps,
        normal_thresh=norm_th,
        min_support=min_supp,
        probability=0.001,
        normal_knn=20,
        max_shapes=30,
    )
    plane_shapes = [s for s in shapes if s["type"] == "plane"]

    summary = {
        "file": frame_name,
        "n_points": len(points),
        "eps": eps, "min_supp": min_supp, "norm_th": norm_th,
        "n_shapes": len(shapes),
        "ground_pts": 0, "ground_pct": 0.0, "ground_z_align": None,
        "n_walls": 0, "wall_pts": 0,
        "n_cylinders": 0, "cylinder_pts": 0,
    }

    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(points_world)
    pcd_rgb.colors = o3d.utility.Vector3dVector(colors_rgb)

    if not shapes:
        if verbose:
            print(f"  {frame_name}: {len(points)} pts, RANSAC found nothing to classify.")
        return pcd_rgb, [], summary

    ground_shape, avg_z, ground_z_align, ground_residual = find_ground_plane_robust(
        plane_shapes, points, z_mode="z_up", horizontal_thresh=horizontal_thresh
    )
    ground_mask = ground_shape["inlier_mask"] if ground_shape is not None else None

    walls = find_wall_planes(plane_shapes, points, wall_thresh=wall_thresh, exclude_mask=ground_mask)
    walls = walls[:max_walls]

    cylinders = find_cylinders(shapes, len(points), max_frac=max_cylinder_frac,
                                max_radius=max_cylinder_radius)[:max_cylinders]

    if ground_shape is not None:
        summary["ground_pts"] = ground_shape["n_points"]
        summary["ground_pct"] = ground_shape["n_points"] / len(points) * 100
        summary["ground_z_align"] = ground_z_align
    summary["n_walls"] = len(walls)
    summary["wall_pts"] = sum(w["n_points"] for w in walls)
    summary["n_cylinders"] = len(cylinders)
    summary["cylinder_pts"] = sum(c["n_points"] for c in cylinders)

    if verbose:
        print(f"  {frame_name}: {len(points)} pts, params=(eps={eps}, min_supp={min_supp}, norm_th={norm_th}), "
              f"{len(shapes)} shape(s) -> ground={summary['ground_pts']} pts ({summary['ground_pct']:.1f}%), "
              f"{len(walls)} wall(s) ({summary['wall_pts']} pts), {len(cylinders)} cylinder(s) ({summary['cylinder_pts']} pts)")
        for i, w in enumerate(walls):
            pct = w["n_points"] / len(points) * 100
            print(f"      wall[{i}]: {w['n_points']} pts ({pct:.1f}%), z_align={w['z_align']:.3f}, "
                  f"residual={w['residual']:.4f}m")
        for i, c in enumerate(cylinders):
            radius_str = f"{c['radius']:.3f}m" if c["radius"] is not None else "?"
            print(f"      cylinder[{i}]: {c['n_points']} pts ({c['frac']*100:.1f}%), radius={radius_str}")

    # ---- Build classified geometries (world coords, uncentered) ----
    geometries = []
    classified_mask = np.zeros(len(points), dtype=bool)
    if ground_mask is not None:
        pcd_ground = o3d.geometry.PointCloud()
        pcd_ground.points = o3d.utility.Vector3dVector(points_world[ground_mask])
        pcd_ground.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(pcd_ground)
        classified_mask |= ground_mask

    for i, w in enumerate(walls):
        mask = w["shape"]["inlier_mask"]
        pcd_wall = o3d.geometry.PointCloud()
        pcd_wall.points = o3d.utility.Vector3dVector(points_world[mask])
        pcd_wall.paint_uniform_color(WALL_COLORS[i % len(WALL_COLORS)])
        geometries.append(pcd_wall)
        classified_mask |= mask

    for i, c in enumerate(cylinders):
        mask = c["inlier_mask"]
        pcd_cyl = o3d.geometry.PointCloud()
        pcd_cyl.points = o3d.utility.Vector3dVector(points_world[mask])
        pcd_cyl.paint_uniform_color(CYLINDER_COLORS[i % len(CYLINDER_COLORS)])
        geometries.append(pcd_cyl)
        classified_mask |= mask

    remaining = ~classified_mask
    if np.any(remaining):
        pcd_rest = o3d.geometry.PointCloud()
        pcd_rest.points = o3d.utility.Vector3dVector(points_world[remaining])
        pcd_rest.paint_uniform_color([0.6, 0.6, 0.6])
        geometries.append(pcd_rest)

    # Offset the classified panel sideways so it renders next to the true-RGB one.
    extent = points_world.max(axis=0) - points_world.min(axis=0)
    offset = np.array([extent[0] * 1.15, 0.0, 0.0], dtype=np.float64)
    for g in geometries:
        g.points = o3d.utility.Vector3dVector(np.asarray(g.points) + offset)

    return pcd_rgb, geometries, summary


def process_frame(model, normalize_obs, ply_path, z_mode, horizontal_thresh, wall_thresh, max_walls,
                   variant, max_cylinders=8, max_cylinder_frac=None, max_cylinder_radius=None, verbose=True,
                   voxel=0.05, tile_walls=False, tile_size=20.0, tile_overlap=3.0,
                   wall_merge_angle_deg=5.0, wall_merge_offset_m=0.20):
    """Runs the model + detect() + ground/wall/cylinder classification on
    one frame. Returns (geometries, summary_dict) -- geometries is a list
    of colored o3d PointClouds (ground=green, each wall its own color,
    each cylinder its own color, rest=gray).

    `voxel` matches load_ply_xyz()'s own default (0.05m, calibrated for a
    single LiDAR sweep) -- pass 0 to disable downsampling entirely (for
    already-sparse/point-bounded clouds), or raise it toward 0.15-0.3m for
    large, dense, tile-scale point clouds (millions of points spanning
    100m+), matching the precedent set by new_data's own viewer scripts.

    Recenters on the point cloud's own centroid before anything else runs,
    via load_ply_xyz(recenter=True) -- computed in float64 and subtracted
    BEFORE that function's internal float32 cast, not after. Real-world
    surveys are often stored in absolute coordinates (e.g. UTM, ~10^6
    scale); recentering post-cast (subtracting a centroid from
    already-float32-quantized points) stops downstream error from
    compounding but can't recover precision already lost at load time --
    recentering first avoids the loss happening at all. Also keeps the
    input in the near-origin range the model's observation space was
    actually trained on. A no-op (negligible effect) for already-near-origin
    data like TartanAir, so this is safe to always apply."""
    points, centroid = load_ply_xyz(ply_path, voxel_size=voxel, recenter=True)
    if verbose:
        print(f"  recentered on centroid {centroid}")
    eps, min_supp, norm_th = choose_params_single_shot(model, normalize_obs, points, variant, z_mode=z_mode)

    shapes, _ = schnabel_ransac.detect(
        points,
        shapes=["plane", "cylinder"] if variant["detect_cylinders"] else ["plane"],
        relative_epsilon=False,
        epsilon=eps,
        normal_thresh=norm_th,
        min_support=min_supp,
        probability=0.001,
        normal_knn=20,
        max_shapes=30,
    )
    plane_shapes = [s for s in shapes if s["type"] == "plane"]

    summary = {
        "file": os.path.basename(ply_path),
        "n_points": len(points),
        "eps": eps, "min_supp": min_supp, "norm_th": norm_th,
        "n_shapes": len(shapes),
        "ground_pts": 0, "ground_pct": 0.0, "ground_z_align": None,
        "n_walls": 0, "wall_pts": 0,
        "n_cylinders": 0, "cylinder_pts": 0,
    }

    if not shapes:
        if verbose:
            print(f"  {summary['file']}: 0 points, RANSAC found nothing to classify.")
        return [], summary

    ground_shape, avg_z, ground_z_align, ground_residual = find_ground_plane_robust(
        plane_shapes, points, z_mode=z_mode, horizontal_thresh=horizontal_thresh
    )
    ground_mask = ground_shape["inlier_mask"] if ground_shape is not None else None

    if tile_walls:
        walls = detect_walls_tiled(points, model, normalize_obs, variant, voxel, z_mode,
                                    tile_size=tile_size, tile_overlap=tile_overlap,
                                    wall_thresh=wall_thresh, horizontal_thresh=horizontal_thresh,
                                    max_walls=max_walls, merge_angle_deg=wall_merge_angle_deg,
                                    merge_offset_m=wall_merge_offset_m, verbose=verbose)
    else:
        walls = find_wall_planes(plane_shapes, points, wall_thresh=wall_thresh, exclude_mask=ground_mask)
    walls = walls[:max_walls]

    cylinders = find_cylinders(shapes, len(points), max_frac=max_cylinder_frac,
                                max_radius=max_cylinder_radius)[:max_cylinders]

    summary["n_shapes"] = len(shapes)
    if ground_shape is not None:
        summary["ground_pts"] = ground_shape["n_points"]
        summary["ground_pct"] = ground_shape["n_points"] / len(points) * 100
        summary["ground_z_align"] = ground_z_align
    summary["n_walls"] = len(walls)
    summary["wall_pts"] = sum(w["n_points"] for w in walls)
    summary["n_cylinders"] = len(cylinders)
    summary["cylinder_pts"] = sum(c["n_points"] for c in cylinders)

    if verbose:
        print(f"  {summary['file']}: {len(points)} pts, params=(eps={eps}, min_supp={min_supp}, norm_th={norm_th}), "
              f"{len(shapes)} shape(s) -> ground={summary['ground_pts']} pts ({summary['ground_pct']:.1f}%), "
              f"{len(walls)} wall(s) ({summary['wall_pts']} pts), {len(cylinders)} cylinder(s) ({summary['cylinder_pts']} pts)")
        for i, w in enumerate(walls):
            pct = w["n_points"] / len(points) * 100
            print(f"      wall[{i}]: {w['n_points']} pts ({pct:.1f}%), z_align={w['z_align']:.3f}, "
                  f"residual={w['residual']:.4f}m")
        for i, c in enumerate(cylinders):
            radius_str = f"{c['radius']:.3f}m" if c["radius"] is not None else "?"
            print(f"      cylinder[{i}]: {c['n_points']} pts ({c['frac']*100:.1f}%), radius={radius_str}")

    # ---- Build geometries: ground=green, walls=distinct colors, cylinders=gold shades, rest=gray ----
    geometries = []
    classified_mask = np.zeros(len(points), dtype=bool)
    if ground_mask is not None:
        pcd_ground = o3d.geometry.PointCloud()
        pcd_ground.points = o3d.utility.Vector3dVector(points[ground_mask])
        pcd_ground.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(pcd_ground)
        classified_mask |= ground_mask

    for i, w in enumerate(walls):
        mask = w["shape"]["inlier_mask"]
        pcd_wall = o3d.geometry.PointCloud()
        pcd_wall.points = o3d.utility.Vector3dVector(points[mask])
        pcd_wall.paint_uniform_color(WALL_COLORS[i % len(WALL_COLORS)])
        geometries.append(pcd_wall)
        classified_mask |= mask

    for i, c in enumerate(cylinders):
        mask = c["inlier_mask"]
        pcd_cyl = o3d.geometry.PointCloud()
        pcd_cyl.points = o3d.utility.Vector3dVector(points[mask])
        pcd_cyl.paint_uniform_color(CYLINDER_COLORS[i % len(CYLINDER_COLORS)])
        geometries.append(pcd_cyl)
        classified_mask |= mask

    remaining = ~classified_mask
    if np.any(remaining):
        pcd_rest = o3d.geometry.PointCloud()
        pcd_rest.points = o3d.utility.Vector3dVector(points[remaining])
        pcd_rest.paint_uniform_color([0.6, 0.6, 0.6])
        geometries.append(pcd_rest)

    return geometries, summary


CSV_FIELDS = [
    "timestamp", "source", "model_variant", "env_or_room", "frame_id", "resolution",
    "n_points", "voxel_downsampling", "eps", "min_support", "normal_thresh",
    "n_shapes", "ground_pts", "ground_pct", "n_walls", "wall_pts",
    "n_cylinders", "cylinder_pts", "processing_time_sec",
]

# Voxel downsampling actually applied per source -- lidar uses load_ply_xyz's
# own default (0.05m, see ransac_env.py); rgbd and s3dis are both raw
# per-point/per-pixel with no downsampling (see backproject_rgbd's and
# load_s3dis_room's docstrings, and Table 6.1 in the generalization report).
VOXEL_DOWNSAMPLING_BY_SOURCE = {
    "lidar": "0.05 m",
    "rgbd": "None (0 m)",
    "s3dis": "None (0 m)",
}


def open_csv_logger(path):
    """Opens (or creates) a CSV log for report-writing purposes -- one row
    per processed frame/room, written and flushed immediately after each
    one so progress survives an interrupted run. Returns a (writer, file)
    pair; caller is responsible for closing the file when done."""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if is_new:
        writer.writeheader()
        f.flush()
    return writer, f


def log_csv_row(writer, file, source, model_variant, env_or_room, frame_id, resolution,
                 summary, processing_time_sec):
    writer.writerow({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "model_variant": model_variant,
        "env_or_room": env_or_room,
        "frame_id": frame_id,
        "resolution": resolution,
        "n_points": summary["n_points"],
        "voxel_downsampling": VOXEL_DOWNSAMPLING_BY_SOURCE[source],
        "eps": summary["eps"],
        "min_support": summary["min_supp"],
        "normal_thresh": summary["norm_th"],
        "n_shapes": summary["n_shapes"],
        "ground_pts": summary["ground_pts"],
        "ground_pct": round(summary["ground_pct"], 2),
        "n_walls": summary["n_walls"],
        "wall_pts": summary["wall_pts"],
        "n_cylinders": summary["n_cylinders"],
        "cylinder_pts": summary["cylinder_pts"],
        "processing_time_sec": round(processing_time_sec, 3),
    })
    file.flush()


def save_screenshot(geometries, out_path, width=1600, height=900):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for g in geometries:
        vis.add_geometry(g)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.light_on = True
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Detect the ground plane AND wall (vertical) planes on LiDAR or RGB-D frame(s).")
    parser.add_argument("--ply", type=str, default=None, help="Path to a single .ply point cloud (lidar source only) -- opens an interactive window.")
    parser.add_argument("--env", type=str, default=None,
                         help="Environment name (e.g. Office, House, Downtown) -- used with --indices.")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                         help="Frame indices to process (with --env), e.g. --indices 481 485 491. "
                              "Saves one screenshot PNG per frame; add --interactive to also open a window per frame.")
    parser.add_argument("--source", type=str, default="lidar", choices=["lidar", "rgbd", "s3dis"],
                         help="lidar (default): sparse single-sweep .ply from data/<env>/.../lidar/. "
                              "rgbd: dense colored point cloud back-projected from "
                              "image_lcam_front/+depth_lcam_front/ (only available for the 8 scenes "
                              "downloaded via download_tartan_rgbd_batch2.py: SeasonalForestAutumn, "
                              "Office, CoalMine, House, OldScandinavia, Downtown, Supermarket, WesternDesertTown). "
                              "s3dis: real indoor room scan(s), use with --s3dis_room instead of --env/--indices.")
    parser.add_argument("--s3dis_room", type=str, nargs="+", default=None,
                         help="one or more S3DIS room directories (each containing coord.npy/color.npy in the "
                              "Pointcept/s3dis-compressed layout), e.g. --s3dis_room data/s3dis_sample/Area_1/"
                              "hallway_1. Use instead of --env/--indices; combine with --interactive to also "
                              "open a window per room.")
    parser.add_argument("--ply_list", type=str, nargs="+", default=None,
                         help="one or more arbitrary LiDAR .ply file paths (no TartanGround directory-naming "
                              "assumption, unlike --env/--indices) -- e.g. real RELLIS-3D frames: --ply_list "
                              "data/RELLIS3D/RELLIS3D_00000/lidar/000000.ply ... Purely visual (ground+wall "
                              "classification only) -- does NOT load or score against any ground-truth labels, "
                              "unlike eval_rellis3d.py's IoU evaluation. Use --z_mode z_up for real sensor data. "
                              "Combine with --interactive to also open a window per frame.")
    parser.add_argument("--interactive", action="store_true",
                         help="with --env/--indices: also open an on-screen window per frame (blocks until "
                              "closed) in addition to saving a screenshot.")
    parser.add_argument("--model_variant", type=str, default="main", choices=list(MODEL_VARIANTS.keys()),
                         help="'main' (default): ppo_ransac_final.zip, trained on real TartanAir/TartanGround "
                              "LiDAR data. 'synthetic': synthetic_ppo_v2.zip, trained on procedurally-generated "
                              "synthetic planes -- DIFFERENT action-space tables (11 eps levels, fraction-based "
                              "min_support) and DIFFERENT scene-feature count (23, not 21). Sets --model/"
                              "--vecnormalize defaults accordingly; only override those explicitly if you want "
                              "a specific checkpoint of the chosen variant (e.g. synthetic_ppo_pilot.zip).")
    parser.add_argument("--model", type=str, default=None, help="default: the --model_variant's own model file.")
    parser.add_argument("--vecnormalize", type=str, default=None,
                         help="default: the --model_variant's own VecNormalize file.")
    parser.add_argument("--z_mode", type=str, default="z_down", choices=["z_down", "z_up"],
                         help="Only used for --source lidar (z_down for TartanAir/TartanGround synthetic "
                              "data, z_up for real sensor data e.g. RELLIS-3D). --source rgbd always uses "
                              "z_up internally regardless of this flag -- see backproject_rgbd()'s docstring.")
    parser.add_argument("--voxel", type=float, default=0.05,
                         help="Voxel downsample size in meters, lidar sources only (--source rgbd/s3dis are "
                              "never downsampled, see VOXEL_DOWNSAMPLING_BY_SOURCE). Default 0.05m matches "
                              "load_ply_xyz()'s existing baseline for a single sensor-centric sweep. Raise "
                              "toward 0.15-0.3m for large, dense, tile-scale clouds (millions of points "
                              "spanning 100m+, matching new_data/'s own viewer scripts); pass 0 to disable "
                              "downsampling entirely for already-sparse or point-bounded clouds.")
    parser.add_argument("--horizontal_thresh", type=float, default=0.80,
                         help="min |normal_z| to count as the ground candidate (matches ransac_env.py's default).")
    parser.add_argument("--wall_thresh", type=float, default=0.3,
                         help="max |normal_z| to count as a wall candidate.")
    parser.add_argument("--max_walls", type=int, default=8, help="cap on how many wall planes to keep/show.")
    parser.add_argument("--tile_walls", action="store_true",
                         help="Detect walls by tiling the cloud into overlapping XY windows and "
                              "merging same-plane detections across tiles, instead of one whole-cloud "
                              "pass. Ground detection is unaffected either way. Fixes a real ceiling: "
                              "with --model_variant synthetic, min_support is a FRACTION of point "
                              "count, so on a large cloud it becomes too high a bar for a single small "
                              "facade to clear on its own -- RANSAC never proposes it as a shape. "
                              "Tiling shrinks the fraction's base so facades clear it (measured "
                              "57.1%%->77.2%% wall coverage vs. Toronto-3D's Building class at 20m "
                              "tiles/3m overlap). VALIDATED WITH --model_variant synthetic ONLY -- "
                              "--model_variant main's min_support is already a fixed absolute count, "
                              "so this mechanism doesn't apply and the benefit is unproven there. Adds "
                              "real cost on top of the already-mandatory whole-cloud pass: one "
                              "whole-cloud detect() alone took 266s on L003's 13.4M downsampled "
                              "points; a full-cloud tiled run adds ~300+ more, individually-smaller "
                              "detect() calls, likely several more minutes. Off by default -- normal "
                              "runs are unaffected. PAIR WITH A HIGHER --max_walls: tiling is meant to "
                              "surface MORE distinct facades than one whole-cloud pass ever finds, so "
                              "the default cap of 8 can silently truncate most of the benefit -- "
                              "measured on a 60m Toronto-3D L003 block, coverage vs. the real Building "
                              "class was 56-57%% (baseline, uncapped) -> only 63.4%% at --max_walls 8 "
                              "but 76.9%% at --max_walls 30 (same tiling, same data, cap raised).")
    parser.add_argument("--tile_size", type=float, default=20.0,
                         help="--tile_walls tile edge length in meters. 20m measured best in the pilot "
                              "sweep (15/20/25m).")
    parser.add_argument("--tile_overlap", type=float, default=3.0,
                         help="--tile_walls overlap between adjacent tiles in meters, needed so a "
                              "facade near a tile boundary is still fully captured in at least one tile.")
    parser.add_argument("--wall_merge_angle_deg", type=float, default=5.0,
                         help="--tile_walls: max angle (degrees) between two tiles' wall normals to "
                              "treat them as the same physical facade and merge.")
    parser.add_argument("--wall_merge_offset_m", type=float, default=0.20,
                         help="--tile_walls: max plane-offset difference (meters) to treat two tiles' "
                              "walls as the same physical facade and merge.")
    parser.add_argument("--max_cylinders", type=int, default=8, help="cap on how many cylinder (curved) shapes to keep/show.")
    parser.add_argument("--max_cylinder_frac", type=float, default=None,
                         help="drop any cylinder whose inlier count exceeds this fraction of the total cloud "
                              "(e.g. 0.15 = reject a cylinder claiming >15%% of all points, a likely degenerate "
                              "loose-epsilon fit rather than a genuine single curved object). Default: no cap, "
                              "raw/unfiltered output.")
    parser.add_argument("--max_cylinder_radius", type=float, default=None,
                         help="drop any cylinder whose fitted radius (meters) exceeds this -- empirically the "
                              "stronger of the two caps (verified radii up to 120m on a scene spanning a few "
                              "meters). Default: no cap, raw/unfiltered output.")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="output folder for batch screenshots (default: screenshots/ground_walls or "
                              "screenshots/ground_walls_rgbd depending on --source).")
    parser.add_argument("--no_cylinders", action="store_true",
                         help="force plane-only detection regardless of --model_variant's own default -- "
                              "e.g. to isolate whether a model-vs-model comparison is really about the model, "
                              "or just about avoiding the plane/cylinder competition (see MODEL_VARIANTS' "
                              "'synthetic' entry docstring for why that competition matters).")
    parser.add_argument("--csv_log", type=str, default=None,
                         help="append one row per processed frame/room to this CSV (report-writing use) -- "
                              "timestamp, source, model_variant, points, resolution, voxel downsampling, "
                              "chosen RANSAC params, shape/ground/wall/cylinder counts, and per-frame wall-clock "
                              "processing time. Written and flushed after each frame, so an interrupted run "
                              "still leaves a complete record of everything processed so far.")
    args = parser.parse_args()

    if not args.ply and not (args.env and args.indices) and not args.s3dis_room and not args.ply_list:
        parser.error("specify --ply <file> (lidar), --env <name> --indices i j k (batch), --s3dis_room <dir...>, "
                     "or --ply_list <file...>.")
    if args.ply and args.source == "rgbd":
        parser.error("--ply is lidar-only; use --env/--indices --source rgbd for RGB-D frames.")

    out_dir = args.out_dir or os.path.join(
        WORKSPACE, "screenshots",
        "ground_walls_s3dis" if args.source == "s3dis" else
        ("ground_walls_rgbd" if args.source == "rgbd" else "ground_walls")
    )

    variant = dict(MODEL_VARIANTS[args.model_variant])  # copy -- --no_cylinders must not mutate the shared dict
    if args.no_cylinders:
        variant["detect_cylinders"] = False
    model_path = args.model or variant["model_path"]
    vecnormalize_path = args.vecnormalize or variant["vecnormalize_path"]

    from stable_baselines3 import PPO
    print(f"Model variant: {args.model_variant}")
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, device="cpu")
    normalize_obs = load_obs_normalizer(vecnormalize_path)

    csv_writer, csv_file = (open_csv_logger(args.csv_log) if args.csv_log else (None, None))
    if csv_writer:
        print(f"CSV log: {args.csv_log}")

    if args.s3dis_room:
        print(f"\nBatch mode: S3DIS, {len(args.s3dis_room)} room(s) -> {out_dir}")
        results = []
        for room_dir in args.s3dis_room:
            if not os.path.isdir(room_dir) or not os.path.exists(os.path.join(room_dir, "coord.npy")):
                print(f"  MISSING or not a room dir (no coord.npy): {room_dir}")
                continue
            t0 = time.perf_counter()
            pcd_rgb, geometries, summary = process_frame_s3dis(
                model, normalize_obs, room_dir,
                args.horizontal_thresh, args.wall_thresh, args.max_walls, variant, args.max_cylinders,
                args.max_cylinder_frac, args.max_cylinder_radius
            )
            elapsed = time.perf_counter() - t0
            view_geoms = [pcd_rgb] + geometries
            room_name = os.path.basename(room_dir.rstrip("/\\"))
            summary["env"] = "S3DIS"
            summary["idx"] = room_name
            results.append(summary)
            if csv_writer:
                log_csv_row(csv_writer, csv_file, "s3dis", args.model_variant, room_dir.split("/")[-2]
                            if "/" in room_dir else "S3DIS", room_name, "n/a (real scan)", summary, elapsed)
            if view_geoms:
                out_path = os.path.join(out_dir, f"S3DIS_{room_name}.png")
                save_screenshot(view_geoms, out_path)
                print(f"      saved {out_path} ({elapsed:.1f}s)")
                if args.interactive:
                    print(f"      opening window for room {room_name} -- close it to continue to the next room...")
                    window_name = (f"S3DIS/{room_name} -- ground={summary['ground_pts']}pts, "
                                    f"{summary['n_walls']} wall(s), {summary['n_cylinders']} cylinder(s)")
                    screenshot_utils.draw_geometries(view_geoms, window_name=window_name, width=1600, height=900)

        print("\n" + "=" * 118)
        print(f"{'env':>10} {'room':>20} {'pts':>7} {'shapes':>7} {'ground_pts':>11} {'ground_%':>9} "
              f"{'n_walls':>8} {'wall_pts':>9} {'n_cyl':>6} {'cyl_pts':>8}")
        for r in results:
            print(f"{r['env']:>10} {str(r['idx']):>20} {r['n_points']:7d} {r['n_shapes']:7d} "
                  f"{r['ground_pts']:11d} {r['ground_pct']:8.1f}% {r['n_walls']:8d} {r['wall_pts']:9d} "
                  f"{r['n_cylinders']:6d} {r['cylinder_pts']:8d}")
        if csv_file:
            csv_file.close()
        return

    if args.ply_list:
        print(f"\nBatch mode: arbitrary .ply list, {len(args.ply_list)} file(s) -> {out_dir}")
        results = []
        for ply_path in args.ply_list:
            if not os.path.exists(ply_path):
                print(f"  MISSING: {ply_path}")
                continue
            t0 = time.perf_counter()
            geometries, summary = process_frame(
                model, normalize_obs, ply_path, args.z_mode,
                args.horizontal_thresh, args.wall_thresh, args.max_walls, variant, args.max_cylinders,
                args.max_cylinder_frac, args.max_cylinder_radius, voxel=args.voxel,
                tile_walls=args.tile_walls, tile_size=args.tile_size, tile_overlap=args.tile_overlap,
                wall_merge_angle_deg=args.wall_merge_angle_deg, wall_merge_offset_m=args.wall_merge_offset_m
            )
            elapsed = time.perf_counter() - t0
            frame_name = os.path.basename(ply_path)
            summary["env"] = os.path.basename(os.path.dirname(os.path.dirname(ply_path)))
            summary["idx"] = frame_name
            results.append(summary)
            if csv_writer:
                log_csv_row(csv_writer, csv_file, "lidar", args.model_variant, summary["env"],
                            frame_name, "n/a (LiDAR sweep)", summary, elapsed)
            if geometries:
                out_path = os.path.join(out_dir, f"{summary['env']}_{os.path.splitext(frame_name)[0]}.png")
                save_screenshot(geometries, out_path)
                print(f"      saved {out_path} ({elapsed:.1f}s)")
                if args.interactive:
                    print(f"      opening window for {frame_name} -- close it to continue to the next frame...")
                    window_name = (f"{summary['env']}/{frame_name} -- ground={summary['ground_pts']}pts, "
                                    f"{summary['n_walls']} wall(s), {summary['n_cylinders']} cylinder(s)")
                    screenshot_utils.draw_geometries(geometries, window_name=window_name, width=1600, height=900)

        print("\n" + "=" * 118)
        print(f"{'env':>16} {'frame':>28} {'pts':>7} {'shapes':>7} {'ground_pts':>11} {'ground_%':>9} "
              f"{'n_walls':>8} {'wall_pts':>9} {'n_cyl':>6} {'cyl_pts':>8}")
        for r in results:
            print(f"{r['env']:>16} {str(r['idx']):>28} {r['n_points']:7d} {r['n_shapes']:7d} "
                  f"{r['ground_pts']:11d} {r['ground_pct']:8.1f}% {r['n_walls']:8d} {r['wall_pts']:9d} "
                  f"{r['n_cylinders']:6d} {r['cylinder_pts']:8d}")
        if csv_file:
            csv_file.close()
        return

    if args.env and args.indices:
        print(f"\nBatch mode: {args.env}, source={args.source}, {len(args.indices)} frame(s) -> {out_dir}")
        results = []
        for idx in args.indices:
            t0 = time.perf_counter()
            if args.source == "rgbd":
                img_path = os.path.join(WORKSPACE, "data", args.env, "Data_omni", "P0000",
                                         "image_lcam_front", f"{idx:06d}_lcam_front.png")
                depth_path = os.path.join(WORKSPACE, "data", args.env, "Data_omni", "P0000",
                                           "depth_lcam_front", f"{idx:06d}_lcam_front_depth.png")
                if not os.path.exists(img_path) or not os.path.exists(depth_path):
                    print(f"  [{idx}] MISSING: {img_path if not os.path.exists(img_path) else depth_path}")
                    continue
                pcd_rgb, geometries, summary = process_frame_rgbd(
                    model, normalize_obs, img_path, depth_path,
                    args.horizontal_thresh, args.wall_thresh, args.max_walls, variant, args.max_cylinders,
                    args.max_cylinder_frac, args.max_cylinder_radius
                )
                view_geoms = [pcd_rgb] + geometries
                resolution = "640x640"
            else:
                ply_path = os.path.join(WORKSPACE, "data", args.env, "Data_omni", "P0000", "lidar",
                                         f"{idx:06d}_lcam_front_lidar.ply")
                if not os.path.exists(ply_path):
                    print(f"  [{idx}] MISSING: {ply_path}")
                    continue
                geometries, summary = process_frame(
                    model, normalize_obs, ply_path, args.z_mode,
                    args.horizontal_thresh, args.wall_thresh, args.max_walls, variant, args.max_cylinders,
                    args.max_cylinder_frac, args.max_cylinder_radius, voxel=args.voxel,
                    tile_walls=args.tile_walls, tile_size=args.tile_size, tile_overlap=args.tile_overlap,
                    wall_merge_angle_deg=args.wall_merge_angle_deg, wall_merge_offset_m=args.wall_merge_offset_m
                )
                view_geoms = geometries
                resolution = "n/a (LiDAR sweep)"
            elapsed = time.perf_counter() - t0

            summary["env"] = args.env
            summary["idx"] = idx
            results.append(summary)
            if csv_writer:
                log_csv_row(csv_writer, csv_file, args.source, args.model_variant, args.env,
                            f"{idx:06d}", resolution, summary, elapsed)
            if view_geoms:
                out_path = os.path.join(out_dir, f"{args.env}_{idx:06d}.png")
                save_screenshot(view_geoms, out_path)
                print(f"      saved {out_path} ({elapsed:.1f}s)")
                if args.interactive:
                    print(f"      opening window for frame {idx} -- close it to continue to the next frame...")
                    window_name = (f"{args.env}/{idx:06d} ({args.source}) -- ground={summary['ground_pts']}pts, "
                                    f"{summary['n_walls']} wall(s), {summary['n_cylinders']} cylinder(s)")
                    screenshot_utils.draw_geometries(view_geoms, window_name=window_name, width=1600, height=900)

        print("\n" + "=" * 118)
        print(f"{'env':>10} {'idx':>6} {'pts':>7} {'shapes':>7} {'ground_pts':>11} {'ground_%':>9} "
              f"{'n_walls':>8} {'wall_pts':>9} {'n_cyl':>6} {'cyl_pts':>8}")
        for r in results:
            print(f"{r['env']:>10} {r['idx']:>6} {r['n_points']:7d} {r['n_shapes']:7d} "
                  f"{r['ground_pts']:11d} {r['ground_pct']:8.1f}% {r['n_walls']:8d} {r['wall_pts']:9d} "
                  f"{r['n_cylinders']:6d} {r['cylinder_pts']:8d}")
        if csv_file:
            csv_file.close()
        return

    # ---- Single-file interactive mode (lidar only) ----
    print(f"Loading point cloud: {args.ply}")
    t0 = time.perf_counter()
    geometries, summary = process_frame(
        model, normalize_obs, args.ply, args.z_mode,
        args.horizontal_thresh, args.wall_thresh, args.max_walls, variant, args.max_cylinders,
        args.max_cylinder_frac, args.max_cylinder_radius, voxel=args.voxel,
        tile_walls=args.tile_walls, tile_size=args.tile_size, tile_overlap=args.tile_overlap,
        wall_merge_angle_deg=args.wall_merge_angle_deg, wall_merge_offset_m=args.wall_merge_offset_m
    )
    elapsed = time.perf_counter() - t0
    if csv_writer:
        log_csv_row(csv_writer, csv_file, "lidar", args.model_variant, os.path.dirname(args.ply),
                    os.path.basename(args.ply), "n/a (LiDAR sweep)", summary, elapsed)
        csv_file.close()
    if not geometries:
        print("Nothing to visualize.")
        return

    print("\nOpening visualizer window (ground=GREEN, walls=distinct colors, cylinders=gold shades, rest=GRAY).")
    print("Rotate: Left Mouse | Pan: Shift+Left Mouse | Zoom: Scroll | Screenshot: 'P' key | Close window to exit.")
    window_name = f"{os.path.basename(args.ply)} -- ground + {summary['n_walls']} wall(s) + {summary['n_cylinders']} cylinder(s)"
    screenshot_utils.draw_geometries(geometries, window_name=window_name)


if __name__ == "__main__":
    main()
