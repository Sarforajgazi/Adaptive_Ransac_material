# -*- coding: utf-8 -*-
"""
tiling_pilot.py -- Phase 1 of the tiled-detection plan.

Question this answers: does splitting a large point cloud into small
overlapping XY tiles, and running the existing per-cloud detection on each
tile independently, raise ground coverage above the single-plane ceiling?

Baseline to beat: on the whole of Toronto-3D L003, a single detect() call
reaches only ~2.8% coverage of the real `Ground` class (95.6% precision) --
not a precision problem, an architectural one: find_ground_plane_robust()
reports exactly ONE plane, and a 250m road has real camber and grade.
(Walls, by contrast, already reach 45.9%/98.9% -- a facade genuinely IS one
plane, so this ceiling is ground-specific.)

Nothing in detect_ground_and_walls.py is modified. This script reuses its
functions directly so the pilot measures the real pipeline, not a
reimplementation of it.

Per-tile adjustments, each with a documented reason:
  * tile-local recentering  -- keeps coordinates near origin (precision) and
    the scene in the ~10m range the agent actually trained on
  * eps = max(eps_agent, K_MIN * voxel)  -- CLIP AS A FLOOR, the agent still
    drives eps. Anchor: voxel 0.02 + eps 0.08 (k=4) worked on L003; voxel
    0.3 + eps 0.08 (k=0.27) found nothing at all. The floor only fires in
    that already-proven-dead regime, so the agent's adaptivity (its
    strongest measured signal, r=0.52) is preserved.
  * normal_knn = min(20, n-1)  -- process_frame hardcodes 20 unguarded;
    small tiles can hold fewer points. Same guard traversability.py:189 uses.
  * min_points_abs scaled per tile -- it is an ABSOLUTE 500 by default,
    tuned for whole ~40k-point frames to reject 168-391 point noise
    fragments (~1% of frame). Applied unscaled to a small tile it would
    reject legitimate ground. Same class of rescaling traversability.py:69-80
    documents for its own small cells.
"""
import os
import sys
import time
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from plyfile import PlyData
from stable_baselines3 import PPO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "schnabel_cython"))

from detect_ground_and_walls import (
    MODEL_VARIANTS, load_obs_normalizer, choose_params_single_shot,
    find_ground_plane_robust, find_wall_planes, plane_normal_and_residual,
)
import schnabel_ransac

DEFAULT_PLY = os.path.join(ROOT, "my_point_clouds", "Toronto_3D", "L003.ply")
GT_GROUND_CLASS = 1        # Toronto-3D scalar_Label: 1 = Ground
GT_BUILDING_CLASS = 4      # 4 = Building
K_MIN = 2.0                # eps floor multiplier (eps >= K_MIN * voxel)
COPLANAR_ANGLE_DEG = 5.0   # fragmentation clustering thresholds
COPLANAR_OFFSET_M = 0.20


def detect_on_points(model, normalize_obs, points, variant, voxel, z_mode="z_up",
                      horizontal_thresh=0.80, wall_thresh=0.3, label="",
                      min_points_abs=None):
    """The detect half of process_frame(), on an in-memory array instead of a
    file. Returns a dict of results + diagnostics (never raises on an empty
    or degenerate input -- a tiled run must not die on one bad tile)."""
    n = len(points)
    out = {
        "n_points": n, "eps_agent": None, "eps_used": None, "eps_clipped": False,
        "min_supp": None, "norm_th": None, "n_shapes": 0,
        "ground_mask": None, "ground_z_align": None, "fallback_fired": False,
        "n_walls": 0, "wall_mask": None,
    }
    if n < 10:
        return out

    # Tile-local recentering: near-origin coords for precision, and an
    # in-distribution scene scale for the agent's observation.
    local = points - points.mean(axis=0)

    eps_agent, min_supp, norm_th = choose_params_single_shot(
        model, normalize_obs, local, variant, z_mode=z_mode)
    eps_used = max(eps_agent, K_MIN * voxel) if voxel and voxel > 0 else eps_agent
    out.update(eps_agent=eps_agent, eps_used=eps_used,
               eps_clipped=bool(eps_used > eps_agent), min_supp=min_supp, norm_th=norm_th)

    try:
        shapes, _ = schnabel_ransac.detect(
            local, shapes=["plane"], relative_epsilon=False,
            epsilon=eps_used, normal_thresh=norm_th, min_support=min_supp,
            probability=0.001, normal_knn=min(20, n - 1), max_shapes=30,
        )
    except Exception as e:                      # one bad tile must not kill the sweep
        print(f"      [{label}] detect() failed: {e}")
        return out

    out["n_shapes"] = len(shapes)
    if not shapes:
        return out

    # min_points_abs is an absolute count tuned for whole frames (see module
    # docstring); scale it to this tile, with a small floor so a sparse tile
    # doesn't accept 3-point "planes". An explicit override lets the caller
    # pin the original hardcoded 500 to isolate its effect.
    min_pts_abs = min_points_abs if min_points_abs is not None else max(25, int(0.01 * n))

    ground_shape, _avg_z, z_align, _resid = find_ground_plane_robust(
        shapes, local, z_mode=z_mode, horizontal_thresh=horizontal_thresh,
        min_points_abs=min_pts_abs)
    if ground_shape is not None:
        out["ground_mask"] = ground_shape["inlier_mask"]
        out["ground_z_align"] = z_align
        # find_ground_plane_robust falls back to "largest shape wins regardless
        # of orientation" when nothing clears horizontal_thresh -- meaning a
        # tile with no real ground still emits one, possibly a wall. Detect it
        # the same way the function decides it, so we can count how often it fires.
        out["fallback_fired"] = bool(z_align is not None and z_align < horizontal_thresh)

    walls = find_wall_planes(shapes, local, wall_thresh=wall_thresh,
                             exclude_mask=out["ground_mask"])
    out["n_walls"] = len(walls)
    if walls:
        wm = np.zeros(n, dtype=bool)
        for w in walls:
            wm |= w["shape"]["inlier_mask"]
        out["wall_mask"] = wm
    return out


def score(pred_mask_full, gt_mask_full):
    tp = int(np.sum(pred_mask_full & gt_mask_full))
    fp = int(np.sum(pred_mask_full & ~gt_mask_full))
    fn = int(np.sum(~pred_mask_full & gt_mask_full))
    cov = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    prec = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, coverage=cov, precision=prec)


def plane_signature(plane_points_block):
    """(sign-normalized normal, offset) in the BLOCK frame, for cross-tile
    coplanarity clustering. Takes the plane's points directly (already in
    block frame) rather than a mask over the whole cloud -- masking the full
    array once per tile would be needlessly O(N) per tile.

    PCA normals have arbitrary sign, so force normal[2] >= 0 before any
    comparison, otherwise the same physical plane can appear twice."""
    all_mask = np.ones(len(plane_points_block), dtype=bool)
    normal, _z, _r = plane_normal_and_residual(plane_points_block, all_mask)
    if not np.all(np.isfinite(normal)) or np.allclose(normal, 0):
        return None, None
    if normal[2] < 0:
        normal = -normal
    d = float(np.dot(normal, plane_points_block.mean(axis=0)))
    return normal, d


def count_coplanar_groups(signatures):
    """Greedy clustering of per-tile ground planes -- how many DISTINCT real
    surfaces did the tiles find? A low group count relative to tile count
    means merging would collapse them (tiling found one road repeatedly);
    a high count means genuinely different planes (real camber/grade)."""
    cos_thresh = np.cos(np.deg2rad(COPLANAR_ANGLE_DEG))
    groups = []
    for n, d in signatures:
        placed = False
        for g in groups:
            if abs(float(np.dot(n, g["n"]))) >= cos_thresh and abs(d - g["d"]) <= COPLANAR_OFFSET_M:
                g["count"] += 1
                placed = True
                break
        if not placed:
            groups.append({"n": n, "d": d, "count": 1})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default=DEFAULT_PLY)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--block", type=float, default=60.0, help="block edge length in meters")
    ap.add_argument("--tile_sizes", type=float, nargs="+", default=[15.0, 20.0, 25.0])
    ap.add_argument("--overlap", type=float, default=3.0)
    ap.add_argument("--z_mode", default="z_up")
    args = ap.parse_args()

    # ---- Load once, float64, global recenter (mirrors load_ply_xyz(recenter=True)) ----
    print(f"Loading {args.ply} ...")
    t0 = time.perf_counter()
    v = PlyData.read(args.ply)["vertex"]
    pts64 = np.stack([np.asarray(v["x"], dtype=np.float64),
                       np.asarray(v["y"], dtype=np.float64),
                       np.asarray(v["z"], dtype=np.float64)], axis=-1)
    labels = np.asarray(v["scalar_Label"], dtype=np.int32)
    pts64 -= pts64.mean(axis=0)
    print(f"  {len(pts64):,} points in {time.perf_counter()-t0:.1f}s")

    # ---- Auto-pick the densest block (the road corridor, not empty space) ----
    cell = 10.0
    gx = ((pts64[:, 0] - pts64[:, 0].min()) // cell).astype(np.int32)
    gy = ((pts64[:, 1] - pts64[:, 1].min()) // cell).astype(np.int32)
    H = np.zeros((gx.max() + 1, gy.max() + 1), dtype=np.int64)
    np.add.at(H, (gx, gy), 1)
    k = int(args.block // cell)
    best = None
    for i in range(max(1, H.shape[0] - k)):
        for j in range(max(1, H.shape[1] - k)):
            s = int(H[i:i + k, j:j + k].sum())
            if best is None or s > best[0]:
                best = (s, i, j)
    x0 = pts64[:, 0].min() + best[1] * cell
    y0 = pts64[:, 1].min() + best[2] * cell
    in_block = ((pts64[:, 0] >= x0) & (pts64[:, 0] < x0 + args.block) &
                (pts64[:, 1] >= y0) & (pts64[:, 1] < y0 + args.block))
    # Cast to float32 and drop the float64 original immediately: the global
    # recenter (the part that NEEDS float64) already happened, and holding a
    # ~1GB float64 copy alongside the KDTree + the C++ detector's own octree
    # is what makes a large --block segfault.
    pts_block = pts64[in_block].astype(np.float32)
    labels_block = labels[in_block]
    del pts64, labels, in_block, H, gx, gy
    print(f"  block: {args.block:.0f}x{args.block:.0f}m at ({x0:.1f}, {y0:.1f}), "
          f"{len(pts_block):,} raw points")
    gt_ground = (labels_block == GT_GROUND_CLASS)
    gt_building = (labels_block == GT_BUILDING_CLASS)
    print(f"  block GT: Ground {gt_ground.sum():,} ({100*gt_ground.mean():.1f}%), "
          f"Building {gt_building.sum():,} ({100*gt_building.mean():.1f}%)")

    # ---- Voxel downsample the block (what detection actually runs on) ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_block)
    pcd = pcd.voxel_down_sample(voxel_size=args.voxel)
    pts_ds = np.asarray(pcd.points, dtype=np.float32)
    del pcd
    print(f"  downsampled (voxel={args.voxel}m): {len(pts_ds):,} points")

    # ---- KDTree lift: downsampled prediction -> full-res block, for GT scoring ----
    # Queried in chunks: a single query over tens of millions of points
    # allocates a large result array on top of the tree itself, and this
    # script has to coexist with the C++ detector's own octree afterwards.
    print("  building KDTree for full-res lift ...")
    t0 = time.perf_counter()
    tree = cKDTree(pts_ds)
    nn_idx = np.empty(len(pts_block), dtype=np.int64)
    CHUNK = 5_000_000
    for s in range(0, len(pts_block), CHUNK):
        _, nn_idx[s:s + CHUNK] = tree.query(pts_block[s:s + CHUNK], k=1, workers=-1)
    del tree, pts_block
    print(f"    done in {time.perf_counter()-t0:.1f}s")

    variant = MODEL_VARIANTS["synthetic"]
    model = PPO.load(variant["model_path"], device="cpu")
    normalize_obs = load_obs_normalizer(variant["vecnormalize_path"])

    results = []

    # ---- Baseline: ONE detect() over the whole block (the thing to beat) ----
    print("\n" + "=" * 78)
    print("BASELINE -- single detect() over the whole block (no tiling)")
    t0 = time.perf_counter()
    b = detect_on_points(model, normalize_obs, pts_ds, variant, args.voxel,
                          z_mode=args.z_mode, label="baseline")
    b_elapsed = time.perf_counter() - t0
    if b["ground_mask"] is not None:
        b_ground_full = b["ground_mask"][nn_idx]
        b_score = score(b_ground_full, gt_ground)
    else:
        b_score = dict(tp=0, fp=0, fn=int(gt_ground.sum()), coverage=0.0, precision=0.0)
    b_wall_score = (score(b["wall_mask"][nn_idx], gt_building)
                    if b["wall_mask"] is not None else None)
    print(f"  {b['n_shapes']} plane(s), eps_agent={b['eps_agent']} -> used {b['eps_used']}"
          f"{' (CLIPPED)' if b['eps_clipped'] else ''}, min_supp={b['min_supp']}, "
          f"norm_th={b['norm_th']}, {b['n_walls']} wall(s), {b_elapsed:.1f}s")
    print(f"  GROUND coverage {b_score['coverage']:.1f}%  precision {b_score['precision']:.1f}%")
    if b_wall_score:
        print(f"  WALLS  coverage {b_wall_score['coverage']:.1f}%  precision {b_wall_score['precision']:.1f}%")
    results.append(("baseline (no tiling)", b_score, b_wall_score, 1, 0, 0, b_elapsed, None))

    # ---- Tiled sweeps ----
    for tsize in args.tile_sizes:
        stride = tsize - args.overlap
        print("\n" + "=" * 78)
        print(f"TILED -- tile={tsize:.0f}m overlap={args.overlap:.0f}m (stride {stride:.0f}m)")
        t_start = time.perf_counter()

        xs = np.arange(x0, x0 + args.block, stride)
        ys = np.arange(y0, y0 + args.block, stride)
        union_ground = np.zeros(len(pts_ds), dtype=bool)
        union_wall = np.zeros(len(pts_ds), dtype=bool)
        n_tiles = n_nonempty = n_found = n_fallback = n_clipped = 0
        tile_pt_counts, signatures, tile_walls = [], [], 0

        for tx in xs:
            for ty in ys:
                n_tiles += 1
                m = ((pts_ds[:, 0] >= tx) & (pts_ds[:, 0] < tx + tsize) &
                     (pts_ds[:, 1] >= ty) & (pts_ds[:, 1] < ty + tsize))
                gidx = np.flatnonzero(m)
                if len(gidx) < 10:
                    continue
                n_nonempty += 1
                tile_pt_counts.append(len(gidx))
                r = detect_on_points(model, normalize_obs, pts_ds[gidx], variant,
                                      args.voxel, z_mode=args.z_mode,
                                      label=f"{tx:.0f},{ty:.0f}")
                if r["eps_clipped"]:
                    n_clipped += 1
                if r["ground_mask"] is not None:
                    n_found += 1
                    if r["fallback_fired"]:
                        n_fallback += 1
                    gmask_global = gidx[r["ground_mask"]]
                    union_ground[gmask_global] = True
                    nrm, d = plane_signature(pts_ds[gmask_global])
                    if nrm is not None:
                        signatures.append((nrm, d))
                if r["wall_mask"] is not None:
                    union_wall[gidx[r["wall_mask"]]] = True
                    tile_walls += r["n_walls"]

        elapsed = time.perf_counter() - t_start
        s = score(union_ground[nn_idx], gt_ground)
        ws = score(union_wall[nn_idx], gt_building) if union_wall.any() else None
        groups = count_coplanar_groups(signatures)

        tpc = np.array(tile_pt_counts) if tile_pt_counts else np.array([0])
        print(f"  tiles: {n_nonempty} non-empty of {n_tiles}; points/tile "
              f"min={tpc.min():,} med={int(np.median(tpc)):,} max={tpc.max():,}")
        print(f"  ground found in {n_found}/{n_nonempty} tiles; "
              f"non-horizontal fallback fired {n_fallback}x; eps clipped {n_clipped}x; "
              f"{tile_walls} wall(s) total")
        print(f"  distinct coplanar ground groups: {len(groups)} "
              f"(from {len(signatures)} per-tile planes)")
        print(f"  GROUND coverage {s['coverage']:.1f}%  precision {s['precision']:.1f}%   [{elapsed:.1f}s]")
        if ws:
            print(f"  WALLS  coverage {ws['coverage']:.1f}%  precision {ws['precision']:.1f}%")
        results.append((f"tiled {tsize:.0f}m/{args.overlap:.0f}m", s, ws,
                        n_nonempty, n_fallback, len(groups), elapsed, tpc))

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("SUMMARY (block: %.0fx%.0fm, voxel=%.2fm)" % (args.block, args.block, args.voxel))
    print(f"{'config':>22} {'grnd cov':>9} {'grnd prec':>10} {'wall cov':>9} "
          f"{'tiles':>6} {'fallbk':>7} {'groups':>7} {'sec':>7}")
    for name, s, ws, ntiles, nfb, ngrp, el, _ in results:
        wc = f"{ws['coverage']:.1f}%" if ws else "-"
        print(f"{name:>22} {s['coverage']:8.1f}% {s['precision']:9.1f}% {wc:>9} "
              f"{ntiles:6d} {nfb:7d} {ngrp:7d} {el:7.1f}")
    base_cov = results[0][1]["coverage"]
    best = max(results[1:], key=lambda r: r[1]["coverage"]) if len(results) > 1 else None
    if best:
        print(f"\nBest tiled config: {best[0]} at {best[1]['coverage']:.1f}% ground coverage "
              f"vs {base_cov:.1f}% baseline "
              f"({best[1]['coverage'] - base_cov:+.1f} pp).")


if __name__ == "__main__":
    main()
