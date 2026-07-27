# -*- coding: utf-8 -*-
"""
Validates the production --tile_walls implementation (detect_walls_tiled in
detect_ground_and_walls.py) on a bounded block instead of the full 39.7M-pt
L003 cloud -- the full-cloud run kept segfaulting (exit 139) 3/3 times deep
into this session, most likely memory pressure (system RAM/SSD showing
strain), not a code bug: the crash point was identical every time and
BEFORE any tile_walls code even runs (during the mandatory whole-cloud
detect() call, which is unchanged by this flag and had already succeeded
4/4 times moments earlier without it).

This mirrors scratchpad/tiling_pilot.py's own block-selection approach
(same densest-60m-block heuristic), but calls the actual PRODUCTION
functions (find_wall_planes vs. detect_walls_tiled from
detect_ground_and_walls.py) instead of the pilot's scratchpad copies --
so this validates the real, shipped code, including the two bugs fixed
during production implementation (ground-exclusion-on-fallback,
sign-convention).
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
    MODEL_VARIANTS, load_obs_normalizer, find_wall_planes, detect_walls_tiled,
)
import schnabel_ransac

DEFAULT_PLY = os.path.join(ROOT, "my_point_clouds", "Toronto_3D", "L003.ply")
GT_BUILDING_CLASS = 4


def score(pred_mask_full, gt_mask_full):
    tp = int(np.sum(pred_mask_full & gt_mask_full))
    fp = int(np.sum(pred_mask_full & ~gt_mask_full))
    fn = int(np.sum(~pred_mask_full & gt_mask_full))
    cov = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    prec = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, coverage=cov, precision=prec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default=DEFAULT_PLY)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--block", type=float, default=60.0)
    ap.add_argument("--tile_size", type=float, default=20.0)
    ap.add_argument("--tile_overlap", type=float, default=3.0)
    ap.add_argument("--max_walls", type=int, default=8)
    args = ap.parse_args()

    print(f"Loading {args.ply} ...")
    v = PlyData.read(args.ply)["vertex"]
    pts64 = np.stack([np.asarray(v["x"], dtype=np.float64),
                       np.asarray(v["y"], dtype=np.float64),
                       np.asarray(v["z"], dtype=np.float64)], axis=-1)
    labels = np.asarray(v["scalar_Label"], dtype=np.int32)
    pts64 -= pts64.mean(axis=0)

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
    inb = ((pts64[:, 0] >= x0) & (pts64[:, 0] < x0 + args.block) &
           (pts64[:, 1] >= y0) & (pts64[:, 1] < y0 + args.block))
    pts_block = pts64[inb].astype(np.float32)
    labels_block = labels[inb]
    del pts64, labels, inb, H, gx, gy
    gt_building = (labels_block == GT_BUILDING_CLASS)
    print(f"  block {args.block:.0f}m: {len(pts_block):,} raw pts, "
          f"Building {100*gt_building.mean():.1f}%")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_block)
    pcd = pcd.voxel_down_sample(voxel_size=args.voxel)
    pts_ds = np.asarray(pcd.points, dtype=np.float32)
    del pcd
    print(f"  downsampled: {len(pts_ds):,} pts")

    tree = cKDTree(pts_ds)
    nn_idx = np.empty(len(pts_block), dtype=np.int64)
    CHUNK = 5_000_000
    for s in range(0, len(pts_block), CHUNK):
        _, nn_idx[s:s + CHUNK] = tree.query(pts_block[s:s + CHUNK], k=1, workers=-1)
    del tree, pts_block

    variant = MODEL_VARIANTS["synthetic"]
    model = PPO.load(variant["model_path"], device="cpu")
    normalize_obs = load_obs_normalizer(variant["vecnormalize_path"])

    print("\n" + "=" * 74)
    print("Whole-block single pass (find_wall_planes, no tiling) ...")
    t0 = time.perf_counter()
    from detect_ground_and_walls import choose_params_single_shot, find_ground_plane_robust
    eps, min_supp, norm_th = choose_params_single_shot(model, normalize_obs, pts_ds, variant, z_mode="z_up")
    shapes, _ = schnabel_ransac.detect(
        pts_ds, shapes=["plane"], relative_epsilon=False,
        epsilon=eps, normal_thresh=norm_th, min_support=min_supp,
        probability=0.001, normal_knn=20, max_shapes=30,
    )
    ground_shape, _avg_z, _za, _res = find_ground_plane_robust(shapes, pts_ds, z_mode="z_up")
    ground_mask = ground_shape["inlier_mask"] if ground_shape is not None else None
    walls_baseline = find_wall_planes(shapes, pts_ds, wall_thresh=0.3, exclude_mask=ground_mask)[:args.max_walls]
    el = time.perf_counter() - t0
    wall_mask = np.zeros(len(pts_ds), dtype=bool)
    for w in walls_baseline:
        wall_mask |= w["shape"]["inlier_mask"]
    s_base = score(wall_mask[nn_idx], gt_building)
    print(f"  {len(walls_baseline)} wall(s), coverage {s_base['coverage']:.1f}%  "
          f"precision {s_base['precision']:.1f}%   [{el:.1f}s]")

    print("\n" + "=" * 74)
    print(f"Tiled (detect_walls_tiled, tile={args.tile_size:.0f}m overlap={args.tile_overlap:.0f}m) ...")
    t0 = time.perf_counter()
    walls_tiled = detect_walls_tiled(pts_ds, model, normalize_obs, variant, args.voxel, "z_up",
                                      tile_size=args.tile_size, tile_overlap=args.tile_overlap,
                                      wall_thresh=0.3, horizontal_thresh=0.80, max_walls=args.max_walls,
                                      verbose=True)
    el = time.perf_counter() - t0
    wall_mask_t = np.zeros(len(pts_ds), dtype=bool)
    for w in walls_tiled:
        wall_mask_t |= w["shape"]["inlier_mask"]
    s_tiled = score(wall_mask_t[nn_idx], gt_building)
    print(f"  {len(walls_tiled)} wall(s), coverage {s_tiled['coverage']:.1f}%  "
          f"precision {s_tiled['precision']:.1f}%   [{el:.1f}s]")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print(f"  baseline (no tiling): {s_base['coverage']:6.1f}% coverage  {s_base['precision']:6.1f}% precision  "
          f"{len(walls_baseline)} wall(s)")
    print(f"  --tile_walls:         {s_tiled['coverage']:6.1f}% coverage  {s_tiled['precision']:6.1f}% precision  "
          f"{len(walls_tiled)} wall(s)   [{s_tiled['coverage']-s_base['coverage']:+.1f} pp]")


if __name__ == "__main__":
    main()
