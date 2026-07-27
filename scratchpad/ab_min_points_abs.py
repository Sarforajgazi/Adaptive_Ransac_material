# -*- coding: utf-8 -*-
"""
Isolation test for ONE variable: find_ground_plane_robust()'s `min_points_abs`.

Why: on the full L003 cloud, compare_toronto3d_gt.py measured 2.8% ground
coverage. tiling_pilot.py, on a 150m block holding 92% of that same cloud,
measured 96.3% -- from the same model, same eps, same voxel, same number of
detected planes. The one differing input is min_points_abs: the default 500
vs. a value scaled to point count.

Hypothesis: 500 is an absolute floor tuned for whole ~40k-point frames (it
was chosen to reject 168-391 point noise fragments). On a multi-million-point
cloud it is effectively zero, so find_ground_plane_robust's "most extreme
average height wins" tie-break can hand the ground label to a tiny sliver --
exactly the failure mode that floor exists to prevent, reappearing because
the floor does not scale.

This runs the SAME block through the SAME detection twice, changing only
min_points_abs. Everything else is held fixed, including the RANSAC seed
problem (see caveat printed at the end).
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

from detect_ground_and_walls import MODEL_VARIANTS, load_obs_normalizer
from tiling_pilot import detect_on_points, score, GT_GROUND_CLASS, GT_BUILDING_CLASS

DEFAULT_PLY = os.path.join(ROOT, "my_point_clouds", "Toronto_3D", "L003.ply")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default=DEFAULT_PLY)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--block", type=float, default=60.0)
    ap.add_argument("--repeats", type=int, default=2,
                    help="repeats per setting -- RANSAC is not seedable, so a "
                         "single run of each cannot separate signal from jitter")
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
    gt_ground = (labels_block == GT_GROUND_CLASS)
    gt_building = (labels_block == GT_BUILDING_CLASS)
    print(f"  block {args.block:.0f}m: {len(pts_block):,} raw pts, "
          f"Ground {100*gt_ground.mean():.1f}%, Building {100*gt_building.mean():.1f}%")

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

    scaled = max(25, int(0.01 * len(pts_ds)))
    settings = [("original (500, hardcoded default)", 500),
                (f"scaled (1% of points = {scaled:,})", scaled)]

    print("\n" + "=" * 74)
    print(f"{'setting':>38} {'run':>4} {'planes':>7} {'grnd cov':>9} {'grnd prec':>10} {'sec':>7}")
    results = {}
    for name, mpa in settings:
        covs = []
        for r in range(args.repeats):
            t0 = time.perf_counter()
            out = detect_on_points(model, normalize_obs, pts_ds, variant, args.voxel,
                                    z_mode="z_up", label=name, min_points_abs=mpa)
            el = time.perf_counter() - t0
            if out["ground_mask"] is not None:
                sc = score(out["ground_mask"][nn_idx], gt_ground)
            else:
                sc = dict(coverage=0.0, precision=0.0)
            covs.append(sc["coverage"])
            print(f"{name:>38} {r+1:>4} {out['n_shapes']:>7} "
                  f"{sc['coverage']:8.1f}% {sc['precision']:9.1f}% {el:7.1f}")
        results[name] = covs

    print("\n" + "=" * 74)
    print("Mean ground coverage by setting:")
    for name, covs in results.items():
        print(f"  {name:>38}: {np.mean(covs):6.1f}%  (runs: {', '.join(f'{c:.1f}' for c in covs)})")
    print("\nCaveat: the Schnabel backend reseeds itself per call and is not "
          "seedable from Python, so runs of the same setting differ. Repeats "
          "above are there to show whether the gap between settings is larger "
          "than that jitter.")


if __name__ == "__main__":
    main()
