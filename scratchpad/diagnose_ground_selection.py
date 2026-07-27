# -*- coding: utf-8 -*-
"""
Why does the FULL L003 cloud score 2.8% ground coverage when a 150m block of
that same cloud scores 96.3%?

Ruled out already:
  * precision / float32 (fixed earlier; residual is 0.04m, not 25m)
  * min_points_abs (A/B tested at 500 vs 66,677 -- identical, 97.9% both)

Remaining suspect: the SELECTION rule. find_ground_plane_robust() picks the
horizontal candidate with the most extreme average height -- for z_up, the
LOWEST. That is a sound tie-break on a single ~40m frame. Over a 273x331m
city block it may be picking a small low-lying patch (a dip, a sunken
entrance, a low car park) in preference to the actual road surface, which
sits slightly higher but is 25x larger.

This prints every plane candidate with its size, average height and
orientation, marks which one the current rule selects, and computes which
one a coverage-optimal oracle would have selected. If the selected plane is
small and low while a much larger horizontal plane sits just above it, the
selection rule is the bug -- not tiling, and not the thresholds.
"""
import os
import sys
import time
import argparse
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData
from stable_baselines3 import PPO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "schnabel_cython"))

from detect_ground_and_walls import (
    MODEL_VARIANTS, load_obs_normalizer, choose_params_single_shot,
    find_ground_plane_robust, plane_normal_and_residual,
)
from ransac_env import load_ply_xyz
import schnabel_ransac

DEFAULT_PLY = os.path.join(ROOT, "my_point_clouds", "Toronto_3D", "L003.ply")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default=DEFAULT_PLY)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--horizontal_thresh", type=float, default=0.80)
    ap.add_argument("--label_field", default="scalar_Label",
                    help="Toronto-3D: scalar_Label (1=Ground). "
                         "OpenTopography/ASPRS: classification (2=Ground).")
    ap.add_argument("--ground_class", type=int, default=1)
    ap.add_argument("--height_band", type=float, default=0.5,
                    help="proposed fix: among horizontal candidates within this "
                         "many meters of the lowest, pick the LARGEST")
    ap.add_argument("--s3dis_room", default=None,
                    help="path to an S3DIS room dir (coord.npy + segment.npy). "
                         "Overrides --ply. Use --ground_class 1 (floor).")
    ap.add_argument("--eps_floor_k", type=float, default=0.0,
                    help="apply the plan's eps floor: eps = max(eps_agent, k*voxel). "
                         "0 disables. Needed on sparse clouds where the agent's "
                         "absolute eps falls below actual point spacing.")
    ap.add_argument("--adaptive_k", type=float, nargs="+", default=[3.0, 5.0, 8.0, 12.0],
                    help="Option 3: multipliers of the largest candidate's own "
                         "residual to test as an adaptive height band.")
    ap.add_argument("--adaptive_floor", type=float, default=0.02,
                    help="minimum adaptive band, meters -- guards against a "
                         "near-zero residual on a suspiciously clean candidate.")
    args = ap.parse_args()

    if args.s3dis_room:
        # S3DIS ships Pointcept-style .npy, not .ply. 13-class segment.npy:
        # 0=ceiling, 1=floor, 2=wall, ... -- so a room contains BOTH a floor
        # and a ceiling, which is exactly the case find_ground_plane_robust's
        # docstring says broke a "largest support wins" rule.
        print(f"Loading S3DIS room {args.s3dis_room} ...")
        raw_xyz = np.load(os.path.join(args.s3dis_room, "coord.npy")).astype(np.float64)
        labels = np.load(os.path.join(args.s3dis_room, "segment.npy")).reshape(-1).astype(np.int32)
        gt_ground = (labels == args.ground_class)
        print(f"  {len(raw_xyz):,} pts, real Floor (segment=={args.ground_class}) = "
              f"{gt_ground.sum():,} ({100*gt_ground.mean():.1f}%), "
              f"Ceiling (segment==0) = {int((labels==0).sum()):,}")
        centroid = raw_xyz.mean(axis=0)
        points = (raw_xyz - centroid).astype(np.float32)
        raw_xyz = points.copy()
        if args.voxel and args.voxel > 0:
            import open3d as o3d
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            points = np.asarray(pc.voxel_down_sample(args.voxel).points, dtype=np.float32)
        print(f"  {len(points):,} pts used for detection (voxel={args.voxel})")
    else:
        print(f"Loading GT labels from {args.ply} ...")
        v = PlyData.read(args.ply)["vertex"]
        raw_xyz = np.stack([np.asarray(v["x"], dtype=np.float64),
                             np.asarray(v["y"], dtype=np.float64),
                             np.asarray(v["z"], dtype=np.float64)], axis=-1)
        labels = np.asarray(v[args.label_field]).astype(np.int32)
        del v
        gt_ground = (labels == args.ground_class)
        print(f"  {len(raw_xyz):,} pts, real Ground ({args.label_field}=="
              f"{args.ground_class}) = {gt_ground.sum():,} ({100*gt_ground.mean():.1f}%)")

        print("Loading + downsampling via the real pipeline path ...")
        points, centroid = load_ply_xyz(args.ply, voxel_size=args.voxel, recenter=True)
        raw_xyz = (raw_xyz - centroid).astype(np.float32)
        print(f"  {len(points):,} downsampled pts")

    tree = cKDTree(points)
    nn_idx = np.empty(len(raw_xyz), dtype=np.int64)
    for s in range(0, len(raw_xyz), 5_000_000):
        _, nn_idx[s:s + 5_000_000] = tree.query(raw_xyz[s:s + 5_000_000], k=1, workers=-1)
    del tree, raw_xyz

    variant = MODEL_VARIANTS["synthetic"]
    model = PPO.load(variant["model_path"], device="cpu")
    normalize_obs = load_obs_normalizer(variant["vecnormalize_path"])
    eps, min_supp, norm_th = choose_params_single_shot(
        model, normalize_obs, points, variant, z_mode="z_up")
    print(f"  agent chose eps={eps} min_supp={min_supp} norm_th={norm_th}")
    if args.eps_floor_k > 0 and args.voxel > 0:
        floored = max(eps, args.eps_floor_k * args.voxel)
        if floored > eps:
            print(f"  eps floor applied: {eps} -> {floored} "
                  f"({args.eps_floor_k} x voxel {args.voxel})")
        eps = floored

    print("Running detect() on the full cloud ...")
    t0 = time.perf_counter()
    shapes, _ = schnabel_ransac.detect(
        points, shapes=["plane"], relative_epsilon=False,
        epsilon=eps, normal_thresh=norm_th, min_support=min_supp,
        probability=0.001, normal_knn=20, max_shapes=30)
    print(f"  {len(shapes)} plane(s) in {time.perf_counter()-t0:.1f}s")

    selected, _avgz, _za, _res = find_ground_plane_robust(
        shapes, points, z_mode="z_up", horizontal_thresh=args.horizontal_thresh)

    print("\n" + "=" * 96)
    print(f"{'#':>3} {'n_points':>11} {'% cloud':>8} {'avg_z':>9} {'z_align':>8} "
          f"{'horiz?':>7} {'residual':>8} {'gt cov':>8} {'gt prec':>8}  {'':>3}")
    rows = []
    for i, sh in enumerate(shapes):
        m = sh["inlier_mask"]
        if m.sum() < 10:
            continue
        _n, z_align, residual = plane_normal_and_residual(points, m)
        avg_z = float(points[m][:, 2].mean())
        pred_full = m[nn_idx]
        tp = int(np.sum(pred_full & gt_ground))
        fp = int(np.sum(pred_full & ~gt_ground))
        cov = 100.0 * tp / max(int(gt_ground.sum()), 1)
        prec = 100.0 * tp / max(tp + fp, 1)
        is_sel = selected is not None and np.array_equal(m, selected["inlier_mask"])
        rows.append((i, int(m.sum()), avg_z, z_align, cov, prec, is_sel, residual))

    for i, n, avg_z, za, cov, prec, is_sel, res in sorted(rows, key=lambda r: r[2]):
        horiz = "YES" if za >= args.horizontal_thresh else "no"
        mark = "<== SELECTED" if is_sel else ""
        print(f"{i:>3} {n:>11,} {100*n/len(points):7.2f}% {avg_z:9.2f} {za:8.3f} "
              f"{horiz:>7} {res:8.4f} {cov:7.1f}% {prec:7.1f}%  {mark}")

    print("\n" + "=" * 96)
    sel_row = next((r for r in rows if r[6]), None)
    if sel_row:
        print(f"Current rule selected plane #{sel_row[0]}: {sel_row[1]:,} pts, "
              f"avg_z={sel_row[2]:.2f}, coverage {sel_row[4]:.1f}%")
    horiz_rows = [r for r in rows if r[3] >= args.horizontal_thresh]
    if horiz_rows:
        oracle = max(horiz_rows, key=lambda r: r[4])
        biggest = max(horiz_rows, key=lambda r: r[1])

        # PROPOSED RULE: among horizontal candidates within `height_band` of
        # the lowest, take the largest. Rationale: a sub-metre height spread
        # is inside normal road camber/grade, so those candidates are the same
        # physical surface level -- at which point support size, not a
        # centimetre of height, is the honest discriminator.
        lowest_z = min(r[2] for r in horiz_rows)
        in_band = [r for r in horiz_rows if r[2] - lowest_z <= args.height_band]
        proposed = max(in_band, key=lambda r: r[1])

        print(f"Coverage-optimal horizontal plane #{oracle[0]}: {oracle[1]:,} pts, "
              f"avg_z={oracle[2]:.2f}, coverage {oracle[4]:.1f}%")
        print(f"Largest horizontal plane        #{biggest[0]}: {biggest[1]:,} pts, "
              f"avg_z={biggest[2]:.2f}, coverage {biggest[4]:.1f}%")
        print(f"PROPOSED rule (largest within {args.height_band}m of lowest) "
              f"picks #{proposed[0]}: {proposed[1]:,} pts, avg_z={proposed[2]:.2f}, "
              f"coverage {proposed[4]:.1f}%  [{len(in_band)} of {len(horiz_rows)} "
              f"horizontal candidates in band]")

        print("\n" + "-" * 96)
        if sel_row:
            delta = proposed[4] - sel_row[4]
            print(f"  current rule : {sel_row[4]:6.1f}% coverage  ({sel_row[1]:,} pts)")
            print(f"  PROPOSED rule: {proposed[4]:6.1f}% coverage  ({proposed[1]:,} pts)   "
                  f"[{delta:+.1f} pp]")
            print(f"  oracle (best): {oracle[4]:6.1f}% coverage  ({oracle[1]:,} pts)")
            if proposed[0] == oracle[0] and proposed[0] != sel_row[0]:
                print("\n=> PROPOSED RULE RECOVERS THE OPTIMAL PLANE on this dataset.")
            elif proposed[0] == sel_row[0] and sel_row[0] == oracle[0]:
                print("\n=> Current rule was already optimal here; proposed rule agrees "
                      "(no regression).")
            elif proposed[0] == sel_row[0]:
                print("\n=> Proposed rule changes nothing here -- the competing planes "
                      "are further apart in height than the band.")
            else:
                print("\n=> Proposed rule differs from current but is NOT optimal here "
                      "-- band width may need tuning, or size is the wrong tie-break "
                      "for this scene.")

        # ---- OPTION 3: adaptive band, scaled by the largest candidate's own
        # measured residual (plane_normal_and_residual's mean abs deviation --
        # a direct measurement of how rough THIS surface actually is in THIS
        # scene) instead of a fixed 0.5m constant. Reported alongside the
        # fixed band, same dataset, same run -- not a separate claim.
        print("\n" + "=" * 96)
        print("OPTION 3: adaptive band = K x residual of the largest horizontal candidate")
        ref_residual = biggest[7]
        print(f"  reference: largest horizontal candidate #{biggest[0]}, "
              f"residual={ref_residual:.4f}m")
        for k in args.adaptive_k:
            band_k = max(args.adaptive_floor, k * ref_residual)
            in_band_k = [r for r in horiz_rows if r[2] - lowest_z <= band_k]
            pick_k = max(in_band_k, key=lambda r: r[1])
            tag = ("MATCHES ORACLE" if pick_k[0] == oracle[0]
                   else "WRONG" if pick_k[0] != sel_row[0] else "no change")
            print(f"  K={k:>4.1f}  band={band_k:7.4f}m  picks #{pick_k[0]:>2} "
                  f"({pick_k[1]:>10,} pts, {pick_k[4]:5.1f}% coverage)  [{tag}]")


if __name__ == "__main__":
    main()
