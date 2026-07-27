# -*- coding: utf-8 -*-
"""
Scores a detect_ground_and_walls.py-style ground prediction against
Toronto-3D's real per-point ground truth (scalar_Label field, class 1 =
Ground). Reuses the exact same functions detect_ground_and_walls.py uses
internally (choose_params_single_shot, find_ground_plane_robust,
find_wall_planes) rather than reimplementing them, so the result reflects
exactly what a real --ply run would have done.

Downsampled prediction vs full-resolution truth are aligned via
nearest-neighbor lookup (scipy.cKDTree) -- each original (full-res) point
inherits its nearest downsampled point's predicted class. This gives a
proper per-point confusion matrix over the ENTIRE original cloud, not just
the predicted-ground subset.
"""
import os
import sys
import time
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
    find_ground_plane_robust, find_wall_planes, WALL_COLORS,
)
from ransac_env import load_ply_xyz
import schnabel_ransac
import screenshot_utils

PLY_PATH = os.path.join(ROOT, "my_point_clouds", "Toronto_3D", "L003.ply")
VOXEL = 0.02  # user's actual value for their L003 run
GT_CLASS_NAMES = {0: "Unclassified", 1: "Ground", 2: "Road_markings", 3: "Natural",
                   4: "Building", 5: "Utility_line", 6: "Pole", 7: "Car", 8: "Fence"}

print(f"Loading ground truth from {PLY_PATH} ...")
t0 = time.perf_counter()
raw = PlyData.read(PLY_PATH)["vertex"]
points_raw = np.stack([raw["x"], raw["y"], raw["z"]], axis=-1).astype(np.float32)
labels_raw = np.asarray(raw["scalar_Label"], dtype=np.int32)
print(f"  {len(points_raw):,} raw points loaded in {time.perf_counter()-t0:.1f}s")

print("\nGround-truth class breakdown (full resolution):")
for cls_id, name in GT_CLASS_NAMES.items():
    n = int(np.sum(labels_raw == cls_id))
    if n > 0:
        print(f"  {name:>14} (id={cls_id}): {n:>10,} pts ({100*n/len(labels_raw):5.2f}%)")

print(f"\nRunning detection (model=synthetic, voxel={VOXEL}m) ...")
variant = MODEL_VARIANTS["synthetic"]
model = PPO.load(variant["model_path"], device="cpu")
normalize_obs = load_obs_normalizer(variant["vecnormalize_path"])

points_pred, centroid = load_ply_xyz(PLY_PATH, voxel_size=VOXEL, recenter=True)
points_raw = points_raw - centroid  # same shift applied to both -- keeps KDTree alignment exact
print(f"  recentered on centroid {centroid} (float64, pre-cast -- matches detect_ground_and_walls.py's process_frame fix)")
eps, min_supp, norm_th = choose_params_single_shot(model, normalize_obs, points_pred, variant, z_mode="z_up")
print(f"  {len(points_pred):,} points after voxel downsample, chosen eps={eps} min_supp={min_supp} norm_th={norm_th}")

shapes, _ = schnabel_ransac.detect(
    points_pred, shapes=["plane"], relative_epsilon=False,
    epsilon=eps, normal_thresh=norm_th, min_support=min_supp,
    probability=0.001, normal_knn=20, max_shapes=30,
)
plane_shapes = [s for s in shapes if s["type"] == "plane"]
print(f"  {len(shapes)} shape(s) returned by RANSAC ({len(plane_shapes)} plane(s))")

ground_shape, avg_z, ground_z_align, ground_residual = find_ground_plane_robust(
    plane_shapes, points_pred, z_mode="z_up", horizontal_thresh=0.80
)
if ground_shape is None:
    print("No ground plane found -- nothing to score.")
    sys.exit(0)
ground_mask_pred = ground_shape["inlier_mask"]
walls = find_wall_planes(plane_shapes, points_pred, wall_thresh=0.3, exclude_mask=ground_mask_pred)
print(f"  ground: {ground_mask_pred.sum():,} pts ({100*ground_mask_pred.sum()/len(points_pred):.1f}%), "
      f"z_align={ground_z_align:.3f}, residual={ground_residual:.4f}m")
print(f"  walls found: {len(walls)}")
for i, w in enumerate(walls):
    print(f"    wall[{i}]: {w['n_points']:,} pts, z_align={w['z_align']:.3f}, residual={w['residual']:.4f}m")

# wall_id_pred[i] = which wall (index) that downsampled point belongs to, -1 if none
wall_id_pred = np.full(len(points_pred), -1, dtype=np.int32)
for i, w in enumerate(walls):
    wall_id_pred[w["shape"]["inlier_mask"]] = i

# ---- Align: nearest-neighbor upsample the downsampled prediction back to full-res ----
print(f"\nAligning {len(points_pred):,} downsampled predictions to {len(points_raw):,} raw points via KDTree ...")
t0 = time.perf_counter()
tree = cKDTree(points_pred)
_, nn_idx = tree.query(points_raw, k=1, workers=-1)
pred_ground_full = ground_mask_pred[nn_idx]
wall_id_full = wall_id_pred[nn_idx]
print(f"  done in {time.perf_counter()-t0:.1f}s")

# ---- Confusion matrix vs scalar_Label==1 (Ground) ----
gt_ground_full = (labels_raw == 1)
tp = int(np.sum(pred_ground_full & gt_ground_full))
fp = int(np.sum(pred_ground_full & ~gt_ground_full))
fn = int(np.sum(~pred_ground_full & gt_ground_full))
tn = int(np.sum(~pred_ground_full & ~gt_ground_full))
total = len(points_raw)

print("\n=== Ground detection vs. real ground truth (scalar_Label==1) ===")
print(f"  TP (correctly predicted ground):     {tp:>12,} ({100*tp/total:5.2f}% of cloud)")
print(f"  FP (predicted ground, actually not): {fp:>12,} ({100*fp/total:5.2f}% of cloud)")
print(f"  FN (missed real ground):             {fn:>12,} ({100*fn/total:5.2f}% of cloud)")
print(f"  TN (correctly excluded):             {tn:>12,} ({100*tn/total:5.2f}% of cloud)")
real_ground_total = tp + fn
pred_ground_total = tp + fp
print(f"\n  Coverage of real ground found: {100*tp/real_ground_total:.1f}% ({tp:,} of {real_ground_total:,} real ground points)")
print(f"  Precision of predicted ground: {100*tp/pred_ground_total:.1f}% ({tp:,} of {pred_ground_total:,} predicted-ground points)")

# ---- Diagnose the "why no walls" question using real Building-class density ----
building_mask = (labels_raw == 4)
n_building = int(building_mask.sum())
if n_building > 0:
    b_pts = points_raw[building_mask]
    b_extent = b_pts.max(axis=0) - b_pts.min(axis=0)
    b_volume = max(b_extent[0] * b_extent[1], 1e-6)
    print(f"\n=== Building-class density check (why walls may be missing) ===")
    print(f"  Real 'Building' points (full-res): {n_building:,} ({100*n_building/total:.2f}% of cloud)")
    print(f"  Ground points (full-res):          {int(gt_ground_full.sum()):,} ({100*gt_ground_full.sum()/total:.2f}% of cloud)")
    print(f"  Ratio ground:building points = {gt_ground_full.sum()/max(n_building,1):.1f}:1")
    # how many raw building points survived to the nearest downsampled point actually being classified as anything
    nn_idx_building = nn_idx[building_mask]
    unique_downsampled_building = len(np.unique(nn_idx_building))
    print(f"  {n_building:,} raw building points map to only {unique_downsampled_building:,} unique "
          f"downsampled points after voxel={VOXEL}m -- i.e. building facades survive downsampling "
          f"at ~{unique_downsampled_building/max(n_building,1)*100:.2f}% the density ground does.")

# ---- Wall accuracy vs real Building class (same idea as ground's confusion matrix) ----
pred_wall_full = wall_id_full >= 0
if n_building > 0:
    w_tp = int(np.sum(pred_wall_full & building_mask))
    w_fp = int(np.sum(pred_wall_full & ~building_mask))
    w_fn = int(np.sum(~pred_wall_full & building_mask))
    print(f"\n=== Wall detection vs. real Building class ===")
    print(f"  TP (wall on real building):     {w_tp:>12,}")
    print(f"  FP (wall, not really building): {w_fp:>12,}")
    print(f"  FN (missed real building):      {w_fn:>12,}")
    if (w_tp + w_fn) > 0:
        print(f"  Coverage of real building found: {100*w_tp/(w_tp+w_fn):.1f}%")
    if (w_tp + w_fp) > 0:
        print(f"  Precision of predicted walls:    {100*w_tp/(w_tp+w_fp):.1f}%")

# ---- Visualize: ground as TP/FP/FN/TN (yellow/red/orange/gray, this project's
# established GT-comparison convention), walls layered on top in their own distinct
# colors (WALL_COLORS, matching detect_ground_and_walls.py's raw-detection view) so
# both the ground accuracy AND the actual wall detections are visible in one pass.
print(f"\nBuilding colored point cloud ({total:,} points) for visualization ...")
colors = np.empty((total, 3), dtype=np.float32)
colors[pred_ground_full & gt_ground_full] = [1.0, 1.0, 0.0]   # ground TP - yellow
colors[pred_ground_full & ~gt_ground_full] = [1.0, 0.0, 0.0]  # ground FP - red
colors[~pred_ground_full & gt_ground_full] = [1.0, 0.5, 0.0]  # ground FN - orange
colors[~pred_ground_full & ~gt_ground_full] = [0.6, 0.6, 0.6]  # TN - gray (default)

# Walls drawn last so they win over the gray TN background (never over ground, since
# find_wall_planes() already excludes ground-assigned points at the source).
for i in range(len(walls)):
    colors[wall_id_full == i] = WALL_COLORS[i % len(WALL_COLORS)]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_raw)
pcd.colors = o3d.utility.Vector3dVector(colors)

print("Opening visualizer (yellow=correct ground, red=false-positive ground, orange=missed ground, "
      "wall colors=detected walls, gray=everything else).")
print("Rotate: Left Mouse | Pan: Shift+Left Mouse | Zoom: Scroll | Screenshot: 'P' key | Close window to exit.")
window_name = (f"Toronto-3D L003 vs ground truth -- ground TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,} "
               f"-- {len(walls)} wall(s) shown in their own colors")
screenshot_utils.draw_geometries([pcd], window_name=window_name, width=1600, height=900)
