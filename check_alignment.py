"""
Diagnostic: test every coordinate transform preset for each held-out scene.
This tells us which transform should be used in eval_tartanair_gt.py.
"""
import glob, os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from debug_alignment import TRANSFORM_PRESETS, apply_transform, score_transform
from precompute_gt_masks import get_ground_colors
from ransac_env import load_ply_xyz

ENVS = ["Gascola", "House", "NordicHarbor", "WesternDesertTown"]
MAP_DOWNSAMPLE = 0.05  # 5% random sample to keep kdtree fast

for env in ENVS:
    print(f"\n=== {env} ===")
    pcd_path = f"data/{env}/{env}_sem.pcd"
    if not os.path.exists(pcd_path):
        print("  sem_pcd missing, skip")
        continue

    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = pcd.random_down_sample(MAP_DOWNSAMPLE)
    gpts  = np.asarray(pcd.points)
    gcols = np.asarray(pcd.colors)
    print(f"  Global map: {len(gpts)} points after {MAP_DOWNSAMPLE*100:.0f}% sample")

    gc = get_ground_colors(env)
    if gc is None or len(gc) == 0:
        print("  No ground colors found")
        gc = None
    else:
        print(f"  Ground colors ({len(gc)} classes):", gc[:3])

    kdtree = cKDTree(gpts)

    files = sorted(glob.glob(f"data/{env}/Data_omni/P0000/lidar/*.ply"))[:3]
    poses = np.loadtxt(f"data/{env}/Data_omni/P0000/pose_lcam_front.txt")

    for fpath in files:
        fidx = int(os.path.basename(fpath).split("_")[0])
        pts = load_ply_xyz(fpath, voxel_size=0.05)
        pose = poses[fidx]
        print(f"\n  Frame {fidx}: {len(pts)} pts after voxel | "
              f"raw range x=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
              f"y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
              f"z=[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
        print(f"  Pose t={pose[:3].round(2)}  q={pose[3:].round(3)}")

        results = []
        for tn in TRANSFORM_PRESETS:
            res = score_transform(pts, pose, kdtree, gcols, gc, tn, 0.5, 0.05)
            results.append(res)
        results.sort(key=lambda r: r["score"], reverse=True)

        print(f"  Transform ranking (match% / ground%):") 
        for row in results[:5]:
            md = "n/a" if row["mean_dist"] is None else f"{row['mean_dist']:.3f}m"
            print(f"    {row['transform']:15s}  match={row['match_rate']*100:5.1f}%  "
                  f"ground={row['ground_fraction']*100:5.1f}%  dist={md}")
        print(f"  --> BEST: {results[0]['transform']}")

print("\nDone.")
