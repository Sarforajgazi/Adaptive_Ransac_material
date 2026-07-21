import argparse
import glob
import json
import os
import zipfile

import numpy as np
import open3d as o3d
import tartanair as ta
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from debug_alignment import TRANSFORM_PRESETS, apply_transform, score_transform


GROUND_NAMES = {
    "ground", "floor", "grass", "road", "sand", "dirt", "rock",
    "terrain", "pavement", "asphalt", "concretefloor", "floordirt"
}


def get_ground_colors(env_name, data_dir="data"):
    label_map_path = os.path.join(data_dir, env_name, "seg_label_map.json")
    if not os.path.exists(label_map_path):
        if env_name == "House":
            return np.array([[132 / 255.0, 237 / 255.0, 195 / 255.0]])
        return None

    with open(label_map_path, "r") as f:
        name_map = json.load(f)["name_map"]

    ground_ids = []
    for name, id_ in name_map.items():
        if any(token in name.lower() for token in GROUND_NAMES):
            ground_ids.append(id_)

    with open(".venv/Lib/site-packages/tartanair/seg_rgbs.txt", "r") as f:
        lines = f.readlines()

    ground_colors_float = []
    for gid in ground_ids:
        b, g, r = [int(x) for x in lines[gid].strip().split(",")]
        ground_colors_float.append([r / 255.0, g / 255.0, b / 255.0])

    return np.array(ground_colors_float)


def download_sem_pcd(env_name, data_dir="data"):
    ta.init(data_dir)
    print(f"[{env_name}] Downloading sem_pcd...")
    try:
        ta.download_ground(
            env=env_name, version="omni", traj="P0000", modality="sem_pcd",
            unzip=False, delete_zip=False, num_workers=1, data_source="huggingface"
        )
        zip_path = os.path.join(data_dir, env_name, f"{env_name}_sem_pcd.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(os.path.join(data_dir, env_name))
    except Exception as e:
        print(f"[{env_name}] Error downloading sem_pcd: {e}")


def choose_transform(env_name, lidar_files, poses, global_points, global_colors,
                     ground_colors, probe_frame=0, radius=0.5, color_tol=0.05):
    if not lidar_files:
        return "swapxy_negz"

    probe_file = lidar_files[min(probe_frame, len(lidar_files) - 1)]
    frame_idx = int(os.path.basename(probe_file).split("_")[0])
    if frame_idx >= len(poses):
        return "swapxy_negz"

    from ransac_env import load_ply_xyz
    points_local = load_ply_xyz(probe_file, voxel_size=0.05)
    if len(points_local) == 0:
        return "swapxy_negz"

    kdtree = cKDTree(global_points)
    results = []
    for transform_name in TRANSFORM_PRESETS:
        results.append(
            score_transform(
                points_local, poses[frame_idx], kdtree, global_colors, ground_colors,
                transform_name, radius, color_tol
            )
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    best = results[0]["transform"]
    print(f"[{env_name}] Auto-selected transform: {best}")
    for row in results[:3]:
        mean_dist = "n/a" if row["mean_dist"] is None else f"{row['mean_dist']:.4f}m"
        print(
            f"    {row['transform']:12s} match={row['match_rate'] * 100:6.2f}% "
            f"ground={row['ground_fraction'] * 100:6.2f}% dist={mean_dist}"
        )
    return best


def precompute_for_env(env_name, data_dir="data", force=False, transform_name="swapxy_negz",
                       auto_transform=False, probe_frame=0, probe_radius=0.5,
                       map_downsample=0.1, color_tol=0.05):
    print(f"\n======================================")
    print(f"Precomputing Masks for {env_name}")
    print(f"======================================")

    pcd_path = os.path.join(data_dir, env_name, f"{env_name}_sem.pcd")
    if not os.path.exists(pcd_path):
        download_sem_pcd(env_name, data_dir)

    if not os.path.exists(pcd_path):
        print(f"[{env_name}] Failed to get sem_pcd.")
        return

    ground_colors = get_ground_colors(env_name, data_dir)
    if ground_colors is None or len(ground_colors) == 0:
        print(f"[{env_name}] No ground classes found.")
        return

    print(f"[{env_name}] Loading global sem_pcd...")
    pcd = o3d.io.read_point_cloud(pcd_path)

    print(f"[{env_name}] Downsampling global map from {len(pcd.points)} points...")
    if 0.0 < map_downsample < 1.0:
        pcd = pcd.random_down_sample(sampling_ratio=map_downsample)

    global_points = np.asarray(pcd.points)
    global_colors = np.asarray(pcd.colors)

    print(f"[{env_name}] Building KDTree with {len(global_points)} points...")
    kdtree = cKDTree(global_points)

    pose_file = os.path.join(data_dir, env_name, "Data_omni", "P0000", "pose_lcam_front.txt")
    if not os.path.exists(pose_file):
        print(f"[{env_name}] Pose file missing! Did you run download_poses.py?")
        return

    poses = np.loadtxt(pose_file)

    lidar_dir = os.path.join(data_dir, env_name, "Data_omni", "P0000", "lidar")
    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))

    if not lidar_files:
        print(f"[{env_name}] No lidar files found.")
        return

    if auto_transform:
        transform_name = choose_transform(
            env_name, lidar_files, poses, global_points, global_colors, ground_colors,
            probe_frame=probe_frame, radius=probe_radius, color_tol=color_tol
        )

    print(f"[{env_name}] Using transform: {transform_name}")
    print(f"[{env_name}] Processing {len(lidar_files)} frames...")

    from ransac_env import load_ply_xyz

    for i, file_path in enumerate(lidar_files):
        mask_path = file_path.replace(".ply", "_gt_mask.npy")
        if os.path.exists(mask_path) and not force:
            continue

        frame_idx = int(os.path.basename(file_path).split("_")[0])
        if frame_idx >= len(poses):
            print(f"[{env_name}] Warning: frame index {frame_idx} out of bounds for poses ({len(poses)}). Skipping.")
            continue

        pose = poses[frame_idx]
        t = pose[:3]
        q = pose[3:]
        rot = R.from_quat(q).as_matrix()

        pts_local = load_ply_xyz(file_path, voxel_size=0.05)
        if len(pts_local) == 0:
            np.save(mask_path, np.array([], dtype=bool))
            continue

        pts_sensor = apply_transform(pts_local, transform_name)
        pts_global = (rot @ pts_sensor.T).T + t

        distances, indices = kdtree.query(pts_global, k=1, distance_upper_bound=probe_radius)
        valid_mask = distances != np.inf
        nn_colors = global_colors[indices[valid_mask]]

        is_floor_gt = np.zeros(len(nn_colors), dtype=bool)
        for color in ground_colors:
            is_floor_gt |= np.linalg.norm(nn_colors - color, axis=1) < color_tol

        gt_labels = np.zeros(len(pts_local), dtype=bool)
        gt_labels[valid_mask] = is_floor_gt
        np.save(mask_path, gt_labels)

        if i % 100 == 0:
            print(f"  Processed {i}/{len(lidar_files)}")

    print(f"[{env_name}] DONE.")


if __name__ == "__main__":
    from download_lidar_frames import HELD_OUT_ENVIRONMENTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="all",
                        help="Single environment name or 'all'")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing *_gt_mask.npy files")
    parser.add_argument("--transform", type=str, default="swapxy_negz",
                        choices=sorted(TRANSFORM_PRESETS.keys()))
    parser.add_argument("--auto_transform", action="store_true",
                        help="Probe several common transforms and pick the best one")
    parser.add_argument("--probe_frame", type=int, default=0)
    parser.add_argument("--probe_radius", type=float, default=0.5)
    parser.add_argument("--map_downsample", type=float, default=0.1)
    parser.add_argument("--color_tol", type=float, default=0.05)
    args = parser.parse_args()

    if args.env == "all":
        envs = HELD_OUT_ENVIRONMENTS
    else:
        envs = [args.env]

    for env in envs:
        if os.path.exists(os.path.join("data", env, "Data_omni", "P0000", "lidar")):
            precompute_for_env(
                env,
                force=args.force,
                transform_name=args.transform,
                auto_transform=args.auto_transform,
                probe_frame=args.probe_frame,
                probe_radius=args.probe_radius,
                map_downsample=args.map_downsample,
                color_tol=args.color_tol,
            )
