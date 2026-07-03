"""
accumulate_trajectory.py

Builds a merged, world-frame point cloud from the per-frame LiDAR scans in
data/<Env>/Data_omni/P0000/lidar/*.ply -- the same per-frame data used for
RL training/evaluation, NOT the pre-made "complete" reconstructions in
schnabel_cython/tartanair_data/.

Each frame's points are stored in sensor-local (ego-centric) coordinates --
confirmed empirically: centroids stay pinned near the origin across every
frame in a trajectory regardless of how far the drone has actually moved.
Concatenating raw frames directly would be meaningless. This module uses
the per-frame pose file (pose_lcam_front.txt: x,y,z,qx,qy,qz,qw, NED) to
transform each frame into one shared world frame before merging.

The pose transform alone gets frames roughly aligned but is not perfectly
consistent frame-to-frame -- confirmed by diagnosing why the grid-based
traversability classifier failed on this data (see TRAVERSABILITY_COMPARISON.md,
Part 2): the same physical floor surface landed at slightly different heights
across different frames' contributions to a cell. ICP refinement (frame-to-
previous-frame, point-to-plane, seeded with the pose transform as initial
alignment) corrects this residual misalignment before merging.

Note the resulting cloud is in NED (Z-down, ground = highest Z) -- the same
convention as the rest of the per-frame pipeline (ransac_env.py etc.), NOT
the Z-up convention of the separate "complete" reconstructions.

Usage:
    python accumulate_trajectory.py --env Downtown
    python accumulate_trajectory.py --env OldScandinavia --stride 5
    python accumulate_trajectory.py --env Downtown --no-icp   # old behavior, for comparison
"""

import os
import glob
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial.transform import Rotation

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(WORKSPACE, "data")


def load_frame_xyz(filepath):
    ply = PlyData.read(filepath)
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)


def voxel_downsample(points, voxel_size):
    if voxel_size is None or voxel_size <= 0.0 or len(points) == 0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points)


def _to_pcd_with_normals(points, knn=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return pcd


def icp_align(source_pts, target_pts, max_corr_dist=0.5):
    """
    Refines source_pts onto target_pts with point-to-plane ICP. Both are
    assumed already roughly aligned (e.g. via the pose transform) -- this
    corrects small residual misalignment, not large unknown transforms.

    Returns (transformation 4x4, fitness, inlier_rmse).
    """
    src = _to_pcd_with_normals(source_pts)
    tgt = _to_pcd_with_normals(target_pts)
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result.transformation, result.fitness, result.inlier_rmse


def accumulate_trajectory(env, stride=10, per_frame_voxel=0.05, final_voxel=0.1,
                           use_icp=True, icp_max_corr_dist=0.5,
                           icp_min_fitness=0.3, icp_max_correction=2.0, verbose=True):
    lidar_dir = os.path.join(DATA_ROOT, env, "Data_omni", "P0000", "lidar")
    pose_path = os.path.join(DATA_ROOT, env, "Data_omni", "P0000", "pose_lcam_front.txt")

    if not os.path.exists(pose_path):
        raise FileNotFoundError(
            f"No pose file at {pose_path}. Download it first, e.g.:\n"
            f"  ta.download_ground(env='{env}', version='omni', traj='P0000', "
            f"modality='meta', unzip=True, delete_zip=True)"
        )

    files = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))
    poses = np.loadtxt(pose_path)
    if len(files) != len(poses):
        raise ValueError(f"{len(files)} lidar frames but {len(poses)} poses -- can't assume 1:1 index match")

    world_chunks = []
    prev_world_pts = None
    indices = range(0, len(files), stride)
    fitness_log = []
    accepted_count = 0
    rejected_count = 0

    for count, i in enumerate(indices):
        local_pts = load_frame_xyz(files[i])
        local_pts = voxel_downsample(local_pts, per_frame_voxel)

        x, y, z, qx, qy, qz, qw = poses[i]
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        world_pts = local_pts @ R.T + np.array([x, y, z])

        if use_icp and prev_world_pts is not None and len(world_pts) > 50 and len(prev_world_pts) > 50:
            transform, fitness, rmse = icp_align(world_pts, prev_world_pts, max_corr_dist=icp_max_corr_dist)
            translation_mag = float(np.linalg.norm(transform[:3, 3]))
            fitness_log.append(fitness)

            if fitness >= icp_min_fitness and translation_mag <= icp_max_correction:
                world_pts_h = np.hstack([world_pts, np.ones((len(world_pts), 1))])
                world_pts = (world_pts_h @ transform.T)[:, :3]
                accepted_count += 1
            else:
                # Bad correspondence quality or an implausibly large correction --
                # trust the pose transform instead of a diverged ICP result.
                rejected_count += 1

        world_chunks.append(world_pts)
        prev_world_pts = world_pts

        if verbose and (count + 1) % 20 == 0:
            msg = f"  [{count + 1}/{len(indices)}] merged frame {i} ({os.path.basename(files[i])})"
            if fitness_log:
                msg += f"  ICP fitness (last 20 avg): {np.mean(fitness_log[-20:]):.3f}  accepted={accepted_count} rejected={rejected_count}"
            print(msg)

    merged = np.concatenate(world_chunks, axis=0).astype(np.float32)
    if verbose:
        print(f"Merged {len(indices)} frames -> {len(merged)} raw points, downsampling final cloud...")
        if use_icp and fitness_log:
            print(f"ICP fitness overall: mean={np.mean(fitness_log):.3f}, min={np.min(fitness_log):.3f}")
            print(f"ICP corrections accepted={accepted_count}, rejected={rejected_count} "
                  f"(rejected = low fitness < {icp_min_fitness} or correction > {icp_max_correction}m -- fell back to pose transform)")

    merged = voxel_downsample(merged, final_voxel)
    if verbose:
        extent = merged.max(axis=0) - merged.min(axis=0)
        print(f"Final accumulated cloud: {len(merged)} points, XYZ extent {extent}")

    return merged.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Accumulate a trajectory's per-frame LiDAR scans into one world-frame point cloud")
    parser.add_argument("--env", type=str, default="Downtown")
    parser.add_argument("--stride", type=int, default=10, help="Use every Nth frame (consecutive frames overlap heavily)")
    parser.add_argument("--per_frame_voxel", type=float, default=0.05)
    parser.add_argument("--final_voxel", type=float, default=0.1)
    parser.add_argument("--icp_max_corr_dist", type=float, default=0.5)
    parser.add_argument("--no-icp", action="store_true", help="Skip ICP refinement, use raw pose transform only (old behavior)")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the merged cloud as .ply")
    args = parser.parse_args()

    points = accumulate_trajectory(args.env, stride=args.stride,
                                    per_frame_voxel=args.per_frame_voxel,
                                    final_voxel=args.final_voxel,
                                    use_icp=not args.no_icp,
                                    icp_max_corr_dist=args.icp_max_corr_dist)

    if args.save:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(args.save, pcd)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
