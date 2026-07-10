"""
rellis3d_convert.py

Converts the raw RELLIS-3D Ouster LiDAR release (downloaded by
download_rellis3d.py into data/RELLIS3D_raw/) into the layout the rest of
this pipeline understands:

    data/RELLIS3D/RELLIS3D_<seq>/lidar/<frame_id>.ply     -- xyz points,
        same minimal format ransac_env.load_ply_xyz already reads.
    data/RELLIS3D/RELLIS3D_<seq>/lidar/<frame_id>_gt.npy  -- uint8
        ground/not-ground mask, one entry per point in the .ply above,
        same order (no downsampling here, so this is the real ground
        truth -- eval_rellis3d.py nearest-neighbor-matches it against
        whatever RansacEnv's own downsampling produces).

Ground truth is derived from RELLIS-3D's point-wise semantic labels via
benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml in
unmannedlab/RELLIS-3D: raw label ids {1: dirt, 3: grass, 10: asphalt,
23: concrete, 31: puddle, 33: mud} are treated as ground/traversable;
everything else (tree, bush, water, barrier, rubble, vehicle, person,
building, fence, pole, log, object, sky, void) is not.

Usage:
    python rellis3d_convert.py
    python rellis3d_convert.py --limit 200   # convert only the first 200
                                              # frames per sequence (smoke test)
"""

import os
import glob
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(WORKSPACE, "data", "RELLIS3D_raw")
OUT_ROOT = os.path.join(WORKSPACE, "data", "RELLIS3D")

# Raw RELLIS-3D semantic label ids that count as ground/traversable terrain.
# See rellis.yaml's `labels:` section: dirt=1, grass=3, asphalt=10,
# concrete=23, puddle=31, mud=33. Water(6), rubble(34) and barrier(27) are
# deliberately excluded -- not traversable ground for this task.
GROUND_RAW_IDS = {1, 3, 10, 23, 31, 33}

BIN_DIRNAME = "os1_cloud_node_kitti_bin"
LABEL_DIRNAME = "os1_cloud_node_semantickitti_label_id"


def find_sequences():
    """
    Locates every '<seq>/os1_cloud_node_kitti_bin' folder anywhere under
    RAW_ROOT (searched recursively since the zip's internal root folder
    name isn't guaranteed), paired with its sibling label folder.
    """
    bin_dirs = glob.glob(os.path.join(RAW_ROOT, "**", BIN_DIRNAME), recursive=True)
    sequences = []
    for bin_dir in sorted(bin_dirs):
        seq_dir = os.path.dirname(bin_dir)
        label_dir = os.path.join(seq_dir, LABEL_DIRNAME)
        if not os.path.isdir(label_dir):
            print(f"WARNING: no label dir next to {bin_dir}, skipping sequence.")
            continue
        seq_name = os.path.basename(seq_dir)
        sequences.append((seq_name, bin_dir, label_dir))
    return sequences


def load_bin_xyz(path):
    arr = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return arr[:, :3]


def load_label_ids(path):
    raw = np.fromfile(path, dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.uint32)  # SemanticKITTI: lower 16 bits = semantic class


def write_ply_xyz(path, xyz):
    vertex = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(path)


def convert_sequence(seq_name, bin_dir, label_dir, limit=None):
    out_dir = os.path.join(OUT_ROOT, f"RELLIS3D_{seq_name}", "lidar")
    os.makedirs(out_dir, exist_ok=True)

    bin_files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
    if limit:
        bin_files = bin_files[:limit]

    n_ok, n_skip = 0, 0
    for bin_path in bin_files:
        frame_id = os.path.splitext(os.path.basename(bin_path))[0]
        label_path = os.path.join(label_dir, frame_id + ".label")
        ply_path = os.path.join(out_dir, frame_id + ".ply")
        gt_path = os.path.join(out_dir, frame_id + "_gt.npy")

        if os.path.exists(ply_path) and os.path.exists(gt_path):
            n_skip += 1
            continue

        if not os.path.exists(label_path):
            print(f"  WARNING: no label for {bin_path}, skipping frame.")
            continue

        xyz = load_bin_xyz(bin_path)
        sem_ids = load_label_ids(label_path)
        if len(xyz) != len(sem_ids):
            print(f"  WARNING: point/label count mismatch for {frame_id} "
                  f"({len(xyz)} vs {len(sem_ids)}), skipping frame.")
            continue

        gt_mask = np.isin(sem_ids, list(GROUND_RAW_IDS)).astype(np.uint8)

        write_ply_xyz(ply_path, xyz)
        np.save(gt_path, gt_mask)
        n_ok += 1

    print(f"  RELLIS3D_{seq_name}: {n_ok} frames converted, {n_skip} already done "
          f"-> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Convert only the first N frames per sequence (smoke test)")
    args = parser.parse_args()

    sequences = find_sequences()
    if not sequences:
        print(f"No sequences found under {RAW_ROOT}. Run download_rellis3d.py first.")
        return

    print(f"Found {len(sequences)} sequence(s) under {RAW_ROOT}.")
    for seq_name, bin_dir, label_dir in sequences:
        convert_sequence(seq_name, bin_dir, label_dir, limit=args.limit)


if __name__ == "__main__":
    main()
