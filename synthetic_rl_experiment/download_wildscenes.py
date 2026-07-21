"""
download_wildscenes.py

Downloads a random sample of WildScenes3d point cloud submaps (.bin) +
their per-point semantic labels (.label) from CSIRO's Data Access Portal
S3 bucket, across all 5 available sequences (K-01, K-03 = Karawatha
forest; V-01, V-02, V-03 = Venman forest).

Format confirmed empirically (not assumed): .bin is raw float32, reshape
(-1, 3) -> x,y,z (no intensity column, unlike SemanticKITTI's raw sweeps --
these are already-aggregated SLAM submaps). .label is uint32, one value per
point, matching the .bin point count exactly. Real per-point terrain
classes (from the devkit's METAINFO, wildscenes/tools/utils3d.py):
bush, dirt, fence, grass, gravel, log, mud, other-object, other-terrain,
rock, sky, structure, tree-foliage, tree-trunk, water -- includes real
"mud" and "water" ground truth, which is what this was pulled for.

Requires temporary S3 credentials from the CSIRO Data Access Portal
(https://data.csiro.au/collection/csiro:61541 -> "S3 Client" tab), valid
~48h. Never hardcode them here -- pass via environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

Usage:
    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \\
        python download_wildscenes.py --frames_per_seq 4 --seed 0
"""
import os
import sys
import argparse
import numpy as np
import boto3
from plyfile import PlyElement, PlyData

ENDPOINT = "https://s3.data.csiro.au"
BUCKET = "dapprd"
BASE_PREFIX = "000061541v003/data/WildScenes/WildScenes3d"
SEQUENCES = ["K-01", "K-03", "V-01", "V-02", "V-03"]

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildscenes_data", "raw")
PLY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildscenes_data", "ply")


def get_client():
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        print("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (temporary CSIRO DAP S3 credentials) first.")
        sys.exit(1)
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=access_key, aws_secret_access_key=secret_key)


def list_frame_stems(s3, seq):
    prefix = f"{BASE_PREFIX}/{seq}/Clouds/"
    stems = []
    token = None
    while True:
        kwargs = dict(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for c in resp.get("Contents", []):
            key = c["Key"]
            if key.endswith(".bin"):
                stems.append(os.path.basename(key)[:-4])
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break
    return stems


def convert_pair(bin_path, label_path, ply_path):
    xyz = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)
    label = np.fromfile(label_path, dtype=np.uint32)
    cidx = (label & 0xFFFF).astype(np.uint8)
    assert len(xyz) == len(cidx), f"point/label count mismatch: {len(xyz)} vs {len(cidx)}"

    vertex = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("classification", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["classification"] = cidx
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(ply_path)
    return len(xyz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_per_seq", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    s3 = get_client()
    rng = np.random.default_rng(args.seed)

    for seq in SEQUENCES:
        print(f"\n=== {seq} ===")
        stems = list_frame_stems(s3, seq)
        print(f"  {len(stems)} frames available")
        if not stems:
            continue
        chosen = rng.choice(len(stems), size=min(args.frames_per_seq, len(stems)), replace=False)

        raw_seq_dir = os.path.join(RAW_DIR, seq)
        ply_seq_dir = os.path.join(PLY_DIR, seq)
        os.makedirs(raw_seq_dir, exist_ok=True)
        os.makedirs(ply_seq_dir, exist_ok=True)

        for idx in chosen:
            stem = stems[idx]
            bin_path = os.path.join(raw_seq_dir, stem + ".bin")
            label_path = os.path.join(raw_seq_dir, stem + ".label")
            ply_path = os.path.join(ply_seq_dir, stem + ".ply")

            s3.download_file(BUCKET, f"{BASE_PREFIX}/{seq}/Clouds/{stem}.bin", bin_path)
            s3.download_file(BUCKET, f"{BASE_PREFIX}/{seq}/Labels/{stem}.label", label_path)
            n = convert_pair(bin_path, label_path, ply_path)
            print(f"  {stem}: {n} points -> {ply_path}")


if __name__ == "__main__":
    main()
