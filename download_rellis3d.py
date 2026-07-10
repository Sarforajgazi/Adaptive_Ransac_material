"""
download_rellis3d.py

Downloads the RELLIS-3D Ouster LiDAR SemanticKITTI-format release
(unmannedlab/RELLIS-3D) via Google Drive, for use as a real-world,
eval-only ground-truth benchmark alongside the synthetic TartanAir
environments in download_lidar_frames.py.

Usage:
    python download_rellis3d.py --list
    python download_rellis3d.py            # downloads + extracts both packages
    python download_rellis3d.py --points-only
    python download_rellis3d.py --labels-only
"""

import os
import sys
import argparse
import zipfile

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = os.path.join(WORKSPACE, "data", "RELLIS3D_raw")

# Google Drive file IDs, from the RELLIS-3D README (unmannedlab/RELLIS-3D)
# as of 2026-07-09. Re-check the README if these ever start 404ing --
# Drive links on that repo have moved before.
PACKAGES = {
    "points": {
        "id": "1lDSVRf_kZrD0zHHMsKJ0V1GN9QATR4wH",
        "zip_name": "os1_cloud_node_kitti_bin.zip",
        "desc": "Ouster LiDAR point clouds, SemanticKITTI .bin format (~14GB)",
    },
    "labels": {
        "id": "12bsblHXtob60KrjV7lGXUQTdC5PhV8Er",
        "zip_name": "os1_cloud_node_semantickitti_label_id.zip",
        "desc": "Ouster LiDAR point-wise annotations, SemanticKITTI .label format (~174MB)",
    },
}


def is_downloaded(key):
    zip_path = os.path.join(RAW_ROOT, PACKAGES[key]["zip_name"])
    return os.path.exists(zip_path)


def download_and_extract(key):
    info = PACKAGES[key]
    os.makedirs(RAW_ROOT, exist_ok=True)
    zip_path = os.path.join(RAW_ROOT, info["zip_name"])

    if not os.path.exists(zip_path):
        try:
            import gdown
        except ImportError:
            print("gdown is required to download from Google Drive: pip install gdown")
            sys.exit(1)

        url = f"https://drive.google.com/uc?id={info['id']}"
        print(f"Downloading {key} ({info['desc']})...")
        # resume=True: if a matching <zip_name>*.part file already exists (e.g.
        # from a stalled prior attempt), gdown picks it up and continues via an
        # HTTP Range request instead of starting over from byte 0.
        gdown.download(url, zip_path, quiet=False, resume=True)

        if not os.path.exists(zip_path):
            print(f"ERROR: download of {key} failed (no file at {zip_path}).")
            return False
    else:
        print(f"SKIP download for {key} -- zip already present at {zip_path}")

    print(f"Extracting {zip_path} -> {RAW_ROOT} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_ROOT)
    print(f"OK {key} extracted.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="Show package status and exit")
    parser.add_argument("--points-only", action="store_true", help="Only download/extract the point clouds")
    parser.add_argument("--labels-only", action="store_true", help="Only download/extract the annotations")
    args = parser.parse_args()

    if args.list:
        print("{:<10} {:>10}  {}".format("Package", "Status", "Description"))
        print("-" * 70)
        for key, info in PACKAGES.items():
            status = "downloaded" if is_downloaded(key) else "--"
            print("{:<10} {:>10}  {}".format(key, status, info["desc"]))
        return

    keys = list(PACKAGES.keys())
    if args.points_only:
        keys = ["points"]
    elif args.labels_only:
        keys = ["labels"]

    for key in keys:
        download_and_extract(key)


if __name__ == "__main__":
    main()
