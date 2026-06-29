import os, sys, warnings, io
warnings.filterwarnings("ignore")

# Force UTF-8 output so tartanair's emoji (📦) doesn't crash on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import tartanair as ta

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(WORKSPACE, "data")

ENVIRONMENTS = [
    "Downtown", "Hospital", "OldScandinavia", "OldTownFall",
    "SeasonalForestAutumn", "SeasonalForestSpring",
    "SeasonalForestWinterNight", "Sewerage", "Supermarket",
]
OUTDOOR = {"Downtown","OldScandinavia","OldTownFall",
           "SeasonalForestAutumn","SeasonalForestSpring","SeasonalForestWinterNight"}

def count_frames(env):
    d = os.path.join(DATA_ROOT, env, "Data_omni", "P0000", "lidar")
    return len([f for f in os.listdir(d) if f.endswith(".ply")]) if os.path.isdir(d) else 0

import zipfile

def download_env(env):
    n = count_frames(env)
    if n > 10:
        print("SKIP {} ({} frames already)".format(env, n))
        return True
    print("Downloading {}...".format(env))
    ta.init(DATA_ROOT)
    try:
        # Download the zip file without extracting via the library (fails on Windows)
        ta.download_ground(env=env, version="omni", traj="P0000",
            modality="lidar", unzip=False, delete_zip=False,
            num_workers=4, data_source="huggingface")
        
        # Manually extract using python's zipfile
        zip_path = os.path.join(DATA_ROOT, env, "Data_omni", "P0000", "lidar.zip")
        extract_path = os.path.join(DATA_ROOT, env, "Data_omni", "P0000")
        if os.path.exists(zip_path):
            print("  Extracting {}...".format(zip_path))
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            os.remove(zip_path)
            print("  Extracted and removed zip.")
        
        n = count_frames(env)
        print("OK {} -> {} frames".format(env, n))
        return n > 0
    except Exception as e:
        print("ERROR {}: {}".format(env, e))
        return False

def main():
    ta.init(DATA_ROOT)
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        print("{:<35} {:>8}  {}".format("Environment", "Frames", "Type"))
        print("-" * 55)
        for env in ENVIRONMENTS:
            n = count_frames(env)
            t = "Outdoor" if env in OUTDOOR else "Indoor"
            print("{:<35} {:>8}  {}".format(env, n if n else "--", t))
        office_n = count_frames("Office")
        print("{:<35} {:>8}  {}".format("Office (existing)", office_n, "Indoor"))
        return
    if len(sys.argv) > 1:
        e = sys.argv[1]
        if e not in ENVIRONMENTS:
            print("Unknown env:", e)
            sys.exit(1)
        download_env(e)
        return
    print("Downloading all environments to:", DATA_ROOT)
    for env in ENVIRONMENTS:
        download_env(env)
    print("Total frames:", sum(count_frames(e) for e in ENVIRONMENTS))

if __name__ == "__main__":
    main()
