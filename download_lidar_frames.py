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
    "SeasonalForestWinterNight", "Sewerage", "Supermarket", "Office",
    "ForestEnv", "GreatMarsh", "Restaurant", "SeasideTown",
    "GothicIsland",
]
# Ocean and ShoreCaves exist in the visual TartanAir dataset but have no
# Data_omni/lidar modality in the TartanGround (ground-vehicle lidar) repo,
# so they can't be downloaded via download_ground(). GothicIsland does have
# lidar (Data_omni/P0000) and is the closest available beach/coastal scene.
OUTDOOR = {"Downtown","OldScandinavia","OldTownFall",
           "SeasonalForestAutumn","SeasonalForestSpring","SeasonalForestWinterNight",
           "ForestEnv", "GreatMarsh", "SeasideTown",
           "GothicIsland"}
# Gascola, House, and HELD_OUT_ENVIRONMENTS below are deliberately NOT in this
# list -- they're the fully held-out (scene-level) test environments, never
# trained on and never folded into "--env all" aggregate stats alongside the
# trained-on environments above. Evaluate them individually: --env Gascola,
# --env House, --env WesternDesertTown, etc.
#
# NordicHarbor is a deliberate near-neighbor of SeasideTown (both coastal) --
# kept held-out specifically to test generalization *across* similar coastal
# scenes, not just "has it ever seen water at all" (which SeasideTown alone
# in training would leave untested).
HELD_OUT_ENVIRONMENTS = [
    "Gascola", "House", "NordicHarbor", "WesternDesertTown"
]

# Real-world data (RELLIS-3D, converted from SemanticKITTI .bin/.label to
# .ply), NOT synthetic TartanGround -- this is the independent ground-truth
# validation set, meant to be scored against but never trained on. Different
# purpose from HELD_OUT_ENVIRONMENTS above (those test cross-scene
# generalization *within* the synthetic distribution), so tracked separately,
# but must be excluded from training just the same. Found the hard way: a
# recursive data_dir=None training scan silently swept in 13,556 RELLIS3D
# frames because nothing outside HELD_OUT_ENVIRONMENTS was being excluded --
# see train_rl.py, which now always excludes this list regardless of what
# --exclude_envs is passed, rather than relying on a manually-typed list that
# has to be remembered and kept in sync by hand every time new data is added.
REAL_WORLD_EVAL_ENVIRONMENTS = ["RELLIS3D", "RELLIS3D_raw"]

# Everything that must never be trained on, for either reason above.
NEVER_TRAIN_ENVIRONMENTS = HELD_OUT_ENVIRONMENTS + REAL_WORLD_EVAL_ENVIRONMENTS

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
            
        # Download metadata.zip which contains pose_lcam_front.txt
        from huggingface_hub import hf_hub_download
        print(f"  Downloading metadata.zip for {env}...")
        try:
            meta_zip = hf_hub_download(repo_id='theairlabcmu/TartanGround', filename=f'{env}/Data_omni/P0000/metadata.zip', repo_type='dataset')
            with zipfile.ZipFile(meta_zip, 'r') as z:
                # only extract pose_lcam_front.txt
                z.extract('pose_lcam_front.txt', extract_path)
            print("  Extracted pose_lcam_front.txt")
        except Exception as e:
            print(f"  Warning: failed to download metadata.zip for {env}: {e}")
        
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
        print("\n-- Held-out (scene-level test, never trained on) --")
        for env in HELD_OUT_ENVIRONMENTS:
            n = count_frames(env)
            print("{:<35} {:>8}".format(env, n if n else "--"))
        return
    if len(sys.argv) > 1:
        e = sys.argv[1]
        if e not in ENVIRONMENTS and e not in HELD_OUT_ENVIRONMENTS:
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
