import os
import zipfile
from huggingface_hub import hf_hub_download
from download_lidar_frames import ENVIRONMENTS, HELD_OUT_ENVIRONMENTS

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

all_envs = ENVIRONMENTS + HELD_OUT_ENVIRONMENTS

for env in all_envs:
    extract_path = os.path.join(DATA_ROOT, env, "Data_omni", "P0000")
    pose_file = os.path.join(extract_path, "pose_lcam_front.txt")
    
    if os.path.exists(pose_file):
        print(f"[{env}] pose_lcam_front.txt already exists.")
        continue
        
    print(f"[{env}] Downloading metadata.zip...")
    try:
        os.makedirs(extract_path, exist_ok=True)
        meta_zip = hf_hub_download(
            repo_id='theairlabcmu/TartanGround', 
            filename=f'{env}/Data_omni/P0000/metadata.zip', 
            repo_type='dataset'
        )
        with zipfile.ZipFile(meta_zip, 'r') as z:
            z.extract('pose_lcam_front.txt', extract_path)
        print(f"[{env}] Extracted pose_lcam_front.txt")
    except Exception as e:
        print(f"[{env}] Error: {e}")
