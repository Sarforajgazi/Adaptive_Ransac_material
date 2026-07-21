import os
import tartanair as ta
from download_lidar_frames import HELD_OUT_ENVIRONMENTS

def download():
    DATA_ROOT = "data"
    ta.init(DATA_ROOT)
    for env in HELD_OUT_ENVIRONMENTS:
        if env == "House":
            continue # already have a hardcoded fallback or downloaded
        try:
            ta.download_ground(env=env, version="omni", traj="P0000",
                               modality="seg_labels", unzip=False, delete_zip=False,
                               num_workers=4, data_source="huggingface")
            
            import zipfile
            zip_path = os.path.join(DATA_ROOT, env, "seg_labels.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(DATA_ROOT, env))
                os.remove(zip_path)
        except Exception as e:
            print(f"Error downloading {env}: {e}")

if __name__ == "__main__":
    download()
