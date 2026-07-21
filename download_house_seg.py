import os
import tartanair as ta

def download():
    DATA_ROOT = "data"
    ta.init(DATA_ROOT)
    ta.download_ground(env="House", version="omni", traj="P0000",
                       modality="seg_labels", unzip=True, delete_zip=True,
                       num_workers=4, data_source="huggingface")

if __name__ == "__main__":
    download()
