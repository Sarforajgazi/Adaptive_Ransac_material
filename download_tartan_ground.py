import os
import tartanair as ta

def main():
    # Define local data directory
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(workspace_dir, "data")
    os.makedirs(data_root, exist_ok=True)
    
    print(f"Initializing TartanAir toolbox with data root: {data_root}")
    ta.init(data_root)
    
    print("Starting download of 'Office' environment (omni version, trajectory P0000, lidar modality)...")
    # This will fetch Office/Data_omni/P0000/lidar.zip (~404 MB) from Hugging Face
    success = ta.download_ground(
        env='Office',
        version='omni',
        traj='P0000',
        modality='lidar',
        unzip=True,
        delete_zip=True,
        num_workers=2,
        data_source='huggingface'
    )
    
    if success:
        print("\n Download and extraction completed successfully!")
        expected_path = os.path.join(data_root, "Office", "Data_omni", "P0000", "lidar")
        print(f"Data should be located at: {expected_path}")
    else:
        print("\n Download failed. Please check the logs above.")

if __name__ == "__main__":
    main()
