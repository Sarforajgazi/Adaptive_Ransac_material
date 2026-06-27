import os
import sys
import zipfile
from huggingface_hub import hf_hub_download

ENV_FILES = {
    "supermarket": "Supermarket/Supermarket_rgb_pcd.zip",
    "forest": "SeasonalForestAutumn/SeasonalForestAutumn_rgb_pcd.zip",
    "tunnel": "AbandonedCable/AbandonedCable_rgb_pcd.zip",
    "meadow": "GreatMarsh/GreatMarsh_rgb_pcd.zip",
    "sewer": "Sewerage/Sewerage_rgb_pcd.zip",
    "city": "Downtown/Downtown_rgb_pcd.zip",
    "hospital": "Hospital/Hospital_rgb_pcd.zip",
    "scandinavia": "OldScandinavia/OldScandinavia_rgb_pcd.zip",
    "spring": "SeasonalForestSpring/SeasonalForestSpring_rgb_pcd.zip",
    "town": "OldTownFall/OldTownFall_rgb_pcd.zip",
    "winternight": "SeasonalForestWinterNight/SeasonalForestWinterNight_rgb_pcd.zip"
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_tartan_pcd.py <environment_type>")
        print("Available environment types:")
        print("  - supermarket : Supermarket (118.9 MB)")
        print("  - forest      : SeasonalForestAutumn (492.6 MB)")
        print("  - tunnel      : AbandonedCable (1.09 GB)")
        print("  - meadow      : GreatMarsh (1.27 GB)")
        print("  - sewer       : Sewerage (120.9 MB)")
        print("  - city        : Downtown (914.4 MB)")
        print("  - hospital    : Hospital (322.2 MB)")
        print("  - scandinavia : OldScandinavia (1.88 GB)")
        print("  - spring      : SeasonalForestSpring (483.7 MB)")
        print("  - town        : OldTownFall (284.4 MB)")
        print("  - winternight : SeasonalForestWinterNight (382.8 MB)")
        sys.exit(1)
        
    env_type = sys.argv[1].strip().lower()
    if env_type not in ENV_FILES:
        print(f"ERROR: Unknown environment type '{env_type}'.")
        print("Choose from: supermarket, forest, tunnel, meadow, sewer, city, hospital, scandinavia, spring, town, winternight")
        sys.exit(1)
        
    hf_filename = ENV_FILES[env_type]
    env_name = hf_filename.split('/')[0]
    
    print(f"\n[1/3] Downloading {hf_filename} from Hugging Face (theairlabcmu/TartanGround)...")
    try:
        local_zip = hf_hub_download(
            repo_id="theairlabcmu/TartanGround",
            filename=hf_filename,
            repo_type="dataset"
        )
        print(f"      Downloaded to cache: {local_zip}")
    except Exception as e:
        print(f"ERROR: Failed to download from Hugging Face: {e}")
        sys.exit(1)
        
    output_dir = os.path.join("tartanair_data", env_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[2/3] Extracting point cloud files to: {output_dir} ...")
    try:
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("      Extraction complete!")
    except Exception as e:
        print(f"ERROR: Failed to extract zip file: {e}")
        sys.exit(1)
        
    print("[3/3] Checking extracted files:")
    extracted_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            full_path = os.path.join(root, file)
            extracted_files.append(full_path)
            print(f"  - {full_path} ({os.path.getsize(full_path)/(1024*1024):.2f} MB)")
            
    print(f"\nSuccessfully downloaded and set up {env_type} environment!")
    if extracted_files:
        pcd_path = extracted_files[0]
        print(f"\nYou can now run ground segmentation with:")
        print(f"  python tartan_ground_segmentation.py {pcd_path} z_down 0.3")

if __name__ == "__main__":
    main()
