import requests
import os
import tyro
from dataclasses import dataclass
from crysreas import Config

@dataclass
class Args:
    """Arguments for downloading CDVAE reference datasets."""
    save_dir: str = "assets/cdvae"

def main(args: Args):
    """Main entry point for downloading CDVAE datasets."""
    download_cdvae(args.save_dir)

def download_cdvae(save_dir: str):
    """Download the CDVAE test, train, and val CSV files."""
    cdvae_urls = {
        "test": "https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/test.csv",
        "train": "https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/train.csv",
        "val": "https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/val.csv"
    }
    
    # 1. Ensure target directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    for key, url in cdvae_urls.items():
        file_path = os.path.join(save_dir, f"{key}.csv")
        
        if not os.path.exists(file_path):
            print(f"Downloading {key}...")
            try:
                # 2. Initiate request with timeout control
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    print(f"Successfully downloaded: {file_path}")
                else:
                    print(f"Download failed: {key}, status code: {response.status_code}")
            except Exception as e:
                print(f"Connection error: {e}")
        else:
            print(f"File already exists, skipping: {file_path}")

if __name__ == "__main__":
    tyro.cli(main)
