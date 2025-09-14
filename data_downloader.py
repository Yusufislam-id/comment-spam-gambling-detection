"""
Module for downloading and organizing datasets from Kaggle and Huggingface 
related to gambling detection.
"""

import os
import shutil
import kaggle
import kagglehub
import json
from pathlib import Path  # Add this import
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Create datasets directory if it doesn't exist
DATASETS_DIR = "datasets"
if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    kaggle_path = os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')
    if not os.path.exists(kaggle_path):
        # Check for kaggle.json in current directory
        local_path = Path(__file__).parent / 'kaggle.json'
        if local_path.exists():
            # Create .kaggle directory if it doesn't exist
            os.makedirs(os.path.dirname(kaggle_path), exist_ok=True)
            # Copy kaggle.json to the correct location
            shutil.copy(local_path, kaggle_path)
            # Set correct permissions
            os.chmod(kaggle_path, 0o600)
            print(f"Copied kaggle.json to {kaggle_path}")
        else:
            raise FileNotFoundError("Please place your kaggle.json file in the project directory")

def download_kaggle_datasets():
    """Download and organize datasets from Kaggle."""
    # Setup Kaggle credentials first
    setup_kaggle_credentials()
    
    dataset_urls = [
        "ferdiansakti/gambling-comments-from-youtube-platform",
        "yaemico/judionline",
        "yaemico/deteksi-judi-online"
    ]

    for dataset_url in dataset_urls:
        try:
            print(f"Downloading Kaggle dataset: {dataset_url}")
            username, dataset_name = dataset_url.split('/')
            
            # Create dataset directory
            dataset_subdir = os.path.join(DATASETS_DIR, "kaggle", dataset_name)
            Path(dataset_subdir).mkdir(parents=True, exist_ok=True)
            
            # Download using Kaggle API
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                dataset_url,
                path=dataset_subdir,
                unzip=True
            )
            
            print(f"Successfully downloaded to {os.path.abspath(dataset_subdir)}")
            
        except Exception as e:
            print(f"Error downloading {dataset_url}: {str(e)}")


def download_huggingface_datasets():
    """Download and organize datasets from Huggingface."""
    dataset_ids = [
        "indirapravianti/online_gambling_yt_comments",
        "KagChi/indonesian-gambling-words"
    ]

    for dataset_id in dataset_ids:
        print(f"Downloading Huggingface dataset: {dataset_id}")
        dataset_name = dataset_id.split('/')[-1]
        dataset_subdir = os.path.join(DATASETS_DIR, "huggingface", dataset_name)
        
        if not os.path.exists(dataset_subdir):
            os.makedirs(dataset_subdir)

        try:
            # Load dataset using datasets library
            dataset = load_dataset(dataset_id)
            
            # Save each split to CSV
            for split in dataset.keys():
                output_file = os.path.join(dataset_subdir, f"{split}.csv")
                dataset[split].to_csv(output_file, index=False)
                print(f"Saved {split} split to {output_file}")
        
        except Exception as e:
            print(f"Error downloading {dataset_id}: {str(e)}")

def main():
    """Main function to execute all downloads."""
    print("Starting dataset downloads...")
    
    # Create subdirectories for each source
    os.makedirs(os.path.join(DATASETS_DIR, "kaggle"), exist_ok=True)
    os.makedirs(os.path.join(DATASETS_DIR, "huggingface"), exist_ok=True)
    
    # Download from both sources
    download_kaggle_datasets()
    download_huggingface_datasets()
    
    print("All datasets downloaded and organized successfully!")

if __name__ == "__main__":
    main()