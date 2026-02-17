import os
import kaggle
import zipfile
import pandas as pd

def download_datasets():
    # Create directories
    os.makedirs('data/parkinsons', exist_ok=True)
    os.makedirs('data/alzheimers', exist_ok=True)
    
    # Download Parkinson's dataset
    print("Downloading Parkinson's dataset...")
    kaggle.api.dataset_download_files('kmader/parkinsons-drawings', 
                                       path='data/parkinsons', 
                                       unzip=True)
    
    # Download Alzheimer's dataset
    print("Downloading Alzheimer's MRI dataset...")
    kaggle.api.dataset_download_files('junesuzi/alzheimer-mri-dataset',
                                       path='data/alzheimers',
                                       unzip=True)
    
    print("Download complete!")

if __name__ == "__main__":
    download_datasets()