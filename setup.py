#!/usr/bin/env python3
"""
Setup script for NeuroMotor Health System
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "saved_models",
        "assets",
        "samples",
        "logs",
        "datasets/raw",
        "datasets/processed",
        "src"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create empty __init__.py files
    for root, dirs, files in os.walk("src"):
        for dir in dirs:
            init_file = Path(root) / dir / "__init__.py"
            init_file.touch(exist_ok=True)
    
    print("✅ Directory structure created")

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    
    requirements = [
        "streamlit==1.28.0",
        "numpy==1.24.3",
        "pandas==2.1.3",
        "opencv-python==4.8.1.78",
        "Pillow==10.1.0",
        "plotly==5.18.0",
        "matplotlib==3.8.2",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "scikit-learn==1.3.2",
        "scipy==1.11.4"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install: {package}")
    
    print("✅ Requirements installation completed")

def create_sample_files():
    """Create sample files if they don't exist"""
    # Create a sample model checkpoint
    sample_model = Path("saved_models") / "best_model.pth"
    if not sample_model.exists():
        import torch
        import torch.nn as nn
        
        # Create a simple model
        class SampleModel(nn.Module):
            def __init__(self):
                super(SampleModel, self).__init__()
                self.fc = nn.Linear(10, 3)
            
            def forward(self, x):
                return self.fc(x)
        
        model = SampleModel()
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': 0,
            'accuracy': 0.85
        }, sample_model)
        print(f"✅ Created sample model: {sample_model}")
    
    # Create placeholder sample images
    samples_dir = Path("samples")
    sample_images = ["normal.png", "mild.png", "severe.png"]
    
    for img_name in sample_images:
        img_path = samples_dir / img_name
        if not img_path.exists():
            # Create placeholder text file
            with open(img_path, 'w') as f:
                f.write(f"Placeholder for {img_name}")
            print(f"✅ Created placeholder: {img_path}")

def main():
    """Main setup function"""
    print("=" * 60)
    print("NeuroMotor Health System Setup")
    print("=" * 60)
    
    create_directory_structure()
    
    print("\n" + "-" * 60)
    response = input("Install Python packages? (y/n): ")
    if response.lower() == 'y':
        install_requirements()
    
    print("\n" + "-" * 60)
    response = input("Create sample files? (y/n): ")
    if response.lower() == 'y':
        create_sample_files()
    
    print("\n" + "=" * 60)
    print("Setup completed!")
    print("\nTo run the application:")
    print("1. streamlit run main.py")
    print("2. Open http://localhost:8501 in your browser")
    print("\nTo train a model (optional):")
    print("python train_model.py")

if __name__ == "__main__":
    main()