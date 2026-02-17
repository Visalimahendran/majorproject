import subprocess
import sys

# Packages that work with Python 3.12
packages = [
    
    "streamlit-option-menu==0.3.6",
    "streamlit-extras==0.3.0",
    "pandas==2.2.1",
    "numpy==1.26.4",
    "plotly==5.19.0",
    "plotly-express==0.4.1",
    "Pillow==10.2.0",
    "matplotlib==3.8.3",
    "seaborn==0.13.2",
    "bokeh==3.4.0",
    "torch==2.2.0",
    "torchvision==0.17.0",
    "torchaudio==2.2.0",
    "scikit-learn==1.4.1.post1",
    "scikit-image==0.22.0",
    "scipy==1.12.0",
    "opencv-python==4.9.0.80",
    "opencv-contrib-python==4.9.0.80",
    "transformers==4.37.2",
    "datasets==2.17.0",
    "sentencepiece>=0.2.0",
    "accelerate==0.27.2",
    "einops==0.7.0",
    "PyWavelets"  # Note: package name is PyWavelets, not pywt
]

for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Installation complete!")