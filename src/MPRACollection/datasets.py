import os
import requests
from pathlib import Path

# Placeholder for dataset URLs. Update with actual links.
DATASETS = {
    "dataset1": "https://your-link-to-dataset1",
    "dataset2": "https://your-link-to-dataset2",
    # Add more datasets here
}

# Data directory inside the package
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def download_dataset(name, overwrite=False):
    """
    Download a dataset by name.
    Args:
        name (str): Name of the dataset (must be a key in DATASETS).
        overwrite (bool): If True, re-download even if file exists.
    Returns:
        Path to the downloaded file.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    url = DATASETS[name]
    filename = url.split("/")[-1]
    dest = DATA_DIR / filename

    if dest.exists() and not overwrite:
        print(f"{filename} already exists at {dest}. Skipping download.")
        return dest

    print(f"Downloading {name} from {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {name} to {dest}")
    return dest

# Optionally, add convenience functions for each dataset
def download_dataset1(overwrite=False):
    return download_dataset("dataset1", overwrite=overwrite)

def download_dataset2(overwrite=False):
    return download_dataset("dataset2", overwrite=overwrite) 