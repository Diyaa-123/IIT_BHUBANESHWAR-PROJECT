"""
download_dataset.py

Downloads the Flickr8k dataset from Hugging Face and saves all images
into the data/images/ folder expected by prepare_embeddings.py.

Usage:
    python download_dataset.py

Requirements:
    pip install huggingface_hub Pillow
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download


# ── Config ────────────────────────────────────────────────────────────────────
DATASET_NAME = "intro/flickr8k"
OUTPUT_DIR = Path("data/images")
# ─────────────────────────────────────────────────────────────────────────────


def download(dataset_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{dataset_name}' images from Hugging Face …")
    
    # Download images using fast multi-threaded snapshot_download (resumes automatically)
    cache_path = snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        allow_patterns=["train/*", "*.jpg", "*.jpeg", "*.png"]
    )
    
    source_dir = Path(cache_path) / "train"
    if not source_dir.exists():
        source_dir = Path(cache_path)

    existing_files = {p.name for p in output_dir.glob("*.jpg")}
    source_files = list(source_dir.glob("*.jpg"))
    
    print(f"\nCache contains {len(source_files)} images.")
    print(f"Destination currently has {len(existing_files)} images.")
    
    new_files = [f for f in source_files if f.name not in existing_files]
    if new_files:
        print(f"Copying {len(new_files)} new images to {output_dir.resolve()} …")
        for f in new_files:
            shutil.copy2(f, output_dir / f.name)
    else:
        print("All cached images are already present in destination folder.")

    total_images = len(list(output_dir.glob("*.jpg")))
    print(f"\n✅ Done!")
    print(f"   Total images ready in {output_dir.resolve()}: {total_images}")


if __name__ == "__main__":
    download(DATASET_NAME, OUTPUT_DIR)

