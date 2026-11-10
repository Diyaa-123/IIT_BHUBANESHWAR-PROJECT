from pathlib import Path
from typing import List
from PIL import Image

from .config import IMAGE_EXTENSIONS


def get_image_paths(dataset_dir: Path) -> List[Path]:
    """Recursively find supported image files in the dataset directory."""
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory was not found: {dataset_dir}"
        )

    image_paths = []

    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    image_paths.sort()
    return image_paths


def is_valid_image(image_path: Path) -> bool:
    """Check whether an image can be opened successfully."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def load_image(image_path: Path) -> Image.Image:
    """Load an image as RGB."""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as error:
        raise ValueError(
            f"Could not load image '{image_path}': {error}"
        )
