from pathlib import Path
import torch

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset and generated model files
DATASET_DIR = BASE_DIR / "data" / "images"
MODEL_DIR = BASE_DIR / "models"
EMBEDDINGS_FILE = MODEL_DIR / "image_embeddings.npy"
IMAGE_PATHS_FILE = MODEL_DIR / "image_paths.json"

# Hugging Face CLIP model
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Search / inference settings
DEFAULT_TOP_K = 5
BATCH_SIZE = 16

# Automatically use GPU if PyTorch can access one
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}
