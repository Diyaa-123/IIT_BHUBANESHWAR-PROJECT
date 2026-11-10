import json

from src.config import (
    DATASET_DIR,
    EMBEDDINGS_FILE,
    IMAGE_PATHS_FILE,
    MODEL_DIR,
    BATCH_SIZE,
)
from src.embeddings import CLIPEmbedder
from src.utils import get_image_paths, is_valid_image


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset directory: {DATASET_DIR}")

    image_paths = get_image_paths(DATASET_DIR)

    if not image_paths:
        raise RuntimeError(
            f"No supported images were found in {DATASET_DIR}. "
            "Add your assigned dataset to data/images/."
        )

    print(f"Images found: {len(image_paths)}")

    valid_paths = [
        path for path in image_paths
        if is_valid_image(path)
    ]

    print(f"Valid images: {len(valid_paths)}")

    if not valid_paths:
        raise RuntimeError(
            "Images were found, but none could be opened successfully."
        )

    embedder = CLIPEmbedder()

    embeddings, saved_paths = embedder.generate_image_embeddings(
        valid_paths,
        batch_size=BATCH_SIZE
    )

    if embeddings.shape[0] == 0:
        raise RuntimeError(
            "No embeddings were generated. Check the dataset and image files."
        )

    import numpy as np
    np.save(EMBEDDINGS_FILE, embeddings)

    with open(IMAGE_PATHS_FILE, "w", encoding="utf-8") as file:
        json.dump(saved_paths, file, indent=2)

    print("\nEmbedding generation completed.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Number of indexed images: {len(saved_paths)}")
    print(f"Saved embeddings: {EMBEDDINGS_FILE}")
    print(f"Saved image paths: {IMAGE_PATHS_FILE}")


if __name__ == "__main__":
    main()
