import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import (
    EMBEDDINGS_FILE,
    IMAGE_PATHS_FILE,
    DEFAULT_TOP_K,
)
from .embeddings import CLIPEmbedder
from .search import top_k_search


class ImageSearchEngine:
    """Complete semantic image retrieval backend."""

    def __init__(
        self,
        embeddings_file: Path = EMBEDDINGS_FILE,
        image_paths_file: Path = IMAGE_PATHS_FILE
    ):
        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_file}\n"
                "Run 'python prepare_embeddings.py' first."
            )

        if not image_paths_file.exists():
            raise FileNotFoundError(
                f"Image path index not found: {image_paths_file}\n"
                "Run 'python prepare_embeddings.py' first."
            )

        self.image_embeddings = np.load(embeddings_file)

        with open(image_paths_file, "r", encoding="utf-8") as file:
            self.image_paths = json.load(file)

        if len(self.image_paths) != len(self.image_embeddings):
            raise ValueError(
                "Stored image paths and embeddings have different lengths."
            )

        if len(self.image_paths) == 0:
            raise ValueError("The image index is empty.")

        self.embedder = CLIPEmbedder()

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[str, float]]:
        """Search the indexed dataset using a natural-language query."""
        query_embedding = self.embedder.generate_text_embedding(query)

        return top_k_search(
            query_embedding,
            self.image_embeddings,
            self.image_paths,
            top_k
        )
