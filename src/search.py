from typing import List, Tuple
import numpy as np


def cosine_similarity(
    query_embedding: np.ndarray,
    image_embeddings: np.ndarray
) -> np.ndarray:
    """Calculate cosine similarity between one query and all images."""
    if image_embeddings.size == 0:
        raise ValueError("No image embeddings are available.")

    query_norm = np.linalg.norm(query_embedding)
    image_norms = np.linalg.norm(image_embeddings, axis=1)

    if query_norm == 0:
        raise ValueError("Query embedding has zero magnitude.")

    safe_norms = np.where(image_norms == 0, 1e-12, image_norms)

    return (image_embeddings @ query_embedding) / (
        safe_norms * query_norm
    )


def top_k_search(
    query_embedding: np.ndarray,
    image_embeddings: np.ndarray,
    image_paths: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Return the Top-K most similar images."""
    if len(image_paths) != len(image_embeddings):
        raise ValueError(
            "Number of image paths does not match number of embeddings."
        )

    if not image_paths:
        raise ValueError("No indexed images found.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    top_k = min(top_k, len(image_paths))

    scores = cosine_similarity(query_embedding, image_embeddings)

    # argpartition avoids fully sorting the whole dataset.
    candidate_indices = np.argpartition(
        -scores, top_k - 1
    )[:top_k]

    # Sort only the Top-K candidates.
    candidate_indices = candidate_indices[
        np.argsort(-scores[candidate_indices])
    ]

    results = [
        (image_paths[index], float(scores[index]))
        for index in candidate_indices
    ]

    return results
