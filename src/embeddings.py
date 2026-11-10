from typing import List, Tuple
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

from .config import CLIP_MODEL_NAME, DEVICE, BATCH_SIZE
from .utils import load_image


class CLIPEmbedder:
    """Reusable CLIP model for image and text embeddings."""

    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = DEVICE):
        self.device = device

        print(f"Loading CLIP model: {model_name}")
        print(f"Using device: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
        """L2-normalize embeddings so dot product equals cosine similarity."""
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    def generate_image_embeddings(
        self,
        image_paths: List,
        batch_size: int = BATCH_SIZE
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate normalized CLIP embeddings for valid images."""
        all_embeddings = []
        valid_paths = []

        for start in tqdm(
            range(0, len(image_paths), batch_size),
            desc="Generating image embeddings"
        ):
            batch_paths = image_paths[start:start + batch_size]
            images = []
            current_paths = []

            for path in batch_paths:
                try:
                    images.append(load_image(path))
                    current_paths.append(path)
                except Exception as error:
                    print(f"Skipping {path}: {error}")

            if not images:
                continue

            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                if hasattr(features, "pooler_output") and features.pooler_output is not None:
                    features = features.pooler_output
                features = self.normalize_embeddings(features)

            all_embeddings.append(features.cpu().numpy())
            valid_paths.extend([str(path) for path in current_paths])

        if not all_embeddings:
            return np.empty((0, 512), dtype=np.float32), []

        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings.astype(np.float32), valid_paths

    def generate_text_embedding(self, query: str) -> np.ndarray:
        """Convert a text query into a normalized CLIP embedding."""
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        inputs = self.processor(
            text=[query.strip()],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            if hasattr(features, "pooler_output") and features.pooler_output is not None:
                features = features.pooler_output
            features = self.normalize_embeddings(features)

        return features.cpu().numpy()[0].astype(np.float32)

