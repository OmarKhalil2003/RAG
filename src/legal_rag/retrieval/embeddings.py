import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


class BGEM3Embedder:
    """
    BGE-M3 multilingual embedding model wrapper.
    Produces 1024-dimensional normalized dense vectors for Arabic and English legal text.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            import torch
            torch.set_num_threads(os.cpu_count() or 4)
            self.model = SentenceTransformer(self.model_name)
            self.model.max_seq_length = 128
        return self.model

    def encode(self, texts: list[str], batch_size: int = 16, show_progress_bar: bool = True) -> list[list[float]]:
        """Encode a list of texts into normalized dense vectors."""
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar
        )
        return [vec.tolist() for vec in embeddings]

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string into a normalized dense vector."""
        model = self._get_model()
        vec = model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()
