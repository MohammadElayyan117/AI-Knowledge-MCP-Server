
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
BATCH_SIZE = 4


class EmbeddingService:
    """Generate local multilingual embeddings using BGE-M3."""

    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_kwargs = (
            {"torch_dtype": torch.float16}
            if self.device == "cuda"
            else {"torch_dtype": torch.float32}
        )

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=self.device,
            model_kwargs=model_kwargs,
        )

        self.model.eval()

        dimension = self.model.get_embedding_dimension()

        if dimension != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Unexpected embedding dimension: {dimension}"
            )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Create normalized embeddings for text inputs."""

        if not texts:
            return []

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "All texts must be non-empty strings."
            )

        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.astype("float32").tolist()

    def encode_query(self, query: str) -> list[float]:
        """Generate an embedding for one search query."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.encode([query])[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Reuse one model instance within the Python process."""
    return EmbeddingService()
