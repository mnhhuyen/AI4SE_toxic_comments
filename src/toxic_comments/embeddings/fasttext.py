"""FastText embedding vectorizer."""

from __future__ import annotations

from pathlib import Path

from toxic_comments.config import EMBEDDINGS_DIR
from toxic_comments.embeddings.pooling import MeanEmbeddingVectorizer, build_pooled_embedding_vectorizer

FASTTEXT_VECTOR_FILE = "fasttext/wiki-news-300d-1M.vec"


class FastTextVectorizer(MeanEmbeddingVectorizer):
    def __init__(
        self,
        embedding_dir: Path = EMBEDDINGS_DIR,
        max_vectors: int | None = None,
    ) -> None:
        super().__init__(
            embedding_path=embedding_dir / FASTTEXT_VECTOR_FILE,
            lowercase=True,
            max_vectors=max_vectors,
        )


def build_fasttext_vectorizer(
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_vectors: int | None = None,
    pooling: str = "mean",
) -> MeanEmbeddingVectorizer:
    return build_pooled_embedding_vectorizer(
        embedding_path=embedding_dir / FASTTEXT_VECTOR_FILE,
        pooling=pooling,
        lowercase=True,
        max_vectors=max_vectors,
    )
