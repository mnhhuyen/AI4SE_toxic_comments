"""Word2Vec embedding vectorizer."""

from __future__ import annotations

from pathlib import Path

from toxic_comments.config import EMBEDDINGS_DIR
from toxic_comments.embeddings.pooling import MeanEmbeddingVectorizer, build_pooled_embedding_vectorizer

WORD2VEC_VECTOR_FILE = "word2vec/GoogleNews-vectors-negative300.txt"


class Word2VecVectorizer(MeanEmbeddingVectorizer):
    def __init__(
        self,
        embedding_dir: Path = EMBEDDINGS_DIR,
        max_vectors: int | None = None,
    ) -> None:
        super().__init__(
            embedding_path=embedding_dir / WORD2VEC_VECTOR_FILE,
            lowercase=False,
            max_vectors=max_vectors,
        )


def build_word2vec_vectorizer(
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_vectors: int | None = None,
    pooling: str = "mean",
) -> MeanEmbeddingVectorizer:
    return build_pooled_embedding_vectorizer(
        embedding_path=embedding_dir / WORD2VEC_VECTOR_FILE,
        pooling=pooling,
        lowercase=False,
        max_vectors=max_vectors,
    )
