"""GloVe Twitter embedding vectorizer."""

from __future__ import annotations

from pathlib import Path

from toxic_comments.config import EMBEDDINGS_DIR
from toxic_comments.embeddings.pooling import MeanEmbeddingVectorizer, build_pooled_embedding_vectorizer

GLOVE_TWITTER_VECTOR_FILE = "glove_twitter/glove.twitter.27B.200d.txt"


class GloveTwitterVectorizer(MeanEmbeddingVectorizer):
    def __init__(
        self,
        embedding_dir: Path = EMBEDDINGS_DIR,
        max_vectors: int | None = None,
    ) -> None:
        super().__init__(
            embedding_path=embedding_dir / GLOVE_TWITTER_VECTOR_FILE,
            lowercase=True,
            max_vectors=max_vectors,
        )


def build_glove_twitter_vectorizer(
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_vectors: int | None = None,
    pooling: str = "mean",
) -> MeanEmbeddingVectorizer:
    return build_pooled_embedding_vectorizer(
        embedding_path=embedding_dir / GLOVE_TWITTER_VECTOR_FILE,
        pooling=pooling,
        lowercase=True,
        max_vectors=max_vectors,
    )
