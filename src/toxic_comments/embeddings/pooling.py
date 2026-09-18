"""Pooling utilities for pretrained token embeddings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TOKEN_RE = re.compile(r"(?u)\b\w+\b")


class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Convert text into averaged pretrained token vectors."""

    def __init__(
        self,
        embedding_path: str | Path,
        lowercase: bool = True,
        encoding: str = "utf-8",
        max_vectors: int | None = None,
    ) -> None:
        self.embedding_path = embedding_path
        self.lowercase = lowercase
        self.encoding = encoding
        self.max_vectors = max_vectors

    def fit(self, x: Iterable[str], y=None):
        self.embedding_ = load_text_embeddings(
            Path(self.embedding_path),
            encoding=self.encoding,
            max_vectors=self.max_vectors,
        )
        self.embedding_dim_ = len(next(iter(self.embedding_.values())))
        return self

    def transform(self, x: Iterable[str]) -> np.ndarray:
        check_is_fitted_embedding(self)
        texts = pd.Series(x).fillna("").astype(str)
        vectors = [self._vectorize_text(text) for text in texts]
        if not vectors:
            return np.empty((0, self.output_dim_), dtype=np.float32)
        return np.vstack(vectors)

    @property
    def output_dim_(self) -> int:
        return self.embedding_dim_

    def _vectorize_text(self, text: str) -> np.ndarray:
        token_vectors = self._token_vectors(text)
        if len(token_vectors) == 0:
            return np.zeros(self.output_dim_, dtype=np.float32)
        return self._pool_token_vectors(token_vectors)

    def _token_vectors(self, text: str) -> np.ndarray:
        tokens = tokenize(text, lowercase=self.lowercase)
        token_vectors = [
            self.embedding_[token]
            for token in tokens
            if token in self.embedding_
        ]
        if not token_vectors:
            return np.empty((0, self.embedding_dim_), dtype=np.float32)
        return np.vstack(token_vectors)

    def _pool_token_vectors(self, token_vectors: np.ndarray) -> np.ndarray:
        return np.mean(token_vectors, axis=0).astype(np.float32)


class MaxEmbeddingVectorizer(MeanEmbeddingVectorizer):
    """Convert text into element-wise max-pooled pretrained token vectors."""

    def _pool_token_vectors(self, token_vectors: np.ndarray) -> np.ndarray:
        return np.max(token_vectors, axis=0).astype(np.float32)


class AttentionEmbeddingVectorizer(MeanEmbeddingVectorizer):
    """Convert text into attention-weighted pretrained token vectors."""

    def fit(self, x: Iterable[str], y=None):
        x = list(x)
        super().fit(x, y)
        self.attention_query_ = self._build_attention_query(x)
        return self

    def _build_attention_query(self, x: Iterable[str]) -> np.ndarray:
        texts = pd.Series(x).fillna("").astype(str)
        total = np.zeros(self.embedding_dim_, dtype=np.float64)
        count = 0
        for text in texts:
            vectors = self._token_vectors(text)
            if len(vectors) > 0:
                total += vectors.sum(axis=0, dtype=np.float64)
                count += len(vectors)

        if count == 0:
            return np.ones(self.embedding_dim_, dtype=np.float32)

        query = (total / count).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm == 0:
            return np.ones(self.embedding_dim_, dtype=np.float32)
        return query / norm

    def _pool_token_vectors(self, token_vectors: np.ndarray) -> np.ndarray:
        check_is_fitted_attention(self)
        scores = token_vectors @ self.attention_query_
        weights = softmax(scores)
        return (token_vectors * weights[:, None]).sum(axis=0).astype(np.float32)


class MaxAttentionEmbeddingVectorizer(AttentionEmbeddingVectorizer):
    """Concatenate max pooling and attention pooling for each text."""

    @property
    def output_dim_(self) -> int:
        return 2 * self.embedding_dim_

    def _pool_token_vectors(self, token_vectors: np.ndarray) -> np.ndarray:
        max_vector = np.max(token_vectors, axis=0).astype(np.float32)
        attention_vector = super()._pool_token_vectors(token_vectors)
        return np.concatenate([max_vector, attention_vector]).astype(np.float32)


def build_pooled_embedding_vectorizer(
    embedding_path: str | Path,
    pooling: str = "mean",
    lowercase: bool = True,
    encoding: str = "utf-8",
    max_vectors: int | None = None,
) -> MeanEmbeddingVectorizer:
    """Build an embedding vectorizer with the requested pooling strategy."""

    vectorizer_class = {
        "mean": MeanEmbeddingVectorizer,
        "max": MaxEmbeddingVectorizer,
        "attention": AttentionEmbeddingVectorizer,
        "max_attention": MaxAttentionEmbeddingVectorizer,
    }.get(pooling)
    if vectorizer_class is None:
        raise ValueError("pooling must be one of: mean, max, attention, max_attention")

    return vectorizer_class(
        embedding_path=embedding_path,
        lowercase=lowercase,
        encoding=encoding,
        max_vectors=max_vectors,
    )


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_RE.findall(text)


def load_text_embeddings(
    embedding_path: Path,
    encoding: str = "utf-8",
    max_vectors: int | None = None,
) -> dict[str, np.ndarray]:
    """Load a word-vector text file: token val1 val2 ..."""

    if max_vectors is not None and max_vectors < 1:
        raise ValueError("max_vectors must be positive or None")

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {embedding_path}. "
            "Download pretrained vectors and place them under data/embeddings/."
        )

    embeddings: dict[str, np.ndarray] = {}
    expected_dim: int | None = None
    with embedding_path.open("r", encoding=encoding, errors="ignore") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.rstrip().split()
            if not parts:
                continue

            if line_number == 1 and _looks_like_word2vec_header(parts):
                expected_dim = int(parts[1])
                continue

            token, values = parts[0], parts[1:]
            if not values:
                continue

            try:
                vector = np.asarray(values, dtype=np.float32)
            except ValueError:
                continue

            if expected_dim is None:
                expected_dim = len(vector)
            if len(vector) != expected_dim:
                continue
            if not np.isfinite(vector).all():
                continue

            embeddings[token] = vector
            if max_vectors is not None and len(embeddings) >= max_vectors:
                break

    if not embeddings:
        raise ValueError(f"No embeddings could be loaded from {embedding_path}")
    return embeddings


def _looks_like_word2vec_header(parts: list[str]) -> bool:
    if len(parts) != 2:
        return False
    return all(part.isdigit() for part in parts)


def check_is_fitted_embedding(vectorizer: MeanEmbeddingVectorizer) -> None:
    if not hasattr(vectorizer, "embedding_") or not hasattr(vectorizer, "embedding_dim_"):
        raise ValueError("MeanEmbeddingVectorizer must be fitted before transform.")


def check_is_fitted_attention(vectorizer: AttentionEmbeddingVectorizer) -> None:
    if not hasattr(vectorizer, "attention_query_"):
        raise ValueError("AttentionEmbeddingVectorizer must be fitted before transform.")


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()
