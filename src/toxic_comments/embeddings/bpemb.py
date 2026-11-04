"""SentencePiece subword embeddings with document pooling."""

from __future__ import annotations

from pathlib import Path

from toxic_comments.config import EMBEDDINGS_DIR
import numpy as np
import pandas as pd

from toxic_comments.embeddings.pooling import (
    AttentionEmbeddingVectorizer,
    MeanEmbeddingVectorizer,
    SupervisedAttentionEmbeddingVectorizer,
    MaxSupervisedAttentionEmbeddingVectorizer,
)


class BPEmbVectorizer(MeanEmbeddingVectorizer):
    """Load BPEmb on fit, retaining its tokenizer and full vocabulary.

    max_vectors is accepted for registry compatibility but does not truncate
    BPEmb: SentencePiece IDs require the complete embedding matrix.
    """

    def __init__(
        self,
        embedding_dir: Path = EMBEDDINGS_DIR,
        max_vectors: int | None = None,
        pooling: str = "mean",
        lang: str = "en",
        vocab_size: int = 10000,
        dim: int = 300,
    ) -> None:
        self.embedding_dir = embedding_dir
        self.max_vectors = max_vectors
        self.pooling = pooling
        self.lang = lang
        self.vocab_size = vocab_size
        self.dim = dim

    def fit(self, x, y=None):
        if self.pooling not in {
            "mean", "max", "attention", "max_attention",
            "supervised_attention", "max_supervised_attention",
        }:
            raise ValueError(
                "pooling must be one of: mean, max, attention, max_attention, "
                "supervised_attention, max_supervised_attention"
            )
        x = pd.Series(x).fillna("").astype(str).tolist()
        try:
            from bpemb import BPEmb
        except ImportError as exc:
            raise ImportError("Install BPEmb with: python -m pip install bpemb") from exc

        self.bpemb_ = BPEmb(
            lang=self.lang,
            vs=self.vocab_size,
            dim=self.dim,
            cache_dir=Path(self.embedding_dir) / "bpemb",
            vs_fallback=False,
        )
        self.embedding_ = self.bpemb_.emb
        self.embedding_dim_ = self.bpemb_.dim
        if self.pooling in {"attention", "max_attention"}:
            self.attention_query_ = AttentionEmbeddingVectorizer._build_attention_query(self, x)
        if self.pooling in {"supervised_attention", "max_supervised_attention"}:
            if y is None:
                raise ValueError("Supervised attention requires multi-label targets y.")
            targets = np.asarray(y, dtype=np.float32)
            if targets.ndim != 2 or len(targets) != len(x):
                raise ValueError("y must be a 2D array with one row per text.")
            if not x:
                raise ValueError("Cannot fit supervised attention on an empty dataset.")
            attention_class = (
                MaxSupervisedAttentionEmbeddingVectorizer
                if self.pooling == "max_supervised_attention"
                else SupervisedAttentionEmbeddingVectorizer
            )
            self.supervised_attention_ = attention_class(None)
            self.supervised_attention_.embedding_dim_ = self.embedding_dim_
            # Reserve zero for padding; SentencePiece ID zero is a real subword.
            matrix = np.vstack([
                np.zeros((1, self.embedding_dim_), dtype=np.float32),
                self.bpemb_.vectors,
            ]).astype(np.float32)
            ids = np.zeros((len(x), self.supervised_attention_.max_length), dtype=np.int64)
            for row, text in enumerate(x):
                encoded = self.bpemb_.encode_ids(text)[:ids.shape[1]]
                ids[row, :len(encoded)] = np.asarray(encoded, dtype=np.int64) + 1
            self.supervised_attention_._fit_attention(matrix, ids, targets)
            self.loss_history_ = self.supervised_attention_.loss_history_
        return self

    @property
    def output_dim_(self) -> int:
        return self.embedding_dim_ * (
            2 if self.pooling in {"max_attention", "max_supervised_attention"} else 1
        )

    def _token_vectors(self, text: str) -> np.ndarray:
        if self.pooling in {"supervised_attention", "max_supervised_attention"}:
            ids = self.bpemb_.encode_ids(text)[:self.supervised_attention_.max_length]
            return np.asarray(self.bpemb_.vectors[ids], dtype=np.float32)
        return self.bpemb_.embed(text)

    def _pool_token_vectors(self, token_vectors: np.ndarray) -> np.ndarray:
        if self.pooling == "mean":
            return token_vectors.mean(axis=0).astype(np.float32)
        if self.pooling == "max":
            return token_vectors.max(axis=0).astype(np.float32)
        if self.pooling in {"supervised_attention", "max_supervised_attention"}:
            return self.supervised_attention_._pool_token_vectors(token_vectors)
        attention = AttentionEmbeddingVectorizer._pool_token_vectors(self, token_vectors)
        if self.pooling == "attention":
            return attention
        return np.concatenate([token_vectors.max(axis=0), attention]).astype(np.float32)


def build_bpemb_vectorizer(
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_vectors: int | None = None,
    pooling: str = "mean",
) -> BPEmbVectorizer:
    return BPEmbVectorizer(
        embedding_dir=embedding_dir,
        pooling=pooling,
        max_vectors=max_vectors,
    )
