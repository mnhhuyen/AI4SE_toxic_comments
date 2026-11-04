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


class _SupervisedAttentionNet:
    """Small train-time network; defined lazily so torch stays optional."""

    def __new__(cls, torch, nn, embedding_matrix, num_labels, dropout, include_max=False):
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding.from_pretrained(
                    torch.as_tensor(embedding_matrix, dtype=torch.float32),
                    freeze=True,
                    padding_idx=0,
                )
                dim = embedding_matrix.shape[1]
                self.projection = nn.Linear(dim, dim)
                self.context = nn.Linear(dim, 1, bias=False)
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(dim * (2 if include_max else 1), num_labels)

            def pool(self, token_ids):
                mask = token_ids != 0
                hidden = self.embedding(token_ids)
                scores = self.context(torch.tanh(self.projection(hidden))).squeeze(-1)
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                weights = torch.softmax(scores, dim=1).masked_fill(~mask, 0.0)
                pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
                if include_max:
                    maximum = hidden.masked_fill(
                        ~mask.unsqueeze(-1), torch.finfo(hidden.dtype).min
                    ).max(dim=1).values
                    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, 0.0)
                    pooled = torch.cat([maximum, pooled], dim=1)
                return pooled

            def forward(self, token_ids):
                return self.classifier(self.dropout(self.pool(token_ids)))

        return Network()


class SupervisedAttentionEmbeddingVectorizer(MeanEmbeddingVectorizer):
    """Learn additive attention from multi-label targets, then pool with numpy."""

    def __init__(
        self, embedding_path, lowercase=True, encoding="utf-8", max_vectors=None,
        max_length=200, num_epochs=3, batch_size=256, learning_rate=1e-3,
        dropout=0.1, random_state=42, device=None,
    ):
        super().__init__(embedding_path, lowercase, encoding, max_vectors)
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.random_state = random_state
        self.device = device

    def fit(self, x, y=None):
        if y is None:
            raise ValueError("Supervised attention requires multi-label targets y.")
        from collections import Counter

        self._validate_max_length()
        texts = pd.Series(x).fillna("").astype(str).tolist()
        targets = np.asarray(y, dtype=np.float32)
        if targets.ndim != 2 or len(targets) != len(texts):
            raise ValueError("y must be a 2D array with one row per text.")
        if not texts:
            raise ValueError("Cannot fit supervised attention on an empty dataset.")

        super().fit(texts, y)
        counts = Counter(
            token for text in texts for token in tokenize(text, lowercase=self.lowercase)
            if token in self.embedding_
        )
        self.vocab_ = {token: index + 1 for index, (token, _) in enumerate(counts.most_common())}
        matrix = np.zeros((len(self.vocab_) + 1, self.embedding_dim_), dtype=np.float32)
        for token, index in self.vocab_.items():
            matrix[index] = self.embedding_[token]

        ids = np.zeros((len(texts), self.max_length), dtype=np.int64)
        for row, text in enumerate(texts):
            tokens = tokenize(text, lowercase=self.lowercase)
            token_ids = [self.vocab_[t] for t in tokens if t in self.vocab_][:self.max_length]
            if token_ids:
                ids[row, :len(token_ids)] = token_ids

        return self._fit_attention(matrix, ids, targets)

    def _validate_max_length(self):
        if not isinstance(self.max_length, (int, np.integer)) or self.max_length < 1:
            raise ValueError("max_length must be a positive integer.")

    def _token_vectors(self, text):
        # Match training: filter unavailable vectors, then truncate known tokens.
        tokens = (token for token in tokenize(text, lowercase=self.lowercase)
                  if token in self.embedding_)
        from itertools import islice
        vectors = [self.embedding_[token] for token in islice(tokens, self.max_length)]
        if not vectors:
            return np.empty((0, self.embedding_dim_), dtype=np.float32)
        return np.vstack(vectors)

    def _fit_attention(self, matrix, ids, targets):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dataset = TensorDataset(torch.as_tensor(ids), torch.as_tensor(targets))
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        model = _SupervisedAttentionNet(
            torch, nn, matrix, targets.shape[1], self.dropout,
            include_max=self.output_dim_ == 2 * matrix.shape[1],
        ).to(self.device_)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        self.loss_history_ = []
        model.train()
        for _ in range(self.num_epochs):
            total = 0.0
            for batch_ids, batch_targets in loader:
                batch_ids = batch_ids.to(self.device_)
                batch_targets = batch_targets.to(self.device_)
                optimizer.zero_grad()
                loss = loss_fn(model(batch_ids), batch_targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
            self.loss_history_.append(total / max(len(loader), 1))

        self.attention_W_ = model.projection.weight.detach().cpu().numpy().T.copy()
        self.attention_b_ = model.projection.bias.detach().cpu().numpy().copy()
        self.attention_v_ = model.context.weight.detach().cpu().numpy().reshape(-1).copy()
        return self

    def _pool_token_vectors(self, token_vectors):
        check_is_fitted_supervised_attention(self)
        hidden = np.tanh(token_vectors @ self.attention_W_ + self.attention_b_)
        weights = softmax(hidden @ self.attention_v_)
        return (token_vectors * weights[:, None]).sum(axis=0).astype(np.float32)


class MaxSupervisedAttentionEmbeddingVectorizer(SupervisedAttentionEmbeddingVectorizer):
    """Concatenate max pooling and label-trained attention pooling."""

    @property
    def output_dim_(self):
        return 2 * self.embedding_dim_

    def _pool_token_vectors(self, token_vectors):
        return np.concatenate([
            np.max(token_vectors, axis=0),
            super()._pool_token_vectors(token_vectors),
        ]).astype(np.float32)


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
        "supervised_attention": SupervisedAttentionEmbeddingVectorizer,
        "max_supervised_attention": MaxSupervisedAttentionEmbeddingVectorizer,
    }.get(pooling)
    if vectorizer_class is None:
        raise ValueError(
            "pooling must be one of: mean, max, attention, max_attention, "
            "supervised_attention, max_supervised_attention"
        )

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


def check_is_fitted_supervised_attention(vectorizer) -> None:
    if not all(hasattr(vectorizer, name) for name in ("attention_W_", "attention_b_", "attention_v_")):
        raise ValueError("SupervisedAttentionEmbeddingVectorizer must be fitted before transform.")


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()
