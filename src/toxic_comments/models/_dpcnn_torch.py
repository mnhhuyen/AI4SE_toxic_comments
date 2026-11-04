from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from toxic_comments.models._rnn_base import (
    PAD_ID,
    RecurrentMultiLabelClassifier,
    _SequenceDataset,
)


def _fixed_length_collate(batch, max_length: int):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    ids = torch.full((len(sequences), max_length), PAD_ID, dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq[:max_length], dtype=torch.long)
    targets = None
    if labels[0] is not None:
        targets = torch.tensor(np.asarray(labels), dtype=torch.float32)
    return ids, lengths, targets


def num_downsampling_blocks(max_length: int) -> int:
    blocks, length = 0, max_length
    while length > 1:
        length = (length + 1) // 2
        blocks += 1
    return blocks


class _ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class _DPCNNNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_size: int,
        num_blocks: int,
        num_labels: int,
        dropout: float,
        embedding_matrix: np.ndarray | None = None,
        train_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = train_embeddings
        self.embedding_dropout = nn.Dropout(dropout)
        self.region_embedding = nn.Conv1d(
            embedding_dim, num_filters, kernel_size, padding=kernel_size // 2
        )
        self.first_block = _ConvBlock(num_filters, kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            _ConvBlock(num_filters, kernel_size) for _ in range(num_blocks)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters, num_labels)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding_dropout(self.embedding(ids))
        x = self.region_embedding(x.transpose(1, 2))
        x = self.first_block(x)
        for block in self.blocks:
            x = block(self.pool(x))
        x = x.max(dim=2).values
        return self.classifier(self.dropout(x))


class DPCNNMultiLabelClassifier(RecurrentMultiLabelClassifier):
    def __init__(
        self,
        embedding_path: str | Path | None = None,
        embedding_dim: int = 300,
        train_embeddings: bool = False,
        max_vocab: int = 100_000,
        max_length: int = 200,
        num_filters: int = 250,
        kernel_size: int = 3,
        dropout: float = 0.3,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        num_epochs: int = 4,
        clip_grad_norm: float = 1.0,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.train_embeddings = train_embeddings
        self.max_vocab = max_vocab
        self.max_length = max_length
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.random_state = random_state
        self.device = device

    def token_attention(self, text: str):
        raise NotImplementedError("DPCNN has no attention layer.")

    def _check_params(self) -> None:
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

    def _log_tag(self) -> str:
        return "dpcnn"

    def _build_model(self, embedding_matrix: np.ndarray | None) -> nn.Module:
        return _DPCNNNet(
            vocab_size=len(self.vocab_) + 2,
            embedding_dim=self.embedding_dim_,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            num_blocks=num_downsampling_blocks(self.max_length),
            num_labels=self.num_labels_,
            dropout=self.dropout,
            embedding_matrix=embedding_matrix,
            train_embeddings=self.train_embeddings or self.embedding_path is None,
        )

    def _make_loader(self, token_lists, y, shuffle: bool) -> DataLoader:
        dataset = _SequenceDataset([self._encode(tokens) for tokens in token_lists], y)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=partial(_fixed_length_collate, max_length=self.max_length),
        )
