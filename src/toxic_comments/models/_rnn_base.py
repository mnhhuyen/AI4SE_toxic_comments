"""Shared PyTorch recurrent classifier: Bi-RNN / Bi-LSTM / Bi-GRU.

All three variants share one architecture and differ only in the recurrent
cell (``cell_type``), so their fold results isolate the effect of the cell:

    token ids -> embedding (pretrained, optionally frozen) -> dropout
              -> bidirectional RNN/LSTM/GRU
              -> [max pooling ; learned attention pooling] -> dropout
              -> linear layer with one logit per label (BCE loss)

Like ``_roberta_base.py``, this module imports torch at module level because
it defines ``nn.Module`` subclasses; ``models/rnn.py``, ``bilstm.py`` and
``bigru.py`` defer importing it so the registry still works without torch.

Design notes
------------
- The vocabulary is built from the *training fold only* inside ``fit``, so
  no information leaks from the held-out fold.
- Pretrained vectors are streamed from the text file and only rows for
  in-vocabulary tokens are kept, instead of loading the whole file
  (``embeddings.pooling.load_text_embeddings`` keeps every vector).
- Constructor parameters are stored verbatim so ``sklearn.base.clone`` works
  (``ThresholdedClassifier`` and k-fold loops rely on it).
- ``__getstate__``/``__setstate__`` move weights to CPU before pickling, as
  in ``_roberta_base.py``, so joblib files trained on GPU load on CPU.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from toxic_comments.embeddings.pooling import tokenize

PAD_ID = 0
UNK_ID = 1
CELL_TYPES = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


def load_vocab_embeddings(
    embedding_path: Path,
    vocab: dict[str, int],
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Return an embedding matrix aligned with ``vocab`` and its coverage.

    Rows for tokens missing from the file are drawn from a normal
    distribution matching the found vectors' mean/std; the padding row is 0.
    """

    embedding_path = Path(embedding_path)
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {embedding_path}. "
            "See data/embeddings/README.md for download instructions."
        )

    found: dict[int, np.ndarray] = {}
    dim: int | None = None
    with embedding_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.rstrip().split(" ")
            if line_number == 1 and len(parts) == 2 and all(p.isdigit() for p in parts):
                continue  # word2vec-style "count dim" header
            if len(parts) < 2:
                continue
            if dim is None:
                dim = len(parts) - 1
            token_id = vocab.get(parts[0])
            if token_id is None or token_id in found or len(parts) - 1 != dim:
                continue
            try:
                found[token_id] = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue

    if dim is None or not found:
        raise ValueError(f"No usable vectors for this vocabulary in {embedding_path}")

    vectors = np.vstack(list(found.values()))
    rng = np.random.default_rng(random_state)
    matrix = rng.normal(
        vectors.mean(), vectors.std(), size=(len(vocab) + 2, dim)
    ).astype(np.float32)
    matrix[PAD_ID] = 0.0
    for token_id, vector in found.items():
        matrix[token_id] = vector
    return matrix, len(found) / max(len(vocab), 1)


class _SequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: np.ndarray | None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        labels = None if self.labels is None else self.labels[index]
        return self.sequences[index], labels


def _collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    ids = torch.full((len(sequences), int(lengths.max())), PAD_ID, dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    targets = None
    if labels[0] is not None:
        targets = torch.tensor(np.asarray(labels), dtype=torch.float32)
    return ids, lengths, targets


class _RecurrentNet(nn.Module):
    """Bidirectional recurrent encoder with max + attention pooling."""

    def __init__(
        self,
        cell_type: str,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
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
        self.rnn = CELL_TYPES[cell_type](
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )
        # Additive attention: score_t = v^T tanh(W h_t)
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.Tanh(),
            nn.Linear(2 * hidden_size, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(4 * hidden_size, num_labels)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor, return_attention: bool = False):
        mask = ids != PAD_ID
        embedded = self.embedding_dropout(self.embedding(ids))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=ids.shape[1])

        max_pooled = outputs.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        scores = self.attention(outputs).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        attention_pooled = (weights.unsqueeze(-1) * outputs).sum(dim=1)

        logits = self.classifier(self.dropout(torch.cat([max_pooled, attention_pooled], dim=1)))
        if return_attention:
            return logits, weights
        return logits


class RecurrentMultiLabelClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible Bi-RNN/Bi-LSTM/Bi-GRU multi-label classifier."""

    def __init__(
        self,
        cell_type: str = "gru",
        embedding_path: str | Path | None = None,
        embedding_dim: int = 300,
        train_embeddings: bool = False,
        max_vocab: int = 100_000,
        max_length: int = 200,
        hidden_size: int = 128,
        dropout: float = 0.3,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        num_epochs: int = 4,
        clip_grad_norm: float = 1.0,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.cell_type = cell_type
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.train_embeddings = train_embeddings
        self.max_vocab = max_vocab
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.random_state = random_state
        self.device = device

    # ---- scikit-learn estimator contract ---------------------------------
    def fit(self, X: pd.Series, y: np.ndarray) -> "RecurrentMultiLabelClassifier":
        if self.cell_type not in CELL_TYPES:
            raise ValueError(f"cell_type must be one of: {', '.join(CELL_TYPES)}")
        y = np.asarray(y, dtype=np.float32)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels_ = y.shape[1]

        token_lists = [tokenize(text) for text in pd.Series(X).fillna("").astype(str)]
        counts = Counter(token for tokens in token_lists for token in tokens)
        self.vocab_ = {
            token: index
            for index, (token, _) in enumerate(counts.most_common(self.max_vocab), start=2)
        }

        embedding_matrix = None
        self.embedding_coverage_ = None
        self.embedding_dim_ = self.embedding_dim
        if self.embedding_path is not None:
            embedding_matrix, self.embedding_coverage_ = load_vocab_embeddings(
                Path(self.embedding_path), self.vocab_, random_state=self.random_state
            )
            self.embedding_dim_ = embedding_matrix.shape[1]
            print(
                f"Vocab {len(self.vocab_):,} tokens — "
                f"{self.embedding_coverage_:.1%} found in {Path(self.embedding_path).name}",
                flush=True,
            )

        self.model_ = self._build_model(embedding_matrix).to(self.device_)
        loader = self._make_loader(token_lists, y, shuffle=True)
        optimizer = torch.optim.Adam(
            [p for p in self.model_.parameters() if p.requires_grad], lr=self.learning_rate
        )

        self.history_: list[float] = []
        self.model_.train()
        for epoch in range(self.num_epochs):
            started = time.perf_counter()
            total_loss, total_rows = 0.0, 0
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False)
            for ids, lengths, targets in progress:
                ids, targets = ids.to(self.device_), targets.to(self.device_)
                optimizer.zero_grad()
                logits = self.model_(ids, lengths)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
                loss.backward()
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.clip_grad_norm)
                optimizer.step()
                total_loss += loss.item() * len(ids)
                total_rows += len(ids)
                progress.set_postfix(loss=f"{loss.item():.4f}")
            self.history_.append(total_loss / total_rows)
            print(
                f"[{self.cell_type}] epoch {epoch + 1}/{self.num_epochs} "
                f"loss={self.history_[-1]:.4f} ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
        return self

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        token_lists = [tokenize(text) for text in pd.Series(X).fillna("").astype(str)]
        loader = self._make_loader(token_lists, None, shuffle=False)
        self.model_.eval()
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for ids, lengths, _ in loader:
                logits = self.model_(ids.to(self.device_), lengths)
                batches.append(torch.sigmoid(logits).cpu().numpy())
        if not batches:
            return np.empty((0, self.num_labels_), dtype=np.float32)
        return np.concatenate(batches, axis=0)

    def predict(self, X: pd.Series, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def token_attention(self, text: str) -> pd.DataFrame:
        """Attention weight per token of one text — for qualitative analysis."""

        tokens = tokenize(str(text))[: self.max_length] or ["<unk>"]
        ids, lengths, _ = _collate([(self._encode(tokens), None)])
        self.model_.eval()
        with torch.no_grad():
            _, weights = self.model_(ids.to(self.device_), lengths, return_attention=True)
        return pd.DataFrame(
            {
                "token": tokens,
                "in_vocab": [token in self.vocab_ for token in tokens],
                "attention": weights[0].cpu().numpy().round(4),
            }
        )

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "model_")

    # ---- GPU-safe pickling (joblib.dump / joblib.load via train.py) ------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            state["_model_state_dict"] = {
                key: value.cpu() for key, value in model.state_dict().items()
            }
        return state

    def __setstate__(self, state: dict) -> None:
        model_state_dict = state.pop("_model_state_dict", None)
        self.__dict__.update(state)
        if model_state_dict is not None:
            self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model_ = self._build_model(None)
            self.model_.load_state_dict(model_state_dict)
            self.model_.to(self.device_)
            self.model_.eval()

    # ---- helpers ------------------------------------------------------------
    def _build_model(self, embedding_matrix: np.ndarray | None) -> nn.Module:
        return _RecurrentNet(
            cell_type=self.cell_type,
            vocab_size=len(self.vocab_) + 2,
            embedding_dim=self.embedding_dim_,
            hidden_size=self.hidden_size,
            num_labels=self.num_labels_,
            dropout=self.dropout,
            embedding_matrix=embedding_matrix,
            # Random init has nothing useful to freeze.
            train_embeddings=self.train_embeddings or self.embedding_path is None,
        )

    def _encode(self, tokens: list[str]) -> list[int]:
        ids = [self.vocab_.get(token, UNK_ID) for token in tokens[: self.max_length]]
        return ids or [UNK_ID]  # packing needs length >= 1 for empty texts

    def _make_loader(
        self, token_lists: list[list[str]], y: np.ndarray | None, shuffle: bool
    ) -> DataLoader:
        dataset = _SequenceDataset([self._encode(tokens) for tokens in token_lists], y)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=_collate
        )
