"""Bi-GRU + (max ‖ attention) pooling multi-label classifier (PyTorch).

Architecture
------------
tokens -> Embedding (pretrained FastText/GloVe or random) -> SpatialDropout
       -> stacked Bi-GRU (packed sequences, padding ignored)
       -> concat[ masked max-pooling , additive attention pooling ]
       -> Dropout -> Linear(num_labels)  (sigmoid applied in predict_proba)

The attention pooling follows Yang et al. (2016): u_t = tanh(W h_t + b),
alpha_t = softmax(u_t · u_w), s = sum_t alpha_t h_t. Bi-GRU + attention was
the strongest single model in van Aken et al. (2018) on the Jigsaw data.

Import pattern
--------------
Like ``roberta_label_dependency.py``, this module defines ``torch.nn.Module``
subclasses, so ``torch`` is imported at module level. ``models/bigru.py``
(the lightweight builder) and ``models/registry.py`` defer importing this
module inside a try/except so the rest of the project keeps working on
machines without torch.

scikit-learn contract
---------------------
Every constructor argument is stored verbatim (no derived state), so
``sklearn.base.clone`` works — required by ``ThresholdedClassifier`` and the
k-fold loop. Fitted state ends with ``_``. Pickling (``joblib.dump`` in
``train.save_model``) is GPU-safe: weights are moved to CPU on pickle and the
network is rebuilt on unpickle, mirroring ``_roberta_base.py``.
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from toxic_comments.config import LABEL_COLUMNS
from toxic_comments.embeddings.pooling import load_text_embeddings, tokenize

PAD_INDEX = 0
UNK_INDEX = 1


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """Additive attention pooling over time (Yang et al., 2016)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, input_dim)
        self.context = nn.Linear(input_dim, 1, bias=False)  # u_w

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden: [B, T, D], mask: [B, T] (True = real token)
        scores = self.context(torch.tanh(self.projection(hidden))).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)  # [B, T]
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)  # [B, D]
        return pooled, weights


class BiGRUMaxAttentionNet(nn.Module):
    """Embedding -> SpatialDropout -> Bi-GRU -> [max ‖ attention] -> Linear."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        num_labels: int,
        spatial_dropout: float,
        dropout: float,
        freeze_embeddings: bool,
        embedding_matrix: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_INDEX)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = not freeze_embeddings

        # Dropout1d on [B, D, T] drops whole embedding channels across all
        # time steps == Keras SpatialDropout1D.
        self.spatial_dropout = nn.Dropout1d(spatial_dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionPooling(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(4 * hidden_size, num_labels)

    def encode(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        """Return Bi-GRU states [B, T, 2H] and the padding mask [B, T]."""

        total_length = input_ids.shape[1]
        mask = input_ids != PAD_INDEX
        embedded = self.embedding(input_ids)  # [B, T, E]
        embedded = self.spatial_dropout(embedded.transpose(1, 2)).transpose(1, 2)

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=total_length)
        return hidden, mask

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor, return_attention: bool = False):
        hidden, mask = self.encode(input_ids, lengths)
        max_pooled = hidden.masked_fill(~mask.unsqueeze(-1), torch.finfo(hidden.dtype).min).max(dim=1).values
        attn_pooled, weights = self.attention(hidden, mask)
        features = self.dropout(torch.cat([max_pooled, attn_pooled], dim=1))  # [B, 4H]
        logits = self.classifier(features)
        return (logits, weights) if return_attention else logits


# --------------------------------------------------------------------------
# scikit-learn estimator
# --------------------------------------------------------------------------
class BiGRUMaxAttentionClassifier(BaseEstimator, ClassifierMixin):
    """Bi-GRU multi-label classifier with a scikit-learn interface.

    Parameters worth tuning: ``max_length``, ``hidden_size``, ``num_layers``,
    ``num_epochs``, ``learning_rate``. Set ``embedding_path=None`` to train
    embeddings from scratch (no download needed, useful for smoke tests).
    """

    def __init__(
        self,
        embedding_path: str | Path | None = None,
        embedding_dim: int = 300,
        max_embedding_vectors: int | None = None,
        freeze_embeddings: bool = True,
        lowercase: bool = True,
        max_vocab_size: int = 100_000,
        min_count: int = 1,
        max_length: int = 200,
        hidden_size: int = 64,
        num_layers: int = 2,
        spatial_dropout: float = 0.2,
        dropout: float = 0.1,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        num_epochs: int = 4,
        grad_clip: float | None = 1.0,
        use_pos_weight: bool = False,
        num_labels: int = len(LABEL_COLUMNS),
        random_state: int = 42,
        device: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.max_embedding_vectors = max_embedding_vectors
        self.freeze_embeddings = freeze_embeddings
        self.lowercase = lowercase
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.spatial_dropout = spatial_dropout
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.use_pos_weight = use_pos_weight
        self.num_labels = num_labels
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    # ---- vocabulary / encoding ---------------------------------------------
    def _tokenize(self, text: str) -> list[str]:
        return tokenize(text, lowercase=self.lowercase)

    def _build_vocab(self, texts: pd.Series) -> dict[str, int]:
        counts = Counter(token for text in texts for token in self._tokenize(text))
        vocab = {"<pad>": PAD_INDEX, "<unk>": UNK_INDEX}
        for token, count in counts.most_common(self.max_vocab_size):
            if count < self.min_count:
                break
            vocab[token] = len(vocab)
        return vocab

    def _encode(self, X) -> tuple[np.ndarray, np.ndarray]:
        texts = pd.Series(X).fillna("").astype(str)
        ids = np.full((len(texts), self.max_length), PAD_INDEX, dtype=np.int32)
        lengths = np.ones(len(texts), dtype=np.int64)  # empty text -> one <unk> step
        for row, text in enumerate(texts):
            token_ids = [self.vocab_.get(token, UNK_INDEX) for token in self._tokenize(text)][: self.max_length]
            if not token_ids:
                token_ids = [UNK_INDEX]
            ids[row, : len(token_ids)] = token_ids
            lengths[row] = len(token_ids)
        return ids, lengths

    def _build_embedding_matrix(self) -> np.ndarray | None:
        if self.embedding_path is None:
            self.embedding_dim_ = self.embedding_dim
            self.embedding_coverage_ = None
            return None

        vectors = load_text_embeddings(Path(self.embedding_path), max_vectors=self.max_embedding_vectors)
        dim = len(next(iter(vectors.values())))
        stacked = np.vstack(list(vectors.values()))
        rng = np.random.default_rng(self.random_state)
        matrix = rng.normal(stacked.mean(), stacked.std(), size=(len(self.vocab_), dim)).astype(np.float32)
        matrix[PAD_INDEX] = 0.0

        hits = 0
        for token, index in self.vocab_.items():
            vector = vectors.get(token)
            if vector is not None:
                matrix[index] = vector
                hits += 1
        self.embedding_dim_ = dim
        self.embedding_coverage_ = hits / max(len(self.vocab_) - 2, 1)
        self._log(f"Embedding {Path(self.embedding_path).name}: phủ {self.embedding_coverage_:.1%} từ vựng ({hits:,}/{len(self.vocab_) - 2:,}).")
        return matrix

    def _build_network(self, embedding_matrix: np.ndarray | None = None) -> BiGRUMaxAttentionNet:
        return BiGRUMaxAttentionNet(
            vocab_size=len(self.vocab_),
            embedding_dim=self.embedding_dim_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_labels=self.num_labels,
            spatial_dropout=self.spatial_dropout,
            dropout=self.dropout,
            # Random-init embeddings must be trainable, otherwise they carry no signal.
            freeze_embeddings=self.freeze_embeddings and embedding_matrix is not None,
            embedding_matrix=embedding_matrix,
        )

    def _make_loader(self, ids, lengths, y=None, shuffle=False) -> DataLoader:
        tensors = [torch.as_tensor(ids), torch.as_tensor(lengths)]
        if y is not None:
            tensors.append(torch.as_tensor(y, dtype=torch.float32))
        generator = torch.Generator().manual_seed(self.random_state)
        return DataLoader(TensorDataset(*tensors), batch_size=self.batch_size, shuffle=shuffle, generator=generator)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # ---- scikit-learn contract ---------------------------------------------
    def fit(self, X, y) -> "BiGRUMaxAttentionClassifier":
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 2 or y.shape[1] != self.num_labels:
            raise ValueError(f"y must have shape (n_samples, {self.num_labels}), got {y.shape}")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        texts = pd.Series(X).fillna("").astype(str)
        start = time.perf_counter()
        self.vocab_ = self._build_vocab(texts)
        embedding_matrix = self._build_embedding_matrix()
        ids, lengths = self._encode(texts)
        self._log(f"Vocab {len(self.vocab_):,} từ, encode {len(texts):,} dòng sau {time.perf_counter() - start:.1f}s.")

        self.model_ = self._build_network(embedding_matrix).to(self.device_)
        loader = self._make_loader(ids, lengths, y, shuffle=True)

        pos_weight = None
        if self.use_pos_weight:
            positives = y.sum(axis=0).clip(min=1.0)
            pos_weight = torch.as_tensor((len(y) - positives) / positives, dtype=torch.float32, device=self.device_)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            [p for p in self.model_.parameters() if p.requires_grad], lr=self.learning_rate
        )

        self.loss_history_ = []
        self.model_.train()
        for epoch in range(self.num_epochs):
            total, batches = 0.0, 0
            progress = tqdm(loader, desc=f"BiGRU epoch {epoch + 1}/{self.num_epochs}", leave=False, disable=not self.verbose)
            for batch_ids, batch_lengths, targets in progress:
                batch_ids = batch_ids.long().to(self.device_)
                targets = targets.to(self.device_)
                optimizer.zero_grad()
                loss = loss_fn(self.model_(batch_ids, batch_lengths), targets)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                optimizer.step()
                total += loss.item()
                batches += 1
                progress.set_postfix(loss=f"{loss.item():.4f}")
            self.loss_history_.append(total / max(batches, 1))
            self._log(f"Epoch {epoch + 1}/{self.num_epochs} — loss trung bình {self.loss_history_[-1]:.4f}")
        self.model_.eval()
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.__sklearn_is_fitted__():
            raise ValueError("BiGRUMaxAttentionClassifier must be fitted before predict_proba.")
        ids, lengths = self._encode(X)
        loader = self._make_loader(ids, lengths)
        self.model_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_ids, batch_lengths in loader:
                logits = self.model_(batch_ids.long().to(self.device_), batch_lengths)
                outputs.append(torch.sigmoid(logits).cpu().numpy())
        if not outputs:
            return np.empty((0, self.num_labels), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def explain(self, text: str) -> pd.DataFrame:
        """Token-level attention weights for one comment (explainability).

        Note: attention weights are an indication, not a faithful
        explanation (Jain & Wallace, 2019).
        """

        if not self.__sklearn_is_fitted__():
            raise ValueError("Model must be fitted before explain().")
        tokens = self._tokenize(str(text))[: self.max_length] or ["<unk>"]
        ids, lengths = self._encode([text])
        self.model_.eval()
        with torch.no_grad():
            _, weights = self.model_(torch.as_tensor(ids).long().to(self.device_), torch.as_tensor(lengths), return_attention=True)
        weights = weights[0, : len(tokens)].cpu().numpy()
        return pd.DataFrame({"token": tokens, "attention": weights})

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "model_")

    # ---- GPU-safe pickling --------------------------------------------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            state["_model_state_dict"] = {key: value.cpu() for key, value in model.state_dict().items()}
        return state

    def __setstate__(self, state: dict) -> None:
        model_state_dict = state.pop("_model_state_dict", None)
        self.__dict__.update(state)
        if model_state_dict is not None:
            self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            # Weights (embedding included) come from the state dict; requires_grad
            # only matters for training, so freezing flags are irrelevant here.
            self.model_ = self._build_network(embedding_matrix=None)
            self.model_.load_state_dict(model_state_dict)
            self.model_.to(self.device_)
            self.model_.eval()
