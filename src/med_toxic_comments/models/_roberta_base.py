"""Shared base class for RoBERTa-based multi-label classifiers.

This module gives Method 1 (RoBERTa+BCE), Method 2 (RoBERTa+ASL), and
Method 3 (RoBERTa + label-dependency layer, this integration) one
tokenization/training/inference loop to share, so each variant only needs to
override the model architecture (``_build_model``) and, if needed, the loss
function (``_compute_loss``). Everything else — including the
``get_params``/``set_params``/``clone`` contract that
``evaluation.cross_validate_model`` relies on via ``sklearn.base.clone`` — is
implemented once here.

Design notes
------------
- Every constructor parameter is stored verbatim as an attribute with the
  same name (no mutation, no derived state) so ``sklearn.base.clone`` can
  rebuild an untrained copy of the estimator from ``get_params()`` alone.
  Anything computed from data (tokenizer, model weights, label statistics)
  is only ever set inside ``fit`` and named with a trailing underscore
  (``self.model_``), which is the scikit-learn convention for "fitted"
  attributes that ``clone`` must NOT copy.
- Fine-tuning a transformer inside a 5-fold CV loop is expensive. See
  ``models/registry.py`` and ``cli.py`` for the ``--include-transformers``
  opt-in flag that keeps the existing fast baseline-only runs unaffected.
- ``cross_validate_model`` clones + fits + discards one model per fold — by
  design, since the point of that loop is comparing methods, not producing
  a deployable artifact. Use ``save``/``load`` below (see
  ``notebooks/train_and_save_roberta_label_dependency.ipynb``) to train once
  on a train/test split and persist that model to disk instead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from toxic_comments.config import LABEL_COLUMNS


class _TextLabelDataset(Dataset):
    """Wraps tokenizer output (+ optional labels) for a torch DataLoader."""

    def __init__(self, encodings: dict[str, torch.Tensor], labels: np.ndarray | None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: tensor[index] for key, tensor in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


class RobertaMultiLabelBase(BaseEstimator, ClassifierMixin):
    """Shared fit/predict loop for RoBERTa-based multi-label classifiers.

    Subclasses must implement ``_build_model`` and may override
    ``_compute_loss``. See ``roberta_label_dependency.py`` for Method 3.
    """

    def __init__(
        self,
        pretrained_model_name: str = "roberta-base",
        max_length: int = 128,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 2,
        num_labels: int = len(LABEL_COLUMNS),
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_labels = num_labels
        self.random_state = random_state
        self.device = device

    # ---- extension points for subclasses --------------------------------
    def _build_model(self, y: np.ndarray) -> nn.Module:
        """Build and return the torch module for this variant.

        Receives the *training fold's* labels so subclasses that need label
        statistics (e.g. Method 3's co-occurrence adjacency) can compute
        them here — this keeps the statistic fold-local and avoids leaking
        information from the held-out fold.
        """

        raise NotImplementedError

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Default loss is plain BCE. Method 2 (ASL) overrides this."""

        return nn.functional.binary_cross_entropy_with_logits(logits, targets)

    # ---- scikit-learn estimator contract ---------------------------------
    def fit(self, X: pd.Series, y: np.ndarray) -> "RobertaMultiLabelBase":
        y = np.asarray(y)
        torch.manual_seed(self.random_state)
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.model_ = self._build_model(y).to(self.device_)

        # Tokenizing the whole fold in one call has no built-in progress
        # output and can take a while (CPU-bound) — print around it so a
        # long pause here doesn't look like a hang.
        print(f"Đang tokenize {len(X):,} dòng...", flush=True)
        tokenize_start = time.perf_counter()
        loader = self._make_loader(X, y, shuffle=True)
        print(f"Tokenize xong sau {time.perf_counter() - tokenize_start:.1f}s — bắt đầu train.", flush=True)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate)

        self.model_.train()
        for epoch in range(self.num_epochs):
            progress = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                leave=False,
            )
            for batch in progress:
                targets = batch.pop("labels").to(self.device_)
                batch = {key: value.to(self.device_) for key, value in batch.items()}

                optimizer.zero_grad()
                logits = self.model_(**batch)
                loss = self._compute_loss(logits, targets)
                loss.backward()
                optimizer.step()
                progress.set_postfix(loss=f"{loss.item():.4f}")
        return self

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        loader = self._make_loader(X, y=None, shuffle=False)
        self.model_.eval()
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device_) for key, value in batch.items()}
                logits = self.model_(**batch)
                batches.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(batches, axis=0)

    def predict(self, X: pd.Series, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ---- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist the fitted model, tokenizer, and reconstruction params.

        Works for any subclass without extra code: ``self.model_.state_dict()``
        already includes registered buffers (e.g. Method 3's co-occurrence
        adjacency), so subclass-specific state is captured automatically.
        """

        if not hasattr(self, "model_"):
            raise RuntimeError("Cannot save an unfitted estimator — call fit() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), path / "model_state_dict.pt")
        self.tokenizer_.save_pretrained(path)
        (path / "params.json").write_text(json.dumps(self.get_params(), indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "RobertaMultiLabelBase":
        """Reconstruct a fitted estimator saved with ``save``. Ready to
        ``predict``/``predict_proba`` immediately — no need to call ``fit``.
        """

        path = Path(path)
        params = json.loads((path / "params.json").read_text())
        estimator = cls(**params)
        estimator.device_ = device or ("cuda" if torch.cuda.is_available() else "cpu")
        estimator.tokenizer_ = AutoTokenizer.from_pretrained(path)

        # _build_model may need label statistics (e.g. Method 3's adjacency).
        # A zero-filled placeholder is fine here: the real values are part of
        # the state_dict loaded right below, which overwrites this anyway.
        placeholder_y = np.zeros((1, estimator.num_labels))
        estimator.model_ = estimator._build_model(placeholder_y).to(estimator.device_)

        state_dict = torch.load(path / "model_state_dict.pt", map_location=estimator.device_)
        estimator.model_.load_state_dict(state_dict)
        estimator.model_.eval()
        return estimator

    # ---- helpers ----------------------------------------------------------
    def _make_loader(self, X: pd.Series, y: np.ndarray | None, shuffle: bool) -> DataLoader:
        encodings = self.tokenizer_(
            list(X),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dataset = _TextLabelDataset(dict(encodings), y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

