"""Shared base class for RoBERTa-based multi-label classifiers.

This module gives every RoBERTa-based method (currently just the
label-dependency model in ``roberta_label_dependency.py``, but written so a
future BCE/ASL variant can reuse it) one tokenization/training/inference
loop, so each variant only needs to override the model architecture
(``_build_model``) and, if needed, the loss function (``_compute_loss``).

Unlike the project's other optional-dependency models (e.g.
``models/dpcnn.py``), this module imports ``torch``/``transformers`` at
module level rather than inside a builder function, because it defines
``torch.nn.Module`` subclasses that need those names to exist at class
definition time. Callers that need to work on machines without torch
installed (mirroring the project's ``MissingDependencyEstimator`` pattern)
should defer *importing this module* until inside a try/except — see
``models/registry.py``, which does exactly that for
``roberta_label_dependency``.

Design notes
------------
- Every constructor parameter is stored verbatim as an attribute with the
  same name (no mutation, no derived state) so ``sklearn.base.clone`` can
  rebuild an untrained copy of the estimator from ``get_params()`` alone —
  this is what ``ThresholdedClassifier`` (thresholds.py) and any k-fold
  loop rely on.
- Persistence goes through this project's existing ``joblib.dump``/
  ``joblib.load`` (see ``train.save_model``), not a custom save/load. The
  ``__getstate__``/``__setstate__`` overrides below exist only to make that
  generic pickling GPU-safe: a naive pickle of a CUDA-resident
  ``torch.nn.Module`` fails to load on a CPU-only machine, so the model's
  weights are moved to CPU before pickling and the architecture is rebuilt
  (then weights reloaded) on unpickling.
"""

from __future__ import annotations

import time

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
    """Shared fit/predict loop for RoBERTa-based multi-label classifiers."""

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
        statistics (e.g. co-occurrence) can compute them here — this keeps
        the statistic fold-local and avoids leaking information from the
        held-out fold. When rebuilding from a pickle (see __setstate__), a
        zero-filled placeholder is passed instead; the real values are part
        of the state_dict loaded right after and overwrite the placeholder.
        """

        raise NotImplementedError

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Default loss is plain BCE."""

        return nn.functional.binary_cross_entropy_with_logits(logits, targets)

    # ---- scikit-learn estimator contract ---------------------------------
    def fit(self, X: pd.Series, y: np.ndarray) -> "RobertaMultiLabelBase":
        y = np.asarray(y)
        torch.manual_seed(self.random_state)
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.model_ = self._build_model(y).to(self.device_)

        print(f"Đang tokenize {len(X):,} dòng...", flush=True)
        tokenize_start = time.perf_counter()
        loader = self._make_loader(X, y, shuffle=True)
        print(f"Tokenize xong sau {time.perf_counter() - tokenize_start:.1f}s — bắt đầu train.", flush=True)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate)

        self.model_.train()
        for epoch in range(self.num_epochs):
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False)
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
            placeholder_y = np.zeros((1, self.num_labels))
            self.model_ = self._build_model(placeholder_y)
            self.model_.load_state_dict(model_state_dict)
            self.model_.to(self.device_)
            self.model_.eval()

    # ---- helpers ------------------------------------------------------------
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

