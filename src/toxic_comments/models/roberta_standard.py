"""Standard RoBERTa multi-label classifier — baseline for the label-dependency model.

RoBERTa-base encoder -> first-token (``<s>``) representation -> one linear
layer (768 -> 6 logits) -> sigmoid, trained with plain BCE.

Everything except the classification head is shared with
``roberta_label_dependency`` through ``RobertaMultiLabelBase``: tokenizer,
``max_length``, batch size, AdamW learning rate, number of epochs, seed,
``<s>`` pooling, loss, fit/predict loop and GPU-safe pickling. Comparing the
two models on the same folds therefore isolates the effect of the
label-dependency head (RQ3).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import AutoModel

from toxic_comments.config import LABEL_COLUMNS
from toxic_comments.models._roberta_base import RobertaMultiLabelBase


class _RobertaWithLinearHead(nn.Module):
    """RoBERTa encoder -> [CLS]/<s> pooling -> Linear(hidden_size, num_labels)."""

    def __init__(self, pretrained_model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # same pooling as roberta_label_dependency
        return self.classifier(pooled)  # [batch, num_labels]


class RobertaStandardClassifier(RobertaMultiLabelBase):
    """Baseline: RoBERTa encoder + linear multi-label head, BCE loss.

    Constructor parameters are re-declared explicitly (not ``**kwargs``) so
    scikit-learn's ``get_params``/``clone`` work, exactly like
    ``RobertaLabelDependencyClassifier``.
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
        super().__init__(
            pretrained_model_name=pretrained_model_name,
            max_length=max_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            num_labels=num_labels,
            random_state=random_state,
            device=device,
        )

    def _build_model(self, y: np.ndarray) -> nn.Module:
        # y is unused: the standard head needs no label statistics.
        return _RobertaWithLinearHead(self.pretrained_model_name, self.num_labels)

    # _compute_loss is inherited unchanged (plain BCE), same as the LD model.


def build_roberta_standard(
    pretrained_model_name: str = "roberta-base",
    max_length: int = 128,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 2,
    device: str | None = None,
) -> RobertaStandardClassifier:
    """Factory matching the project's ``build_*`` convention (same defaults as LD)."""

    return RobertaStandardClassifier(
        pretrained_model_name=pretrained_model_name,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device,
    )
