"""Method 3 — RoBERTa + explicit label-dependency graph layer + BCE.

Research role (see the 5-stage research design)
-------------------------------------------------
Representation (RoBERTa encoder) and loss (plain BCE) are held IDENTICAL to
Method 1. The only change relative to Method 1 is one new mechanism: a small
graph message-passing layer over the six toxicity labels, using the
*empirical label co-occurrence* from the training fold as a fixed adjacency
— e.g. ``severe_toxic`` almost always co-occurring with ``toxic``. This
isolates the contribution of explicit label-dependency modeling from
representation changes (Method 1) and imbalance-handling changes (Method 2),
so that Method 4's combined model can later be attributed correctly in the
ablations.

This is intentionally NOT the same mechanism as Method 4's label-attention
head (which attends over *tokens* to build a per-label representation).
Method 3 instead lets each label's *own* representation be refined by its
*correlated labels'* representations — a GraphSAGE-style aggregation step
over a 6-node label graph, using the co-occurrence matrix as a fixed,
data-driven adjacency instead of a learned one.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import AutoModel

from toxic_comments.config import LABEL_COLUMNS
from toxic_comments.models._roberta_base import RobertaMultiLabelBase


def compute_cooccurrence_adjacency(y: np.ndarray) -> torch.Tensor:
    """Row-normalized empirical label co-occurrence from the training fold.

    ``adjacency[i, j]`` approximates P(label_j = 1 | label_i = 1), estimated
    only from the labels passed in (the training fold), so no information
    leaks from the held-out fold. The diagonal is zeroed so a label does not
    "propagate to itself" in the message-passing step below.
    """

    y = np.asarray(y, dtype=float)
    co_occurrence = y.T @ y
    label_counts = np.clip(y.sum(axis=0), a_min=1.0, a_max=None)
    adjacency = co_occurrence / label_counts[:, None]
    np.fill_diagonal(adjacency, 0.0)
    return torch.tensor(adjacency, dtype=torch.float32)


class LabelDependencyGraphLayer(nn.Module):
    """One message-passing step over the six toxicity labels.

    Each label starts from its own hidden vector (a per-label linear
    projection of the pooled RoBERTa representation). It is then updated
    using a weighted sum of the *other* labels' vectors, weighted by the
    fixed co-occurrence adjacency, before a final per-label linear scorer
    produces the logit.
    """

    def __init__(self, hidden_size: int, dep_dim: int, num_labels: int, adjacency: torch.Tensor):
        super().__init__()
        self.num_labels = num_labels
        self.dep_dim = dep_dim
        self.register_buffer("adjacency", adjacency)
        self.label_projection = nn.Linear(hidden_size, num_labels * dep_dim)
        self.self_transform = nn.Linear(dep_dim, dep_dim)
        self.neighbor_transform = nn.Linear(dep_dim, dep_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(dep_dim, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        batch_size = pooled.shape[0]
        label_hidden = self.label_projection(pooled).view(batch_size, self.num_labels, self.dep_dim)

        # For each label l: sum_j adjacency[l, j] * label_hidden[:, j, :]
        neighbor_hidden = torch.einsum("lj,bjd->bld", self.adjacency, label_hidden)

        updated = self.activation(
            self.self_transform(label_hidden) + self.neighbor_transform(neighbor_hidden)
        )
        logits = self.classifier(updated).squeeze(-1)  # [batch, num_labels]
        return logits


class _RobertaWithLabelDependency(nn.Module):
    """RoBERTa encoder -> [CLS] pooling -> LabelDependencyGraphLayer."""

    def __init__(self, pretrained_model_name: str, num_labels: int, dep_dim: int, adjacency: torch.Tensor):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dependency_layer = LabelDependencyGraphLayer(hidden_size, dep_dim, num_labels, adjacency)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        return self.dependency_layer(pooled)


class RobertaLabelDependencyClassifier(RobertaMultiLabelBase):
    """Method 3: RoBERTa encoder + explicit label-dependency graph layer, BCE loss.

    All constructor parameters are re-declared explicitly (rather than via
    ``**kwargs``) because scikit-learn's ``get_params``/``clone`` machinery
    introspects each estimator's own ``__init__`` signature — a subclass
    that swallows parent params into ``**kwargs`` breaks that contract.
    """

    def __init__(
        self,
        dep_dim: int = 64,
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
        self.dep_dim = dep_dim

    def _build_model(self, y: np.ndarray) -> nn.Module:
        adjacency = compute_cooccurrence_adjacency(y)
        return _RobertaWithLabelDependency(
            pretrained_model_name=self.pretrained_model_name,
            num_labels=self.num_labels,
            dep_dim=self.dep_dim,
            adjacency=adjacency,
        )
    # _compute_loss is inherited unchanged from the base class (plain BCE) —
    # this is deliberate: Method 3 isolates the dependency-layer variable only.


def build_roberta_label_dependency(
    pretrained_model_name: str = "roberta-base",
    dep_dim: int = 64,
    max_length: int = 128,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 2,
    device: str | None = None,
) -> RobertaLabelDependencyClassifier:
    """Factory matching the project's existing ``build_*`` model convention.

    ``device`` defaults to ``None``, which auto-detects GPU vs CPU inside
    ``fit`` (see ``_roberta_base.RobertaMultiLabelBase.fit``). Pass
    ``device="cuda"`` explicitly if you want fit() to raise immediately when
    no GPU is available, instead of silently falling back to a very slow
    CPU run.
    """

    return RobertaLabelDependencyClassifier(
        dep_dim=dep_dim,
        pretrained_model_name=pretrained_model_name,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device,
    )

